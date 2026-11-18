"""V14: capacity-constrained ML inference sensitivity experiment.

Run pilot: python experiment_v14.py
Run overnight: python experiment_v14.py --full
Does not overwrite V12/V13 outputs.
"""
import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

import joblib
import simpy

from experiment_v10 import generate_traffic
from experiment_v9 import OnlineFeatureExtractor
from experiment_v12 import SCENARIOS, TokenBucket

SIMULATION_END = 60.0
SERVER_CAPACITY = 100
QUEUE_LIMIT = 500
SERVICE_TIME = 0.1

# Sensitivity assumption based on V13 pilot's approximate 7.6 ms mean.
# NOT a calibrated real-world CPU model.
INFERENCE_SERVICE_TIME = 0.0076
INFERENCE_QUEUE_LIMIT = 500
WORKER_COUNTS = [1, 4, 16]
SEEDS = list(range(2026, 2036))
DEFENSES = ["NONE", "FIXED", "ADAPTIVE", "ML"]


def percentile95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def percent(num, den):
    return 100 * num / den if den else None


def run_one(scenario, defense, model, seed, workers=None):
    traffic = [
        request for request in generate_traffic(seed=seed, groups=SCENARIOS[scenario])
        if request["arrival_time"] <= SIMULATION_END
    ]
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=SERVER_CAPACITY)
    inference = simpy.Resource(env, capacity=workers) if defense == "ML" else None
    extractor = OnlineFeatureExtractor()
    buckets = {}
    stats = Counter()
    response_times = []
    inference_wait_times = []
    measured_predict_ms = []

    def handle(request):
        label = request["label"]
        arrival = request["arrival_time"]
        client_id = request["client_id"]
        kind = "legitimate" if label == 0 else "attack"
        stats[f"{kind}_total"] += 1
        features = extractor.observe(client_id, env.now)
        allowed = True

        if defense in ("FIXED", "ADAPTIVE"):
            if client_id not in buckets:
                buckets[client_id] = TokenBucket(rate=3.0, burst=10)
            bucket = buckets[client_id]
            if defense == "ADAPTIVE":
                load = (server.count + len(server.queue)) / (SERVER_CAPACITY + QUEUE_LIMIT)
                bucket.rate = 1.5 if load >= 0.5 else 3.0
                stats["adaptive_reduced" if load >= 0.5 else "adaptive_normal"] += 1
            allowed = bucket.allow(env.now)

        elif defense == "ML":
            # Bound the waiting queue separately from occupied inference workers.
            if inference.count + len(inference.queue) >= workers + INFERENCE_QUEUE_LIMIT:
                stats[f"{kind}_inference_overflow"] += 1
                return
            queued_at = env.now
            with inference.request() as ticket:
                yield ticket
                inference_wait_times.append(env.now - queued_at)
                row = [[
                    features["request_count"],
                    features["request_rate"],
                    features["mean_interval"],
                    features["interval_std"],
                ]]
                start = time.perf_counter_ns()
                prediction = model.predict(row)[0]
                measured_predict_ms.append((time.perf_counter_ns() - start) / 1e6)
                stats["ml_predictions"] += 1
                yield env.timeout(INFERENCE_SERVICE_TIME)
                allowed = prediction == 0

        if not allowed:
            stats[f"{kind}_defense_rejected"] += 1
            return

        if server.count + len(server.queue) >= SERVER_CAPACITY + QUEUE_LIMIT:
            stats[f"{kind}_capacity_rejected"] += 1
            return

        with server.request() as ticket:
            yield ticket
            yield env.timeout(SERVICE_TIME)
            stats[f"{kind}_completed"] += 1
            if label == 0:
                response_times.append(env.now - arrival)

    def arrivals():
        for request in traffic:
            yield env.timeout(max(0.0, request["arrival_time"] - env.now))
            env.process(handle(request))

    env.process(arrivals())
    env.run()  # drain all admitted work after the 60 s arrival horizon

    legit = stats["legitimate_total"]
    attack = stats["attack_total"]
    # Integrity: every arrival must have exactly one terminal outcome.
    for kind in ("legitimate", "attack"):
        accounted = sum(stats[f"{kind}_{suffix}"] for suffix in (
            "completed", "defense_rejected", "capacity_rejected", "inference_overflow"
        ))
        assert accounted == stats[f"{kind}_total"], (scenario, defense, seed, kind, accounted)

    return {
        "scenario": scenario,
        "seed": seed,
        "defense": defense,
        "ml_workers": workers if defense == "ML" else "",
        "inference_service_ms_assumed": INFERENCE_SERVICE_TIME * 1000 if defense == "ML" else "",
        "completion_pct": percent(stats["legitimate_completed"], legit),
        "legitimate_defense_rejected_pct": percent(stats["legitimate_defense_rejected"], legit),
        "legitimate_capacity_rejected_pct": percent(stats["legitimate_capacity_rejected"], legit),
        "legitimate_inference_overflow_pct": percent(stats["legitimate_inference_overflow"], legit),
        "attack_defense_rejected_pct": percent(stats["attack_defense_rejected"], attack),
        "attack_capacity_rejected_pct": percent(stats["attack_capacity_rejected"], attack),
        "attack_inference_overflow_pct": percent(stats["attack_inference_overflow"], attack),
        "p95_legitimate_response_s": percentile95(response_times),
        "p95_inference_wait_s": percentile95(inference_wait_times),
        "mean_measured_predict_ms": statistics.mean(measured_predict_ms) if measured_predict_ms else None,
        "legitimate_total": legit,
        "legitimate_completed": stats["legitimate_completed"],
        "attack_total": attack,
        "ml_predictions": stats["ml_predictions"],
        "adaptive_reduced": stats["adaptive_reduced"],
        "adaptive_normal": stats["adaptive_normal"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="10 seeds instead of 1")
    args = parser.parse_args()
    seeds = SEEDS if args.full else SEEDS[:1]
    output = Path("v14_full_results.csv" if args.full else "v14_pilot_results.csv")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing {output}; rename it first.")

    model = joblib.load("random_forest_model.joblib")
    model.n_jobs = 1
    configs = [(d, None) for d in DEFENSES if d != "ML"] + [("ML", n) for n in WORKER_COUNTS]
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for defense, workers in configs:
                print(f"Running {scenario} / {defense} / workers={workers} / seed={seed}", flush=True)
                rows.append(run_one(scenario, defense, model, seed, workers))

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

        print("\n--- V14 SUMMARY (mean over seeds) ---")
    for scenario in SCENARIOS:
        for defense, workers in configs:
            subset = [
                r for r in rows
                if r["scenario"] == scenario
                and r["defense"] == defense
                and (defense != "ML" or r["ml_workers"] == workers)
            ]

            def avg(key):
                vals = [r[key] for r in subset if r[key] is not None]
                return f"{statistics.mean(vals):.3f}" if vals else "N/A"

            print(
                f"{scenario:12} {defense:8} workers={str(workers):4} "
                f"completion={avg('completion_pct')}% "
                f"inference_overflow={avg('legitimate_inference_overflow_pct')}%"
            )


if __name__ == "__main__":
    main()
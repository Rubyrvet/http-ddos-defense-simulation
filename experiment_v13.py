
import csv
import random
import statistics
import time
from collections import Counter

import joblib
import simpy

from experiment_v10 import generate_traffic
from experiment_v9 import OnlineFeatureExtractor
from experiment_v12 import SCENARIOS, TokenBucket


SIMULATION_END = 60.0
SERVER_CAPACITY = 100
QUEUE_LIMIT = 500
SERVICE_TIME = 0.1

DEFENSES = ["NONE", "FIXED", "ADAPTIVE", "ML"]

# First run a small pilot before repeating all experiments.
PILOT_SEEDS = [2026]

# Sensitivity analysis: additional simulated ML decision delays.
# These are hypothetical delays, NOT measured CPU overhead.
ML_DELAYS = [0.0, 0.001, 0.005]


def percentile_95(values):
    if not values:
        return None

    ordered = sorted(values)
    index = int(0.95 * (len(ordered) - 1))
    return ordered[index]


def run_experiment_v13(scenario, defense, model, seed, ml_delay=0.0):
    traffic = generate_traffic(seed=seed, groups=SCENARIOS[scenario])
    traffic = [
        request for request in traffic
        if request["arrival_time"] <= SIMULATION_END
    ]

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=SERVER_CAPACITY)

    extractor = OnlineFeatureExtractor()
    buckets = {}

    stats = Counter()
    response_times = []
    inference_times_ms = []

    def handle_request(request):
        arrival = request["arrival_time"]
        client_id = request["client_id"]
        label = request["label"]

        if label == 0:
            stats["legitimate_total"] += 1
        else:
            stats["attack_total"] += 1

        features = extractor.observe(client_id, env.now)
        allowed = True

        if defense in ("FIXED", "ADAPTIVE"):
            if client_id not in buckets:
                buckets[client_id] = TokenBucket(
                    rate=3.0,
                    burst=10
                )

            bucket = buckets[client_id]

            if defense == "ADAPTIVE":
                load = (
                    server.count + len(server.queue)
                ) / (SERVER_CAPACITY + QUEUE_LIMIT)

                if load >= 0.5:
                    bucket.rate = 1.5
                    stats["adaptive_reduced_decisions"] += 1
                else:
                    bucket.rate = 3.0
                    stats["adaptive_normal_decisions"] += 1

            allowed = bucket.allow(env.now)

        elif defense == "ML":
            row = [[
                features["request_count"],
                features["request_rate"],
                features["mean_interval"],
                features["interval_std"]
            ]]

            start_ns = time.perf_counter_ns()
            prediction = model.predict(row)[0]
            elapsed_ms = (
                time.perf_counter_ns() - start_ns
            ) / 1_000_000

            inference_times_ms.append(elapsed_ms)
            stats["ml_predictions"] += 1

            # Simulated decision delay, applied to every ML request.
            # This is not the measured wall-clock inference time.
            if ml_delay > 0:
                yield env.timeout(ml_delay)

            allowed = prediction == 0

        if not allowed:
            if label == 0:
                stats["legitimate_defense_rejected"] += 1
            else:
                stats["attack_defense_rejected"] += 1
            return

        if (
            server.count + len(server.queue)
            >= SERVER_CAPACITY + QUEUE_LIMIT
        ):
            if label == 0:
                stats["legitimate_capacity_rejected"] += 1
            else:
                stats["attack_capacity_rejected"] += 1
            return

        with server.request() as resource_request:
            yield resource_request
            yield env.timeout(SERVICE_TIME)

            response_time = env.now - arrival

            if label == 0:
                stats["legitimate_completed"] += 1
                response_times.append(response_time)
            else:
                stats["attack_completed"] += 1

    def arrival_process():
        for request in traffic:
            wait = max(
                0.0,
                request["arrival_time"] - env.now
            )
            yield env.timeout(wait)
            env.process(handle_request(request))

    env.process(arrival_process())
    env.run()

    legitimate_total = stats["legitimate_total"]
    attack_total = stats["attack_total"]

    def percent(numerator, denominator):
        return (
            100.0 * numerator / denominator
            if denominator else None
        )

    return {
        "scenario": scenario,
        "seed": seed,
        "defense": defense,
        "ml_delay_s": ml_delay if defense == "ML" else 0.0,
        "completion": percent(
            stats["legitimate_completed"], legitimate_total
        ),
        "legitimate_defense_rejection": percent(
            stats["legitimate_defense_rejected"], legitimate_total
        ),
        "legitimate_capacity_rejection": percent(
            stats["legitimate_capacity_rejected"], legitimate_total
        ),
        "attack_defense_rejection": percent(
            stats["attack_defense_rejected"], attack_total
        ),
        "attack_capacity_rejection": percent(
            stats["attack_capacity_rejected"], attack_total
        ),
        "p95": percentile_95(response_times),
        "legitimate_total": legitimate_total,
        "legitimate_completed": stats["legitimate_completed"],
        "adaptive_reduced_decisions": stats[
            "adaptive_reduced_decisions"
        ],
        "adaptive_normal_decisions": stats[
            "adaptive_normal_decisions"
        ],
        "ml_predictions": stats["ml_predictions"],
        "mean_inference_ms": (
            statistics.mean(inference_times_ms)
            if inference_times_ms else None
        ),
        "p95_inference_ms": (
            percentile_95(inference_times_ms)
            if inference_times_ms else None
        )
    }


def main():
    model = joblib.load("random_forest_model.joblib")
    model.n_jobs = 1

    results = []

    for scenario in SCENARIOS:
        for seed in PILOT_SEEDS:
            for defense in DEFENSES:
                delays = (
                    ML_DELAYS if defense == "ML" else [0.0]
                )

                for delay in delays:
                    print(
                        f"Running {scenario} / {defense} / "
                        f"seed={seed} / ML delay={delay}",
                        flush=True
                    )

                    result = run_experiment_v13(
                        scenario, defense, model, seed, delay
                    )
                    results.append(result)

    output_file = "v13_pilot_results.csv"

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n--- V13 PILOT SUMMARY ---")

    for result in results:
        print(
            f"{result['scenario']:12} "
            f"{result['defense']:8} "
            f"delay={result['ml_delay_s']:.3f}s | "
            f"completion={result['completion']:.3f}% | "
            f"capacity_reject="
            f"{result['legitimate_capacity_rejection']:.3f}% | "
            f"adaptive_reduced="
            f"{result['adaptive_reduced_decisions']}"
        )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()

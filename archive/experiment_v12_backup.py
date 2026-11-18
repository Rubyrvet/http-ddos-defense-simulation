
import random
import statistics
from collections import deque

import joblib
import simpy

from experiment_v10 import generate_traffic
from experiment_v9 import OnlineFeatureExtractor


SIMULATION_END = 60.0
SERVER_CAPACITY = 100
QUEUE_LIMIT = 500
SERVICE_TIME = 0.1

SCENARIOS = {
    "NORMAL": [
        ("normal", 200, 0.5, 0)
    ],
    "FLASH_CROWD": [
        ("flash", 400, 2.0, 0)
    ],
    "ATTACK": [
        ("normal", 200, 0.5, 0),
        ("attack", 40, 12.0, 1)
    ],
    "COMBINED": [
        ("flash", 400, 2.0, 0),
        ("attack", 40, 12.0, 1)
    ]
}


class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_time = 0.0

    def allow(self, now):
        elapsed = max(0.0, now - self.last_time)
        self.tokens = min(
            self.burst,
            self.tokens + elapsed * self.rate
        )
        self.last_time = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True

        return False


def percentile_95(values):
    if not values:
        return None

    ordered = sorted(values)
    index = int(0.95 * (len(ordered) - 1))
    return ordered[index]


def run_experiment(scenario, defense, model):
    traffic = generate_traffic(
        seed=2026,
        groups=SCENARIOS[scenario]
    )

    # Use only requests inside the simulation horizon
    traffic = [
        request for request in traffic
        if request["arrival_time"] <= SIMULATION_END
    ]

    env = simpy.Environment()
    server = simpy.Resource(
        env,
        capacity=SERVER_CAPACITY
    )

    extractor = OnlineFeatureExtractor()
    buckets = {}

    results = {
        "legitimate_total": 0,
        "legitimate_completed": 0,
        "legitimate_defense_rejected": 0,
        "legitimate_capacity_rejected": 0,
        "attack_total": 0,
        "attack_defense_rejected": 0,
        "attack_completed": 0,
        "response_times": []
    }

    def handle_request(request):
        arrival = request["arrival_time"]
        client_id = request["client_id"]
        label = request["label"]

        if label == 0:
            results["legitimate_total"] += 1
        else:
            results["attack_total"] += 1

        # Observe the current request before predicting
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

                bucket.rate = (
                    3.0 if load < 0.5 else 1.5
                )

            allowed = bucket.allow(env.now)

        elif defense == "ML":
            row = [[
                features["request_count"],
                features["request_rate"],
                features["mean_interval"],
                features["interval_std"]
            ]]

            prediction = model.predict(row)[0]
            allowed = prediction == 0

        if not allowed:
            if label == 0:
                results["legitimate_defense_rejected"] += 1
            else:
                results["attack_defense_rejected"] += 1
            return

        # Finite-capacity server
        if (
            server.count + len(server.queue)
            >= SERVER_CAPACITY + QUEUE_LIMIT
        ):
            if label == 0:
                results["legitimate_capacity_rejected"] += 1
            return

        with server.request() as resource_request:
            yield resource_request

            yield env.timeout(SERVICE_TIME)

            response_time = env.now - arrival

            if label == 0:
                results["legitimate_completed"] += 1
                results["response_times"].append(
                    response_time
                )
            else:
                results["attack_completed"] += 1

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

    legitimate_total = results["legitimate_total"]
    attack_total = results["attack_total"]

    completion = (
        100 * results["legitimate_completed"]
        / legitimate_total
        if legitimate_total else 0
    )

    false_positive = (
        100 * results["legitimate_defense_rejected"]
        / legitimate_total
        if legitimate_total else 0
    )

    attack_blocked = (
        100 * results["attack_defense_rejected"]
        / attack_total
        if attack_total else 0
    )

    p95 = percentile_95(
        results["response_times"]
    )

    return {
        "completion": completion,
        "false_positive": false_positive,
        "attack_blocked": attack_blocked,
        "p95": p95,
        "legitimate_total": legitimate_total,
        "legitimate_completed": results[
            "legitimate_completed"
        ]
    }


def main():
    model = joblib.load(
        "random_forest_model.joblib"
    )
    model.n_jobs = 1

    defenses = [
        "NONE",
        "FIXED",
        "ADAPTIVE",
        "ML"
    ]

    print("\n--- EXPERIMENT V12 RESULTS ---")

    for scenario in SCENARIOS:
        print(f"\nSCENARIO: {scenario}")

        for defense in defenses:
            result = run_experiment(
                scenario,
                defense,
                model
            )

            p95_text = (
                f"{result['p95']:.4f}s"
                if result["p95"] is not None
                else "N/A"
            )

            print(
                f"{defense:8} | "
                f"Completion: {result['completion']:.2f}% | "
                f"False Positive: {result['false_positive']:.2f}% | "
                f"Attack Blocked: {result['attack_blocked']:.2f}% | "
                f"P95: {p95_text}"
            )


if __name__ == "__main__":
    main()

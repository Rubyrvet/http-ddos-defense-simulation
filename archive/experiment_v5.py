
import simpy
import statistics

from experiment_v3 import (
    generate_requests,
    SERVER_CAPACITY,
    QUEUE_LIMIT,
    PROCESSING_TIME
)

from experiment_v4 import TokenBucket

# Adaptive rate limiting settings
LOW_RATE = 500
MEDIUM_RATE = 800
HIGH_RATE = 1200

BUCKET_CAPACITY = 800


def calculate_adaptive_rate(server):
    # Current server load
    active_ratio = server.count / SERVER_CAPACITY
    queue_ratio = len(server.queue) / QUEUE_LIMIT

    load = max(active_ratio, queue_ratio)

    if load >= 0.8:
        return LOW_RATE
    elif load >= 0.5:
        return MEDIUM_RATE
    else:
        return HIGH_RATE


def handle_request(env, server, request, results):
    request_class = request["request_class"]
    arrival_time = env.now

    if (
        server.count >= SERVER_CAPACITY
        and len(server.queue) >= QUEUE_LIMIT
    ):
        results[request_class]["capacity_rejected"] += 1
        return

    with server.request() as slot:
        yield slot

        yield env.timeout(PROCESSING_TIME)

        results[request_class]["completed"] += 1
        results[request_class]["response_times"].append(
            env.now - arrival_time
        )


def send_requests(env, server, requests, results, defense):
    limiter = TokenBucket(800, BUCKET_CAPACITY)

    for request in requests:
        yield env.timeout(
            max(0, request["arrival_time"] - env.now)
        )

        request_class = request["request_class"]

        if defense == "fixed":
            limiter.rate = 800

        elif defense == "adaptive":
            limiter.rate = calculate_adaptive_rate(server)

        if defense != "none":
            if not limiter.allow(env.now):
                results[request_class]["defense_rejected"] += 1
                continue

        env.process(
            handle_request(env, server, request, results)
        )


def run_experiment(scenario, defense):
    requests = generate_requests(scenario)

    env = simpy.Environment()

    server = simpy.Resource(
        env,
        capacity=SERVER_CAPACITY
    )

    results = {
        "legitimate": {
            "completed": 0,
            "capacity_rejected": 0,
            "defense_rejected": 0,
            "response_times": []
        },
        "attack": {
            "completed": 0,
            "capacity_rejected": 0,
            "defense_rejected": 0,
            "response_times": []
        }
    }

    env.process(
        send_requests(
            env, server, requests, results, defense
        )
    )

    env.run()

    legitimate_total = sum(
        r["request_class"] == "legitimate"
        for r in requests
    )

    attack_total = len(requests) - legitimate_total

    legitimate = results["legitimate"]
    attack = results["attack"]

    completion_rate = (
        legitimate["completed"] / legitimate_total * 100
        if legitimate_total else 0
    )

    false_positive_rate = (
        legitimate["defense_rejected"] / legitimate_total * 100
        if legitimate_total else 0
    )

    attack_block_rate = (
        attack["defense_rejected"] / attack_total * 100
        if attack_total else 0
    )

    avg_response = (
        statistics.mean(legitimate["response_times"])
        if legitimate["response_times"] else 0
    )

    return {
        "scenario": scenario,
        "defense": defense,
        "completion_rate": completion_rate,
        "false_positive_rate": false_positive_rate,
        "attack_block_rate": attack_block_rate,
        "avg_response": avg_response,
        "legitimate_completed": legitimate["completed"],
        "legitimate_rejected": (
            legitimate["capacity_rejected"]
            + legitimate["defense_rejected"]
        )
    }


def main():
    scenarios = [
        "normal",
        "flash_crowd",
        "attack",
        "combined"
    ]

    defenses = ["none", "fixed", "adaptive"]

    print("\n--- EXPERIMENT V5 RESULTS ---")

    for scenario in scenarios:
        print(f"\nSCENARIO: {scenario.upper()}")

        for defense in defenses:
            result = run_experiment(scenario, defense)

            print(
                f"{defense.upper():10} | "
                f"Completion: {result['completion_rate']:.2f}% | "
                f"False Positive: {result['false_positive_rate']:.2f}% | "
                f"Attack Blocked: {result['attack_block_rate']:.2f}% | "
                f"Avg Response: {result['avg_response']:.4f}s"
            )


if __name__ == "__main__":
    main()

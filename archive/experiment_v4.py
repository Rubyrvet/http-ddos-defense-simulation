
import simpy
import statistics

from experiment_v3 import (
    generate_requests,
    SERVER_CAPACITY,
    QUEUE_LIMIT,
    PROCESSING_TIME
)

# Fixed Rate Limiting parameters
RATE_LIMIT = 800       # tokens per second
BUCKET_CAPACITY = 800  # maximum burst size


class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = 0.0

    def allow(self, current_time):
        elapsed = current_time - self.last_update

        # Refill tokens over time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )

        self.last_update = current_time

        if self.tokens >= 1:
            self.tokens -= 1
            return True

        return False


def handle_request(env, server, request, results):
    request_class = request["request_class"]
    arrival_time = env.now

    # Server capacity rejection
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
    limiter = TokenBucket(
        RATE_LIMIT,
        BUCKET_CAPACITY
    )

    for request in requests:
        yield env.timeout(
            max(0, request["arrival_time"] - env.now)
        )

        request_class = request["request_class"]

        # Apply rate limiting before the server
        if defense == "fixed":
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

    legitimate_rate = (
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

    print(f"\n--- {scenario.upper()} | {defense.upper()} ---")
    print("Legitimate requests:", legitimate_total)
    print("Legitimate completed:", legitimate["completed"])
    print("Legitimate completion rate:", round(legitimate_rate, 2), "%")
    print("Legitimate capacity rejected:", legitimate["capacity_rejected"])
    print("Legitimate defense rejected:", legitimate["defense_rejected"])
    print("Defense false positive rate:", round(false_positive_rate, 2), "%")
    print("Attack requests:", attack_total)
    print("Attack completed:", attack["completed"])
    print("Attack defense rejected:", attack["defense_rejected"])
    print("Attack blocking rate:", round(attack_block_rate, 2), "%")
    print("Average legitimate response time:", round(avg_response, 4))

    assert (
        legitimate["completed"]
        + legitimate["capacity_rejected"]
        + legitimate["defense_rejected"]
        == legitimate_total
    ), "Legitimate request counts do not match!"

    assert (
        attack["completed"]
        + attack["capacity_rejected"]
        + attack["defense_rejected"]
        == attack_total
    ), "Attack request counts do not match!"


def main():
    scenarios = [
        "normal",
        "flash_crowd",
        "attack",
        "combined"
    ]

    for scenario in scenarios:
        for defense in ["none", "fixed"]:
            run_experiment(scenario, defense)


if __name__ == "__main__":
    main()

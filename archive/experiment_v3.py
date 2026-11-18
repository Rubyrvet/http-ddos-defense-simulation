
import random
import statistics
import simpy

# Simulation parameters
SIMULATION_TIME = 60
SERVER_CAPACITY = 100
QUEUE_LIMIT = 500
PROCESSING_TIME = 0.1

NORMAL_RATE = 300
FLASH_RATE = 1500
ATTACK_RATE = 1000

FLASH_START = 20
FLASH_END = 40

RANDOM_SEED = 42


def generate_requests(scenario):
    rng = random.Random(RANDOM_SEED)
    requests = []
    request_id = 0

    periods = [
        (0, 20),
        (20, 40),
        (40, 60)
    ]

    for start, end in periods:

        legitimate_rate = NORMAL_RATE
        attack_rate = 0

        if scenario in ("flash_crowd", "combined"):
            if start == FLASH_START:
                legitimate_rate = FLASH_RATE

        if scenario in ("attack", "combined"):
            if start == FLASH_START:
                attack_rate = ATTACK_RATE

        for request_class, rate in [
            ("legitimate", legitimate_rate),
            ("attack", attack_rate)
        ]:
            if rate == 0:
                continue

            current_time = start

            while True:
                current_time += rng.expovariate(rate)

                if current_time >= end:
                    break

                request_id += 1

                requests.append({
                    "id": request_id,
                    "arrival_time": current_time,
                    "request_class": request_class
                })

    requests.sort(key=lambda x: x["arrival_time"])

    return requests


def handle_request(env, server, request, results):
    request_class = request["request_class"]
    arrival_time = env.now

    if (
        server.count >= SERVER_CAPACITY
        and len(server.queue) >= QUEUE_LIMIT
    ):
        results[request_class]["rejected"] += 1
        return

    with server.request() as slot:
        yield slot

        yield env.timeout(PROCESSING_TIME)

        results[request_class]["completed"] += 1

        response_time = env.now - arrival_time

        results[request_class]["response_times"].append(
            response_time
        )


def send_requests(env, server, requests, results):
    for request in requests:
        yield env.timeout(
            max(0, request["arrival_time"] - env.now)
        )

        env.process(
            handle_request(env, server, request, results)
        )


def run_experiment(scenario):
    requests = generate_requests(scenario)

    env = simpy.Environment()

    server = simpy.Resource(
        env,
        capacity=SERVER_CAPACITY
    )

    results = {
        "legitimate": {
            "completed": 0,
            "rejected": 0,
            "response_times": []
        },
        "attack": {
            "completed": 0,
            "rejected": 0,
            "response_times": []
        }
    }

    env.process(
        send_requests(env, server, requests, results)
    )

    env.run()

    legitimate_total = sum(
        1 for r in requests
        if r["request_class"] == "legitimate"
    )

    legitimate_completed = results["legitimate"]["completed"]
    legitimate_rejected = results["legitimate"]["rejected"]

    completion_rate = (
        legitimate_completed / legitimate_total * 100
        if legitimate_total else 0
    )

    response_times = results["legitimate"]["response_times"]

    avg_response = (
        statistics.mean(response_times)
        if response_times else 0
    )

    print(f"\n--- {scenario.upper()} ---")
    print("Total requests:", len(requests))
    print("Legitimate requests:", legitimate_total)
    print("Legitimate completed:", legitimate_completed)
    print("Legitimate rejected:", legitimate_rejected)
    print("Legitimate completion rate:", round(completion_rate, 2), "%")
    print("Attack completed:", results["attack"]["completed"])
    print("Attack rejected:", results["attack"]["rejected"])
    print("Average legitimate response time:", round(avg_response, 4))


def main():
    scenarios = [
        "normal",
        "flash_crowd",
        "attack",
        "combined"
    ]

    for scenario in scenarios:
        run_experiment(scenario)


if __name__ == "__main__":
    main()

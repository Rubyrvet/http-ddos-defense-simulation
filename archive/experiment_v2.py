
import random
import simpy

# Experiment settings
RANDOM_SEED = 42
SIMULATION_TIME = 60

NORMAL_RATE = 300
FLASH_RATE = 1500

FLASH_START = 20
FLASH_END = 40

SERVER_CAPACITY = 100
QUEUE_LIMIT = 500
PROCESSING_TIME = 0.1


def generate_traffic():
    rng = random.Random(RANDOM_SEED)
    traffic = []
    request_id = 0

    # Each period has its own arrival rate
    periods = [
        (0, FLASH_START, NORMAL_RATE),
        (FLASH_START, FLASH_END, FLASH_RATE),
        (FLASH_END, SIMULATION_TIME, NORMAL_RATE)
    ]

    for start, end, rate in periods:
        current_time = start

        while True:
            current_time += rng.expovariate(rate)

            if current_time >= end:
                break

            request_id += 1

            traffic.append({
                "id": request_id,
                "arrival_time": current_time,
                "request_class": "legitimate",
                "scenario": (
                    "flash_crowd"
                    if start == FLASH_START
                    else "normal"
                )
            })

    return traffic


def handle_request(env, server, request, results):
    arrival_time = env.now

    # Reject requests when the server and queue are full
    if (
        server.count >= SERVER_CAPACITY
        and len(server.queue) >= QUEUE_LIMIT
    ):
        results["rejected"] += 1
        return

    with server.request() as slot:
        yield slot

        results["waiting_times"].append(
            env.now - arrival_time
        )

        yield env.timeout(PROCESSING_TIME)

        results["completed"] += 1

        results["response_times"].append(
            env.now - arrival_time
        )


def send_requests(env, server, traffic, results):
    for request in traffic:
        yield env.timeout(
            max(0, request["arrival_time"] - env.now)
        )

        env.process(
            handle_request(env, server, request, results)
        )


def main():
    traffic = generate_traffic()

    env = simpy.Environment()
    server = simpy.Resource(
        env,
        capacity=SERVER_CAPACITY
    )

    results = {
        "completed": 0,
        "rejected": 0,
        "waiting_times": [],
        "response_times": []
    }

    env.process(
        send_requests(env, server, traffic, results)
    )

    # Allow accepted requests to finish
    env.run()

    total = len(traffic)

    print("\n--- EXPERIMENT V2 RESULTS ---")
    print("Total requests:", total)
    print("Completed:", results["completed"])
    print("Rejected:", results["rejected"])

    if total:
        print(
            "Completion rate:",
            round(results["completed"] / total * 100, 2),
            "%"
        )

    if results["response_times"]:
        response_times = results["response_times"]

        print(
            "Average response time:",
            round(sum(response_times) / len(response_times), 4)
        )

        print(
            "Maximum response time:",
            round(max(response_times), 4)
        )


if __name__ == "__main__":
    main()

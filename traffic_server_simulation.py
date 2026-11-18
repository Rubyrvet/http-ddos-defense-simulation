
import simpy
import statistics

from traffic_generator import generate_traffic

SERVER_CAPACITY = 2
QUEUE_LIMIT = 3
PROCESSING_TIME = 2


def handle_request(env, server, request, results):
    arrival_time = env.now

    if (
        server.count >= SERVER_CAPACITY
        and len(server.queue) >= QUEUE_LIMIT
    ):
        results["rejected"] += 1
        return

    with server.request() as req:
        yield req

        waiting_time = env.now - arrival_time

        yield env.timeout(PROCESSING_TIME)

        response_time = env.now - arrival_time

        results["completed"] += 1
        results["waiting_times"].append(waiting_time)
        results["response_times"].append(response_time)


def send_requests(env, server, traffic, results):
    for request in traffic:
        delay = request["arrival_time"] - env.now

        yield env.timeout(max(0, delay))

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

    env.run()

    print("\n--- TRAFFIC SERVER RESULTS ---")
    print("Total requests:", len(traffic))
    print("Completed:", results["completed"])
    print("Rejected:", results["rejected"])

    if results["response_times"]:
        print(
            "Average waiting time:",
            round(statistics.mean(results["waiting_times"]), 2)
        )
        print(
            "Average response time:",
            round(statistics.mean(results["response_times"]), 2)
        )


if __name__ == "__main__":
    main()

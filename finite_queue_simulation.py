
import simpy

# Server settings
SERVER_CAPACITY = 2
QUEUE_LIMIT = 3
PROCESSING_TIME = 2

results = {
    "completed": 0,
    "rejected": 0,
    "waiting_times": [],
    "response_times": []
}


def handle_request(env, server, request_id):
    arrival_time = env.now

    # Reject if both processing slots and queue are full
    if server.count >= SERVER_CAPACITY and len(server.queue) >= QUEUE_LIMIT:
        results["rejected"] += 1
        print(f"Request {request_id}: REJECTED at {env.now}")
        return

    with server.request() as req:
        yield req

        waiting_time = env.now - arrival_time
        results["waiting_times"].append(waiting_time)

        print(f"Request {request_id}: START at {env.now}")

        yield env.timeout(PROCESSING_TIME)

        response_time = env.now - arrival_time
        results["response_times"].append(response_time)
        results["completed"] += 1

        print(f"Request {request_id}: COMPLETED at {env.now}")


def main():
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=SERVER_CAPACITY)

    # All 8 requests arrive at the same time
    for request_id in range(1, 9):
        env.process(handle_request(env, server, request_id))

    env.run()

    print("\n--- SIMULATION RESULTS ---")
    print("Completed:", results["completed"])
    print("Rejected:", results["rejected"])

    if results["waiting_times"]:
        avg_wait = sum(results["waiting_times"]) / len(results["waiting_times"])
        avg_response = sum(results["response_times"]) / len(results["response_times"])

        print("Average waiting time:", round(avg_wait, 2))
        print("Average response time:", round(avg_response, 2))


if __name__ == "__main__":
    main()

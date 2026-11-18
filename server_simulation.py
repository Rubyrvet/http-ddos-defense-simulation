
import simpy
import statistics

# Store simulation results
results = []

def handle_request(env, server, request_id):
    arrival_time = env.now
    print(f"Time {env.now}: Request {request_id} arrived")

    with server.request() as req:
        yield req

        start_time = env.now
        waiting_time = start_time - arrival_time

        print(f"Time {env.now}: Request {request_id} started")

        # Each request needs 2 time units
        yield env.timeout(2)

        completion_time = env.now
        response_time = completion_time - arrival_time

        print(f"Time {env.now}: Request {request_id} completed")

        results.append({
            "request_id": request_id,
            "waiting_time": waiting_time,
            "response_time": response_time
        })


# Create simulation environment
env = simpy.Environment()

# Server can process two request at a time
server = simpy.Resource(env, capacity=2)

# Generate five requests at time 0
for i in range(1, 6):
    env.process(handle_request(env, server, i))

# Run simulation
env.run()

# Calculate performance metrics
average_waiting = statistics.mean(
    r["waiting_time"] for r in results
)

average_response = statistics.mean(
    r["response_time"] for r in results
)

print("\n--- SIMULATION RESULTS ---")
print(f"Total completed requests: {len(results)}")
print(f"Average waiting time: {average_waiting:.2f}")
print(f"Average response time: {average_response:.2f}")

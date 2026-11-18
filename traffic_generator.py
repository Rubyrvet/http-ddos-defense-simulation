
import random

# Simulation settings
SIMULATION_TIME = 60
RANDOM_SEED = 42

# Average requests per time unit
NORMAL_RATE = 2
FLASH_CROWD_RATE = 10

# Flash crowd occurs between time 20 and 40
FLASH_START = 20
FLASH_END = 40


def generate_traffic():
    random.seed(RANDOM_SEED)

    requests = []
    request_id = 1
    current_time = 0.0

    while current_time < SIMULATION_TIME:

        # Select traffic intensity
        if FLASH_START <= current_time < FLASH_END:
            rate = FLASH_CROWD_RATE
            traffic_type = "flash_crowd"
        else:
            rate = NORMAL_RATE
            traffic_type = "normal"

        # Generate time until next request
        interarrival_time = random.expovariate(rate)
        next_time = current_time + interarrival_time

        if next_time >= SIMULATION_TIME:
            break

        # Keep the flash crowd boundary exact
        if current_time < FLASH_START <= next_time:
            current_time = FLASH_START
            continue

        if current_time < FLASH_END <= next_time:
            current_time = FLASH_END
            continue

        current_time = next_time

        requests.append({
            "id": request_id,
            "arrival_time": round(current_time, 4),
            "traffic_type": traffic_type
        })

        request_id += 1

    return requests


if __name__ == "__main__":
    traffic = generate_traffic()

    print("--- TRAFFIC GENERATION RESULTS ---")
    print("Total requests:", len(traffic))

    normal_count = sum(
        1 for request in traffic
        if request["traffic_type"] == "normal"
    )

    flash_count = sum(
        1 for request in traffic
        if request["traffic_type"] == "flash_crowd"
    )

    print("Normal requests:", normal_count)
    print("Flash crowd requests:", flash_count)

    print("\nFirst 10 requests:")

    for request in traffic[:10]:
        print(request)

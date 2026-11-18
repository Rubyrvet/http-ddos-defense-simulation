
import csv
import random
import statistics

RANDOM_SEED = 42
SIMULATION_TIME = 60

NORMAL_CLIENTS = 200
FLASH_CLIENTS = 600
ATTACK_CLIENTS = 40

# Requests per second, per client
NORMAL_CLIENT_RATE = 0.5
FLASH_CLIENT_RATE = 0.5
ATTACK_CLIENT_RATE = 25.0

WINDOW_SIZE = 5.0


def create_client_requests(rng, client_id, rate, label):
    requests = []
    current_time = 0.0

    while True:
        current_time += rng.expovariate(rate)

        if current_time >= SIMULATION_TIME:
            break

        requests.append({
            "client_id": client_id,
            "arrival_time": current_time,
            "label": label
        })

    return requests


def generate_dataset():
    rng = random.Random(RANDOM_SEED)
    requests = []

    groups = [
        ("normal", NORMAL_CLIENTS, NORMAL_CLIENT_RATE, 0),
        ("flash", FLASH_CLIENTS, FLASH_CLIENT_RATE, 0),
        ("attack", ATTACK_CLIENTS, ATTACK_CLIENT_RATE, 1)
    ]

    for group_name, count, rate, label in groups:
        for i in range(count):
            client_id = f"{group_name}_{i}"

            requests.extend(
                create_client_requests(
                    rng, client_id, rate, label
                )
            )

    requests.sort(
        key=lambda r: r["arrival_time"]
    )

    return requests


def extract_features(requests):
    # Group requests by client
    clients = {}

    for request in requests:
        client_id = request["client_id"]

        clients.setdefault(client_id, []).append(
            request["arrival_time"]
        )

    dataset = []

    for client_id, times in clients.items():
        times.sort()

        # Use the first observation window only
        observed = [
            t for t in times
            if t < WINDOW_SIZE
        ]

        if len(observed) < 2:
            continue

        intervals = [
            observed[i] - observed[i - 1]
            for i in range(1, len(observed))
        ]

        request_count = len(observed)

        request_rate = request_count / WINDOW_SIZE

        mean_interval = statistics.mean(intervals)

        interval_std = (
            statistics.pstdev(intervals)
            if len(intervals) > 1
            else 0.0
        )

        label = (
            1 if client_id.startswith("attack_")
            else 0
        )

        dataset.append({
            "client_id": client_id,
            "request_count": request_count,
            "request_rate": round(request_rate, 4),
            "mean_interval": round(mean_interval, 6),
            "interval_std": round(interval_std, 6),
            "label": label
        })

    return dataset


def save_dataset(dataset):
    filename = "synthetic_client_dataset.csv"

    columns = [
        "client_id",
        "request_count",
        "request_rate",
        "mean_interval",
        "interval_std",
        "label"
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(dataset)

    return filename


def main():
    requests = generate_dataset()
    dataset = extract_features(requests)

    filename = save_dataset(dataset)

    legitimate = sum(
        row["label"] == 0
        for row in dataset
    )

    attack = sum(
        row["label"] == 1
        for row in dataset
    )

    print("\n--- EXPERIMENT V6 RESULTS ---")
    print("Total generated requests:", len(requests))
    print("Total clients with features:", len(dataset))
    print("Legitimate clients:", legitimate)
    print("Attack clients:", attack)
    print("Dataset saved as:", filename)

    print("\nFirst 5 rows:")
    for row in dataset[:5]:
        print(row)


if __name__ == "__main__":
    main()

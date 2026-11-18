
from collections import defaultdict, deque
import statistics

WINDOW_SIZE = 5.0


class OnlineFeatureExtractor:
    def __init__(self):
        self.history = defaultdict(deque)

    def observe(self, client_id, current_time):
        times = self.history[client_id]

        # Remove requests outside the observation window
        while times and times[0] <= current_time - WINDOW_SIZE:
            times.popleft()

        # Include the current request
        times.append(current_time)

        request_count = len(times)
        request_rate = request_count / WINDOW_SIZE

        if request_count >= 2:
            intervals = [
                times[i] - times[i - 1]
                for i in range(1, request_count)
            ]

            mean_interval = statistics.mean(intervals)

            interval_std = (
                statistics.pstdev(intervals)
                if len(intervals) > 1
                else 0.0
            )
        else:
            mean_interval = WINDOW_SIZE
            interval_std = 0.0

        return {
            "request_count": request_count,
            "request_rate": request_rate,
            "mean_interval": mean_interval,
            "interval_std": interval_std
        }


def main():
    extractor = OnlineFeatureExtractor()

    example_requests = [
        ("client_1", 1.0),
        ("client_1", 2.0),
        ("client_2", 2.5),
        ("client_1", 2.8),
        ("client_1", 4.1),
        ("client_1", 5.0),
        ("client_1", 8.0)
    ]

    print("\n--- EXPERIMENT V9: ONLINE FEATURES ---")

    for client_id, current_time in example_requests:
        features = extractor.observe(
            client_id,
            current_time
        )

        print(
            f"Time: {current_time:.1f}s | "
            f"Client: {client_id} | "
            f"Count: {features['request_count']} | "
            f"Rate: {features['request_rate']:.2f}/s | "
            f"Mean interval: {features['mean_interval']:.2f}s"
        )


if __name__ == "__main__":
    main()

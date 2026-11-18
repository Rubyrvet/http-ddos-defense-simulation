
import random
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from experiment_v6 import create_client_requests
from experiment_v9 import OnlineFeatureExtractor


FEATURES = [
    "request_count",
    "request_rate",
    "mean_interval",
    "interval_std"
]


def generate_traffic(seed, groups):
    rng = random.Random(seed)
    requests = []

    for name, count, rate, label in groups:
        for i in range(count):
            client_id = f"{seed}_{name}_{i}"

            requests.extend(
                create_client_requests(
                    rng,
                    client_id,
                    rate,
                    label
                )
            )

    requests.sort(key=lambda r: r["arrival_time"])
    return requests


def build_online_dataset(requests):
    extractor = OnlineFeatureExtractor()
    X, y = [], []

    for request in requests:
        features = extractor.observe(
            request["client_id"],
            request["arrival_time"]
        )

        X.append([
            features[name]
            for name in FEATURES
        ])

        y.append(request["label"])

    return X, y


def main():
    # Training conditions
    train_groups = [
        ("normal", 200, 0.5, 0),
        ("flash", 600, 0.5, 0),
        ("attack", 40, 25.0, 1)
    ]

    # Different clients and unseen rates
    test_groups = [
        ("normal", 200, 0.8, 0),
        ("flash", 400, 2.0, 0),
        ("attack_slow", 60, 1.5, 1),
        ("attack_fast", 40, 12.0, 1)
    ]

    train_requests = generate_traffic(
        seed=42,
        groups=train_groups
    )

    test_requests = generate_traffic(
        seed=2026,
        groups=test_groups
    )

    X_train, y_train = build_online_dataset(
        train_requests
    )

    X_test, y_test = build_online_dataset(
        test_requests
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    print("\n--- EXPERIMENT V10 RESULTS ---")
    print("Training requests:", len(X_train))
    print("Testing requests:", len(X_test))
    print("Training classes:", Counter(y_train))
    print("Testing classes:", Counter(y_test))

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print(f"\n--- {name} ---")
        print(classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        ))

        print("Confusion matrix:")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ))


if __name__ == "__main__":
    main()

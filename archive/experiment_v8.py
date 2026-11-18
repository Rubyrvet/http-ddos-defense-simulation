
import random
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from experiment_v6 import (
    create_client_requests,
    extract_features,
    generate_dataset
)

FEATURES = [
    "request_count",
    "request_rate",
    "mean_interval",
    "interval_std"
]

WINDOW_SIZE = 5.0


def prepare_data(rows):
    X = [
        [float(row[feature]) for feature in FEATURES]
        for row in rows
    ]
    y = [int(row["label"]) for row in rows]
    return X, y


def generate_unseen_test(seed=2026):
    rng = random.Random(seed)
    requests = []

    # New clients, with overlapping request rates
    groups = [
        ("normal_test", 200, 0.8, 0),
        ("flash_test", 400, 2.0, 0),
        ("attack_slow", 60, 1.5, 1),
        ("attack_fast", 40, 12.0, 1)
    ]

    for name, count, rate, label in groups:
        for i in range(count):
            client_id = f"{name}_{i}"
            requests.extend(
                create_client_requests(
                    rng, client_id, rate, label
                )
            )

    requests.sort(key=lambda r: r["arrival_time"])

    # Keep original labels from generation
    labels = {
        request["client_id"]: request["label"]
        for request in requests
    }

    rows = extract_features(requests)

    for row in rows:
        row["label"] = labels[row["client_id"]]

    return rows


def main():
    train_rows = extract_features(generate_dataset())
    test_rows = generate_unseen_test()

    X_train, y_train = prepare_data(train_rows)
    X_test, y_test = prepare_data(test_rows)

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
            class_weight="balanced"
        )
    }

    print("\n--- EXPERIMENT V8: ROBUSTNESS TEST ---")
    print("Training clients:", len(X_train))
    print("Unseen testing clients:", len(X_test))

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

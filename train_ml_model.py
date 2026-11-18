
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from experiment_v10 import (
    generate_traffic,
    build_online_dataset
)

TRAIN_GROUPS = [
    ("normal", 200, 0.5, 0),
    ("flash", 600, 0.5, 0),
    ("attack", 40, 25.0, 1)
]

TEST_GROUPS = [
    ("normal", 200, 0.8, 0),
    ("flash", 400, 2.0, 0),
    ("attack_slow", 60, 1.5, 1),
    ("attack_fast", 40, 12.0, 1)
]


def main():
    train_requests = generate_traffic(
        seed=42,
        groups=TRAIN_GROUPS
    )

    test_requests = generate_traffic(
        seed=2026,
        groups=TEST_GROUPS
    )

    X_train, y_train = build_online_dataset(
        train_requests
    )

    X_test, y_test = build_online_dataset(
        test_requests
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)
    from collections import defaultdict
    
    group_results = defaultdict(lambda: [0, 0])
    
    for request, prediction in zip(test_requests, predictions):
        group = request["client_id"].rsplit("_", 1)[0].split("_", 1)[1]
        group_results[group][0] += 1
    
        if prediction == request["label"]:
            group_results[group][1] += 1
    
    print("\n--- PERFORMANCE BY TRAFFIC GROUP ---")
    
    for group, (total, correct) in group_results.items():
        print(
            f"{group}: "
            f"{correct}/{total} correct "
            f"({100 * correct / total:.2f}%)"
        )
    

    print("\n--- TRAIN PERFORMANCE ---")
    print(classification_report(y_train, train_predictions, digits=4))

    print("\n--- TEST PERFORMANCE ---")
    print(classification_report(y_test, predictions, digits=4))

    print("\n--- ML MODEL VALIDATION ---")
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

    joblib.dump(model, "random_forest_model.joblib")

    print("\nModel saved: random_forest_model.joblib")


if __name__ == "__main__":
    main()

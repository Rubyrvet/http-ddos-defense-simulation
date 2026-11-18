
import csv
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

FEATURES = [
    "request_count",
    "request_rate",
    "mean_interval",
    "interval_std"
]

X = []
y = []

with open("synthetic_client_dataset.csv", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        X.append([
            float(row[feature])
            for feature in FEATURES
        ])
        y.append(int(row["label"]))

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
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
        class_weight="balanced"
    )
}

print("\n--- EXPERIMENT V7: ML RESULTS ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training classes:", Counter(y_train))
print("Testing classes:", Counter(y_test))

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

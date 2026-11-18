
import csv
import statistics

import joblib

from experiment_v12 import SCENARIOS, run_experiment


SEEDS = list(range(2026, 2036))
DEFENSES = ["NONE", "FIXED", "ADAPTIVE", "ML"]
METRICS = ["completion", "false_positive", "attack_blocked", "p95"]


def main():
    model = joblib.load("random_forest_model.joblib")
    model.n_jobs = 1

    results = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            print(f"Running {scenario}, seed={seed}", flush=True)

            for defense in DEFENSES:
                result = run_experiment(
                    scenario,
                    defense,
                    model,
                    seed=seed
                )

                results.append({
                    "scenario": scenario,
                    "seed": seed,
                    "defense": defense,
                    **result
                })

    with open("repeated_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n--- SUMMARY: MEAN AND STANDARD DEVIATION ---")

    for scenario in SCENARIOS:
        for defense in DEFENSES:
            rows = [
                row for row in results
                if row["scenario"] == scenario
                and row["defense"] == defense
            ]

            print(f"\n{scenario} / {defense}")

            for metric in METRICS:
                values = [
                    row[metric]
                    for row in rows
                    if row[metric] is not None
                ]

                if values:
                    mean = statistics.mean(values)
                    std = statistics.stdev(values)

                    print(
                        f"  {metric}: "
                        f"{mean:.4f} ± {std:.4f}"
                    )

    print("\nSaved: repeated_results.csv")


if __name__ == "__main__":
    main()

 
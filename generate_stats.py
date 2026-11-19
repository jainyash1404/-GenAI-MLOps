import pandas as pd
import json

df = pd.read_csv("data/student_performance.csv")

features = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours"
]

stats = {}

for feature in features:

    stats[feature] = {
        "mean": float(df[feature].mean()),
        "std": float(df[feature].std())
    }

with open("data/training_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print("Training statistics generated successfully!")
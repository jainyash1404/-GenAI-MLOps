import pandas as pd
import numpy as np

np.random.seed(42)

# Generate student data
n = 600

data = {
    "attendance": np.random.randint(35, 101, n),
    "study_hours": np.random.uniform(0, 8, n),
    "previous_score": np.random.randint(30, 101, n),
    "assignment_completion": np.random.randint(20, 101, n),
    "sleep_hours": np.random.uniform(3, 10, n)
}

df = pd.DataFrame(data)

# Calculate final score
df["final_score"] = (
    0.22 * df["attendance"]
    + 3 * df["study_hours"]
    + 0.42 * df["previous_score"]
    + 0.12 * df["assignment_completion"]
    + 1.2 * df["sleep_hours"]
    - 12
    + np.random.normal(0, 5, n)
)

df["final_score"] = df["final_score"].clip(0, 100)

# Save dataset
df.to_csv("data/student_performance.csv", index=False)

print("Dataset generated successfully!")
print(df.head())

print("\nDataset Shape:", df.shape)
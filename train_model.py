import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours"
]

TARGET = "final_score"

DATA = Path("data/student_performance.csv")
MODEL = Path("models/student_model.pkl")
STATS = Path("data/training_stats.json")


def train():

    print("Starting Model Training...")

    df = pd.read_csv(DATA)

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    params = {
        "n_estimators": 100,
        "max_depth": 8,
        "random_state": 42
    }

    model = RandomForestRegressor(**params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(y_test, predictions) ** 0.5

    r2 = r2_score(y_test, predictions)

    MODEL.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL)

    stats = {}

    for feature in FEATURES:

        stats[feature] = {
            "mean": float(X[feature].mean()),
            "std": float(X[feature].std())
        }

    STATS.write_text(json.dumps(stats, indent=4))

    mlflow.set_experiment("EduPredict-AI")

    with mlflow.start_run():

        mlflow.log_params(params)

        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        mlflow.sklearn.log_model(
            model,
            artifact_path="random_forest_model"
        )

    print("MODEL TRAINING COMPLETED")

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    print("Model saved successfully!")


if __name__ == "__main__":
    train()
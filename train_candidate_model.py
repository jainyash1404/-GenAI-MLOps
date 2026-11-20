"""Train a candidate model using the original data plus the new labelled batch."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
]
TARGET = "final_score"
TRAINING_DATA = Path("data/student_performance.csv")
NEW_DATA = Path("data/new_student_performance.csv")
CANDIDATE_PATH = Path("models/student_model_candidate.pkl")
REPORT_PATH = Path("reports/candidate_training.json")


def train_candidate() -> dict:
    """Train and save a candidate model without changing the champion artifact."""
    for path in (TRAINING_DATA, NEW_DATA):
        if not path.exists():
            raise FileNotFoundError(f"Training input not found: {path}")

    original = pd.read_csv(TRAINING_DATA)
    new_batch = pd.read_csv(NEW_DATA)
    combined = pd.concat([original, new_batch], ignore_index=True)
    required_columns = FEATURES + [TARGET]
    if combined[required_columns].isna().any().any():
        raise ValueError("Combined training data contains missing values")

    x_train, x_test, y_train, y_test = train_test_split(
        combined[FEATURES],
        combined[TARGET],
        test_size=0.2,
        random_state=42,
    )
    params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}
    candidate = RandomForestRegressor(**params)
    candidate.fit(x_train, y_train)
    predictions = candidate.predict(x_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }

    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, CANDIDATE_PATH)

    mlflow_logged = False
    mlflow_run_id = None
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("EduPredict-AI")
        with mlflow.start_run(run_name="candidate-random-forest") as run:
            mlflow.log_params(params)
            mlflow.log_param("training_rows", len(combined))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(candidate, artifact_path="candidate_model")
            mlflow_logged = True
            mlflow_run_id = run.info.run_id
    except ImportError:
        pass

    report = {
        "training_data": [str(TRAINING_DATA), str(NEW_DATA)],
        "training_rows": int(len(combined)),
        "candidate_model": str(CANDIDATE_PATH),
        "metrics": metrics,
        "mlflow_logged": mlflow_logged,
        "mlflow_run_id": mlflow_run_id,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = train_candidate()
    print(f"Candidate trained on {result['training_rows']} rows")
    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.4f}")
    print(f"Candidate model: {CANDIDATE_PATH}")
    print(f"MLflow logged: {result['mlflow_logged']}")

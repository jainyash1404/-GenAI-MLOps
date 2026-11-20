"""Evaluate the existing champion model on a newly labelled batch."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
]
TARGET = "final_score"
MODEL_PATH = Path("models/student_model.pkl")
DATA_PATH = Path("data/new_student_performance.csv")
REPORT_PATH = Path("reports/existing_model_evaluation.json")


def evaluate_model(
    model_path: Path = MODEL_PATH,
    data_path: Path = DATA_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    """Generate regression metrics for the existing model on the new batch."""
    if not model_path.exists():
        raise FileNotFoundError(f"Existing model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Labelled batch not found: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Labelled batch is missing columns: {missing_columns}")
    if frame[required_columns].isna().any().any():
        raise ValueError("Labelled batch contains missing feature or target values")
    if frame.empty:
        raise ValueError("Labelled batch is empty")

    model = joblib.load(model_path)
    predictions = model.predict(frame[FEATURES])
    actuals = frame[TARGET]
    metrics = {
        "MAE": float(mean_absolute_error(actuals, predictions)),
        "RMSE": float(mean_squared_error(actuals, predictions) ** 0.5),
        "R2": float(r2_score(actuals, predictions)),
    }
    report = {
        "model": str(model_path),
        "data": str(data_path),
        "rows_evaluated": int(len(frame)),
        "metrics": metrics,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = evaluate_model()
    print(f"Evaluated rows: {result['rows_evaluated']}")
    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.4f}")
    print(f"Evaluation report: {REPORT_PATH}")

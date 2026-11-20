"""Compare champion and candidate models and record a promotion decision."""

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
DATA_PATH = Path("data/new_student_performance.csv")
CHAMPION_PATH = Path("models/student_model.pkl")
CANDIDATE_PATH = Path("models/student_model_candidate.pkl")
REPORT_PATH = Path("reports/quality_gate_decision.json")


def metrics_for(model, frame: pd.DataFrame) -> dict:
    predictions = model.predict(frame[FEATURES])
    actuals = frame[TARGET]
    return {
        "MAE": float(mean_absolute_error(actuals, predictions)),
        "RMSE": float(mean_squared_error(actuals, predictions) ** 0.5),
        "R2": float(r2_score(actuals, predictions)),
    }


def run_quality_gate() -> dict:
    for path in (DATA_PATH, CHAMPION_PATH, CANDIDATE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Quality-gate input not found: {path}")

    frame = pd.read_csv(DATA_PATH)
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Quality-gate data contains missing values")

    champion_metrics = metrics_for(joblib.load(CHAMPION_PATH), frame)
    candidate_metrics = metrics_for(joblib.load(CANDIDATE_PATH), frame)
    eligible = candidate_metrics["RMSE"] < champion_metrics["RMSE"]
    decision = "eligible_for_promotion" if eligible else "retain_champion"

    report = {
        "data": str(DATA_PATH),
        "rows_compared": int(len(frame)),
        "primary_metric": "RMSE",
        "metric_direction": "lower_is_better",
        "criterion": "Candidate RMSE must be lower than champion RMSE on the same labelled batch.",
        "champion": {
            "model": str(CHAMPION_PATH),
            "metrics": champion_metrics,
        },
        "candidate": {
            "model": str(CANDIDATE_PATH),
            "metrics": candidate_metrics,
        },
        "decision": decision,
        "champion_overwritten": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run_quality_gate()
    print(f"Quality-gate decision: {result['decision']}")
    print(f"Champion overwritten: {result['champion_overwritten']}")
    print(f"Decision report: {REPORT_PATH}")

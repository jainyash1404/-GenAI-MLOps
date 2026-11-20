"""Run the complete data validation, evaluation, retraining, and gating workflow."""

import json
from pathlib import Path

from evaluate_existing_model import evaluate_model
from prepare_new_data import validate_batch
from quality_gate import run_quality_gate
from train_candidate_model import train_candidate


SUMMARY_PATH = Path("reports/monitoring_workflow_summary.json")


def run_workflow() -> dict:
    validation = validate_batch(
        Path("data/new_student_performance.csv"),
        Path("reports/new_data_validation.json"),
    )
    evaluation = evaluate_model()
    training = train_candidate()
    gate = run_quality_gate()
    summary = {
        "status": "completed",
        "steps": {
            "task_a_data_preparation": {
                "status": "completed",
                "rows": validation["row_count"],
                "report": "reports/new_data_validation.json",
            },
            "task_b_existing_model_evaluation": {
                "status": "completed",
                "metrics": evaluation["metrics"],
                "report": "reports/existing_model_evaluation.json",
            },
            "task_c_candidate_training": {
                "status": "completed",
                "training_rows": training["training_rows"],
                "candidate_model": training["candidate_model"],
                "mlflow_logged": training["mlflow_logged"],
                "report": "reports/candidate_training.json",
            },
            "task_d_quality_gate": {
                "status": "completed",
                "decision": gate["decision"],
                "report": "reports/quality_gate_decision.json",
            },
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    result = run_workflow()
    print(f"Workflow status: {result['status']}")
    print(f"Quality-gate decision: {result['steps']['task_d_quality_gate']['decision']}")
    print(f"Summary: {SUMMARY_PATH}")

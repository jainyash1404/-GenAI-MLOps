import json
import pandas as pd

from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours"
]


def run_drift_check():

    print("Starting Data Drift Monitoring...")

    reference = pd.read_csv(
        "data/student_performance.csv"
    )

    current = pd.read_csv(
        "logs/predictions.csv"
    )

    reference = reference[FEATURES].dropna()

    current = current[FEATURES].dropna()

    if len(current) < 10:

        return {
            "status": "waiting",
            "drift_detected": False,
            "message": "At least 10 predictions required.",
            "rows_checked": len(current)
        }

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference,
        current_data=current
    )

    Path("reports").mkdir(
        parents=True,
        exist_ok=True
    )

    report.save_html(
        "reports/data_drift_report.html"
    )

    result = report.as_dict()

    def find_drift(data):

        if isinstance(data, dict):

            if "dataset_drift" in data:
                return data["dataset_drift"]

            for value in data.values():

                found = find_drift(value)

                if found is not None:
                    return found

        elif isinstance(data, list):

            for value in data:

                found = find_drift(value)

                if found is not None:
                    return found

        return None

    drift = find_drift(result)

    if drift is None:

        status = "review"

        detected = False

    else:

        status = "drift" if drift else "stable"

        detected = bool(drift)

    output = {
        "status": status,
        "drift_detected": detected,
        "message": (
            "Data drift detected!"
            if detected
            else "Data distribution is stable."
            if status == "stable"
            else "Review generated drift report."
        ),
        "rows_checked": len(current)
    }

    with open(
        "reports/drift_status.json",
        "w"
    ) as file:

        json.dump(output, file, indent=4)

    return output


if __name__ == "__main__":

    print(
        json.dumps(
            run_drift_check(),
            indent=4
        )
    )
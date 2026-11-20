"""Prepare and validate a labelled batch for model monitoring."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
]
TARGET = "final_score"
REQUIRED_COLUMNS = FEATURES + [TARGET]
DEFAULT_INPUT = Path("data/new_student_performance.csv")
DEFAULT_REPORT = Path("reports/new_data_validation.json")


def generate_batch(output_path: Path, rows: int = 120) -> None:
    """Create a deterministic labelled batch that represents incoming data."""
    rng = np.random.default_rng(20261120)
    batch = pd.DataFrame(
        {
            "attendance": rng.integers(35, 101, rows),
            "study_hours": rng.uniform(0, 8, rows),
            "previous_score": rng.integers(30, 101, rows),
            "assignment_completion": rng.integers(20, 101, rows),
            "sleep_hours": rng.uniform(3, 10, rows),
        }
    )
    batch[TARGET] = (
        0.22 * batch["attendance"]
        + 3 * batch["study_hours"]
        + 0.42 * batch["previous_score"]
        + 0.12 * batch["assignment_completion"]
        + 1.2 * batch["sleep_hours"]
        - 12
        + rng.normal(0, 5, rows)
    ).clip(0, 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(output_path, index=False)


def validate_batch(input_path: Path, report_path: Path) -> dict:
    """Validate schema, missing values, duplicates, and target summary."""
    if not input_path.exists():
        raise FileNotFoundError(f"Incoming batch not found: {input_path}")

    frame = pd.read_csv(input_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    missing_values = {
        column: int(count)
        for column, count in frame[REQUIRED_COLUMNS].isna().sum().items()
        if count
    } if not missing_columns else {}
    duplicate_rows = int(frame.duplicated().sum())

    report = {
        "input_file": str(input_path),
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "target_summary": (
            frame[TARGET].describe().to_dict()
            if TARGET in frame
            else {}
        ),
        "valid": not (
            frame.empty
            or missing_columns
            or missing_values
            or duplicate_rows
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=float) + "\n")

    if not report["valid"]:
        raise ValueError(f"Incoming batch failed validation; see {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate the deterministic sample incoming batch before validating it.",
    )
    args = parser.parse_args()

    if args.generate:
        generate_batch(args.input)

    report = validate_batch(args.input, args.report)
    print(f"Validated batch: {report['row_count']} rows")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Target summary: {json.dumps(report['target_summary'], default=float)}")
    print(f"Validation report: {args.report}")


if __name__ == "__main__":
    main()

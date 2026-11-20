# EduPredict AI

EduPredict AI is a student performance prediction application built with
Python, Streamlit, and a Random Forest regression model. It includes a basic
MLOps workflow for validating new labelled data, evaluating the deployed
model, retraining a candidate, and applying a quality gate before promotion.

## Features

- Student final-score prediction from attendance, study hours, previous score,
  assignment completion, and sleep hours.
- Risk classification and optional Gemini-powered study recommendations.
- Prediction history stored in `logs/predictions.csv`.
- Input distribution monitoring against the training statistics.
- Evidently data-drift report generation.
- Champion-versus-candidate quality gate using RMSE.
- MLflow integration when MLflow is installed and available.

## Project structure

```text
app.py                         Streamlit application
train_model.py                Original champion-model training
prepare_new_data.py           New-batch generation and validation
evaluate_existing_model.py    Champion evaluation on the new batch
train_candidate_model.py      Candidate training on combined data
quality_gate.py               Candidate promotion decision
run_monitoring_workflow.py    End-to-end monitoring workflow
mlops_monitor.py              Evidently drift monitoring
data/student_performance.csv  Original labelled training data
data/new_student_performance.csv
models/student_model.pkl      Champion model
models/student_model_candidate.pkl
reports/                      Validation, evaluation, training, and gate reports
```

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlops.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The sidebar provides:

- Overview
- Student Prediction
- AI Insights
- MLOps Monitoring
- Prediction History

Gemini recommendations are optional. To enable them, create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys.

## Run the monitoring workflow

Run the complete workflow with:

```bash
python run_monitoring_workflow.py
```

The workflow performs these steps:

1. Validates the new labelled batch for required columns, missing values, and
   duplicate rows.
2. Evaluates the champion model using MAE, RMSE, and R².
3. Combines the original and new data and trains a separate candidate model.
4. Compares champion and candidate on the same labelled batch.
5. Records a promotion decision without overwriting the champion.

Individual steps can also be run:

```bash
python prepare_new_data.py --generate
python evaluate_existing_model.py
python train_candidate_model.py
python quality_gate.py
```

## Quality gate

RMSE is the primary promotion metric because it penalizes larger prediction
errors more strongly. A candidate is marked `eligible_for_promotion` only when
its RMSE is lower than the champion's RMSE on the same labelled batch.
The champion model is never overwritten automatically.

The decision is saved to:

```text
reports/quality_gate_decision.json
```

## Current demonstration results

The included workflow was successfully executed on 120 new labelled rows:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Champion | 5.3828 | 6.7249 | 0.7185 |
| Candidate | 3.0942 | 3.9121 | 0.9047 |

The candidate was marked eligible for promotion, while the champion artifact
remained unchanged. See
`reports/monitoring_workflow_summary.json` for the complete run summary.

## MLflow

Candidate training logs parameters, metrics, and the model artifact to the
`EduPredict-AI` MLflow experiment when MLflow is available:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open [http://localhost:5000](http://localhost:5000). The candidate
training report records whether MLflow logging succeeded.


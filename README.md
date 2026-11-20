# EduPredict AI

> An end-to-end student performance prediction and model-quality monitoring
> platform built with Python, Streamlit, and scikit-learn.

EduPredict AI predicts a student's final score from academic and lifestyle
inputs, assigns a risk level, and can generate personalized study guidance.
The project also demonstrates a practical MLOps lifecycle: validate incoming
data, evaluate the deployed champion, train a separate candidate, and record a
safe promotion decision.

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Using the application](#using-the-application)
- [Running the MLOps workflow](#running-the-mlops-workflow)
- [Quality gate](#quality-gate)
- [Execution evidence](#execution-evidence)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Highlights

- Random Forest regression for `final_score` prediction.
- Student risk categories: Low, Moderate, and High.
- Streamlit dashboard with prediction history.
- Optional Gemini-powered academic recommendations.
- Incoming-batch validation for schema, missing values, and duplicates.
- Champion evaluation with MAE, RMSE, and R².
- Separate candidate artifact trained from combined historical and new data.
- Evidently data-drift report generation.
- Auditable RMSE-based quality gate.
- Champion model is never overwritten automatically.
- MLflow logging support when MLflow is available.

## Architecture

```text
Incoming labelled batch
          |
          v
  prepare_new_data.py
          |
          v
 Champion evaluation  ------> reports/existing_model_evaluation.json
          |
          v
 Combined-data candidate training
          |
          v
 Champion vs candidate quality gate
          |
          +----> eligible_for_promotion
          |
          +----> retain_champion
```

The deployed champion remains at `models/student_model.pkl`. The newly trained
candidate is always written to `models/student_model_candidate.pkl`.

## Quick start

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements-mlops.txt
```

### 3. Start the dashboard

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Using the application

Use the sidebar to switch between:

| Page | Purpose |
|---|---|
| **Overview** | View model status and project KPIs |
| **Student Prediction** | Enter student data and generate a score |
| **AI Insights** | Generate a personalized study plan |
| **MLOps Monitoring** | Run the Evidently drift check and download its report |
| **Prediction History** | Review logged predictions |

Every prediction is appended to `logs/predictions.csv`. The MLOps Monitoring
page compares current input data with the training statistics and writes:

- `reports/data_drift_report.html`
- `reports/drift_status.json`

## Running the MLOps workflow

Run all monitoring and retraining stages in one command:

```bash
python run_monitoring_workflow.py
```

The workflow executes:

1. **Prepare new data** — validates the incoming labelled batch.
2. **Evaluate champion** — predicts on the new batch and records regression
   metrics.
3. **Train candidate** — combines the original 600 rows with the new 120-row
   batch and saves a separate model.
4. **Apply quality gate** — compares both models on the same labelled batch.
5. **Record evidence** — writes a complete execution summary.

To run stages independently:

```bash
python prepare_new_data.py --generate
python evaluate_existing_model.py
python train_candidate_model.py
python quality_gate.py
```

## Quality gate

The primary metric is **RMSE**. It is appropriate here because this is a
regression problem and RMSE penalizes larger prediction errors more strongly
than smaller errors.

The decision rule is:

```text
if candidate_RMSE < champion_RMSE:
    eligible_for_promotion
else:
    retain_champion
```

Both models are evaluated on the same labelled batch to make the comparison
fair. The decision is stored in
`reports/quality_gate_decision.json`, and the field
`champion_overwritten` explicitly records that the champion was not replaced.

## Execution evidence

The workflow has been executed successfully on 120 new labelled rows.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Champion | 5.3828 | 6.7249 | 0.7185 |
| Candidate | 3.0942 | 3.9121 | 0.9047 |

Result: **`eligible_for_promotion`**

Evidence artifacts:

| Artifact | Location |
|---|---|
| New-data validation | `reports/new_data_validation.json` |
| Champion evaluation | `reports/existing_model_evaluation.json` |
| Candidate training report | `reports/candidate_training.json` |
| Candidate model | `models/student_model_candidate.pkl` |
| Quality-gate decision | `reports/quality_gate_decision.json` |
| Complete run summary | `reports/monitoring_workflow_summary.json` |

## Project structure

```text
.
├── app.py
├── requirements-mlops.txt
├── data/
│   ├── student_performance.csv
│   ├── new_student_performance.csv
│   └── training_stats.json
├── models/
│   ├── student_model.pkl
│   └── student_model_candidate.pkl
├── reports/
│   ├── new_data_validation.json
│   ├── existing_model_evaluation.json
│   ├── candidate_training.json
│   ├── quality_gate_decision.json
│   └── monitoring_workflow_summary.json
├── logs/
│   └── predictions.csv
├── prepare_new_data.py
├── evaluate_existing_model.py
├── train_candidate_model.py
├── quality_gate.py
├── run_monitoring_workflow.py
├── mlops_monitor.py
└── README.md
```

## Configuration

Gemini recommendations are optional. To enable them, create a local `.env`
file:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit API keys or `.env` files.

### MLflow

Candidate training logs parameters, metrics, and the model artifact to the
`EduPredict-AI` experiment when MLflow is installed:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open [http://localhost:5000](http://localhost:5000). The field
`mlflow_logged` in `reports/candidate_training.json` records whether logging
succeeded for that run.

## Troubleshooting

### `streamlit: command not found`

Install the requirements and use the module form:

```bash
python -m pip install -r requirements-mlops.txt
python -m streamlit run app.py
```

### MLOps Monitoring page fails to load

Evidently is required for drift monitoring:

```bash
python -m pip install evidently==0.4.40
```

### Gemini recommendations are unavailable

The prediction dashboard still works without Gemini. Configure
`GEMINI_API_KEY` in `.env` to enable AI recommendations.


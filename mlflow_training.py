import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")

X = df.drop("final_score", axis=1)
y = df["final_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("EduPredict-AI")

# Start experiment
with mlflow.start_run(run_name="Random-Forest-Model"):

    # Model creation
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Save model locally
    joblib.dump(model, "models/student_model.pkl")

    # Log model as artifact
    mlflow.log_artifact("models/student_model.pkl")

    print("Model artifact logged successfully!")

    print("MLflow Experiment Completed!")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 2))

print("All parameters, metrics and model logged!")


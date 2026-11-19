import subprocess
import sys

from mlops_monitor import run_drift_check


def main():

    print("================================")

    print("EDUPREDICT AI MLOPS PIPELINE")

    print("================================")

    result = run_drift_check()

    print("Monitoring Status:", result["status"])

    print("Message:", result["message"])

    if result["drift_detected"]:

        print("Data Drift Detected!")

        print("Starting Automatic Model Retraining...")

        subprocess.run(
            [sys.executable, "train_model.py"],
            check=True
        )

        print("Model Retrained Successfully!")

    else:

        print("No Confirmed Drift Detected.")

        print("Retraining Not Required.")


if __name__ == "__main__":

    main()
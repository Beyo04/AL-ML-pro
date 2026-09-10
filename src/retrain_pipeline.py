import os
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from src.monitor_drift import check_data_drift
except ModuleNotFoundError:
    from monitor_drift import check_data_drift

def automated_continuous_training():
    print("Checking production data for drift...")
    drift_detected = check_data_drift()

    if drift_detected:
        print("Drift threshold breached! Triggering automated retraining...")
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # Explicitly set tracking URI to avoid Windows path percent-encoding issues
        db_path = Path.cwd() / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment("iris_classification")

        with mlflow.start_run(run_name="automated_retrain_run"):
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            mlflow.log_param("trigger", "data_drift")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name="IrisRandomForest"
            )

            print(f"Retraining completed. New Model Accuracy: {acc:.4f}")
            print("Updated version registered to MLflow Model Registry.")
    else:
        print("No significant drift detected. Retraining skipped.")

if __name__ == "__main__":
    automated_continuous_training()
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Bypass space in Windows username by directing storage to C:\mlflow_data
os.makedirs("C:/mlflow_data", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///C:/mlflow_data/mlflow.db")

# 2. Enable MLflow tracking
mlflow.set_experiment("iris_classification")

# 3. Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

n_estimators = 50
max_depth = 3

with mlflow.start_run() as run:
    # 4. Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate metrics
    accuracy = model.score(X_test, y_test)
    
    # 6. Log parameters, metrics, and artifact model to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # Register the model version into MLflow Model Registry
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisRandomForest")
    
    print(f"Model logged with Run ID: {run.info.run_id}")
    print(f"Test Accuracy: {accuracy:.4f}")
from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# 1. Point to the tracking database you created in train.py
mlflow.set_tracking_uri("sqlite:///C:/mlflow_data/mlflow.db")

# 2. Fetch model version 1 directly from MLflow Model Registry
MODEL_URI = "models:/IrisRandomForest/1"
model = mlflow.sklearn.load_model(MODEL_URI)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisInput):
    # Prepare input dataframe with Scikit-learn's default feature names
    data_df = pd.DataFrame([list(features.model_dump().values())], columns=[
        "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ])
    
    # Inference
    prediction = model.predict(data_df)
    return {"class_id": int(prediction[0])}
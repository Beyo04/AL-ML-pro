import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"
TRACKING_URI = "sqlite:///mlflow.db"


def load_latest_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        # Dynamically load latest version registered in MLflow
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    except Exception as e:
        try:
            # Fallback if alias / latest tag isn't set yet
            return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        except Exception:
            try:
                return mlflow.sklearn.load_model("models/model.pkl")
            except Exception:
                return None


model = load_latest_model()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model artifact unavailable."
        )

    # Scikit-learn feature formatting
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    data_df = pd.DataFrame(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        columns=feature_names,
    )

    try:
        pred = model.predict(data_df)
    except Exception:
        # Fallback to plain array if model was trained without feature names
        pred = model.predict(
            [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        )

    return {"prediction": int(pred[0])}
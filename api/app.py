from fastapi import FastAPI, status, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
from .schemas import InputSchema

from hydra import initialize_config_dir, compose
from pathlib import Path

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config")

with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
    cfg = compose(config_name="config")

app = FastAPI()

MLFLOW_TRACKING_URI = cfg.envm.mlflow_tracking_uri
MODEL_NAME = cfg.envm.model_name
MODEL_VERSION = cfg.envm.model_version

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)


@app.post("/predict")
async def predict(data: InputSchema):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)

        return {
            "prediction": int(prediction[0]),
            "status": status.HTTP_201_CREATED
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
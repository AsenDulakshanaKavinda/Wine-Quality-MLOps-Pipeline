from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from .schemas import InputSchema

from hydra import initialize, compose

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(config_name="config")

app = FastAPI()
MLFLOW_TRACKING_URI =  cfg.envm.mlflow_tracking_uri# "http://localhost:5000"
MODEL_NAME = cfg.envm.model_name
MODEL_VERSION = cfg.envm.model_version


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

@app.post('/predict')
async def predict(data: InputSchema):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)
        return {
            "prediction": int(prediction[0]),
            "status": status.HTTP_201_CREATED
        }
    # todo: - add correct exception handling
    except Exception as e:
        HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
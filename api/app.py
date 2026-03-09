from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from .schemas import InputSchema

app = FastAPI()
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "rest-reg-1"
MODEL_VERSION = 1 


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

@app.post('/predict')
async def predict(data: InputSchema):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
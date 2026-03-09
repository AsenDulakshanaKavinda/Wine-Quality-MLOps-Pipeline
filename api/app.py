# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# 1️⃣ Define input schema
# -------------------------
class InputSchema(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    def to_dataframe(self):
        # convert single input to DataFrame (MLflow models expect DataFrame)
        return pd.DataFrame([self.model_dump()])

# -------------------------
# 2️⃣ Initialize FastAPI
# -------------------------
app = FastAPI(title="Wine Quality Prediction API")

# Optional: enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://127.0.0.1:5000"] in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 3️⃣ Load MLflow model
# -------------------------
# Point to your running MLflow server (DO NOT use sqlite directly)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow server
MODEL_NAME = "wine-quality-model"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model("models:/wine-quality-model/Production")

# -------------------------
# 4️⃣ Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(input_data: InputSchema):
    df = input_data.to_dataframe()
    predictions = model.predict(df)
    return {"prediction": predictions.tolist()}

# -------------------------
# 5️⃣ Health check endpoint
# -------------------------
@app.get("/health")
def health_check():
    return {"status": "API is running ✅"}
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000/")

model = mlflow.pyfunc.load_model(
    "models:/wine-quality-model/Production"
)

def predict(data: dict) -> dict:


    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return int(prediction[0])

from fastapi import FastAPI, status, HTTPException
from src.predicting.predict import predict
from api.schemas.pred_schemas import InputSchema
app = FastAPI()

@app.get("/health/")
def check_health():
    try:
        return {
            "message": "API running...",
            "status code" : status.HTTP_200_OK
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/prediction")
def pred(payload: InputSchema):
    result = predict(payload)
    return {"prediction": result}
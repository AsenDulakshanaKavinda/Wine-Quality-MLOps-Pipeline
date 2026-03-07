from fastapi import FastAPI, status, HTTPException

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



mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000

#mlflow server \
#  --backend-store-uri sqlite:////media/raptor/volume01/Wine-Quality-MLOps-Pipeline/mlflow/mlflow.db \
#  --default-artifact-root /media/raptor/volume01/Wine-Quality-MLOps-Pipeline/mlflow/artifacts \
#  --host 127.0.0.1 \
#  --port 5000
    
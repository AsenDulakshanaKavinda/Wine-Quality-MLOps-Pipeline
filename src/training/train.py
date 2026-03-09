
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig
from utils import log
from src.preprocessing import DataPreprocessing
from src.training.create_artifacts import create_prediction_plot


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # Set experiment and tracking URI
    mlflow.set_tracking_uri(cfg.envm.mlflow_uri)
    mlflow.set_experiment("Wine quality experiment")
    
    # Initialize preprocessing
    data_processing = DataPreprocessing(cfg=cfg)
    df = data_processing.data_ingestion()

    X_train, X_test, y_train, y_test = data_processing.data_split(df)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
        ))
    ])

    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.autolog(log_models=False)

        pipeline.fit(X_train, y_train) 
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Log metrics per iteration
        mlflow.log_metric("accuracy", acc)
        log.info(f"Accuracy: {acc:.4f}")

        # Log parameters
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
        })

        # Create prediction plot artifact
        artifact_path = "mlflow/tmp/predictions.png"
        create_prediction_plot(predictions=predictions, artifact_path=artifact_path)

        # Log artifact
        mlflow.log_artifact(artifact_path)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model", 
            registered_model_name="wine-quality-model"
        )

        log.info(f"MLflow run completed: {run.info.run_id}")

if __name__ == "__main__":
    train()


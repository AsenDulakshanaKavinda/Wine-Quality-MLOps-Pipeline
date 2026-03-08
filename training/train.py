
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig
from utils import log

from preprocessing import DataPreprocessing

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Set experiment and tracking URI
    mlflow.set_tracking_uri(cfg.env.mlflow_uri)
    mlflow.set_experiment("Wine quality experiment")
    
    # Initialize preprocessing
    data_processing = DataPreprocessing(cfg=cfg)
    df = data_processing.data_ingestion()
    df = data_processing.data_normalization(df)

    X_train, X_test, y_train, y_test = data_processing.data_split(df)

    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.autolog(log_models=False)

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            # random_state=cfg.model.random_state
        )

        # Train with optional iteration stopping
        max_epochs = cfg.env.max_epochs
        for epoch in range(max_epochs):
            model.fit(X_train, y_train) 
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log metrics per iteration
            mlflow.log_metric("accuracy", acc, step=epoch)
            log.info(f"Epoch {epoch+1}/{max_epochs} - Accuracy: {acc:.4f}")

        # Log parameters
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            # "random_state": cfg.model.random_state,
            "max_epochs": max_epochs
        })

        # Create prediction plot artifact
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(preds)), preds, alpha=0.6)
        plt.title("Prediction Distribution")
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Class")
        artifact_path = "mlflow/tmp/predictions.png"
        os.makedirs("mlflow/tmp", exist_ok=True)
        plt.savefig(artifact_path)
        plt.close() 

        # Log artifact
        mlflow.log_artifact(artifact_path)

        # Log the model
        mlflow.sklearn.log_model(
            model, 
            name="model", 
            registered_model_name="wine-quality-model"
        )

        print(f"MLflow run completed: {run.info.run_id}")

if __name__ == "__main__":
    train()


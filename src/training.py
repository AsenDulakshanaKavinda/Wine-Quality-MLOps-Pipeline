import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import hydra
from omegaconf import DictConfig

from .preprocessing import DataPreprocess

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wine-quality-experiment")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):

    preprocessing = DataPreprocess(cfg=cfg)
    df = preprocessing.load_dataset()
    X_train, X_test, y_train, y_test = preprocessing.split_dataset(df=df)

    with mlflow.start_run() as run:
        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", RandomForestClassifier(
                n_estimators=cfg.model.n_estimators,
                max_depth=cfg.model.max_depth,
            ))
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy:.4f}")

        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
        })

        mlflow.log_metric("accuracy", pipeline.score(X_test, y_test))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path='sklearn-model',
            registered_model_name='wine-quality-model'
        )

        print(f"MLflow run completed: {run.info.run_id}")


    

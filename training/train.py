import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

import hydra
from omegaconf import DictConfig

from preprocessing import DataPreprocessing

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    mlflow.set_experiment("wine-quality-classifier")
    mlflow.set_tracking_uri(cfg.env.dev.mlflow_uri)

    data_processing = DataPreprocessing(cfg=cfg)

    X_train, X_test, y_train, y_test = data_processing.data_split()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=cfg.random_forest_classifier.n_estimators,
            max_depth=cfg.random_forest_classifier.max_depth
        )

    print(cfg.dataset.name)
    # print(cfg.model.type)
    # print(cfg.optimizer.lr)
    # print(cfg.env.mlflow_uri)

if __name__ == "__main__":
    train()
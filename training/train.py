import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import plot
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

        model.fit(X_train, y_test)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth            
        })

        # log metric
        mlflow.log_metric("accuracy", acc)

        # create artifact (plot)
        plt.scatter(range(len(preds)), preds)
        plt.title("Prediction Distribution")

        plt.savefig("predictions.png")

        # log artifact
        mlflow.log_artifact("predictions.png")

        # log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
import matplotlib.pyplot as plt  
import os

def create_prediction_plot(predictions, artifact_path):
        # Create prediction plot artifact
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(predictions)), predictions, alpha=0.6)
        plt.title("Prediction Distribution")
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Class")
        os.makedirs("mlflow/tmp", exist_ok=True)
        plt.savefig(artifact_path)
        plt.close() 

import os

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing():
    def __init__(self):
        pass


    def data_ingestion(self, filepath: Path):
        if not filepath:
            raise ValueError("Filepath missing")
        
        if not os.path.exists(filepath):
            raise FileExistsError("File dose not exist")

        if not str(filepath).lower().endswith(".csv"):
            raise ValueError("File is not a CSV file.")
        
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            RuntimeError(f"Error while data ingestion, error: {str(e)}") 

    def data_normalization(self, df, num_col_list: list[str]):
        if not df:
            raise ValueError("Missing dataframe")

        if not num_col_list:
            raise ValueError("Missing numerical columns list")

        scale = MinMaxScaler()

        try:
            df[num_col_list] = scale.fit_transform(df[num_col_list])
            return df
        except Exception as e:
            raise RuntimeError(f"Error while normalizing data, error: {str(e)}")

    def data_split(self, df, target_col: str, test_size: float=0.2, random_state: int=42):
        if not df:
            raise ValueError("Missing dataframe")  

        if not target_col:
            raise ValueError("Missing target column") 

        try:
            X = df.drop(target_col, axis=1)    
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise RuntimeError(f"Error while splitting data, error: {str(e)}")

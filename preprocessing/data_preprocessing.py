import os
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing():
    def __init__(self, cfg):
        self.filepath: Path = Path(cfg.data.filepath)
        self.target_col: str = cfg.data.target_column
        self.num_col_list: list = cfg.data.numerical_columns
        self.test_size: float = cfg.split.test_size
        self.random_state: int = cfg.split.random_state

    def _data_ingestion(self) -> DataFrame:
        if not self.filepath:
            raise ValueError("Filepath missing")
        
        if not os.path.exists(self.filepath):
            raise FileExistsError("File dose not exist")

        if not str(self.filepath).lower().endswith(".csv"):
            raise ValueError("File is not a CSV file.")
        
        try:
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            raise RuntimeError(f"Error while data ingestion, error: {str(e)}") 

    def _data_normalization(self, df: DataFrame) -> DataFrame:
        if df is None or df.empty:
            raise ValueError("Missing dataframe")

        scale = MinMaxScaler()

        try:
            df[self.num_col_list] = scale.fit_transform(df[self.num_col_list])
            return df
        except Exception as e:
            raise RuntimeError(f"Error while normalizing data, error: {str(e)}")



    def data_split(self) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
            
            df = self._data_ingestion()
            df = self._data_normalization(df)

            try:
                X = df.drop(self.target_col, axis=1)    
                y = df[self.target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

                return X_train, X_test, y_train, y_test
            
            except Exception as e:
                raise RuntimeError(f"Error while splitting data, error: {str(e)}")

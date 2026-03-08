import os
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import log


class DataPreprocessing():
    def __init__(self, cfg):
        self.filepath: Path = Path(cfg.data.filepath)
        self.target_col: str = cfg.data.target_column
        self.num_col_list: list = cfg.data.numerical_columns
        self.test_size: float = cfg.split.test_size
        self.random_state: int = cfg.split.random_state

    def data_ingestion(self) -> DataFrame:

        filepath = self.filepath

        if not filepath:
            log.error("Filepath is missing")
            raise ValueError("Filepath missing")
        
        if not os.path.exists(filepath):
            log.error("File dose not exist")
            raise FileExistsError("File dose not exist")

        if not str(filepath).lower().endswith(".csv"):
            log.error("File is not in correct type of file")
            raise ValueError("File is not a CSV file.")
        
        try:
            df = pd.read_csv(filepath)
            log.info("Data ingestion completed.")
            return df
        except Exception as e:
            log.error("Error while ingesting data")
            raise RuntimeError(f"Error while data ingestion, error: {str(e)}") 


    def data_normalization(self, df: DataFrame) -> DataFrame:
        if df is None or df.empty:
            log.error("Missing dataframe in normalization")
            raise ValueError("Missing dataframe")

        scale = MinMaxScaler()

        try:
            df[self.num_col_list] = scale.fit_transform(df[self.num_col_list])
            log.info("Data normalization is completed")
            return df
        except Exception as e:
            raise RuntimeError(f"Error while normalizing data, error: {str(e)}")



    def data_split(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
            if df is None or df.empty:
                log.error("Missing dataframe in data splitting")
                raise ValueError("Missing dataframe")

            try:
                X = df.drop(self.target_col, axis=1)    
                y = df[self.target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, 
                    y, 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )

                log.info("Data splitting completed.")
                return X_train, X_test, y_train, y_test
            
            except Exception as e:
                log.info("Error while splitting data")
                raise RuntimeError(f"Error while splitting data, error: {str(e)}")

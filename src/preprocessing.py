import os
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocess:
    def __init__(self, cfg):
        self.filepath: Path = Path(cfg.dataset.filepath)
        self.target_col:str = cfg.dataset.target_col
        self.test_size: float = cfg.dataset.test_size
        self.random_state: int = cfg.dataset.random_state

    def load_dataset(self) -> pd.DataFrame:
        source = self.filepath
        if not str(source).lower().endswith(".csv"):
            raise ValueError("Error: source file is not a valid file")
        
        try:
            return pd.read_csv(source)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error while reading dataset: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error while reading dataset: {str(e)} ")
        

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not df or df.empty:
            raise ValueError("Error: dataframe is missing in splitting dataset")
        
        try:
            X = df.drop(self.target_col, axis=1)
            y = df[self.target_col]

            return train_test_split(
                X, 
                y,
                test_size=self.test_size,
                random_state=self.random_state
            )
        
        except Exception as e:
            raise RuntimeError(f"Error while splitting dataframe: {str(e)}")










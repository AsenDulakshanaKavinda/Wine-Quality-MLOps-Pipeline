import pytest
import pandas as pd
from pathlib import Path
from preprocessing.data_preprocessing import DataPreprocessing


class DummyConfig:
    class data:
        filepath = ""
        target_column = "quality"
        numerical_columns = ["feature1", "feature2"]

    class split:
        test_size = 0.25
        random_state = 42

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1,2,3,4],
        "feature2": [10,20,30,40],
        "quality": [0,1,0,1]
    })

@pytest.fixture
def cfg(tmp_path):
    config = DummyConfig()
    config.data.filepath = tmp_path / "data.csv"
    return config

@pytest.fixture
def dp(cfg):
    return DataPreprocessing(cfg)


# Data ingestion
def test_data_ingestion_success(tmp_path, cfg, dp):
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    file = cfg.data.filepath

    df.to_csv(file, index=False)

    result = dp._data_ingestion()

    assert isinstance(result, pd.DataFrame)
    assert not result.empty

def test_data_ingestion_file_not_exist(cfg):
    cfg.data.filepath = Path("fake.csv")
    dp = DataPreprocessing(cfg)

    with pytest.raises(FileExistsError):
        dp._data_ingestion()


# Data normalization
def test_data_normalization_success(sample_df, dp):
    result = dp._data_normalization(sample_df)

    assert result["feature1"].max() <= 1
    assert result["feature1"].min() >= 0

def test_data_normalization_missing_df(dp):
    with pytest.raises(ValueError):
        dp._data_normalization(None)


# Data split
def test_data_split_success(sample_df, cfg):
    sample_df.to_csv(cfg.data.filepath, index=False)

    dp = DataPreprocessing(cfg)
    cfg.data.numerical_columns = ["feature1", "feature2"]

    X_train, X_test, y_train, y_test = dp.data_split()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

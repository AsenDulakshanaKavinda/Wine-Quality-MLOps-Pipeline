import pytest
import pandas as pd
from pathlib import Path
from preprocessing.data_preprocessing import DataPreprocessing

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1,2,3,4],
        "feature2": [10, 20, 30, 40],
        "quality": [0,1,0,1]
    })

@pytest.fixture
def dp():
    return DataPreprocessing()

# data ingestion
def test_data_ingestion_success(tmp_path, dp):
    file = tmp_path / "data.csv"

    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    df.to_csv(file, index=False)

    result = dp.data_ingestion(file)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty

# file not exist
def test_data_ingestion_file_not_exist(dp):
    with pytest.raises(FileExistsError):
        dp.data_ingestion("fake.csv")

# data normalization
def test_data_normalization_success(sample_df, dp):
    result = dp.data_normalization(sample_df, ["feature1", "feature2"])

    assert result["feature1"].max() <= 1
    assert result["feature1"].min() >= 0

def test_data_normalization_missing_df(dp):
    with pytest.raises(ValueError):
        dp.data_normalization(None, ["feature1"])

def test_data_normalization_missing_columns(sample_df, dp):
    with pytest.raises(ValueError):
        dp.data_normalization(sample_df, [])

# data split
def test_data_split_success(sample_df, dp):
    X_train, X_test, y_train, y_test = dp.data_split(sample_df, "quality")

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_data_split_missing_target(sample_df, dp):
    with pytest.raises(ValueError):
        dp.data_split(sample_df, "")
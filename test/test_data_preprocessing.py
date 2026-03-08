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
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40],
        "quality": [0, 1, 0, 1]
    })


@pytest.fixture
def cfg(tmp_path):
    config = DummyConfig()
    config.data.filepath = tmp_path / "data.csv"
    return config


@pytest.fixture
def dp(cfg):
    return DataPreprocessing(cfg)


# DATA INGESTION TESTS
def test_data_ingestion_success(cfg, dp):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(cfg.data.filepath, index=False)

    result = dp.data_ingestion()

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ["a", "b"]


def test_data_ingestion_file_not_exist(cfg):
    cfg.data.filepath = Path("fake.csv")
    dp = DataPreprocessing(cfg)

    with pytest.raises(FileExistsError):
        dp.data_ingestion()


def test_data_ingestion_wrong_file_type(tmp_path, cfg):
    file = tmp_path / "data.txt"
    file.write_text("invalid")

    cfg.data.filepath = file
    dp = DataPreprocessing(cfg)

    with pytest.raises(ValueError):
        dp.data_ingestion()


# DATA NORMALIZATION TESTS
def test_data_normalization_success(sample_df, dp):
    result = dp.data_normalization(sample_df)

    assert result["feature1"].min() >= 0
    assert result["feature1"].max() <= 1
    assert result["feature2"].min() >= 0
    assert result["feature2"].max() <= 1


def test_data_normalization_missing_df(dp):
    with pytest.raises(ValueError):
        dp.data_normalization(None)


def test_data_normalization_empty_df(dp):
    with pytest.raises(ValueError):
        dp.data_normalization(pd.DataFrame())


# DATA SPLIT TESTS
def test_data_split_success(sample_df, dp):
    X_train, X_test, y_train, y_test = dp.data_split(sample_df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_data_split_missing_df(dp):
    with pytest.raises(ValueError):
        dp.data_split(None)


def test_data_split_empty_df(dp):
    with pytest.raises(ValueError):
        dp.data_split(pd.DataFrame())

def test_data_split_missing_target(dp, sample_df):
    df = sample_df.drop("quality", axis=1)

    with pytest.raises(RuntimeError):
        dp.data_split(df)

def test_data_normalization_missing_column(dp, sample_df):
    dp.num_col_list = ["invalid"]

    with pytest.raises(RuntimeError):
        dp.data_normalization(sample_df)
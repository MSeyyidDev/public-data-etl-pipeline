"""Tests for loaders."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from etl.loaders import CSVLoader, ParquetLoader, SQLiteLoader


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "city": ["A", "B"],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "temperature_c": [10.0, 20.0],
        "humidity": [50.0, 60.0],
        "pressure_hpa": [1013.0, 1014.0],
        "wind_kmh": [5.0, 7.0],
        "precipitation_mm": [0.0, 1.0],
        "condition": ["Clear", "Rain"],
        "heat_index": [10.0, 20.0],
        "comfort_score": [80.0, 70.0],
        "is_extreme": [False, False],
        "date": ["2025-01-01", "2025-01-02"],
        "hour": [0, 0],
    })


def test_sqlite_loader_writes_table(sample_df, tmp_path: Path):
    db = tmp_path / "test.db"
    url = f"sqlite:///{db.as_posix()}"
    loader = SQLiteLoader(url)
    rows = loader.load(sample_df, "weather_facts")
    assert rows == len(sample_df)

    engine = create_engine(url, future=True)
    df_back = pd.read_sql_table("weather_facts", engine)
    assert len(df_back) == len(sample_df)
    assert set(df_back.columns) == set(sample_df.columns)


def test_sqlite_loader_in_memory(sample_df):
    loader = SQLiteLoader("sqlite:///:memory:")
    rows = loader.load(sample_df, "weather_facts")
    assert rows == len(sample_df)
    engine = loader.engine
    df_back = pd.read_sql_table("weather_facts", engine)
    assert len(df_back) == len(sample_df)


def test_csv_loader(tmp_path: Path, sample_df):
    loader = CSVLoader(tmp_path)
    rows = loader.load(sample_df, "weather_facts")
    assert rows == len(sample_df)
    out_file = tmp_path / "weather_facts.csv"
    assert out_file.exists()
    df_back = pd.read_csv(out_file)
    assert len(df_back) == len(sample_df)


def test_parquet_loader(tmp_path: Path, sample_df):
    loader = ParquetLoader(tmp_path)
    rows = loader.load(sample_df, "weather_facts")
    assert rows == len(sample_df)
    out_file = tmp_path / "weather_facts.parquet"
    assert out_file.exists()
    df_back = pd.read_parquet(out_file)
    assert len(df_back) == len(sample_df)


def test_sqlite_loader_unknown_table_falls_back(sample_df, tmp_path: Path):
    db = tmp_path / "fb.db"
    url = f"sqlite:///{db.as_posix()}"
    loader = SQLiteLoader(url)
    n = loader.load(sample_df, "ad_hoc_table")
    assert n == len(sample_df)
    df_back = pd.read_sql_table("ad_hoc_table", loader.engine)
    assert len(df_back) == len(sample_df)

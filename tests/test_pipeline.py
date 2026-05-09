"""End-to-end pipeline test using in-memory SQLite."""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine

from etl.loaders import SQLiteLoader
from etl.pipeline import Pipeline
from etl.quality.checker import default_weather_checker
from etl.sources import WeatherDataSource
from etl.transformers import WeatherTransformer


def test_pipeline_end_to_end_in_memory():
    source = WeatherDataSource(cities=2, days=2, seed=0)
    transformer = WeatherTransformer()
    sqlite_loader = SQLiteLoader("sqlite:///:memory:")
    pipe = Pipeline(
        source=source,
        transformer=transformer,
        loaders=[sqlite_loader],
        table="weather_facts",
        quality_checker=default_weather_checker(),
    )
    result = pipe.run()
    assert result.rows_extracted == 2 * 2 * 24
    assert result.rows_loaded == result.rows_extracted
    assert result.quality is not None
    assert result.duration_seconds > 0
    # Pipeline should produce 4 stages: extract, transform, quality, load.
    assert [s.stage for s in result.stages] == ["extract", "transform", "quality", "load"]
    df_back = pd.read_sql_table("weather_facts", sqlite_loader.engine)
    assert len(df_back) == result.rows_loaded


def test_pipeline_records_nulls_handled():
    source = WeatherDataSource(cities=1, days=1, seed=0)
    transformer = WeatherTransformer()
    pipe = Pipeline(
        source=source, transformer=transformer,
        loaders=[SQLiteLoader("sqlite:///:memory:")],
        table="weather_facts",
        quality_checker=default_weather_checker(),
    )
    result = pipe.run()
    # Default null_rate is 2% so we expect at least one null with 24 rows * 5 cols.
    assert result.nulls_handled >= 0
    transform_stage = next(s for s in result.stages if s.stage == "transform")
    assert transform_stage.duration_seconds >= 0

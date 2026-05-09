"""Tests for the quality checker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.quality import DataQualityChecker
from etl.quality.checker import (
    default_sales_checker,
    default_traffic_checker,
    default_weather_checker,
)


def test_required_columns_present_pass():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    checker = DataQualityChecker(source="x", required_columns=["a", "b"])
    report = checker.run(df)
    schema_results = [r for r in report.results if r.check_name == "schema_required_columns"]
    assert schema_results[0].passed


def test_required_columns_missing_fail():
    df = pd.DataFrame({"a": [1]})
    checker = DataQualityChecker(source="x", required_columns=["a", "b"])
    report = checker.run(df)
    schema = [r for r in report.results if r.check_name == "schema_required_columns"][0]
    assert not schema.passed
    assert "b" in schema.details


def test_non_null_check_respects_threshold():
    df = pd.DataFrame({"a": [1, None, None, 4]})
    strict = DataQualityChecker(
        source="x", non_null_columns=["a"], max_null_fraction=0.0,
    )
    relaxed = DataQualityChecker(
        source="x", non_null_columns=["a"], max_null_fraction=0.6,
    )
    strict_result = next(r for r in strict.run(df).results if r.check_name == "non_null:a")
    relaxed_result = next(r for r in relaxed.run(df).results if r.check_name == "non_null:a")
    assert not strict_result.passed
    assert relaxed_result.passed


def test_range_check_detects_violations():
    df = pd.DataFrame({"x": [0, 5, 10, 99]})
    checker = DataQualityChecker(source="t", ranges={"x": (0, 10)})
    res = next(r for r in checker.run(df).results if r.check_name == "range:x")
    assert not res.passed
    assert "out_of_range=1" in res.details


def test_unique_subset_check_detects_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    checker = DataQualityChecker(source="t", unique_subset=["a", "b"])
    res = next(r for r in checker.run(df).results if r.check_name.startswith("unique:"))
    assert not res.passed


def test_default_weather_checker_on_clean_frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "city": ["A"] * 24,
        "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
        "temperature_c": rng.uniform(0, 25, 24),
        "humidity": rng.uniform(40, 80, 24),
        "pressure_hpa": rng.uniform(1000, 1020, 24),
        "wind_kmh": rng.uniform(0, 20, 24),
        "precipitation_mm": rng.uniform(0, 2, 24),
        "condition": "Clear",
        "heat_index": rng.uniform(0, 25, 24),
        "comfort_score": rng.uniform(50, 100, 24),
    })
    report = default_weather_checker().run(df)
    assert report.all_passed, report.to_frame()


def test_default_traffic_checker_on_clean_frame():
    df = pd.DataFrame({
        "sensor_id": ["A"] * 24,
        "road": ["R"] * 24,
        "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
        "vehicle_count": [100] * 24,
        "avg_speed_kmh": [50.0] * 24,
        "occupancy_pct": [40.0] * 24,
        "congestion_index": [0.5] * 24,
        "peak_flag": [False] * 24,
        "free_flow_ratio": [0.5] * 24,
    })
    assert default_traffic_checker().run(df).all_passed


def test_default_sales_checker_on_clean_frame():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "product_id": [f"P-{i:04d}" for i in range(10)],
        "product_name": [f"Prod {i}" for i in range(10)],
        "category": ["Apparel"] * 10,
        "units_sold": [5] * 10,
        "unit_price": [10.0] * 10,
        "revenue": [50.0] * 10,
        "returns": [0] * 10,
        "margin": [0.5] * 10,
        "gross_profit": [25.0] * 10,
        "revenue_7d_avg": [50.0] * 10,
    })
    assert default_sales_checker().run(df).all_passed

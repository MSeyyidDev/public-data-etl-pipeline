"""Tests for data sources."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from etl.sources import (
    BusinessSalesDataSource,
    TrafficDataSource,
    WeatherDataSource,
)


def test_weather_source_shape():
    src = WeatherDataSource(cities=2, days=3, seed=1)
    df = src.extract()
    expected_rows = 2 * 3 * 24  # cities * days * hours
    assert len(df) == expected_rows
    expected_cols = {
        "city", "timestamp", "temperature_c", "humidity",
        "pressure_hpa", "wind_kmh", "precipitation_mm", "condition",
    }
    assert expected_cols.issubset(df.columns)
    assert df["timestamp"].dtype.kind == "M"  # datetime64
    # Nulls should have been injected.
    assert df.isna().sum().sum() > 0


def test_weather_source_is_deterministic():
    a = WeatherDataSource(cities=2, days=2, seed=7).extract()
    b = WeatherDataSource(cities=2, days=2, seed=7).extract()
    pd.testing.assert_frame_equal(a, b)


def test_traffic_source_shape():
    src = TrafficDataSource(sensors=3, days=2, seed=1)
    df = src.extract()
    assert len(df) == 3 * 2 * 24
    assert {"sensor_id", "road", "timestamp", "vehicle_count",
            "avg_speed_kmh", "occupancy_pct"}.issubset(df.columns)
    assert (df["vehicle_count"] >= 0).all()


def test_traffic_source_weekday_vs_weekend_pattern():
    src = TrafficDataSource(sensors=1, days=14, seed=2)
    df = src.extract()
    df["dow"] = df["timestamp"].dt.dayofweek
    weekday_8 = df[(df["dow"] < 5) & (df["timestamp"].dt.hour == 8)]["vehicle_count"].mean()
    weekend_8 = df[(df["dow"] >= 5) & (df["timestamp"].dt.hour == 8)]["vehicle_count"].mean()
    # 8am is rush hour on weekdays, much busier than weekends.
    assert weekday_8 > weekend_8


def test_sales_source_extract_all_keys():
    src = BusinessSalesDataSource(products=4, days=10, customers=20, seed=1)
    bundle = src.extract_all()
    assert set(bundle.keys()) == {"sales", "customers", "products"}
    assert len(bundle["sales"]) == 4 * 10
    assert len(bundle["customers"]) == 20
    assert len(bundle["products"]) == 4
    assert bundle["customers"]["customer_id"].is_unique
    assert bundle["products"]["product_id"].is_unique


def test_sales_extract_returns_fact_only():
    src = BusinessSalesDataSource(products=3, days=5, customers=10, seed=1)
    df = src.extract()
    assert isinstance(df, pd.DataFrame)
    assert {"date", "product_id", "units_sold", "unit_price", "revenue"}.issubset(df.columns)


def test_default_source_sizes_have_expected_magnitude():
    # Weather: 5 * 365 * 24 = 43_800
    src = WeatherDataSource(cities=5, days=365, seed=42)
    # Use a fast path: just check the planned timestamp count without retaining the frame.
    n = len(src.extract())
    assert n == 43_800

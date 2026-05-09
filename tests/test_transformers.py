"""Tests for transformers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.transformers import (
    BusinessSalesTransformer,
    TrafficTransformer,
    WeatherTransformer,
)


# ---- weather ----

def test_weather_transformer_handles_nulls(small_weather_raw):
    t = WeatherTransformer()
    df = t.transform(small_weather_raw)
    # No nulls remain in numeric columns.
    for col in ("temperature_c", "humidity", "pressure_hpa", "wind_kmh", "precipitation_mm"):
        assert df[col].isna().sum() == 0
    assert t.nulls_handled > 0


def test_weather_transformer_drops_outliers_into_imputation(small_weather_raw):
    t = WeatherTransformer()
    df = t.transform(small_weather_raw)
    # Out-of-range values in raw should not survive.
    assert df["temperature_c"].max() < 60
    assert df["temperature_c"].min() > -60
    assert df["humidity"].between(0, 100).all()


def test_weather_transformer_adds_derived_features(small_weather_raw):
    t = WeatherTransformer()
    df = t.transform(small_weather_raw)
    for col in ("heat_index", "comfort_score", "is_extreme", "date", "hour"):
        assert col in df.columns
    assert df["comfort_score"].between(0, 100).all()
    assert df["hour"].between(0, 23).all()


def test_weather_transformer_recomputes_missing_condition(small_weather_raw):
    t = WeatherTransformer()
    df = t.transform(small_weather_raw)
    # No empty conditions remain.
    assert (df["condition"].astype(str).str.len() > 0).all()


def test_weather_transformer_empty_input():
    t = WeatherTransformer()
    df = t.transform(pd.DataFrame())
    assert df.empty
    assert t.nulls_handled == 0


# ---- traffic ----

def test_traffic_transformer_no_nulls(small_traffic_raw):
    t = TrafficTransformer()
    df = t.transform(small_traffic_raw)
    assert df["avg_speed_kmh"].isna().sum() == 0
    assert df["occupancy_pct"].isna().sum() == 0
    assert t.nulls_handled > 0


def test_traffic_transformer_derived_features(small_traffic_raw):
    t = TrafficTransformer()
    df = t.transform(small_traffic_raw)
    for col in ("hour", "day_of_week", "is_weekend", "peak_flag",
                "congestion_index", "free_flow_ratio"):
        assert col in df.columns
    assert df["congestion_index"].between(0, 1).all()
    assert df["free_flow_ratio"].between(0, 1).all()


def test_traffic_transformer_peak_flag_logic():
    # Build a minimal frame: weekday 8am should be peak, Sunday 8am should not.
    df = pd.DataFrame([
        {"sensor_id": "X", "road": "R", "timestamp": "2025-01-06 08:00",  # Mon
         "vehicle_count": 100, "avg_speed_kmh": 50.0, "occupancy_pct": 60.0},
        {"sensor_id": "X", "road": "R", "timestamp": "2025-01-05 08:00",  # Sun
         "vehicle_count": 100, "avg_speed_kmh": 50.0, "occupancy_pct": 60.0},
    ])
    out = TrafficTransformer().transform(df)
    out_sorted = out.sort_values("timestamp").reset_index(drop=True)
    assert out_sorted.loc[0, "peak_flag"] == False  # Sun
    assert out_sorted.loc[1, "peak_flag"] == True   # Mon


def test_traffic_transformer_empty():
    df = TrafficTransformer().transform(pd.DataFrame())
    assert df.empty


# ---- sales ----

def test_sales_transformer_basic(small_sales_raw, small_products_dim):
    t = BusinessSalesTransformer(products_dim=small_products_dim)
    df = t.transform(small_sales_raw)
    for col in ("revenue", "margin", "gross_profit", "revenue_7d_avg",
                "year_month", "monthly_revenue", "mom_growth", "net_revenue"):
        assert col in df.columns
    assert df["unit_price"].isna().sum() == 0
    assert t.nulls_handled >= 1


def test_sales_transformer_revenue_is_units_times_price(small_sales_raw, small_products_dim):
    t = BusinessSalesTransformer(products_dim=small_products_dim)
    df = t.transform(small_sales_raw)
    diff = (df["revenue"] - df["units_sold"] * df["unit_price"]).abs()
    assert (diff < 0.01).all()


def test_sales_transformer_margin_in_range(small_sales_raw, small_products_dim):
    t = BusinessSalesTransformer(products_dim=small_products_dim)
    df = t.transform(small_sales_raw)
    assert df["margin"].between(-1.0, 1.0).all()


def test_sales_transformer_without_products_dim(small_sales_raw):
    t = BusinessSalesTransformer(products_dim=None)
    df = t.transform(small_sales_raw)
    assert "margin" in df.columns
    # Fallback assumes 50% margin.
    assert (df["margin"] == 0.5).all()


def test_sales_transformer_empty():
    df = BusinessSalesTransformer().transform(pd.DataFrame())
    assert df.empty

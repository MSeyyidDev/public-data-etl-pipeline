"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_weather_raw() -> pd.DataFrame:
    """Tiny synthetic weather frame for fast unit tests."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range(datetime(2025, 1, 1), periods=48, freq="h")
    rows = []
    for city in ("Berlin", "Tokyo"):
        for ts in timestamps:
            rows.append(
                {
                    "city": city,
                    "timestamp": ts,
                    "temperature_c": float(rng.normal(15, 5)),
                    "humidity": float(rng.uniform(40, 90)),
                    "pressure_hpa": float(rng.uniform(995, 1025)),
                    "wind_kmh": float(rng.uniform(0, 30)),
                    "precipitation_mm": float(max(0.0, rng.normal(0.5, 1.0))),
                    "condition": "Cloudy",
                }
            )
    df = pd.DataFrame(rows)
    # Inject a few nulls and one out-of-range outlier.
    df.loc[3, "temperature_c"] = np.nan
    df.loc[10, "humidity"] = np.nan
    df.loc[20, "temperature_c"] = 200.0  # out of range
    df.loc[25, "humidity"] = -10.0       # out of range
    df.loc[30, "condition"] = ""
    return df


@pytest.fixture(scope="session")
def small_traffic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range(datetime(2025, 1, 1), periods=48, freq="h")
    rows = []
    for sid, road in (("S-001", "A1-North"), ("S-002", "B96-City")):
        for ts in timestamps:
            rows.append(
                {
                    "sensor_id": sid,
                    "road": road,
                    "timestamp": ts,
                    "vehicle_count": int(rng.integers(0, 1500)),
                    "avg_speed_kmh": float(rng.uniform(15, 110)),
                    "occupancy_pct": float(rng.uniform(5, 95)),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[5, "avg_speed_kmh"] = np.nan
    df.loc[15, "occupancy_pct"] = np.nan
    return df


@pytest.fixture(scope="session")
def small_sales_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(datetime(2025, 1, 1), periods=60, freq="D")
    rows = []
    products = [
        ("P-0001", "Cool Apparel", "Apparel", 12.0, 24.0),
        ("P-0002", "Hot Electronic", "Electronics", 60.0, 120.0),
    ]
    for pid, pname, cat, cost, price in products:
        for d in dates:
            units = int(rng.poisson(8))
            rows.append({
                "date": d,
                "product_id": pid,
                "product_name": pname,
                "category": cat,
                "units_sold": units,
                "unit_price": price,
                "revenue": units * price,
                "returns": int(rng.binomial(units, 0.02)),
            })
    df = pd.DataFrame(rows)
    df.loc[5, "unit_price"] = np.nan
    return df


@pytest.fixture(scope="session")
def small_products_dim() -> pd.DataFrame:
    return pd.DataFrame([
        {"product_id": "P-0001", "product_name": "Cool Apparel",
         "category": "Apparel", "unit_cost": 12.0, "list_price": 24.0},
        {"product_id": "P-0002", "product_name": "Hot Electronic",
         "category": "Electronics", "unit_cost": 60.0, "list_price": 120.0},
    ])

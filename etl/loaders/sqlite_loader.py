"""SQLite loader using SQLAlchemy 2.0 with explicit dtype mapping."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine,
)
from sqlalchemy.engine import Engine

from etl.loaders.base import Loader


# Per-table column type hints. Columns not listed default to inferred types via pandas.
TABLE_COLUMN_TYPES: dict[str, dict[str, Any]] = {
    "weather_facts": {
        "city": String(64),
        "timestamp": DateTime,
        "temperature_c": Float,
        "humidity": Float,
        "pressure_hpa": Float,
        "wind_kmh": Float,
        "precipitation_mm": Float,
        "condition": String(32),
        "heat_index": Float,
        "comfort_score": Float,
        "is_extreme": Boolean,
        "date": String(16),
        "hour": Integer,
    },
    "traffic_facts": {
        "sensor_id": String(16),
        "road": String(64),
        "timestamp": DateTime,
        "vehicle_count": Integer,
        "avg_speed_kmh": Float,
        "occupancy_pct": Float,
        "hour": Integer,
        "day_of_week": String(16),
        "is_weekend": Boolean,
        "peak_flag": Boolean,
        "congestion_index": Float,
        "free_flow_ratio": Float,
    },
    "sales_facts": {
        "date": DateTime,
        "product_id": String(16),
        "product_name": String(128),
        "category": String(32),
        "units_sold": Integer,
        "unit_price": Float,
        "revenue": Float,
        "returns": Integer,
        "unit_cost": Float,
        "margin": Float,
        "gross_profit": Float,
        "revenue_7d_avg": Float,
        "year_month": String(8),
        "monthly_revenue": Float,
        "mom_growth": Float,
        "net_revenue": Float,
        "weekday": String(16),
        "is_weekend": Boolean,
    },
    "dim_customers": {
        "customer_id": String(16),
        "name": String(128),
        "email": String(128),
        "country": String(8),
        "signup_date": DateTime,
        "segment": String(16),
    },
    "dim_products": {
        "product_id": String(16),
        "product_name": String(128),
        "category": String(32),
        "unit_cost": Float,
        "list_price": Float,
    },
    "quality_reports": {
        "source": String(32),
        "check_name": String(64),
        "passed": Boolean,
        "details": String(512),
        "timestamp": DateTime,
    },
}


class SQLiteLoader(Loader):
    """Writes a DataFrame to a SQLite database table with declared types."""

    name = "sqlite"

    def __init__(self, url: str, if_exists: str = "replace") -> None:
        self.url = url
        self.if_exists = if_exists
        self._engine: Engine = create_engine(url, future=True)

    @property
    def engine(self) -> Engine:
        return self._engine

    def _ensure_table(self, df: pd.DataFrame, table: str) -> None:
        if table not in TABLE_COLUMN_TYPES:
            return  # let pandas to_sql infer
        meta = MetaData()
        columns: list[Column] = []
        for col in df.columns:
            sa_type = TABLE_COLUMN_TYPES[table].get(col, String(255))
            columns.append(Column(col, sa_type))
        Table(table, meta, *columns)
        with self._engine.begin() as conn:
            if self.if_exists == "replace":
                meta.drop_all(conn, checkfirst=True)
            meta.create_all(conn, checkfirst=True)

    def load(self, df: pd.DataFrame, table: str) -> int:
        self._ensure_table(df, table)
        # Use `append` once we've created the table ourselves to keep our types.
        mode = "append" if table in TABLE_COLUMN_TYPES else self.if_exists
        df.to_sql(table, self._engine, if_exists=mode, index=False, chunksize=5000)
        return len(df)

"""Traffic transformer: cleans nulls, derives congestion_index, peak_flag, etc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.transformers.base import Transformer


class TrafficTransformer(Transformer):
    """Adds congestion_index, peak_flag, day_of_week, is_weekend, free-flow ratio."""

    name = "traffic"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self._record_nulls(0)
            return df.copy()

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"])

        nulls_before = int(out[["avg_speed_kmh", "occupancy_pct"]].isna().sum().sum())

        # Forward/backward fill within each sensor for short gaps.
        out = out.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
        out[["avg_speed_kmh", "occupancy_pct"]] = (
            out.groupby("sensor_id")[["avg_speed_kmh", "occupancy_pct"]]
            .transform(lambda s: s.ffill().bfill())
        )

        # Vehicle count -- ensure non-negative integer.
        out["vehicle_count"] = out["vehicle_count"].clip(lower=0).astype(int)

        # Derived features.
        out["hour"] = out["timestamp"].dt.hour.astype(int)
        out["day_of_week"] = out["timestamp"].dt.day_name().astype("string")
        out["is_weekend"] = out["timestamp"].dt.dayofweek >= 5

        # Peak hours: 7-9 and 16-19 on weekdays.
        out["peak_flag"] = (
            (~out["is_weekend"])
            & (((out["hour"] >= 7) & (out["hour"] <= 9))
               | ((out["hour"] >= 16) & (out["hour"] <= 19)))
        )

        # Congestion index = 1 - speed/free_speed_per_road, clipped 0..1.
        free_speed = out.groupby("road")["avg_speed_kmh"].transform("max")
        congestion = 1.0 - (out["avg_speed_kmh"] / free_speed)
        out["congestion_index"] = congestion.clip(0.0, 1.0).round(3)
        out["free_flow_ratio"] = (out["avg_speed_kmh"] / free_speed).clip(0.0, 1.0).round(3)

        out["sensor_id"] = out["sensor_id"].astype("string")
        out["road"] = out["road"].astype("string")
        for col in ("avg_speed_kmh", "occupancy_pct"):
            out[col] = out[col].astype(float).round(2)

        self._record_nulls(nulls_before)
        return out.reset_index(drop=True)

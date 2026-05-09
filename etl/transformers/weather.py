"""Weather transformer: clean nulls, drop impossible values, derive features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.transformers.base import Transformer


# Plausible physical ranges. Values outside these are treated as data errors.
RANGES = {
    "temperature_c": (-50.0, 55.0),
    "humidity": (0.0, 100.0),
    "pressure_hpa": (900.0, 1080.0),
    "wind_kmh": (0.0, 250.0),
    "precipitation_mm": (0.0, 400.0),
}


class WeatherTransformer(Transformer):
    """Cleans weather data and derives heat_index, comfort_score, is_extreme."""

    name = "weather"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self._record_nulls(0)
            return df.copy()

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"])

        # Replace out-of-range values with NaN before imputation.
        for col, (lo, hi) in RANGES.items():
            if col in out.columns:
                bad = (out[col] < lo) | (out[col] > hi)
                out.loc[bad, col] = np.nan

        nulls_before = int(out[list(RANGES)].isna().sum().sum())

        # Impute numeric columns by per-city per-month mean, falling back to global mean.
        out["_month"] = out["timestamp"].dt.month
        for col in RANGES:
            if col not in out.columns:
                continue
            grp_mean = out.groupby(["city", "_month"])[col].transform("mean")
            out[col] = out[col].fillna(grp_mean)
            out[col] = out[col].fillna(out[col].mean())

        out = out.drop(columns="_month")

        # Categorical: fill missing condition by recomputing from numerics.
        if "condition" in out.columns:
            missing_cond = out["condition"].isna() | (out["condition"].astype(str).str.len() == 0)
            if missing_cond.any():
                out.loc[missing_cond, "condition"] = self._infer_condition(out.loc[missing_cond])

        # Derived features.
        out["heat_index"] = self._heat_index(out["temperature_c"], out["humidity"])
        out["comfort_score"] = self._comfort_score(
            out["temperature_c"], out["humidity"], out["wind_kmh"]
        )
        out["is_extreme"] = (
            (out["temperature_c"] >= 35.0)
            | (out["temperature_c"] <= -10.0)
            | (out["wind_kmh"] >= 80.0)
            | (out["precipitation_mm"] >= 20.0)
        )
        out["date"] = out["timestamp"].dt.date.astype("string")
        out["hour"] = out["timestamp"].dt.hour.astype(int)

        # Final type pass.
        out["city"] = out["city"].astype("string")
        out["condition"] = out["condition"].astype("string")
        for col in ("temperature_c", "humidity", "pressure_hpa", "wind_kmh", "precipitation_mm",
                    "heat_index", "comfort_score"):
            out[col] = out[col].astype(float).round(2)

        self._record_nulls(nulls_before)
        return out.reset_index(drop=True)

    @staticmethod
    def _heat_index(t_c: pd.Series, rh: pd.Series) -> pd.Series:
        """Approximate heat index in Celsius (Steadman formula simplification)."""
        t_f = t_c * 9 / 5 + 32
        hi_f = (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 0.00683783 * t_f**2
            - 0.05481717 * rh**2
            + 0.00122874 * t_f**2 * rh
            + 0.00085282 * t_f * rh**2
            - 0.00000199 * t_f**2 * rh**2
        )
        # Below 27 C the heat index isn't physical; fall back to the temperature.
        hi_c = (hi_f - 32) * 5 / 9
        return np.where(t_c >= 27, hi_c, t_c)

    @staticmethod
    def _comfort_score(t_c: pd.Series, rh: pd.Series, wind: pd.Series) -> pd.Series:
        """0..100 comfort score: 100 = ideal (~22 C, 50% RH, gentle breeze)."""
        ideal_t = 22.0
        ideal_rh = 50.0
        t_term = 1.0 - np.minimum(np.abs(t_c - ideal_t) / 25.0, 1.0)
        rh_term = 1.0 - np.minimum(np.abs(rh - ideal_rh) / 50.0, 1.0)
        wind_term = 1.0 - np.minimum(wind / 80.0, 1.0)
        return (40 * t_term + 30 * rh_term + 30 * wind_term).clip(0, 100)

    @staticmethod
    def _infer_condition(df: pd.DataFrame) -> pd.Series:
        precip = df["precipitation_mm"]
        humidity = df["humidity"]
        temp = df["temperature_c"]
        return pd.Series(
            np.where(precip > 5.0, "Storm",
            np.where(precip > 0.2, "Rain",
            np.where(humidity > 85.0, "Fog",
            np.where(temp < 0.0, "Snow",
            np.where(humidity < 35.0, "Clear", "Cloudy"))))),
            index=df.index,
        )

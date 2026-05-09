"""Synthetic but realistic weather data source.

Generates hourly observations for several cities over a year, with diurnal and
seasonal patterns, occasional storms (anomalies), realistic correlations
between humidity / temperature / precipitation, plus injected nulls and
out-of-range outliers so the cleaning step is non-trivial.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from etl.sources.base import DataSource


@dataclass(frozen=True)
class CityProfile:
    name: str
    base_temp_c: float       # mean annual temperature
    seasonal_amp_c: float    # peak-to-trough seasonal amplitude
    diurnal_amp_c: float     # day-night swing
    base_humidity: float     # %
    base_pressure: float     # hPa
    storm_prob: float        # per-hour
    storm_intensity: float   # mm precipitation when storming


DEFAULT_CITIES: tuple[CityProfile, ...] = (
    CityProfile("Berlin",    9.5,  18.0, 8.0, 70.0, 1013.0, 0.012, 6.0),
    CityProfile("Madrid",    15.0, 22.0, 12.0, 55.0, 1015.0, 0.006, 4.0),
    CityProfile("Reykjavik", 5.0,  12.0, 5.0, 80.0, 1009.0, 0.020, 8.0),
    CityProfile("Cairo",     22.0, 14.0, 14.0, 45.0, 1014.0, 0.002, 2.0),
    CityProfile("Tokyo",     16.0, 20.0, 9.0, 65.0, 1012.0, 0.010, 7.0),
)


class WeatherDataSource(DataSource):
    """Hourly weather observations across multiple cities."""

    name = "weather"

    def __init__(
        self,
        cities: int = 5,
        days: int = 365,
        start: datetime | None = None,
        seed: int = 42,
        null_rate: float = 0.02,
        outlier_rate: float = 0.001,
    ) -> None:
        self.cities = cities
        self.days = days
        self.start = start or datetime(2025, 1, 1)
        self.seed = seed
        self.null_rate = null_rate
        self.outlier_rate = outlier_rate

    def _city_profiles(self) -> list[CityProfile]:
        if self.cities <= len(DEFAULT_CITIES):
            return list(DEFAULT_CITIES[: self.cities])
        # Repeat with suffixed names if more cities are requested.
        out: list[CityProfile] = []
        i = 0
        while len(out) < self.cities:
            base = DEFAULT_CITIES[i % len(DEFAULT_CITIES)]
            suffix = "" if i < len(DEFAULT_CITIES) else f"-{i // len(DEFAULT_CITIES)}"
            out.append(CityProfile(
                name=f"{base.name}{suffix}",
                base_temp_c=base.base_temp_c,
                seasonal_amp_c=base.seasonal_amp_c,
                diurnal_amp_c=base.diurnal_amp_c,
                base_humidity=base.base_humidity,
                base_pressure=base.base_pressure,
                storm_prob=base.storm_prob,
                storm_intensity=base.storm_intensity,
            ))
            i += 1
        return out

    def _generate_for_city(
        self, city: CityProfile, timestamps: pd.DatetimeIndex, rng: np.random.Generator
    ) -> pd.DataFrame:
        n = len(timestamps)
        day_of_year = np.asarray(timestamps.dayofyear)
        hour = np.asarray(timestamps.hour)

        # Seasonal: cosine peak around day 200 (mid-summer in northern hemisphere).
        seasonal = city.seasonal_amp_c / 2 * np.cos(
            2 * np.pi * (day_of_year - 200) / 365.0
        )
        # Diurnal: warmest around 15:00, coldest around 03:00.
        diurnal = city.diurnal_amp_c / 2 * np.cos(2 * np.pi * (hour - 15) / 24.0)
        noise = rng.normal(0.0, 1.5, n)
        temperature_c = city.base_temp_c + seasonal + diurnal + noise

        # Storms: Poisson-like flag.
        storm = rng.random(n) < city.storm_prob
        precipitation_mm = np.where(storm, rng.gamma(2.0, city.storm_intensity / 2.0, n), 0.0)
        # Light rain occasionally even without storms.
        light_rain = (rng.random(n) < 0.05) & (~storm)
        precipitation_mm = np.where(
            light_rain, rng.gamma(1.5, 0.4, n), precipitation_mm
        )

        humidity = (
            city.base_humidity
            + 12.0 * (precipitation_mm > 0).astype(float)
            - 0.4 * (temperature_c - city.base_temp_c)
            + rng.normal(0.0, 4.0, n)
        )
        humidity = np.clip(humidity, 5.0, 100.0)

        pressure_hpa = (
            city.base_pressure
            - 4.0 * (precipitation_mm > 0).astype(float)
            + rng.normal(0.0, 2.0, n)
        )

        wind_kmh = np.clip(
            rng.gamma(2.0, 5.0, n) + 12.0 * storm.astype(float),
            0.0,
            220.0,
        )

        # Categorical condition derived from numerics.
        condition = np.where(
            precipitation_mm > 5.0, "Storm",
            np.where(precipitation_mm > 0.2, "Rain",
            np.where(humidity > 85.0, "Fog",
            np.where(temperature_c < 0.0, "Snow",
            np.where(humidity < 35.0, "Clear", "Cloudy")))))

        df = pd.DataFrame(
            {
                "city": city.name,
                "timestamp": timestamps,
                "temperature_c": temperature_c.round(2),
                "humidity": humidity.round(1),
                "pressure_hpa": pressure_hpa.round(1),
                "wind_kmh": wind_kmh.round(1),
                "precipitation_mm": precipitation_mm.round(2),
                "condition": condition,
            }
        )
        return df

    def _inject_nulls_and_outliers(
        self, df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        n = len(df)
        numeric_cols = [
            "temperature_c", "humidity", "pressure_hpa", "wind_kmh", "precipitation_mm"
        ]
        for col in numeric_cols:
            mask = rng.random(n) < self.null_rate
            df.loc[mask, col] = np.nan
        # Out-of-range outliers.
        n_out = max(1, int(n * self.outlier_rate))
        idx = rng.choice(n, size=n_out, replace=False)
        df.loc[idx, "temperature_c"] = rng.choice([-99.0, 150.0], size=n_out)
        # A few impossible humidity values.
        n_h = max(1, n_out // 2)
        idx_h = rng.choice(n, size=n_h, replace=False)
        df.loc[idx_h, "humidity"] = rng.choice([-5.0, 130.0], size=n_h)
        return df

    def extract(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        end = self.start + timedelta(days=self.days)
        timestamps = pd.date_range(self.start, end, freq="h", inclusive="left")
        cities = self._city_profiles()
        frames = [self._generate_for_city(c, timestamps, rng) for c in cities]
        df = pd.concat(frames, ignore_index=True)
        df = self._inject_nulls_and_outliers(df, rng)
        return df

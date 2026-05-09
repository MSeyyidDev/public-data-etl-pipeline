"""Synthetic traffic sensor data with realistic weekday/weekend rush-hour patterns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from etl.sources.base import DataSource


@dataclass(frozen=True)
class RoadProfile:
    road: str
    capacity: int       # vehicles/hour at full flow
    free_speed: float   # km/h at low occupancy
    weekend_factor: float  # how much busier (or quieter) on weekends


DEFAULT_ROADS: tuple[RoadProfile, ...] = (
    RoadProfile("A1-North",        2400, 110.0, 0.6),
    RoadProfile("A1-South",        2400, 110.0, 0.6),
    RoadProfile("A100-Ring-East",  2800, 100.0, 0.7),
    RoadProfile("A100-Ring-West",  2800, 100.0, 0.7),
    RoadProfile("B96-City",        1400, 60.0, 0.8),
    RoadProfile("B1-Boulevard",    1200, 50.0, 0.9),
    RoadProfile("Hauptstrasse",    900,  45.0, 1.0),
    RoadProfile("Marktplatz",      700,  35.0, 1.2),
    RoadProfile("Industriegebiet", 1100, 60.0, 0.4),
    RoadProfile("Hafenstrasse",    1300, 65.0, 0.5),
)


class TrafficDataSource(DataSource):
    """Hourly readings for a fleet of road sensors."""

    name = "traffic"

    def __init__(
        self,
        sensors: int = 20,
        days: int = 90,
        start: datetime | None = None,
        seed: int = 42,
        null_rate: float = 0.01,
    ) -> None:
        self.sensors = sensors
        self.days = days
        self.start = start or datetime(2025, 7, 1)
        self.seed = seed
        self.null_rate = null_rate

    def _sensor_assignments(self) -> list[tuple[str, RoadProfile]]:
        out: list[tuple[str, RoadProfile]] = []
        for i in range(self.sensors):
            road = DEFAULT_ROADS[i % len(DEFAULT_ROADS)]
            sensor_id = f"S-{i + 1:03d}"
            out.append((sensor_id, road))
        return out

    @staticmethod
    def _hour_demand_curve(hours: np.ndarray, weekend: np.ndarray) -> np.ndarray:
        """Produce a 0..1 demand multiplier per hour."""
        morning = np.exp(-0.5 * ((hours - 8) / 1.6) ** 2)
        evening = np.exp(-0.5 * ((hours - 17) / 1.8) ** 2)
        weekday_curve = 0.15 + 0.85 * np.maximum(morning, evening)
        midday = np.exp(-0.5 * ((hours - 13) / 3.0) ** 2)
        weekend_curve = 0.20 + 0.55 * midday
        return np.where(weekend, weekend_curve, weekday_curve)

    def _generate_for_sensor(
        self,
        sensor_id: str,
        road: RoadProfile,
        timestamps: pd.DatetimeIndex,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = len(timestamps)
        hours = np.asarray(timestamps.hour)
        weekend = np.asarray(timestamps.dayofweek) >= 5
        demand = self._hour_demand_curve(hours, weekend)
        weekend_mult = np.where(weekend, road.weekend_factor, 1.0)
        noise = rng.normal(1.0, 0.08, n)

        vehicle_count = (road.capacity * demand * weekend_mult * noise).clip(0)
        vehicle_count = vehicle_count.round().astype(int)

        occupancy_pct = (
            100.0 * (vehicle_count / road.capacity).clip(0, 1.05)
            + rng.normal(0.0, 2.0, n)
        ).clip(0.0, 100.0)

        avg_speed_kmh = road.free_speed / (1.0 + 0.6 * (occupancy_pct / 100.0) ** 4)
        avg_speed_kmh = (avg_speed_kmh + rng.normal(0.0, 2.0, n)).clip(5.0, 200.0)

        df = pd.DataFrame(
            {
                "sensor_id": sensor_id,
                "road": road.road,
                "timestamp": timestamps,
                "vehicle_count": vehicle_count,
                "avg_speed_kmh": avg_speed_kmh.round(1),
                "occupancy_pct": occupancy_pct.round(1),
            }
        )
        return df

    def _inject_nulls(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        n = len(df)
        for col in ("avg_speed_kmh", "occupancy_pct"):
            mask = rng.random(n) < self.null_rate
            df.loc[mask, col] = np.nan
        return df

    def extract(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1)
        end = self.start + timedelta(days=self.days)
        timestamps = pd.date_range(self.start, end, freq="h", inclusive="left")
        frames = [
            self._generate_for_sensor(sid, road, timestamps, rng)
            for sid, road in self._sensor_assignments()
        ]
        df = pd.concat(frames, ignore_index=True)
        df = self._inject_nulls(df, rng)
        return df

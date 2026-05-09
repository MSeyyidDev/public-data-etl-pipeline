"""Composable data-quality checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class QualityCheckResult:
    source: str
    check_name: str
    passed: bool
    details: str = ""
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "check_name": self.check_name,
            "passed": self.passed,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class QualityReport:
    results: list[QualityCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])


class DataQualityChecker:
    """Runs schema / nullability / range / duplicate checks against a DataFrame."""

    def __init__(
        self,
        source: str,
        required_columns: Iterable[str] = (),
        non_null_columns: Iterable[str] = (),
        ranges: dict[str, tuple[float, float]] | None = None,
        unique_subset: Iterable[str] | None = None,
        max_null_fraction: float = 0.0,
    ) -> None:
        self.source = source
        self.required_columns = tuple(required_columns)
        self.non_null_columns = tuple(non_null_columns)
        self.ranges = ranges or {}
        self.unique_subset = tuple(unique_subset) if unique_subset else None
        self.max_null_fraction = max_null_fraction

    def run(self, df: pd.DataFrame) -> QualityReport:
        results: list[QualityCheckResult] = []

        # Non-empty
        results.append(QualityCheckResult(
            self.source, "non_empty", not df.empty,
            f"row_count={len(df)}",
        ))

        # Schema: required columns present
        missing = [c for c in self.required_columns if c not in df.columns]
        results.append(QualityCheckResult(
            self.source, "schema_required_columns", not missing,
            f"missing={missing}" if missing else "all required columns present",
        ))

        # Non-null columns
        for col in self.non_null_columns:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            frac = null_count / max(len(df), 1)
            passed = frac <= self.max_null_fraction
            results.append(QualityCheckResult(
                self.source, f"non_null:{col}", passed,
                f"nulls={null_count} fraction={frac:.4f} threshold={self.max_null_fraction}",
            ))

        # Range checks
        for col, (lo, hi) in self.ranges.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                results.append(QualityCheckResult(
                    self.source, f"range:{col}", True, "no data after dropna",
                ))
                continue
            out_of_range = int(((series < lo) | (series > hi)).sum())
            results.append(QualityCheckResult(
                self.source, f"range:{col}", out_of_range == 0,
                f"out_of_range={out_of_range} bounds=[{lo},{hi}]",
            ))

        # Duplicate check on subset
        if self.unique_subset:
            dup = int(df.duplicated(subset=list(self.unique_subset)).sum())
            results.append(QualityCheckResult(
                self.source, f"unique:{','.join(self.unique_subset)}", dup == 0,
                f"duplicates={dup}",
            ))

        return QualityReport(results=results)


def default_weather_checker() -> DataQualityChecker:
    return DataQualityChecker(
        source="weather",
        required_columns=[
            "city", "timestamp", "temperature_c", "humidity",
            "pressure_hpa", "wind_kmh", "precipitation_mm", "condition",
            "heat_index", "comfort_score",
        ],
        non_null_columns=[
            "city", "timestamp", "temperature_c", "humidity",
            "pressure_hpa", "wind_kmh", "precipitation_mm",
        ],
        ranges={
            "temperature_c": (-50.0, 55.0),
            "humidity": (0.0, 100.0),
            "pressure_hpa": (900.0, 1080.0),
            "wind_kmh": (0.0, 250.0),
            "precipitation_mm": (0.0, 400.0),
            "comfort_score": (0.0, 100.0),
        },
        unique_subset=["city", "timestamp"],
    )


def default_traffic_checker() -> DataQualityChecker:
    return DataQualityChecker(
        source="traffic",
        required_columns=[
            "sensor_id", "road", "timestamp", "vehicle_count",
            "avg_speed_kmh", "occupancy_pct", "congestion_index", "peak_flag",
        ],
        non_null_columns=[
            "sensor_id", "road", "timestamp", "vehicle_count",
            "avg_speed_kmh", "occupancy_pct",
        ],
        ranges={
            "vehicle_count": (0, 10_000),
            "avg_speed_kmh": (0.0, 250.0),
            "occupancy_pct": (0.0, 100.0),
            "congestion_index": (0.0, 1.0),
            "free_flow_ratio": (0.0, 1.0),
        },
        unique_subset=["sensor_id", "timestamp"],
    )


def default_sales_checker() -> DataQualityChecker:
    return DataQualityChecker(
        source="sales",
        required_columns=[
            "date", "product_id", "product_name", "category",
            "units_sold", "unit_price", "revenue", "returns",
            "margin", "gross_profit", "revenue_7d_avg",
        ],
        non_null_columns=[
            "date", "product_id", "category", "units_sold",
            "unit_price", "revenue",
        ],
        ranges={
            "units_sold": (0, 1_000_000),
            "unit_price": (0.0, 1_000_000.0),
            "revenue": (0.0, 1e12),
            "returns": (0, 1_000_000),
            "margin": (-1.0, 1.0),
        },
        unique_subset=["product_id", "date"],
    )

"""Project-wide configuration loaded from environment with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Construct via :py:meth:`load`."""

    project_root: Path
    output_dir: Path
    db_filename: str
    random_seed: int

    weather_cities: int
    weather_days: int

    traffic_sensors: int
    traffic_days: int

    sales_products: int
    sales_days: int
    sales_customers: int

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = project_root or Path(__file__).resolve().parent.parent
        out = Path(os.getenv("ETL_OUTPUT_DIR", "output"))
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            output_dir=out,
            db_filename=os.getenv("ETL_DB_FILENAME", "etl.db"),
            random_seed=_int("ETL_RANDOM_SEED", 42),
            weather_cities=_int("ETL_WEATHER_CITIES", 5),
            weather_days=_int("ETL_WEATHER_DAYS", 365),
            traffic_sensors=_int("ETL_TRAFFIC_SENSORS", 20),
            traffic_days=_int("ETL_TRAFFIC_DAYS", 90),
            sales_products=_int("ETL_SALES_PRODUCTS", 50),
            sales_days=_int("ETL_SALES_DAYS", 365),
            sales_customers=_int("ETL_SALES_CUSTOMERS", 1000),
        )

    @property
    def db_path(self) -> Path:
        return self.output_dir / self.db_filename

    @property
    def sqlite_url(self) -> str:
        # Use forward slashes for SQLAlchemy URL portability.
        return f"sqlite:///{self.db_path.as_posix()}"


SETTINGS = Settings.load()

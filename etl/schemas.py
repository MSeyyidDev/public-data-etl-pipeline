"""Pydantic v2 row-level schemas. Used for declarative validation in tests
and (optionally) for sampling rows during quality checks."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WeatherRow(BaseModel):
    city: str
    timestamp: datetime
    temperature_c: float = Field(..., ge=-60.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    pressure_hpa: float = Field(..., ge=850.0, le=1100.0)
    wind_kmh: float = Field(..., ge=0.0, le=300.0)
    precipitation_mm: float = Field(..., ge=0.0, le=500.0)
    condition: str

    @field_validator("city", "condition")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v


class TrafficRow(BaseModel):
    sensor_id: str
    road: str
    timestamp: datetime
    vehicle_count: int = Field(..., ge=0, le=10_000)
    avg_speed_kmh: float = Field(..., ge=0.0, le=250.0)
    occupancy_pct: float = Field(..., ge=0.0, le=100.0)


class SalesRow(BaseModel):
    date: datetime
    product_id: str
    product_name: str
    category: str
    units_sold: int = Field(..., ge=0)
    unit_price: float = Field(..., ge=0.0)
    revenue: float = Field(..., ge=0.0)
    returns: int = Field(..., ge=0)


class CustomerRow(BaseModel):
    customer_id: str
    name: str
    email: str
    country: str
    signup_date: datetime
    segment: str


class ProductRow(BaseModel):
    product_id: str
    product_name: str
    category: str
    unit_cost: float = Field(..., ge=0.0)
    list_price: float = Field(..., ge=0.0)


class StageMetric(BaseModel):
    stage: str
    rows_in: int = 0
    rows_out: int = 0
    nulls_handled: int = 0
    duration_seconds: float = 0.0
    notes: Optional[str] = None

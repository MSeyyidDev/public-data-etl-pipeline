"""Factories for assembling pipelines per source."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from etl.config import SETTINGS, Settings
from etl.loaders import CSVLoader, Loader, ParquetLoader, SQLiteLoader
from etl.pipeline import Pipeline
from etl.quality.checker import (
    DataQualityChecker,
    default_sales_checker,
    default_traffic_checker,
    default_weather_checker,
)
from etl.sources import (
    BusinessSalesDataSource,
    DataSource,
    TrafficDataSource,
    WeatherDataSource,
)
from etl.transformers import (
    BusinessSalesTransformer,
    Transformer,
    TrafficTransformer,
    WeatherTransformer,
)


SOURCE_NAMES = ("weather", "traffic", "sales")


def build_default_loaders(settings: Settings | None = None) -> list[Loader]:
    s = settings or SETTINGS
    return [SQLiteLoader(s.sqlite_url), ParquetLoader(s.output_dir)]


def build_source(name: str, settings: Settings | None = None) -> DataSource:
    s = settings or SETTINGS
    if name == "weather":
        return WeatherDataSource(
            cities=s.weather_cities, days=s.weather_days, seed=s.random_seed,
        )
    if name == "traffic":
        return TrafficDataSource(
            sensors=s.traffic_sensors, days=s.traffic_days, seed=s.random_seed,
        )
    if name == "sales":
        return BusinessSalesDataSource(
            products=s.sales_products,
            days=s.sales_days,
            customers=s.sales_customers,
            seed=s.random_seed,
        )
    raise ValueError(f"Unknown source '{name}'. Valid: {SOURCE_NAMES}")


def build_transformer(name: str, products_dim: pd.DataFrame | None = None) -> Transformer:
    if name == "weather":
        return WeatherTransformer()
    if name == "traffic":
        return TrafficTransformer()
    if name == "sales":
        return BusinessSalesTransformer(products_dim=products_dim)
    raise ValueError(f"Unknown source '{name}'")


def build_quality_checker(name: str) -> DataQualityChecker:
    if name == "weather":
        return default_weather_checker()
    if name == "traffic":
        return default_traffic_checker()
    if name == "sales":
        return default_sales_checker()
    raise ValueError(f"Unknown source '{name}'")


TABLE_FOR_SOURCE = {
    "weather": "weather_facts",
    "traffic": "traffic_facts",
    "sales": "sales_facts",
}


def build_pipeline(
    name: str,
    settings: Settings | None = None,
    loaders: Iterable[Loader] | None = None,
    products_dim: pd.DataFrame | None = None,
) -> Pipeline:
    s = settings or SETTINGS
    source = build_source(name, s)
    transformer = build_transformer(name, products_dim=products_dim)
    qc = build_quality_checker(name)
    loaders_list = list(loaders) if loaders is not None else build_default_loaders(s)
    return Pipeline(
        source=source,
        transformer=transformer,
        loaders=loaders_list,
        table=TABLE_FOR_SOURCE[name],
        quality_checker=qc,
    )

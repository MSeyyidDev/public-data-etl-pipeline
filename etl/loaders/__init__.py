"""Loaders package."""
from etl.loaders.base import Loader
from etl.loaders.sqlite_loader import SQLiteLoader
from etl.loaders.csv_loader import CSVLoader
from etl.loaders.parquet_loader import ParquetLoader

__all__ = ["Loader", "SQLiteLoader", "CSVLoader", "ParquetLoader"]

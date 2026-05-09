"""Parquet snapshot loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl.loaders.base import Loader


class ParquetLoader(Loader):
    name = "parquet"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, df: pd.DataFrame, table: str) -> int:
        path = self.output_dir / f"{table}.parquet"
        # PyArrow handles pandas string dtype, but we coerce just in case.
        out = df.copy()
        for col in out.select_dtypes(include="string").columns:
            out[col] = out[col].astype(object)
        out.to_parquet(path, index=False)
        return len(df)

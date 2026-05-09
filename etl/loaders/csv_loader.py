"""CSV loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl.loaders.base import Loader


class CSVLoader(Loader):
    name = "csv"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, df: pd.DataFrame, table: str) -> int:
        path = self.output_dir / f"{table}.csv"
        df.to_csv(path, index=False)
        return len(df)

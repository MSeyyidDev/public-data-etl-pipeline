"""Loader abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    """Persists a DataFrame somewhere durable."""

    name: str = "abstract"

    @abstractmethod
    def load(self, df: pd.DataFrame, table: str) -> int:
        """Load `df` into `table` (or filename). Returns rows written."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"

"""Abstract base for transformers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Transformer(ABC):
    """A transformer takes a raw DataFrame and produces a cleaned, typed,
    feature-enriched DataFrame ready for loading."""

    name: str = "abstract"

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def nulls_handled(self) -> int:
        """How many null values the most recent transform fixed."""
        return getattr(self, "_nulls_handled", 0)

    def _record_nulls(self, count: int) -> None:
        self._nulls_handled = int(count)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"

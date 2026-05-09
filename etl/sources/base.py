"""Abstract base class for ETL data sources."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract data source. Concrete sources implement :py:meth:`extract`.

    Real-world public-API sources (e.g. NOAA, Open-Meteo, Eurostat, city open
    data portals) can be implemented as drop-in subclasses without changing the
    rest of the pipeline.
    """

    name: str = "abstract"

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Return the raw extracted DataFrame."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"

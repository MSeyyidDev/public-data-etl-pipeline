"""Synthetic e-commerce sales fact + customer/product dimension tables."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from etl.sources.base import DataSource


CATEGORIES = (
    "Apparel", "Electronics", "Home", "Beauty", "Sports", "Books", "Toys"
)

CUSTOMER_SEGMENTS = ("Bronze", "Silver", "Gold", "Platinum")


@dataclass(frozen=True)
class ProductSpec:
    product_id: str
    product_name: str
    category: str
    unit_cost: float
    list_price: float
    base_demand: float       # daily mean units
    seasonality: float       # 0..1, how much yearly seasonality
    promo_prob: float        # daily probability of a promo spike


class BusinessSalesDataSource(DataSource):
    """Daily product sales plus dimension tables (customers, products)."""

    name = "sales"

    def __init__(
        self,
        products: int = 50,
        days: int = 365,
        customers: int = 1000,
        start: datetime | None = None,
        seed: int = 42,
    ) -> None:
        self.products = products
        self.days = days
        self.customers = customers
        self.start = start or datetime(2025, 1, 1)
        self.seed = seed

    # ----- helpers ----------------------------------------------------------

    def _build_product_catalog(self, rng: np.random.Generator, fake: Faker) -> list[ProductSpec]:
        catalog: list[ProductSpec] = []
        for i in range(self.products):
            cat = CATEGORIES[i % len(CATEGORIES)]
            unit_cost = float(round(rng.uniform(3.0, 80.0), 2))
            margin = float(rng.uniform(0.25, 0.7))
            list_price = round(unit_cost / (1 - margin), 2)
            catalog.append(
                ProductSpec(
                    product_id=f"P-{i + 1:04d}",
                    product_name=f"{fake.word().title()} {cat[:-1] if cat.endswith('s') else cat}",
                    category=cat,
                    unit_cost=unit_cost,
                    list_price=list_price,
                    base_demand=float(rng.uniform(2.0, 25.0)),
                    seasonality=float(rng.uniform(0.1, 0.6)),
                    promo_prob=float(rng.uniform(0.01, 0.04)),
                )
            )
        return catalog

    def _build_customer_table(self, fake: Faker, rng: np.random.Generator) -> pd.DataFrame:
        # Sample customers with a distribution of segments.
        segments = rng.choice(
            CUSTOMER_SEGMENTS,
            size=self.customers,
            p=[0.45, 0.30, 0.18, 0.07],
        )
        countries = rng.choice(
            ["DE", "FR", "ES", "IT", "NL", "AT", "CH", "PL", "SE", "DK"],
            size=self.customers,
            p=[0.35, 0.15, 0.10, 0.10, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03],
        )
        signup_offsets = rng.integers(0, max(1, self.days), size=self.customers)
        rows = []
        for i in range(self.customers):
            rows.append(
                {
                    "customer_id": f"C-{i + 1:05d}",
                    "name": fake.name(),
                    "email": fake.unique.email(),
                    "country": countries[i],
                    "signup_date": self.start + timedelta(days=int(signup_offsets[i])),
                    "segment": segments[i],
                }
            )
        return pd.DataFrame(rows)

    def _build_product_dim(self, catalog: list[ProductSpec]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "unit_cost": p.unit_cost,
                    "list_price": p.list_price,
                }
                for p in catalog
            ]
        )

    def _build_fact(
        self,
        catalog: list[ProductSpec],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        dates = pd.date_range(self.start, periods=self.days, freq="D")
        day_of_year = dates.dayofyear.to_numpy()
        weekday = dates.dayofweek.to_numpy()

        rows: list[dict] = []
        for prod in catalog:
            seasonal = 1.0 + prod.seasonality * np.sin(
                2 * np.pi * (day_of_year - 60) / 365.0
            )
            weekday_boost = np.where(weekday >= 5, 1.25, 1.0)  # weekend bump
            growth = np.linspace(1.0, 1.15, num=self.days)     # mild yearly trend
            promo = rng.random(self.days) < prod.promo_prob
            promo_mult = np.where(promo, rng.uniform(1.6, 2.6, self.days), 1.0)

            mean_units = prod.base_demand * seasonal * weekday_boost * growth * promo_mult
            units = rng.poisson(mean_units)

            # Price discounts during promo
            unit_price = np.where(
                promo, prod.list_price * rng.uniform(0.7, 0.9, self.days), prod.list_price
            ).round(2)
            revenue = (units * unit_price).round(2)
            returns = rng.binomial(units, 0.02)

            for i in range(self.days):
                rows.append(
                    {
                        "date": dates[i],
                        "product_id": prod.product_id,
                        "product_name": prod.product_name,
                        "category": prod.category,
                        "units_sold": int(units[i]),
                        "unit_price": float(unit_price[i]),
                        "revenue": float(revenue[i]),
                        "returns": int(returns[i]),
                    }
                )

        df = pd.DataFrame(rows)
        # Inject a few nulls into unit_price to exercise the cleaning step.
        n = len(df)
        mask = rng.random(n) < 0.005
        df.loc[mask, "unit_price"] = np.nan
        return df

    # ----- public -----------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """Return the sales fact table. Dimension tables are exposed via
        :py:meth:`extract_all` for the loader to write side tables."""
        return self.extract_all()["sales"]

    def extract_all(self) -> dict[str, pd.DataFrame]:
        """Return fact + dimension DataFrames keyed by table name."""
        rng = np.random.default_rng(self.seed + 2)
        fake = Faker()
        Faker.seed(self.seed + 2)
        catalog = self._build_product_catalog(rng, fake)
        return {
            "sales": self._build_fact(catalog, rng),
            "customers": self._build_customer_table(fake, rng),
            "products": self._build_product_dim(catalog),
        }

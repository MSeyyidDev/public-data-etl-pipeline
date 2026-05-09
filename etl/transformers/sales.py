"""Sales transformer: derives revenue, margin, rolling avg, MoM growth, etc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from etl.transformers.base import Transformer


class BusinessSalesTransformer(Transformer):
    """Computes revenue, margin, rolling 7d average, MoM growth and time features."""

    name = "sales"

    def __init__(self, products_dim: pd.DataFrame | None = None) -> None:
        self.products_dim = products_dim

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self._record_nulls(0)
            return df.copy()

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])

        # Track null impact for unit_price.
        nulls_before = int(out["unit_price"].isna().sum())

        # Fill missing unit_price with per-product mean, falling back to category mean.
        prod_mean = out.groupby("product_id")["unit_price"].transform("mean")
        out["unit_price"] = out["unit_price"].fillna(prod_mean)
        cat_mean = out.groupby("category")["unit_price"].transform("mean")
        out["unit_price"] = out["unit_price"].fillna(cat_mean)
        out["unit_price"] = out["unit_price"].fillna(out["unit_price"].mean())

        # Recompute revenue defensively.
        out["revenue"] = (out["units_sold"] * out["unit_price"]).round(2)

        # Margin requires unit_cost from product dim.
        if self.products_dim is not None and "unit_cost" in self.products_dim.columns:
            cost_map = self.products_dim.set_index("product_id")["unit_cost"]
            out["unit_cost"] = out["product_id"].map(cost_map).astype(float)
            out["margin"] = (
                (out["unit_price"] - out["unit_cost"]) / out["unit_price"]
            ).round(4)
            out["gross_profit"] = (
                (out["unit_price"] - out["unit_cost"]) * out["units_sold"]
            ).round(2)
        else:
            # Fallback: assume 50% margin if cost unknown.
            out["unit_cost"] = (out["unit_price"] * 0.5).round(2)
            out["margin"] = 0.5
            out["gross_profit"] = (out["revenue"] * 0.5).round(2)

        out = out.sort_values(["product_id", "date"]).reset_index(drop=True)

        # Rolling 7-day average revenue per product.
        out["revenue_7d_avg"] = (
            out.groupby("product_id")["revenue"]
            .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
            .round(2)
        )

        # Month-over-month growth on monthly aggregated revenue per product.
        out["year_month"] = out["date"].dt.to_period("M").astype(str)
        monthly = (
            out.groupby(["product_id", "year_month"])["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "monthly_revenue"})
        )
        monthly["mom_growth"] = (
            monthly.groupby("product_id")["monthly_revenue"].pct_change().round(4)
        )
        out = out.merge(
            monthly[["product_id", "year_month", "monthly_revenue", "mom_growth"]],
            on=["product_id", "year_month"],
            how="left",
        )

        # Net revenue after returns (assume returns at unit_price).
        out["net_revenue"] = (out["revenue"] - out["returns"] * out["unit_price"]).round(2)

        out["weekday"] = out["date"].dt.day_name().astype("string")
        out["is_weekend"] = out["date"].dt.dayofweek >= 5

        # Type tightening.
        out["product_id"] = out["product_id"].astype("string")
        out["category"] = out["category"].astype("string")
        out["product_name"] = out["product_name"].astype("string")
        out["units_sold"] = out["units_sold"].clip(lower=0).astype(int)
        out["returns"] = out["returns"].clip(lower=0).astype(int)

        self._record_nulls(nulls_before)
        return out.reset_index(drop=True)

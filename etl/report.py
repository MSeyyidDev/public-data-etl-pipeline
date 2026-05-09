"""Optional HTML report generator (Plotly + Jinja2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import create_engine

from etl.config import SETTINGS


pio.templates.default = "plotly_white"


def _read_table(engine, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_table(table, engine)
    except Exception:  # noqa: BLE001 - any failure means table missing
        return pd.DataFrame()


def _weather_section(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"present": False}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.assign(date=df["timestamp"].dt.date)
    daily = (
        df.groupby(["date", "city"], as_index=False)
        .agg(temperature_c=("temperature_c", "mean"),
             precipitation_mm=("precipitation_mm", "sum"))
    )
    fig_temp = px.line(
        daily, x="date", y="temperature_c", color="city",
        title="Average daily temperature by city",
    )
    fig_precip = px.bar(
        daily, x="date", y="precipitation_mm", color="city",
        title="Daily precipitation totals by city",
    )
    return {
        "present": True,
        "rows": len(df),
        "cities": df["city"].nunique(),
        "time_min": str(df["timestamp"].min()),
        "time_max": str(df["timestamp"].max()),
        "fig_temp": fig_temp.to_html(full_html=False, include_plotlyjs=False),
        "fig_precip": fig_precip.to_html(full_html=False, include_plotlyjs=False),
    }


def _traffic_section(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"present": False}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hourly = df.groupby("hour", as_index=False)["vehicle_count"].mean()
    fig_hour = px.bar(
        hourly, x="hour", y="vehicle_count",
        title="Average vehicle count by hour of day",
    )
    by_road = (
        df.groupby("road", as_index=False)["congestion_index"].mean()
        .sort_values("congestion_index", ascending=False)
    )
    fig_road = px.bar(
        by_road, x="road", y="congestion_index",
        title="Mean congestion index by road",
    )
    return {
        "present": True,
        "rows": len(df),
        "sensors": df["sensor_id"].nunique(),
        "time_min": str(df["timestamp"].min()),
        "time_max": str(df["timestamp"].max()),
        "fig_hour": fig_hour.to_html(full_html=False, include_plotlyjs=False),
        "fig_road": fig_road.to_html(full_html=False, include_plotlyjs=False),
    }


def _sales_section(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"present": False}
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date", as_index=False)["revenue"].sum()
    fig_rev = px.line(daily, x="date", y="revenue", title="Daily total revenue")
    by_cat = (
        df.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"))
        .sort_values("revenue", ascending=False)
    )
    fig_cat = px.bar(
        by_cat, x="category", y="revenue", title="Revenue by category",
    )
    return {
        "present": True,
        "rows": len(df),
        "products": df["product_id"].nunique(),
        "time_min": str(df["date"].min()),
        "time_max": str(df["date"].max()),
        "total_revenue": float(df["revenue"].sum()),
        "fig_rev": fig_rev.to_html(full_html=False, include_plotlyjs=False),
        "fig_cat": fig_cat.to_html(full_html=False, include_plotlyjs=False),
    }


def _quality_section(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"present": False}
    return {
        "present": True,
        "checks": df.to_dict(orient="records"),
        "passed": int(df["passed"].sum()),
        "failed": int((~df["passed"].astype(bool)).sum()),
    }


def generate_html_report(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(SETTINGS.sqlite_url, future=True)
    weather = _weather_section(_read_table(engine, "weather_facts"))
    traffic = _traffic_section(_read_table(engine, "traffic_facts"))
    sales = _sales_section(_read_table(engine, "sales_facts"))
    quality = _quality_section(_read_table(engine, "quality_reports"))

    templates_dir = SETTINGS.project_root / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        weather=weather, traffic=traffic, sales=sales, quality=quality,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path

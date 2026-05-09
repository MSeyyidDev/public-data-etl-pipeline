"""Typer-based command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from etl.config import SETTINGS
from etl.factory import (
    SOURCE_NAMES,
    TABLE_FOR_SOURCE,
    build_default_loaders,
    build_pipeline,
    build_quality_checker,
    build_source,
    build_transformer,
)
from etl.loaders import CSVLoader, ParquetLoader, SQLiteLoader
from etl.logging_setup import configure_logging, get_logger
from etl.pipeline import Pipeline, PipelineResult
from etl.sources import BusinessSalesDataSource


app = typer.Typer(
    add_completion=False,
    help="Public-data ETL pipeline (synthetic but realistic).",
    no_args_is_help=True,
)
console = Console()
log = get_logger("etl.cli")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _print_summary(results: list[PipelineResult]) -> None:
    table = Table(title="ETL Pipeline Summary", show_lines=False)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Table", style="magenta")
    table.add_column("Rows extracted", justify="right")
    table.add_column("Rows loaded", justify="right")
    table.add_column("Nulls handled", justify="right")
    table.add_column("Quality P/F", justify="right")
    table.add_column("Duration (s)", justify="right")
    for r in results:
        if r.quality is not None:
            qpf = f"{r.quality.passed}/{r.quality.failed}"
        else:
            qpf = "-"
        table.add_row(
            r.source,
            r.table,
            f"{r.rows_extracted:,}",
            f"{r.rows_loaded:,}",
            f"{r.nulls_handled:,}",
            qpf,
            f"{r.duration_seconds:.2f}",
        )
    console.print(table)
    total_rows = sum(r.rows_loaded for r in results)
    console.print(f"[bold green]Total rows loaded:[/bold green] {total_rows:,}")


def _run_sales_with_dims() -> PipelineResult:
    """Sales requires loading dim tables alongside the fact table."""
    source = build_source("sales")
    assert isinstance(source, BusinessSalesDataSource)
    bundle = source.extract_all()
    products_dim = bundle["products"]
    transformer = build_transformer("sales", products_dim=products_dim)

    # Build a pipeline manually so we can inject the already-extracted fact frame.
    qc = build_quality_checker("sales")
    loaders = build_default_loaders()

    # Wrap the source so its extract() returns the prebuilt fact frame.
    class _PreBuiltSalesSource(type(source)):  # type: ignore[misc]
        def __init__(self, fact: pd.DataFrame) -> None:
            self._fact = fact

        def extract(self) -> pd.DataFrame:  # type: ignore[override]
            return self._fact

    pre = _PreBuiltSalesSource(bundle["sales"])
    pre.name = "sales"
    pipe = Pipeline(
        source=pre,
        transformer=transformer,
        loaders=loaders,
        table=TABLE_FOR_SOURCE["sales"],
        quality_checker=qc,
    )
    result = pipe.run()

    # Also load dimension tables to SQLite + parquet.
    sqlite = next((l for l in loaders if isinstance(l, SQLiteLoader)), None)
    parquet = next((l for l in loaders if isinstance(l, ParquetLoader)), None)
    if sqlite is not None:
        sqlite.load(bundle["customers"], "dim_customers")
        sqlite.load(bundle["products"], "dim_products")
    if parquet is not None:
        parquet.load(bundle["customers"], "dim_customers")
        parquet.load(bundle["products"], "dim_products")
    result.extras["dim_customers"] = len(bundle["customers"])
    result.extras["dim_products"] = len(bundle["products"])
    return result


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def run(
    source: str = typer.Option(..., "--source", "-s", help="weather|traffic|sales"),
) -> None:
    """Run the pipeline for a single source."""
    configure_logging()
    if source not in SOURCE_NAMES:
        typer.secho(
            f"Unknown source '{source}'. Valid: {', '.join(SOURCE_NAMES)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    if source == "sales":
        result = _run_sales_with_dims()
    else:
        result = build_pipeline(source).run()
    _print_summary([result])


@app.command("run-all")
def run_all() -> None:
    """Run all pipelines sequentially."""
    configure_logging()
    results: list[PipelineResult] = []
    for name in SOURCE_NAMES:
        if name == "sales":
            results.append(_run_sales_with_dims())
        else:
            results.append(build_pipeline(name).run())
    _print_summary(results)


@app.command("quality-report")
def quality_report() -> None:
    """Re-run pipelines (in-memory) and print a consolidated quality table."""
    configure_logging()
    rows: list[dict] = []
    for name in SOURCE_NAMES:
        source = build_source(name)
        if isinstance(source, BusinessSalesDataSource):
            bundle = source.extract_all()
            df = bundle["sales"]
            transformer = build_transformer(name, products_dim=bundle["products"])
        else:
            df = source.extract()
            transformer = build_transformer(name)
        clean = transformer.transform(df)
        report = build_quality_checker(name).run(clean)
        for r in report.results:
            rows.append(r.to_dict())

    df_report = pd.DataFrame(rows)
    table = Table(title="Data Quality Report")
    table.add_column("Source", style="cyan")
    table.add_column("Check")
    table.add_column("Passed", justify="center")
    table.add_column("Details")
    for _, row in df_report.iterrows():
        table.add_row(
            str(row["source"]),
            str(row["check_name"]),
            "[green]OK[/green]" if row["passed"] else "[red]FAIL[/red]",
            str(row["details"]),
        )
    console.print(table)
    passed = int(df_report["passed"].sum())
    total = len(df_report)
    console.print(f"[bold]{passed}/{total}[/bold] checks passed")

    # Persist quality report to SQLite as well.
    loader = SQLiteLoader(SETTINGS.sqlite_url, if_exists="replace")
    loader.load(df_report, "quality_reports")


@app.command()
def preview(
    source: str = typer.Option(..., "--source", "-s", help="weather|traffic|sales"),
    rows: int = typer.Option(10, "--rows", "-n", help="Rows to display"),
) -> None:
    """Preview the head of an extracted+transformed source."""
    configure_logging()
    if source not in SOURCE_NAMES:
        typer.secho(
            f"Unknown source '{source}'. Valid: {', '.join(SOURCE_NAMES)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    src = build_source(source)
    if isinstance(src, BusinessSalesDataSource):
        bundle = src.extract_all()
        raw = bundle["sales"]
        transformer = build_transformer(source, products_dim=bundle["products"])
    else:
        raw = src.extract()
        transformer = build_transformer(source)
    clean = transformer.transform(raw)
    console.print(f"[bold]rows:[/bold] {len(clean):,}  [bold]columns:[/bold] {len(clean.columns)}")
    console.print(clean.head(rows).to_string())


@app.command()
def export(
    fmt: str = typer.Option("csv", "--format", "-f", help="csv|json"),
    source: str = typer.Option(
        "all", "--source", "-s", help="weather|traffic|sales|all",
    ),
) -> None:
    """Export already-loaded SQLite tables as CSV or JSON."""
    configure_logging()
    from sqlalchemy import create_engine

    engine = create_engine(SETTINGS.sqlite_url, future=True)
    targets = (
        ["weather_facts", "traffic_facts", "sales_facts", "dim_customers", "dim_products"]
        if source == "all"
        else [TABLE_FOR_SOURCE[source]]
    )
    out_dir: Path = SETTINGS.output_dir
    for table in targets:
        try:
            df = pd.read_sql_table(table, engine)
        except ValueError:
            log.warning("table %s not present yet, skipping", table)
            continue
        if fmt == "csv":
            df.to_csv(out_dir / f"{table}.csv", index=False)
        elif fmt == "json":
            df.to_json(out_dir / f"{table}.json", orient="records", date_format="iso")
        else:
            typer.secho(f"Unknown format '{fmt}'", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        console.print(f"exported {table} -> {fmt} ({len(df):,} rows)")


@app.command()
def report(
    output: Path = typer.Option(
        None, "--output", "-o", help="HTML report path (default: output/report.html)",
    ),
) -> None:
    """Generate an HTML report (Plotly + Jinja2) summarizing the loaded data."""
    configure_logging()
    from etl.report import generate_html_report

    out_path = output or (SETTINGS.output_dir / "report.html")
    path = generate_html_report(out_path)
    console.print(f"[green]Report written:[/green] {path}")


@app.command()
def info() -> None:
    """Print runtime configuration."""
    cfg = {
        "project_root": str(SETTINGS.project_root),
        "output_dir": str(SETTINGS.output_dir),
        "sqlite_url": SETTINGS.sqlite_url,
        "weather": {
            "cities": SETTINGS.weather_cities,
            "days": SETTINGS.weather_days,
        },
        "traffic": {
            "sensors": SETTINGS.traffic_sensors,
            "days": SETTINGS.traffic_days,
        },
        "sales": {
            "products": SETTINGS.sales_products,
            "days": SETTINGS.sales_days,
            "customers": SETTINGS.sales_customers,
        },
    }
    console.print_json(data=cfg)


if __name__ == "__main__":
    app()

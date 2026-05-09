"""Smoke tests for the Typer CLI app."""
from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from etl.cli import app


runner = CliRunner()


def test_cli_info_runs():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "project_root" in result.stdout


def test_cli_run_unknown_source_returns_2():
    result = runner.invoke(app, ["run", "--source", "lava"])
    assert result.exit_code == 2


def test_cli_preview_weather(tmp_path: Path, monkeypatch):
    # Use a small footprint by injecting env vars before module import is too late;
    # the source classes accept explicit args via factories which read SETTINGS at
    # import-time. So we just call the command -- it should still complete fine,
    # just slower. Use a small horizon by overriding env BEFORE the test runs.
    monkeypatch.setenv("ETL_WEATHER_CITIES", "1")
    monkeypatch.setenv("ETL_WEATHER_DAYS", "2")
    # Reload SETTINGS by re-importing the module dynamically.
    import importlib
    import etl.config as cfg
    importlib.reload(cfg)
    import etl.factory as fc
    importlib.reload(fc)
    import etl.cli as cli_mod
    importlib.reload(cli_mod)
    result = runner.invoke(cli_mod.app, ["preview", "--source", "weather", "--rows", "3"])
    assert result.exit_code == 0
    assert "rows" in result.stdout

"""Composable ETL pipeline orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from etl.loaders.base import Loader
from etl.logging_setup import get_logger
from etl.quality import DataQualityChecker, QualityReport
from etl.sources.base import DataSource
from etl.transformers.base import Transformer


@dataclass
class StageResult:
    stage: str
    rows_in: int = 0
    rows_out: int = 0
    nulls_handled: int = 0
    duration_seconds: float = 0.0
    notes: str = ""


@dataclass
class PipelineResult:
    source: str
    table: str
    rows_extracted: int = 0
    rows_loaded: int = 0
    nulls_handled: int = 0
    duration_seconds: float = 0.0
    stages: list[StageResult] = field(default_factory=list)
    quality: QualityReport | None = None
    extras: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "table": self.table,
            "rows_extracted": self.rows_extracted,
            "rows_loaded": self.rows_loaded,
            "nulls_handled": self.nulls_handled,
            "duration_seconds": round(self.duration_seconds, 3),
            "quality_passed": self.quality.passed if self.quality else None,
            "quality_failed": self.quality.failed if self.quality else None,
        }


class Pipeline:
    """Wires a DataSource through a Transformer into one or more Loaders.

    The same instance can be reused; each call to :py:meth:`run` returns a
    fresh :class:`PipelineResult`.
    """

    def __init__(
        self,
        source: DataSource,
        transformer: Transformer,
        loaders: Iterable[Loader],
        table: str,
        quality_checker: DataQualityChecker | None = None,
    ) -> None:
        self.source = source
        self.transformer = transformer
        self.loaders = list(loaders)
        self.table = table
        self.quality_checker = quality_checker
        self.log = get_logger(f"etl.pipeline.{source.name}")

    def run(self) -> PipelineResult:
        result = PipelineResult(source=self.source.name, table=self.table)
        t_start = time.perf_counter()

        # Extract
        ts = time.perf_counter()
        raw = self.source.extract()
        extract_secs = time.perf_counter() - ts
        result.rows_extracted = len(raw)
        result.stages.append(StageResult(
            stage="extract", rows_in=0, rows_out=len(raw),
            duration_seconds=extract_secs,
        ))
        self.log.info(
            "extract: source=%s rows=%d duration=%.2fs",
            self.source.name, len(raw), extract_secs,
        )

        # Transform
        ts = time.perf_counter()
        clean = self.transformer.transform(raw)
        transform_secs = time.perf_counter() - ts
        result.nulls_handled = self.transformer.nulls_handled
        result.stages.append(StageResult(
            stage="transform", rows_in=len(raw), rows_out=len(clean),
            nulls_handled=self.transformer.nulls_handled,
            duration_seconds=transform_secs,
        ))
        self.log.info(
            "transform: rows_in=%d rows_out=%d nulls_handled=%d duration=%.2fs",
            len(raw), len(clean), self.transformer.nulls_handled, transform_secs,
        )

        # Quality
        if self.quality_checker is not None:
            ts = time.perf_counter()
            report = self.quality_checker.run(clean)
            quality_secs = time.perf_counter() - ts
            result.quality = report
            result.stages.append(StageResult(
                stage="quality",
                rows_in=len(clean), rows_out=len(clean),
                duration_seconds=quality_secs,
                notes=f"passed={report.passed} failed={report.failed}",
            ))
            self.log.info(
                "quality: passed=%d failed=%d duration=%.2fs",
                report.passed, report.failed, quality_secs,
            )
            if report.failed:
                for r in report.results:
                    if not r.passed:
                        self.log.warning(
                            "quality_issue: %s -> %s", r.check_name, r.details
                        )

        # Load (run all loaders sequentially)
        ts = time.perf_counter()
        rows_loaded = 0
        for loader in self.loaders:
            n = loader.load(clean, self.table)
            rows_loaded = max(rows_loaded, n)
            self.log.info(
                "load: loader=%s table=%s rows=%d", loader.name, self.table, n,
            )
        load_secs = time.perf_counter() - ts
        result.rows_loaded = rows_loaded
        result.stages.append(StageResult(
            stage="load", rows_in=len(clean), rows_out=rows_loaded,
            duration_seconds=load_secs,
        ))

        result.duration_seconds = time.perf_counter() - t_start
        self.log.info(
            "pipeline_done: source=%s extracted=%d loaded=%d duration=%.2fs",
            self.source.name, result.rows_extracted, result.rows_loaded,
            result.duration_seconds,
        )
        return result

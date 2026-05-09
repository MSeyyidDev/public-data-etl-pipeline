"""Structured, Rich-powered logging setup."""
from __future__ import annotations

import logging
from logging import Logger

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
    )
    # Quiet down noisy third parties.
    logging.getLogger("faker").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    configure_logging()
    return logging.getLogger(name)

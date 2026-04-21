"""Smoke tests for the logging module."""

from __future__ import annotations

from football_analysis.logging import configure_logging, get_logger


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging(level="DEBUG")
    logger = get_logger("test")
    # bound logger has a log method
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")


def test_get_logger_returns_bound_logger() -> None:
    logger = get_logger("test.module")
    logger.info("hello", key="value")  # should not raise

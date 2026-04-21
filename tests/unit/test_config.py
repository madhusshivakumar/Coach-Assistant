"""Smoke tests for the settings singleton. Proves the scaffold builds + imports cleanly."""

from pathlib import Path

from football_analysis.config import Settings, get_settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.data_dir == Path("./data")
    assert s.log_level == "INFO"


def test_settings_derived_paths() -> None:
    s = Settings(_env_file=None, data_dir=Path("/tmp/fa"))  # type: ignore[call-arg]
    assert s.raw_dir == Path("/tmp/fa/raw")
    assert s.processed_dir == Path("/tmp/fa/processed")
    assert s.features_dir == Path("/tmp/fa/features")
    assert s.catalog_path == Path("/tmp/fa/catalog.duckdb")


def test_get_settings_is_singleton() -> None:
    assert get_settings() is get_settings()

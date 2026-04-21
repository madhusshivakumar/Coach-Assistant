"""Shared pytest fixtures. Keep lean — domain fixtures live under tests/fixtures/*."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures/."""
    return Path(__file__).parent / "fixtures"

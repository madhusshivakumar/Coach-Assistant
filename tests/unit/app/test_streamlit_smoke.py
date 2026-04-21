"""Streamlit smoke tests via streamlit.testing.v1.AppTest.

These run the Streamlit scripts in-process without a browser — catches import errors,
widget-key collisions, and runtime exceptions in CI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture()
def data_env(tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seed a tmp FA_DATA_DIR with the mini fixture, ingest it, and return the data root."""
    data = tmp_path / "data"
    raw_sb = data / "raw" / "statsbomb"
    (raw_sb / "events").mkdir(parents=True)
    (raw_sb / "matches" / "43").mkdir(parents=True)
    shutil.copy(fixtures_dir / "statsbomb" / "events_mini.json", raw_sb / "events" / "1.json")
    shutil.copy(fixtures_dir / "statsbomb" / "matches_mini.json", raw_sb / "matches" / "43" / "106.json")
    monkeypatch.setenv("FA_DATA_DIR", str(data))

    import football_analysis.config as cfg
    cfg._settings = None

    # Ingest fixture
    from typer.testing import CliRunner

    from football_analysis.data.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "statsbomb", "--match-id", "1"])
    assert result.exit_code == 0, result.stdout
    return data


def test_main_page_loads(data_env: Path) -> None:
    at = AppTest.from_file("src/football_analysis/app/main.py", default_timeout=30)
    at.run()
    assert not at.exception
    # Title element must be present
    assert len(at.title) >= 1
    assert "football-analysis" in at.title[0].value
    # At least one metric widget (Ingested matches)
    assert len(at.metric) >= 1


def test_match_overview_page_loads(data_env: Path) -> None:
    at = AppTest.from_file(
        "src/football_analysis/app/pages/01_match_overview.py", default_timeout=60
    )
    at.run()
    assert not at.exception, at.exception

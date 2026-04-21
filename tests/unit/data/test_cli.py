"""CLI smoke tests via Typer's CliRunner.

We test the ingest path against local fixtures (offline); fetch is covered by the source tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from football_analysis.data.cli import app


@pytest.fixture()
def configured_env(tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point FA_DATA_DIR at a tmp_path seeded with the mini StatsBomb fixture."""
    data = tmp_path / "data"
    raw_sb = data / "raw" / "statsbomb"
    (raw_sb / "events").mkdir(parents=True)
    (raw_sb / "matches" / "43").mkdir(parents=True)

    shutil.copy(fixtures_dir / "statsbomb" / "events_mini.json", raw_sb / "events" / "1.json")
    shutil.copy(fixtures_dir / "statsbomb" / "matches_mini.json", raw_sb / "matches" / "43" / "106.json")

    monkeypatch.setenv("FA_DATA_DIR", str(data))
    # Reset the singleton so the env override takes effect
    import football_analysis.config as cfg

    cfg._settings = None
    return data


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Football-analysis data pipeline" in result.stdout


def test_ingest_statsbomb_then_validate_then_status(configured_env: Path) -> None:
    runner = CliRunner()
    # Ingest
    result = runner.invoke(app, ["ingest", "statsbomb", "--match-id", "1"])
    assert result.exit_code == 0, result.stdout
    assert "ingested statsbomb:1" in result.stdout

    # Processed parquet is present
    parquet = next((configured_env / "processed" / "events").rglob("*.parquet"))
    assert parquet.exists()

    # Validate passes
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0, result.stdout
    assert "OK" in result.stdout

    # Status lists the ingested match
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "statsbomb:1" in result.stdout


def test_catalog_rebuild_view_reads_events(configured_env: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["ingest", "statsbomb", "--match-id", "1"])
    result = runner.invoke(app, ["catalog", "rebuild"])
    assert result.exit_code == 0, result.stdout

    import duckdb

    con = duckdb.connect(str(configured_env / "catalog.duckdb"), read_only=True)
    try:
        n = con.execute("SELECT count(*) FROM events").fetchone()
        assert n is not None and n[0] > 0
    finally:
        con.close()


def test_validate_with_no_files_is_noop(configured_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "no parquet files" in result.stdout


def test_validate_specific_match_filter(configured_env: Path) -> None:
    runner = CliRunner()
    runner.invoke(app, ["ingest", "statsbomb", "--match-id", "1"])
    result = runner.invoke(app, ["validate", "--match-id", "statsbomb:1"])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_ingest_fails_for_missing_match(configured_env: Path) -> None:
    # Overwrite the matches manifest with an empty list so match_id=1 is not found
    matches_path = configured_env / "raw" / "statsbomb" / "matches" / "43" / "106.json"
    matches_path.write_text("[]", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "statsbomb", "--match-id", "1"])
    assert result.exit_code != 0


def test_fixture_events_match_expected_shape(fixtures_dir: Path) -> None:
    """Keeps the fixture honest — if someone edits events_mini.json we notice."""
    body = json.loads((fixtures_dir / "statsbomb" / "events_mini.json").read_text(encoding="utf-8"))
    assert len(body) == 11
    type_names = {ev["type"]["name"] for ev in body}
    assert {"Pass", "Shot", "Carry", "Interception", "Clearance", "Dribble", "Foul Committed"} <= type_names

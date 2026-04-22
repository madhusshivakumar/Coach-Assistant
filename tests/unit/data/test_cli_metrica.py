"""Smoke tests for the Metrica fetch + ingest CLI commands.

We patch `kloppy.metrica.load_open_data` to return a tiny in-memory dataset so
the test is network-free and sub-second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from football_analysis.data.cli import app

# --- Tiny fake kloppy TrackingDataset ---


@dataclass(frozen=True)
class _Coord:
    x: float
    y: float


@dataclass(frozen=True)
class _Team:
    team_id: str


@dataclass(frozen=True)
class _Player:
    player_id: str
    team: _Team


@dataclass
class _PData:
    coordinates: _Coord | None


@dataclass
class _Period:
    id: int


@dataclass
class _Frame:
    frame_id: int
    period: _Period
    timestamp: timedelta
    ball_coordinates: _Coord | None
    players_data: dict = field(default_factory=dict)


@dataclass
class _Dataset:
    frames: list[_Frame]


def _tiny_dataset() -> _Dataset:
    frames = []
    h1 = _Player("h1", _Team("home"))
    a1 = _Player("a1", _Team("away"))
    for i in range(3):
        frames.append(
            _Frame(
                frame_id=i + 1,
                period=_Period(id=1),
                timestamp=timedelta(seconds=0.04 * (i + 1)),
                ball_coordinates=_Coord(0.5, 0.5),
                players_data={
                    h1: _PData(_Coord(0.2 + i * 0.01, 0.3)),
                    a1: _PData(_Coord(0.8 - i * 0.01, 0.7)),
                },
            )
        )
    return _Dataset(frames=frames)


@pytest.fixture()
def configured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("FA_DATA_DIR", str(data))
    import football_analysis.config as cfg

    cfg._settings = None
    return data


def test_fetch_metrica_invokes_kloppy_and_reports(configured_env: Path) -> None:
    runner = CliRunner()
    with patch("kloppy.metrica.load_open_data", return_value=_tiny_dataset()) as mock:
        result = runner.invoke(app, ["fetch", "metrica", "--match-id", "2", "--limit", "10"])
    assert result.exit_code == 0, result.stdout
    assert "fetched metrica match 2" in result.stdout
    mock.assert_called_once_with(match_id="2", limit=10, sample_rate=None)


def test_ingest_metrica_writes_parquet_and_records_ingest(configured_env: Path) -> None:
    runner = CliRunner()
    with patch("kloppy.metrica.load_open_data", return_value=_tiny_dataset()):
        result = runner.invoke(app, ["ingest", "metrica", "--match-id", "1"])
    assert result.exit_code == 0, result.stdout
    assert "ingested metrica:1" in result.stdout

    # Parquet under data/processed/tracking/...
    parquets = list((configured_env / "processed" / "tracking").rglob("*.parquet"))
    assert len(parquets) == 1, f"expected 1 period parquet, got {parquets}"

    # Catalog status picks it up
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "metrica:1" in result.stdout


def test_ingest_metrica_empty_dataset_exits_nonzero(configured_env: Path) -> None:
    runner = CliRunner()
    with patch("kloppy.metrica.load_open_data", return_value=_Dataset(frames=[])):
        result = runner.invoke(app, ["ingest", "metrica", "--match-id", "3"])
    assert result.exit_code != 0


def test_catalog_rebuild_includes_tracking_view(configured_env: Path) -> None:
    runner = CliRunner()
    with patch("kloppy.metrica.load_open_data", return_value=_tiny_dataset()):
        runner.invoke(app, ["ingest", "metrica", "--match-id", "1"])
    result = runner.invoke(app, ["catalog", "rebuild"])
    assert result.exit_code == 0, result.stdout

    import duckdb

    con = duckdb.connect(str(configured_env / "catalog.duckdb"), read_only=True)
    try:
        rows = con.execute("SELECT count(*) FROM tracking").fetchone()
        assert rows is not None and rows[0] > 0
    finally:
        con.close()

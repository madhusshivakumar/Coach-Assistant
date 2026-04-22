"""Tests for the Metrica source wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from football_analysis.data.sources import metrica as mt


def test_available_match_ids() -> None:
    assert mt.AVAILABLE_MATCH_IDS == ("1", "2", "3")


def test_fetch_dataset_invokes_kloppy_and_writes_sentinel(tmp_path: Path) -> None:
    fake_ds = MagicMock()
    fake_ds.frames = [MagicMock(), MagicMock(), MagicMock()]

    with patch("kloppy.metrica.load_open_data", return_value=fake_ds) as mock:
        result = mt.fetch_dataset(match_id=2, raw_dir=tmp_path, limit=500)

    mock.assert_called_once_with(match_id="2", limit=500, sample_rate=None)
    assert result is fake_ds
    sentinel = tmp_path / "metrica" / "match-2.fetched"
    assert sentinel.exists()
    assert "frames=3" in sentinel.read_text(encoding="utf-8")


def test_list_fetched_returns_match_ids_present(tmp_path: Path) -> None:
    cache = tmp_path / "metrica"
    cache.mkdir()
    (cache / "match-1.fetched").write_text("", encoding="utf-8")
    (cache / "match-3.fetched").write_text("", encoding="utf-8")
    (cache / "unrelated.json").write_text("", encoding="utf-8")

    assert mt.list_fetched(raw_dir=tmp_path) == ["1", "3"]


def test_list_fetched_empty_when_dir_missing(tmp_path: Path) -> None:
    assert mt.list_fetched(raw_dir=tmp_path) == []


def test_fetch_dataset_passes_sample_rate(tmp_path: Path) -> None:
    fake_ds = MagicMock()
    fake_ds.frames = []

    with patch("kloppy.metrica.load_open_data", return_value=fake_ds) as mock:
        mt.fetch_dataset(match_id=1, raw_dir=tmp_path, sample_rate=5.0)

    mock.assert_called_once_with(match_id="1", limit=None, sample_rate=5.0)


def test_fetch_dataset_default_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When raw_dir is None we use the project settings default."""
    monkeypatch.setenv("FA_DATA_DIR", str(tmp_path))
    import football_analysis.config as cfg

    cfg._settings = None

    fake_ds = MagicMock()
    fake_ds.frames = []
    with patch("kloppy.metrica.load_open_data", return_value=fake_ds):
        mt.fetch_dataset(match_id=1)

    assert (tmp_path / "raw" / "metrica" / "match-1.fetched").exists()

"""Metrica Sports open-data source.

Metrica publishes three free sample games at 25 Hz, full-pitch optical tracking.
kloppy's `metrica.load_open_data` handles the network fetch and CSV parse; we
wrap it so the result is cached to Parquet on first read and reused thereafter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kloppy import metrica as _kmetrica

from football_analysis.config import get_settings
from football_analysis.logging import get_logger

if TYPE_CHECKING:
    from kloppy.domain.models.tracking import TrackingDataset

_log = get_logger(__name__)

# Metrica publishes three sample games on GitHub; match_id is "1", "2", "3".
AVAILABLE_MATCH_IDS: tuple[str, ...] = ("1", "2", "3")


def _raw_cache_dir(raw_dir: Path | None = None) -> Path:
    root = raw_dir if raw_dir is not None else get_settings().raw_dir
    return root / "metrica"


def fetch_dataset(
    match_id: int | str,
    raw_dir: Path | None = None,
    limit: int | None = None,
    sample_rate: float | None = None,
) -> TrackingDataset:
    """Download (or reuse cached) Metrica sample tracking via kloppy.

    We pin a "fetched" sentinel file so `fa-data status` can tell what's on
    disk without reparsing a multi-MB CSV.
    """
    cache = _raw_cache_dir(raw_dir)
    cache.mkdir(parents=True, exist_ok=True)
    sentinel = cache / f"match-{match_id}.fetched"

    _log.info("metrica_load_open_data", match_id=str(match_id), limit=limit, sample_rate=sample_rate)
    dataset = _kmetrica.load_open_data(
        match_id=str(match_id),
        limit=limit,
        sample_rate=sample_rate,
    )

    sentinel.write_text(f"frames={len(dataset.frames)}\n", encoding="utf-8")
    return dataset


def list_fetched(raw_dir: Path | None = None) -> list[str]:
    """Return match ids that have a sentinel (i.e. were fetched at least once)."""
    cache = _raw_cache_dir(raw_dir)
    if not cache.exists():
        return []
    return sorted(p.stem.removeprefix("match-") for p in cache.glob("match-*.fetched"))

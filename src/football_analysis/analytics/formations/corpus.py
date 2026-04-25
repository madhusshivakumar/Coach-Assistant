"""Build a corpus of ``FormationProfile`` objects across many ingested matches.

A real baseline needs a population of matches, not just the focus match. This module
walks ``data/processed/events`` (or any directory of canonical SPADL parquets), fits a
single shared xT grid over the union, and emits one events profile per (team, match).

Used by ``scripts/demo_phase3b.py`` to score Argentina–France against WC2022 instead
of against itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from football_analysis.analytics.formations.profile import (
    FormationProfile,
    extract_events_profile,
)
from football_analysis.analytics.possession_value.xt import XTGrid, apply_xt, fit


def iter_events_parquets(events_dir: Path, glob: str = "**/*.parquet") -> Iterator[Path]:
    """Yield every parquet under ``events_dir`` matching ``glob``, sorted for determinism."""
    yield from sorted(events_dir.glob(glob))


def _match_key_from_path(path: Path) -> str:
    """Stable identifier for a match parquet, e.g. ``statsbomb-3869685``."""
    return path.stem


def fit_corpus_xt_grid(parquets: Iterable[Path], min_rows_per_match: int = 100) -> XTGrid:
    """Fit a single xT grid over the union of all corpus matches.

    Tiny fixture parquets (e.g. the 10-row smoke file) are skipped — they distort the grid.
    """
    frames: list[pd.DataFrame] = []
    for p in parquets:
        df = pd.read_parquet(p)
        if len(df) >= min_rows_per_match:
            frames.append(df)
    if not frames:
        raise ValueError("no usable parquets to fit corpus xT grid")
    union = pd.concat(frames, ignore_index=True)
    return fit(union)


def build_events_corpus_profiles(
    events_dir: Path,
    glob: str = "**/*.parquet",
    min_rows_per_match: int = 100,
    xt_grid: XTGrid | None = None,
) -> tuple[list[FormationProfile], XTGrid | None]:
    """Run the events-side profile extractor over every match in ``events_dir``.

    Args:
        events_dir: directory holding processed SPADL parquets (recursive glob).
        glob: pattern, default ``**/*.parquet``.
        min_rows_per_match: skip parquets below this row count (smoke fixtures).
        xt_grid: optional pre-fit grid; if None, fits one on the corpus union.

    Returns:
        ``(profiles, grid)``. ``profiles`` is a flat list (~ 2 × n_matches); match
        identity is *not* attached — corpus baseline only needs the population.
        ``grid`` is the fitted (or passed-through) xT grid; callers should reuse it
        to score the focus match so its z-scores are on the same scale. ``None`` when
        no usable parquets were found.
    """
    parquets = list(iter_events_parquets(events_dir, glob))
    if not parquets:
        return [], None

    grid = xt_grid if xt_grid is not None else fit_corpus_xt_grid(parquets, min_rows_per_match=min_rows_per_match)

    profiles: list[FormationProfile] = []
    for p in parquets:
        df = pd.read_parquet(p)
        if len(df) < min_rows_per_match:
            continue
        enriched = apply_xt(df, grid)
        team_ids = sorted(t for t in enriched["team_id"].dropna().unique() if str(t))
        if len(team_ids) < 2:
            continue
        # All-pairs would explode for >2-team data; SPADL events parquets are per-match
        # so this loop is bounded to 2.
        for i, team in enumerate(team_ids):
            opponent = team_ids[1 - i] if len(team_ids) == 2 else team_ids[(i + 1) % len(team_ids)]
            profiles.append(extract_events_profile(enriched, team_id=team, opponent_id=opponent))
    return profiles, grid

"""Tests for the events corpus loader (multi-match baseline builder)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from football_analysis.analytics.formations.corpus import (
    build_events_corpus_profiles,
    fit_corpus_xt_grid,
    iter_events_parquets,
)


def _synthetic_match_df(team_a: str, team_b: str, n_passes_per_side: int = 60) -> pd.DataFrame:
    """A barely-enough-rows synthetic SPADL frame with a couple shots so xT.fit works."""
    rows: list[dict] = []
    for i in range(n_passes_per_side):
        rows.append(
            {
                "team_id": team_a,
                "action_type": "pass",
                "result": "success",
                "start_x": 30.0 + (i % 5),
                "start_y": 30.0 + (i % 7),
                "end_x": 50.0 + (i % 5),
                "end_y": 30.0 + (i % 7),
            }
        )
        rows.append(
            {
                "team_id": team_b,
                "action_type": "pass",
                "result": "success",
                "start_x": 40.0 + (i % 5),
                "start_y": 30.0 + (i % 7),
                "end_x": 55.0 + (i % 5),
                "end_y": 30.0 + (i % 7),
            }
        )
    # A few shots so apply_xt has something to land on.
    for team, x in ((team_a, 100.0), (team_b, 95.0)):
        rows.append(
            {
                "team_id": team,
                "action_type": "shot",
                "result": "success",
                "start_x": x,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
            }
        )
    # And a defensive action so PPDA isn't infinite for everyone.
    rows.append(
        {
            "team_id": team_a,
            "action_type": "tackle",
            "result": "success",
            "start_x": 25.0,
            "start_y": 34.0,
            "end_x": None,
            "end_y": None,
        }
    )
    rows.append(
        {
            "team_id": team_b,
            "action_type": "tackle",
            "result": "success",
            "start_x": 70.0,
            "start_y": 34.0,
            "end_x": None,
            "end_y": None,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Three synthetic matches plus a tiny smoke fixture below the row-count threshold."""
    out = tmp_path / "events"
    out.mkdir()
    for i, (a, b) in enumerate([("H1", "A1"), ("H2", "A2"), ("H3", "A3")]):
        _synthetic_match_df(a, b).to_parquet(out / f"match-{i}.parquet")
    # Smoke fixture — should be ignored by min_rows_per_match.
    pd.DataFrame(
        [
            {
                "team_id": "X",
                "action_type": "pass",
                "result": "success",
                "start_x": 1.0,
                "start_y": 1.0,
                "end_x": 2.0,
                "end_y": 2.0,
            }
        ]
        * 5
    ).to_parquet(out / "smoke.parquet")
    return out


def test_iter_events_parquets_is_sorted(corpus_dir: Path) -> None:
    paths = list(iter_events_parquets(corpus_dir))
    assert paths == sorted(paths)
    assert len(paths) == 4  # 3 matches + smoke


def test_fit_corpus_xt_grid_skips_tiny_files(corpus_dir: Path) -> None:
    grid = fit_corpus_xt_grid(iter_events_parquets(corpus_dir), min_rows_per_match=100)
    # XTGrid carries `values` (xT lookup), `shoot_prob`, and `move_prob` 2D arrays.
    assert grid.values.shape == grid.shoot_prob.shape == grid.move_prob.shape


def test_fit_corpus_xt_grid_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="no usable parquets"):
        fit_corpus_xt_grid([])


def test_build_events_corpus_profiles_emits_two_per_match(corpus_dir: Path) -> None:
    profiles, grid = build_events_corpus_profiles(corpus_dir)
    # 3 usable matches x 2 teams = 6 profiles. Smoke parquet skipped.
    assert len(profiles) == 6
    assert grid is not None
    team_ids = {p.team_id for p in profiles}
    # All 6 distinct teams present.
    assert team_ids == {"H1", "A1", "H2", "A2", "H3", "A3"}


def test_build_events_corpus_profiles_returns_empty_when_dir_empty(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    profiles, grid = build_events_corpus_profiles(empty)
    assert profiles == []
    assert grid is None


def test_build_events_corpus_profiles_accepts_external_grid(corpus_dir: Path) -> None:
    """Passing a pre-fit grid should skip refit and return that exact grid back."""
    pre_grid = fit_corpus_xt_grid(iter_events_parquets(corpus_dir))
    profiles, grid = build_events_corpus_profiles(corpus_dir, xt_grid=pre_grid)
    assert grid is pre_grid
    assert len(profiles) == 6

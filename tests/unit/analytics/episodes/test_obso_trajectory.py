"""Tests for OBSO time-series + decisive-moment detection."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.episodes.obso_trajectory import (
    DecisiveMoment,
    find_decisive_moment,
)


def _trajectory(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": 100 + i,
                "time_s": 4.0 + i * 0.5,
                "rel_time_s": i * 0.5,
                "obso_max": v,
                "obso_argmax_x": 95.0,
                "obso_argmax_y": 34.0,
            }
            for i, v in enumerate(values)
        ]
    )


def test_find_decisive_moment_returns_first_frame_above_threshold() -> None:
    """Trajectory rises 0.05 → 0.10 → 0.30 → 0.55 → 0.60 (peak). 50% of peak = 0.30."""
    traj = _trajectory([0.05, 0.10, 0.30, 0.55, 0.60])
    dm = find_decisive_moment(traj, threshold_pct=0.5)
    assert isinstance(dm, DecisiveMoment)
    assert dm.frame_id == 102  # the 0.30 entry
    assert dm.peak_obso == 0.60
    assert dm.threshold_pct == 0.5


def test_find_decisive_moment_with_lower_threshold_finds_earlier_frame() -> None:
    traj = _trajectory([0.05, 0.10, 0.30, 0.55, 0.60])
    dm_50 = find_decisive_moment(traj, threshold_pct=0.5)
    dm_15 = find_decisive_moment(traj, threshold_pct=0.15)
    assert dm_50 is not None and dm_15 is not None
    assert dm_15.frame_id < dm_50.frame_id


def test_find_decisive_moment_returns_none_for_empty_trajectory() -> None:
    assert find_decisive_moment(pd.DataFrame()) is None


def test_find_decisive_moment_returns_none_when_peak_is_zero() -> None:
    traj = _trajectory([0.0, 0.0, 0.0])
    assert find_decisive_moment(traj) is None


def test_find_decisive_moment_decisive_at_first_frame_when_already_above_threshold() -> None:
    """If episode starts above threshold, decisive frame is frame 0."""
    traj = _trajectory([0.40, 0.50, 0.60])
    dm = find_decisive_moment(traj, threshold_pct=0.5)
    assert dm is not None
    assert dm.frame_id == 100  # starts at 0.40, threshold=0.30 → frame 0 already above

"""Tests for OBSO time-series + decisive-moment detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.obso_trajectory import (
    DecisiveMoment,
    compute_obso_trajectory,
    find_decisive_moment,
)
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


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


def _episode_with_tracking(n_snaps: int = 5) -> tuple[EpisodeRecord, pd.DataFrame]:
    """Build a tiny episode + tracking pair for the GPU-vs-CPU parity test."""
    boundary = EpisodeBoundary(
        episode_id=0,
        start_frame=1,
        end_frame=n_snaps,
        start_time_s=0.0,
        end_time_s=float(n_snaps) * 0.5,
        duration_s=float(n_snaps) * 0.5,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=0,
        end_reason="match_end",
        reached_final_third=True,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=80.0,
        end_ball_y=34.0,
        end_ball_speed=4.0,
        duration_s=float(n_snaps) * 0.5,
    )
    states = pd.DataFrame([{"frame_id": i + 1, "time_s": i * 0.5, "rel_time_s": i * 0.5} for i in range(n_snaps)])
    record = EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=None)

    rows = []
    for f in range(1, n_snaps + 1):
        for i in range(11):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": (f - 1) * 0.5,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": 30 + i * 2,
                    "y": 30 + i,
                    "vx": 0.5,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": (f - 1) * 0.5,
                    "player_id": f"a{i}",
                    "team_id": "away",
                    "x": 60 + i * 2,
                    "y": 30 + i,
                    "vx": 0.0,
                    "vy": 0.5,
                    "is_ball": False,
                    "visible": True,
                }
            )
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": (f - 1) * 0.5,
                "player_id": "ball",
                "team_id": "home",
                "x": 50.0 + f,
                "y": 34.0,
                "vx": 5.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return record, pd.DataFrame(rows)


def test_compute_obso_trajectory_gpu_matches_cpu() -> None:
    """The use_gpu=True path should produce numerically-equivalent OBSO peaks
    to the use_gpu=False path (within the same tolerance as direct module test)."""
    pytest.importorskip("torch")
    record, tracking = _episode_with_tracking(n_snaps=5)
    cpu = compute_obso_trajectory(record, tracking, "home", "away", rows=20, cols=30)
    gpu = compute_obso_trajectory(
        record,
        tracking,
        "home",
        "away",
        rows=20,
        cols=30,
        use_gpu=True,
    )
    assert len(cpu) == len(gpu) == 5
    # OBSO max per frame should agree closely.
    np.testing.assert_allclose(
        cpu["obso_max"].to_numpy(),
        gpu["obso_max"].to_numpy(),
        atol=1e-3,
        rtol=1e-2,
    )

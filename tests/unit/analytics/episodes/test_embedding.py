"""Tests for episode embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.embedding import (
    EPISODE_FEATURE_NAMES,
    embed_episode,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _record(states: pd.DataFrame, dominant_phase: str | None = "progression") -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=1,
        start_frame=1,
        end_frame=100,
        start_time_s=0.0,
        end_time_s=4.0,
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=1,
        end_reason="match_end",
        reached_final_third=True,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=80.0,
        end_ball_y=34.0,
        end_ball_speed=4.0,
        duration_s=4.0,
    )
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=dominant_phase)


def _trajectory(start_x: float = 30.0, end_x: float = 80.0, n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rel_time_s": np.linspace(0.0, 4.0, n),
            "ball_x": np.linspace(start_x, end_x, n),
            "ball_y": [34.0] * n,
            "ball_x_oriented": np.linspace(start_x, end_x, n),
            "ball_speed": [3.0] * n,
            "attackers_mean_x_oriented": np.linspace(40, 65, n),
            "defenders_line_height_oriented": [85.0] * n,
            "attackers_length": [40.0] * n,
            "attackers_width": [50.0] * n,
            "attackers_visible": [11] * n,
            "defenders_visible": [10] * n,
            "defenders_compactness_x": [3.5] * n,
        }
    )


def test_embed_episode_returns_correct_shape() -> None:
    rec = _record(_trajectory())
    vec = embed_episode(rec)
    assert vec.shape == (len(EPISODE_FEATURE_NAMES),)
    assert vec.dtype == np.float64
    assert np.isfinite(vec).all()


def test_embed_episode_phase_one_hot_set_correctly() -> None:
    rec = _record(_trajectory(), dominant_phase="finishing")
    vec = embed_episode(rec)
    idx = EPISODE_FEATURE_NAMES.index("phase_finishing")
    assert vec[idx] == 1.0
    # Other phase one-hots should be zero.
    other_phase_idxs = [
        i for i, name in enumerate(EPISODE_FEATURE_NAMES) if name.startswith("phase_") and name != "phase_finishing"
    ]
    assert all(vec[i] == 0.0 for i in other_phase_idxs)


def test_embed_episode_unknown_phase_yields_all_zero_phase_one_hot() -> None:
    rec = _record(_trajectory(), dominant_phase=None)
    vec = embed_episode(rec)
    phase_idxs = [i for i, name in enumerate(EPISODE_FEATURE_NAMES) if name.startswith("phase_")]
    assert all(vec[i] == 0.0 for i in phase_idxs)


def test_embed_episode_ball_x_displacement_is_max_minus_start() -> None:
    rec = _record(_trajectory(start_x=30.0, end_x=80.0))
    vec = embed_episode(rec)
    idx_disp = EPISODE_FEATURE_NAMES.index("ball_x_displacement")
    # Trajectory is monotonically increasing → max == end → displacement = 50.
    assert abs(vec[idx_disp] - 50.0) < 1e-6


def test_embed_episode_partial_mode_uses_only_prefix() -> None:
    rec = _record(_trajectory(start_x=30.0, end_x=80.0, n=8))
    full = embed_episode(rec)
    prefix = embed_episode(rec, max_rel_time_s=2.0)
    # Prefix end_ball_x should be < full end_ball_x (we cut at half-time).
    idx_end = EPISODE_FEATURE_NAMES.index("end_ball_x_oriented")
    assert prefix[idx_end] < full[idx_end]


def test_embed_episode_empty_trajectory_returns_zeros() -> None:
    rec = _record(pd.DataFrame())
    vec = embed_episode(rec)
    assert vec.shape == (len(EPISODE_FEATURE_NAMES),)
    assert (vec == 0).all()


def test_embed_episode_handles_all_nan_ball_columns() -> None:
    states = _trajectory()
    states["ball_x_oriented"] = float("nan")
    states["ball_speed"] = float("nan")
    rec = _record(states)
    vec = embed_episode(rec)
    # Should produce no NaNs anywhere — we substitute 0.
    assert np.isfinite(vec).all()

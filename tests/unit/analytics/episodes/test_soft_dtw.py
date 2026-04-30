"""Tests for soft-DTW similarity on role-anchored trajectories.

Soft-DTW (Cuturi & Blondel, 2017) is a smoothed, differentiable variant of
Dynamic Time Warping. We use it to compare two episodes by their role-aligned
player trajectories — episodes of unequal length can still be compared because
the warping path absorbs time differences.

These tests pin the public contract (``soft_dtw``, ``soft_dtw_from_tensor``,
``soft_dtw_from_trajectory``) plus the mathematical properties M2 clustering
relies on (identity, symmetry, monotone in difference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.episodes.soft_dtw import (
    soft_dtw,
    soft_dtw_from_tensor,
    soft_dtw_from_trajectory,
)
from football_analysis.analytics.formations.roles import (
    FORMATION_4_3_3,
    episode_role_trajectory,
)


def _line(start: float, end: float, n: int) -> np.ndarray:
    """1D series along x with constant y, shape (n, 2)."""
    return np.column_stack([np.linspace(start, end, n), np.full(n, 34.0)])


# --- soft_dtw on bare 2-D arrays ---------------------------------------------


def test_soft_dtw_identity_is_zero() -> None:
    """Soft-DTW(X, X) ≈ 0 because the diagonal alignment has zero cost everywhere."""
    x = _line(0.0, 50.0, 10)
    assert soft_dtw(x, x) < 1e-6


def test_soft_dtw_is_symmetric() -> None:
    x = _line(0.0, 50.0, 10)
    y = _line(0.0, 30.0, 8)
    a = soft_dtw(x, y)
    b = soft_dtw(y, x)
    assert abs(a - b) < 1e-6


def test_soft_dtw_distinguishes_similar_from_different() -> None:
    """A series compared to a near-copy gets a much smaller distance than to a
    very different series. This is the property that makes soft-DTW useful for
    clustering."""
    base = _line(0.0, 50.0, 10)
    near = base + np.array([0.5, 0.0])  # 0.5 m offset everywhere
    far = _line(50.0, 0.0, 10)  # walking the other way
    assert soft_dtw(base, near) < soft_dtw(base, far)


def test_soft_dtw_handles_unequal_length() -> None:
    """The whole point of DTW: T1 ≠ T2 still produces a finite scalar."""
    x = _line(0.0, 50.0, 10)
    y = _line(0.0, 50.0, 25)  # same path, sampled denser
    val = soft_dtw(x, y)
    assert np.isfinite(val)
    # Same path different sampling → small distance after warping
    assert val < soft_dtw(x, _line(50.0, 0.0, 25))


def test_soft_dtw_lower_gamma_approaches_hard_dtw() -> None:
    """Soft-DTW with γ → 0 ⇒ hard DTW. So smaller γ ⇒ tighter (smaller-or-equal)
    smoothing slack ⇒ value is at least as large as the soft version's."""
    x = _line(0.0, 50.0, 10)
    y = _line(2.0, 48.0, 10)
    soft = soft_dtw(x, y, gamma=1.0)
    harder = soft_dtw(x, y, gamma=0.01)
    assert harder >= soft - 1e-6  # may be slightly larger or close


def test_soft_dtw_validates_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        soft_dtw(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_soft_dtw_validates_matching_feature_dim() -> None:
    x = np.zeros((5, 2))
    y = np.zeros((5, 3))
    with pytest.raises(ValueError, match="feature dim"):
        soft_dtw(x, y)


def test_soft_dtw_single_snapshot() -> None:
    """T=1 on either side: distance = squared-Euclidean of the single pair."""
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 4.0]])
    val = soft_dtw(x, y)
    # Hard cost = 3² + 4² = 25; soft-DTW with one cell ≡ that exact cost.
    assert abs(val - 25.0) < 1e-6


def test_soft_dtw_validates_gamma_positive() -> None:
    x = _line(0.0, 50.0, 5)
    with pytest.raises(ValueError, match="gamma"):
        soft_dtw(x, x, gamma=0.0)


# --- soft_dtw_from_tensor — (T, 10, 2) shape used by the role pipeline -------


def test_soft_dtw_from_tensor_flattens_correctly() -> None:
    """A (T, 10, 2) tensor of identical content → distance ≈ 0."""
    rng = np.random.default_rng(42)
    tensor = rng.uniform(0, 105, size=(8, 10, 2))
    assert soft_dtw_from_tensor(tensor, tensor) < 1e-6


def test_soft_dtw_from_tensor_validates_shape() -> None:
    with pytest.raises(ValueError, match="3-D"):
        soft_dtw_from_tensor(np.zeros((5, 2)), np.zeros((5, 2)))


def test_soft_dtw_from_tensor_validates_n_roles_match() -> None:
    a = np.zeros((5, 10, 2))
    b = np.zeros((5, 11, 2))
    with pytest.raises(ValueError, match="role count"):
        soft_dtw_from_tensor(a, b)


# --- soft_dtw_from_trajectory — works on the long-form DataFrame -------------


def _synth_tracking(n_frames: int, x_offset: float = 0.0) -> pd.DataFrame:
    """Tracking with home on 4-3-3 slots (optionally x-shifted by ``x_offset``)
    and an away placeholder so episode_role_trajectory has both teams."""
    tpl = FORMATION_4_3_3
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = f / 25.0
        for i, (tx, ty) in enumerate(zip(tpl.xs, tpl.ys, strict=True)):
            rows.append(
                {
                    "frame_id": f, "period": 1, "time_seconds": t,
                    "player_id": f"home_{i}", "team_id": "home",
                    "x": tx + x_offset, "y": ty,
                    "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
                }
            )
        rows.append(
            {
                "frame_id": f, "period": 1, "time_seconds": t,
                "player_id": "home_GK", "team_id": "home",
                "x": 5.0 + x_offset, "y": 34.0,
                "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
            }
        )
        rows.append(
            {
                "frame_id": f, "period": 1, "time_seconds": t,
                "player_id": "away_0", "team_id": "away",
                "x": 100.0, "y": 34.0,
                "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
            }
        )
        rows.append(
            {
                "frame_id": f, "period": 1, "time_seconds": t,
                "player_id": "ball", "team_id": None,
                "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0,
                "is_ball": True, "visible": True,
            }
        )
    return pd.DataFrame(rows)


def _ep(start: int, end: int) -> EpisodeBoundary:
    return EpisodeBoundary(
        episode_id=1,
        start_frame=start,
        end_frame=end,
        start_time_s=start / 25.0,
        end_time_s=end / 25.0,
        duration_s=(end - start) / 25.0,
        possession_team="home",
        end_reason="match_end",
    )


def test_soft_dtw_from_trajectory_identity() -> None:
    tracking = _synth_tracking(50)
    traj = episode_role_trajectory(tracking, _ep(1, 50), team_id="home")
    assert soft_dtw_from_trajectory(traj, traj) < 1e-6


def test_soft_dtw_from_trajectory_distinguishes_offset_teams() -> None:
    """Two episodes where one team's positions are shifted by 20 m get a
    much larger distance than two near-identical episodes."""
    tracking_a = _synth_tracking(50, x_offset=0.0)
    tracking_b = _synth_tracking(50, x_offset=0.5)
    tracking_c = _synth_tracking(50, x_offset=20.0)
    traj_a = episode_role_trajectory(tracking_a, _ep(1, 50), team_id="home")
    traj_b = episode_role_trajectory(tracking_b, _ep(1, 50), team_id="home")
    traj_c = episode_role_trajectory(tracking_c, _ep(1, 50), team_id="home")
    near = soft_dtw_from_trajectory(traj_a, traj_b)
    far = soft_dtw_from_trajectory(traj_a, traj_c)
    assert near < far


def test_soft_dtw_from_trajectory_handles_unequal_episode_lengths() -> None:
    """Two episodes of different durations still produce a finite scalar."""
    tracking = _synth_tracking(100)
    short = episode_role_trajectory(tracking, _ep(1, 30), team_id="home")
    long_ = episode_role_trajectory(tracking, _ep(1, 100), team_id="home")
    val = soft_dtw_from_trajectory(short, long_)
    assert np.isfinite(val)


def test_soft_dtw_from_trajectory_empty_input_raises() -> None:
    """Empty role-trajectory ⇒ no series to compare ⇒ caller bug."""
    empty = pd.DataFrame(
        columns=["snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"]
    )
    with pytest.raises(ValueError, match="empty"):
        soft_dtw_from_trajectory(empty, empty)


def test_soft_dtw_from_trajectory_uses_consistent_role_ordering() -> None:
    """Permuting the rows in the DataFrame must not change the distance —
    the function must sort by ``(snapshot_idx, role)`` internally before
    reshaping. Otherwise two equivalent trajectories would look different."""
    tracking = _synth_tracking(50)
    traj = episode_role_trajectory(tracking, _ep(1, 50), team_id="home")
    shuffled = traj.sample(frac=1.0, random_state=7).reset_index(drop=True)
    d1 = soft_dtw_from_trajectory(traj, shuffled)
    assert d1 < 1e-6

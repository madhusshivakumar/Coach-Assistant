"""Tests for the torch-backed OBSO compute path.

Two requirements: (1) numerical agreement with the numpy implementation within
a small tolerance, and (2) the API works on CPU even when no GPU is present so
CI/non-GPU dev machines pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from football_analysis.analytics.pitch_control.obso import compute_obso_frame
from football_analysis.analytics.pitch_control.obso_gpu import (
    compute_obso_batch_torch,
    compute_obso_frame_torch_from_tracking,
)


def _synth_frame(seed: int = 0, n_per_team: int = 11) -> pd.DataFrame:
    """One-frame canonical tracking with random-but-seeded positions."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_per_team):
        rows.append(
            {
                "frame_id": 1,
                "period": 1,
                "time_seconds": 0.04,
                "player_id": f"h{i}",
                "team_id": "home",
                "x": float(rng.uniform(0, 105)),
                "y": float(rng.uniform(0, 68)),
                "vx": float(rng.uniform(-2, 2)),
                "vy": float(rng.uniform(-2, 2)),
                "is_ball": False,
                "visible": True,
            }
        )
        rows.append(
            {
                "frame_id": 1,
                "period": 1,
                "time_seconds": 0.04,
                "player_id": f"a{i}",
                "team_id": "away",
                "x": float(rng.uniform(0, 105)),
                "y": float(rng.uniform(0, 68)),
                "vx": float(rng.uniform(-2, 2)),
                "vy": float(rng.uniform(-2, 2)),
                "is_ball": False,
                "visible": True,
            }
        )
    rows.append(
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.04,
            "player_id": "ball",
            "team_id": "home",
            "x": float(rng.uniform(20, 85)),
            "y": float(rng.uniform(20, 48)),
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "visible": True,
        }
    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_torch_obso_matches_numpy_within_tol(seed: int) -> None:
    """Numerical agreement on a coarse grid for several random seeds."""
    df = _synth_frame(seed=seed)
    cpu_result = compute_obso_frame(
        df,
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=20,
        cols=30,
    )
    torch_result = compute_obso_frame_torch_from_tracking(
        df,
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=20,
        cols=30,
        device=torch.device("cpu"),  # force CPU for deterministic comparison
    )
    # OBSO arrays match.
    np.testing.assert_allclose(
        torch_result.obso[0],
        cpu_result.obso,
        atol=1e-4,
        rtol=1e-3,
    )
    # Control surfaces match.
    np.testing.assert_allclose(
        torch_result.control[0],
        cpu_result.control,
        atol=1e-4,
        rtol=1e-3,
    )


def test_torch_obso_handles_missing_ball() -> None:
    """When the ball is missing, the implementation must not crash."""
    df = _synth_frame(seed=0)
    # Drop the ball row.
    df = df[~df["is_ball"]].reset_index(drop=True)
    # Numpy path returns OBSO with arrival=1 everywhere.
    np_result = compute_obso_frame(df, 1, "home", "away", rows=12, cols=20)
    assert np_result.obso.shape == (12, 20)
    # Torch path falls back to centre-of-pitch (close enough that result has finite values).
    t_result = compute_obso_frame_torch_from_tracking(
        df,
        1,
        "home",
        "away",
        rows=12,
        cols=20,
        device=torch.device("cpu"),
    )
    assert t_result.obso.shape == (1, 12, 20)
    assert np.isfinite(t_result.obso).all()


def test_torch_batch_processes_multiple_frames_at_once() -> None:
    """Batched call returns one OBSO per input frame."""
    home_pos = [
        np.array([[40, 30], [50, 30], [60, 30]], dtype=float),
        np.array([[35, 25], [45, 30], [55, 35]], dtype=float),
        np.array([[30, 20], [40, 30], [50, 40]], dtype=float),
    ]
    home_vel = [np.zeros_like(p) for p in home_pos]
    away_pos = [
        np.array([[70, 30], [80, 30], [90, 30]], dtype=float),
        np.array([[75, 25], [80, 30], [85, 35]], dtype=float),
        np.array([[60, 20], [70, 30], [80, 40]], dtype=float),
    ]
    away_vel = [np.zeros_like(p) for p in away_pos]
    ball_xy = np.array([[55, 30], [50, 30], [45, 30]], dtype=float)

    result = compute_obso_batch_torch(
        home_pos,
        home_vel,
        away_pos,
        away_vel,
        ball_xy,
        attacking_is_home=np.array([True, True, True]),
        rows=20,
        cols=30,
        device=torch.device("cpu"),
    )
    assert result.obso.shape == (3, 20, 30)
    assert (result.obso >= 0).all() and (result.obso <= 1).all()


def test_torch_batch_handles_empty_team() -> None:
    """A frame with one team missing must not crash; control goes to the other team."""
    home_pos = [np.array([[40, 30], [50, 30]], dtype=float)]
    away_pos = [np.empty((0, 2))]
    result = compute_obso_batch_torch(
        home_pos,
        [np.zeros_like(home_pos[0])],
        away_pos,
        [np.empty((0, 2))],
        np.array([[55, 30]], dtype=float),
        attacking_is_home=np.array([True]),
        rows=10,
        cols=15,
        device=torch.device("cpu"),
    )
    # Home (attacker) has no opposition → control should be ~1 everywhere.
    assert (result.control[0] >= 0.99).all()

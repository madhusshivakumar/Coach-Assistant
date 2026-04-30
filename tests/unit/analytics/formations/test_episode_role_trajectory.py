"""Tests for ``episode_role_trajectory`` — M1 task #1 of the v2 redesign.

The trajectory is the soft-DTW input shape: for each snapshot in an episode, the
10 outfielders of one team are mapped onto the slots of a formation template via
the existing Bialkowski Hungarian assignment. The output is long-form (one row
per (snapshot, role) pair) so callers can pivot to a (T, 10, 2) tensor or
inspect per-role per-snapshot displacement directly.

These tests exercise the contract M2 clustering and M1 task #3 (soft-DTW) will
depend on, so the function's shape is treated as part of the contract here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.formations.roles import (
    FORMATION_4_3_3,
    episode_role_trajectory,
)


def _full_team_frame(
    frame_id: int,
    time_s: float,
    team_id: str,
    template_xs: tuple[float, ...],
    template_ys: tuple[float, ...],
    gk_x: float,
    gk_y: float,
    jitter: float = 0.5,
) -> list[dict]:
    """11 rows: 10 outfielders near template slots + 1 GK at (gk_x, gk_y)."""
    rows: list[dict] = []
    for i, (tx, ty) in enumerate(zip(template_xs, template_ys, strict=True)):
        rows.append(
            {
                "frame_id": frame_id,
                "period": 1,
                "time_seconds": time_s,
                "player_id": f"{team_id}_{i}",
                "team_id": team_id,
                "x": tx + (jitter if i % 2 == 0 else -jitter),
                "y": ty + (jitter if i % 3 == 0 else -jitter / 2),
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            }
        )
    rows.append(
        {
            "frame_id": frame_id,
            "period": 1,
            "time_seconds": time_s,
            "player_id": f"{team_id}_GK",
            "team_id": team_id,
            "x": gk_x,
            "y": gk_y,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    return rows


def _ball_row(frame_id: int, time_s: float, x: float = 50.0, y: float = 34.0) -> dict:
    return {
        "frame_id": frame_id,
        "period": 1,
        "time_seconds": time_s,
        "player_id": "ball",
        "team_id": None,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "is_ball": True,
        "visible": True,
    }


def _synth_tracking(n_frames: int = 50, fps: int = 25) -> pd.DataFrame:
    """4-3-3 home + arbitrary away; both have a clear GK near own goal."""
    tpl = FORMATION_4_3_3
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = f / fps
        rows.extend(
            _full_team_frame(f, t, "home", tpl.xs, tpl.ys, gk_x=5.0, gk_y=34.0)
        )
        rows.extend(
            _full_team_frame(
                f,
                t,
                "away",
                tuple(105.0 - x for x in tpl.xs),
                tuple(68.0 - y for y in tpl.ys),
                gk_x=100.0,
                gk_y=34.0,
            )
        )
        rows.append(_ball_row(f, t))
    return pd.DataFrame(rows)


def _ep(start: int = 1, end: int = 50, fps: int = 25, team: str = "home") -> EpisodeBoundary:
    return EpisodeBoundary(
        episode_id=1,
        start_frame=start,
        end_frame=end,
        start_time_s=start / fps,
        end_time_s=end / fps,
        duration_s=(end - start) / fps,
        possession_team=team,
        end_reason="match_end",
    )


def test_returns_one_row_per_snapshot_per_role() -> None:
    tracking = _synth_tracking(n_frames=50)  # 2 s
    out = episode_role_trajectory(
        tracking,
        _ep(),
        team_id="home",
        template=FORMATION_4_3_3,
        snapshot_hz=2.0,  # ~5 snapshots over 2s
    )
    assert not out.empty
    n_snaps = out["snapshot_idx"].nunique()
    # 10 roles per snapshot, every role present at every snapshot
    assert len(out) == n_snaps * 10
    for snap in out["snapshot_idx"].unique():
        roles_at_snap = set(out[out["snapshot_idx"] == snap]["role"])
        assert roles_at_snap == set(FORMATION_4_3_3.roles)


def test_excludes_goalkeeper() -> None:
    """The GK player_id (``home_GK``) must never appear — only the 10 outfielders."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(), team_id="home", template=FORMATION_4_3_3)
    assert "home_GK" not in set(out["player_id"])
    # All assigned player_ids belong to the requested team.
    assert all(pid.startswith("home_") for pid in out["player_id"])


def test_only_returns_requested_team() -> None:
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(team="home"), team_id="home")
    # No away player should leak in
    assert not any(pid.startswith("away_") for pid in out["player_id"])


def test_columns_are_stable_contract() -> None:
    """soft-DTW (M1 #3) and clustering (M2) consume this — schema is contractual."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(), team_id="home")
    expected = {"snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"}
    assert expected <= set(out.columns)


def test_template_recovery_when_team_sits_on_slots() -> None:
    """Outfielders sit (with tiny jitter) on the 4-3-3 template slots → assigned
    role for each player should match the slot index."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(), team_id="home", template=FORMATION_4_3_3)
    # For each snapshot the player with index i should get role[i] of the template
    first = out[out["snapshot_idx"] == out["snapshot_idx"].min()]
    by_pid = dict(zip(first["player_id"], first["role"], strict=True))
    for i, expected_role in enumerate(FORMATION_4_3_3.roles):
        assert by_pid[f"home_{i}"] == expected_role
    # And displacements stay small with jitter ≤ 0.5
    assert (first["displacement"] <= 1.0).all()


def test_attacking_left_team_mirrored_correctly() -> None:
    """Away team is laid out mirrored on the pitch (slots flipped to the right side).
    With ``attacking_right=False`` the function mirrors before matching, so the role
    map must be identical to the home (attacking_right=True) result."""
    tracking = _synth_tracking(n_frames=50)
    home_out = episode_role_trajectory(
        tracking, _ep(team="home"), team_id="home", attacking_right=True
    )
    away_out = episode_role_trajectory(
        tracking, _ep(team="away"), team_id="away", attacking_right=False
    )
    home_first = home_out[home_out["snapshot_idx"] == home_out["snapshot_idx"].min()]
    away_first = away_out[away_out["snapshot_idx"] == away_out["snapshot_idx"].min()]
    # Same role assignment by player index ('home_i' ↔ 'away_i' sit on mirrored slots)
    home_by_pid = dict(zip(home_first["player_id"], home_first["role"], strict=True))
    away_by_pid = dict(zip(away_first["player_id"], away_first["role"], strict=True))
    for i in range(10):
        assert home_by_pid[f"home_{i}"] == away_by_pid[f"away_{i}"]


def test_zero_duration_episode_returns_single_snapshot() -> None:
    tracking = _synth_tracking(n_frames=10)
    ep = EpisodeBoundary(
        episode_id=1,
        start_frame=1,
        end_frame=1,
        start_time_s=0.04,
        end_time_s=0.04,
        duration_s=0.0,
        possession_team="home",
        end_reason="match_end",
    )
    out = episode_role_trajectory(tracking, ep, team_id="home")
    assert out["snapshot_idx"].nunique() == 1
    assert len(out) == 10  # 10 roles


def test_handles_explicit_goalkeeper_id() -> None:
    """If caller knows the GK, they can pass it explicitly and skip auto-detection."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(
        tracking, _ep(), team_id="home", goalkeeper_id="home_GK"
    )
    assert "home_GK" not in set(out["player_id"])
    assert len(out) > 0


def test_skips_snapshots_with_too_few_visible_outfielders() -> None:
    """If a snapshot has fewer than 10 visible outfielders, that snapshot is dropped
    rather than crashing the whole episode."""
    tpl = FORMATION_4_3_3
    rows: list[dict] = []
    for f in range(1, 51):
        t = f / 25.0
        # Frame 25 only has 8 home outfielders visible — skip it
        if f == 25:
            for i in range(8):
                rows.append(
                    {
                        "frame_id": f,
                        "period": 1,
                        "time_seconds": t,
                        "player_id": f"home_{i}",
                        "team_id": "home",
                        "x": tpl.xs[i],
                        "y": tpl.ys[i],
                        "vx": 0.0,
                        "vy": 0.0,
                        "is_ball": False,
                        "visible": True,
                    }
                )
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": "home_GK",
                    "team_id": "home",
                    "x": 5.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        else:
            rows.extend(_full_team_frame(f, t, "home", tpl.xs, tpl.ys, gk_x=5.0, gk_y=34.0))
        rows.append(_ball_row(f, t))

    out = episode_role_trajectory(pd.DataFrame(rows), _ep(), team_id="home", snapshot_hz=25.0)
    # Function did not crash, and frame 25 is absent from output
    assert 25 not in set(out["frame_id"])


def test_empty_tracking_returns_empty_df() -> None:
    out = episode_role_trajectory(
        pd.DataFrame(columns=["frame_id", "time_seconds", "player_id", "team_id", "x", "y", "is_ball", "visible"]),
        _ep(),
        team_id="home",
    )
    assert out.empty


def test_invalid_team_raises() -> None:
    tracking = _synth_tracking(n_frames=50)
    with pytest.raises(ValueError, match="team_id"):
        episode_role_trajectory(tracking, _ep(), team_id="nonexistent")


def test_displacement_units_are_metres() -> None:
    """Sanity check: displacement is the Euclidean distance from slot in metres,
    so with jitter ≤ 0.5 m the max displacement at any snapshot is ≤ ~1.0 m."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(), team_id="home")
    assert out["displacement"].max() < 2.0
    assert (out["displacement"] >= 0).all()


def test_pivot_to_tensor_shape_for_soft_dtw() -> None:
    """Caller pivot: (T, 10, 2). M1 task #3 (soft-DTW) will use this exact reshape."""
    tracking = _synth_tracking(n_frames=50)
    out = episode_role_trajectory(tracking, _ep(), team_id="home")
    n_snaps = out["snapshot_idx"].nunique()
    # Order roles consistently for reshape stability
    roles_order = list(FORMATION_4_3_3.roles)
    out_sorted = out.set_index(["snapshot_idx", "role"]).sort_index()
    coords = (
        out_sorted.loc[(slice(None), roles_order), ["x", "y"]]
        .to_numpy()
        .reshape(n_snaps, 10, 2)
    )
    assert coords.shape == (n_snaps, 10, 2)
    # Coordinates are finite (no NaN slipped through)
    assert np.isfinite(coords).all()

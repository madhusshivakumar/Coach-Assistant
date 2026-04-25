"""Tests for the per-snapshot episode state-trajectory builder."""

from __future__ import annotations

import math

import pandas as pd

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.episodes.state import episode_state_trajectory


def _tracking_with_ball_moving(n_frames: int = 100, fps: int = 25) -> pd.DataFrame:
    """Synthetic tracking where the ball travels +x at 5 m/s, home attacks +x."""
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = round(f / fps, 4)
        for i in range(5):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": 30.0 + i * 2,
                    "y": 30.0 + i * 2,
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
                    "player_id": f"a{i}",
                    "team_id": "away",
                    "x": 70.0 + i * 0.5,
                    "y": 30.0 + i * 2,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        ball_x = 30.0 + 5.0 * (f / fps)  # 5 m/s
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "ball",
                "team_id": "home",
                "x": ball_x,
                "y": 34.0,
                "vx": 5.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def _episode(start_frame: int = 1, end_frame: int = 100, fps: int = 25) -> EpisodeBoundary:
    return EpisodeBoundary(
        episode_id=0,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=start_frame / fps,
        end_time_s=end_frame / fps,
        duration_s=round((end_frame - start_frame) / fps, 3),
        possession_team="home",
        end_reason="match_end",
    )


def test_state_trajectory_returns_one_row_per_snapshot() -> None:
    tracking = _tracking_with_ball_moving(n_frames=100)
    ep = _episode(1, 100)
    df = episode_state_trajectory(
        tracking,
        ep,
        home_team_id="home",
        away_team_id="away",
        snapshot_hz=2.0,
        attacking_to_right=True,
    )
    # 4 s episode at 2 Hz → ~9 unique snapshots (8 intervals + endpoint).
    assert 7 <= len(df) <= 12
    assert {
        "episode_id",
        "frame_id",
        "time_s",
        "ball_x",
        "ball_x_oriented",
        "attackers_visible",
        "defenders_visible",
    } <= set(df.columns)


def test_state_trajectory_orientation_flip() -> None:
    """If attacking_to_right=False, ball_x_oriented = 105 - ball_x."""
    tracking = _tracking_with_ball_moving(n_frames=50)
    ep = _episode(1, 50)
    right = episode_state_trajectory(tracking, ep, "home", "away", attacking_to_right=True)
    left = episode_state_trajectory(tracking, ep, "home", "away", attacking_to_right=False)
    # Same ball_x in both, but ball_x_oriented mirrors.
    for _i, r in right.iterrows():
        l_row = left[left["frame_id"] == r["frame_id"]]
        assert not l_row.empty
        assert math.isclose(r["ball_x_oriented"] + l_row["ball_x_oriented"].iloc[0], 105.0, rel_tol=1e-6)


def test_state_trajectory_ball_not_visible_yields_nan() -> None:
    """When the ball is invisible at a snapshot frame, ball_x is NaN."""
    tracking = _tracking_with_ball_moving(n_frames=50)
    # mark all ball rows invisible
    tracking.loc[tracking["is_ball"], "visible"] = False
    ep = _episode(1, 50)
    df = episode_state_trajectory(tracking, ep, "home", "away", attacking_to_right=True)
    assert df["ball_x"].isna().all()


def test_state_trajectory_zero_duration_episode_emits_one_row() -> None:
    tracking = _tracking_with_ball_moving(n_frames=10)
    ep = EpisodeBoundary(
        episode_id=0,
        start_frame=5,
        end_frame=5,
        start_time_s=0.2,
        end_time_s=0.2,
        duration_s=0.0,
        possession_team="home",
        end_reason="match_end",
    )
    df = episode_state_trajectory(tracking, ep, "home", "away")
    assert len(df) == 1
    assert df["frame_id"].iloc[0] == 5


def test_state_trajectory_attacker_count_matches_team_size() -> None:
    """5 home outfielders → attackers_visible=5 in every snapshot."""
    tracking = _tracking_with_ball_moving(n_frames=50)
    ep = _episode(1, 50)
    df = episode_state_trajectory(tracking, ep, "home", "away")
    assert (df["attackers_visible"] == 5).all()
    assert (df["defenders_visible"] == 5).all()


def test_state_trajectory_handles_empty_episode_window() -> None:
    """If the episode's frame window is outside the tracking, output is empty."""
    tracking = _tracking_with_ball_moving(n_frames=50)
    ep = EpisodeBoundary(
        episode_id=99,
        start_frame=1000,
        end_frame=1100,
        start_time_s=40.0,
        end_time_s=44.0,
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    df = episode_state_trajectory(tracking, ep, "home", "away")
    assert df.empty

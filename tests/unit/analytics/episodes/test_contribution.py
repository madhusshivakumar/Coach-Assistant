"""Tests for the leave-one-out OBSO attribution layer."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.episodes.contribution import (
    EpisodeAttribution,
    compute_episode_attribution,
    find_peak_frame,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _build_record(
    *,
    shot_like: bool = False,
    ended_in_box: bool = False,
    reached_final_third: bool = False,
    state_trajectory: pd.DataFrame | None = None,
    end_frame: int = 100,
) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=0,
        start_frame=1,
        end_frame=end_frame,
        start_time_s=0.0,
        end_time_s=4.0,
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=0,
        end_reason="match_end",
        reached_final_third=reached_final_third,
        ended_in_box=ended_in_box,
        shot_like=shot_like,
        end_ball_x=95.0,
        end_ball_y=34.0,
        end_ball_speed=15.0,
        duration_s=4.0,
    )
    if state_trajectory is None:
        state_trajectory = pd.DataFrame(
            [{"frame_id": 50, "ball_x_oriented": 75.0}, {"frame_id": 100, "ball_x_oriented": 95.0}]
        )
    return EpisodeRecord(
        boundary=boundary,
        outcome=outcome,
        state_trajectory=state_trajectory,
        dominant_phase="finishing",
    )


def test_find_peak_frame_returns_end_for_shot_like() -> None:
    rec = _build_record(shot_like=True, ended_in_box=True, reached_final_third=True, end_frame=200)
    assert find_peak_frame(rec) == 200


def test_find_peak_frame_returns_end_for_ended_in_box() -> None:
    rec = _build_record(ended_in_box=True, reached_final_third=True, end_frame=150)
    assert find_peak_frame(rec) == 150


def test_find_peak_frame_returns_deepest_when_only_final_third() -> None:
    states = pd.DataFrame(
        [
            {"frame_id": 10, "ball_x_oriented": 60.0},
            {"frame_id": 25, "ball_x_oriented": 80.0},  # deepest
            {"frame_id": 40, "ball_x_oriented": 72.0},
        ]
    )
    rec = _build_record(reached_final_third=True, state_trajectory=states)
    assert find_peak_frame(rec) == 25


def test_find_peak_frame_returns_none_when_no_threat() -> None:
    rec = _build_record(reached_final_third=False)
    assert find_peak_frame(rec) is None


def test_find_peak_frame_returns_none_for_empty_states() -> None:
    rec = _build_record(reached_final_third=True, state_trajectory=pd.DataFrame())
    assert find_peak_frame(rec) is None


def test_find_peak_frame_returns_none_when_ball_x_oriented_all_nan() -> None:
    states = pd.DataFrame(
        [
            {"frame_id": 10, "ball_x_oriented": float("nan")},
            {"frame_id": 20, "ball_x_oriented": float("nan")},
        ]
    )
    rec = _build_record(reached_final_third=True, state_trajectory=states)
    assert find_peak_frame(rec) is None


def _tracking_with_one_attacker_running(n_frames: int = 100) -> pd.DataFrame:
    """Synthetic tracking: 11 attackers, 11 defenders, 1 ball.

    Player ``runner`` makes a forward run from x=40 to x=95 (over the run of frames),
    everyone else stays put. The defending team holds a flat back four near x=85.
    """
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = round(f / 25, 4)
        # Attackers: 10 stationary at midfield + 1 runner.
        for i in range(10):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": 40.0 + i,
                    "y": 30.0 + i,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        run_x = 40.0 + 55.0 * (f - 1) / max(1, n_frames - 1)
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "runner",
                "team_id": "home",
                "x": run_x,
                "y": 34.0,
                "vx": 5.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            }
        )
        # Defenders: flat-ish around x=85.
        for i in range(11):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"a{i}",
                    "team_id": "away",
                    "x": 85.0 + (i - 5) * 0.5,
                    "y": 14.0 + i * 4,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        # Ball glued to runner.
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "ball",
                "team_id": "home",
                "x": run_x,
                "y": 34.0,
                "vx": 5.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def test_attribution_returns_none_for_episode_without_peak() -> None:
    rec = _build_record(reached_final_third=False)
    tracking = _tracking_with_one_attacker_running(n_frames=20)
    attr = compute_episode_attribution(rec, tracking, "home", "away")
    assert attr is None


def test_attribution_returns_struct_with_per_attacker_contributions() -> None:
    """11 visible attackers at peak → 11 entries in contributions dict."""
    tracking = _tracking_with_one_attacker_running(n_frames=100)
    states = pd.DataFrame(
        [
            {"frame_id": 50, "ball_x_oriented": 70.0},
            {"frame_id": 100, "ball_x_oriented": 95.0},
        ]
    )
    rec = _build_record(
        shot_like=True,
        ended_in_box=True,
        reached_final_third=True,
        state_trajectory=states,
        end_frame=100,
    )
    attr = compute_episode_attribution(rec, tracking, "home", "away", rows=20, cols=30)
    assert isinstance(attr, EpisodeAttribution)
    assert attr.peak_frame == 100
    assert attr.peak_obso >= 0.0
    # All 11 attackers should be in the contributions dict.
    assert len(attr.contributions) == 11
    # The runner moved a lot — its baseline should differ from the others.
    assert "runner" in attr.contributions


def test_attribution_runner_has_nonzero_baseline_diff() -> None:
    """The player who moved most should have a nonzero counterfactual baseline."""
    tracking = _tracking_with_one_attacker_running(n_frames=100)
    states = pd.DataFrame(
        [
            {"frame_id": 50, "ball_x_oriented": 70.0},
            {"frame_id": 100, "ball_x_oriented": 95.0},
        ]
    )
    rec = _build_record(
        shot_like=True,
        ended_in_box=True,
        reached_final_third=True,
        state_trajectory=states,
        end_frame=100,
    )
    attr = compute_episode_attribution(rec, tracking, "home", "away", rows=20, cols=30)
    assert attr is not None
    # Stationary attackers (h0..h9): peak == baseline (no movement => 0 contribution).
    stationary_contribs = [attr.contributions[f"h{i}"] for i in range(10)]
    assert all(abs(c) < 1e-6 for c in stationary_contribs)
    # The runner moved — its baseline should be different from peak.
    assert attr.baseline_obso_per_player["runner"] != attr.peak_obso

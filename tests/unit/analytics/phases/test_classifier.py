"""Tests for the rules-based phase-of-play classifier."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.phases.classifier import (
    PHASE_LABELS,
    PhaseConfig,
    classify_frames,
    segment_phases,
)


def _frame(
    frame_id: int,
    time_seconds: float,
    ball_xy: tuple[float, float] | None,
    possessor_xy: tuple[float, float] | None = None,
    possessor_team: str = "home",
    threshold_close: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    if ball_xy is not None:
        rows.append(
            {
                "frame_id": frame_id,
                "period": 1,
                "time_seconds": time_seconds,
                "player_id": None,
                "team_id": None,
                "x": ball_xy[0],
                "y": ball_xy[1],
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    # Add one home + one away player. Put the possessor on top of the ball (or far from it).
    if ball_xy is not None and threshold_close:
        px, py = possessor_xy if possessor_xy is not None else (ball_xy[0] + 0.5, ball_xy[1] + 0.5)
    else:
        px, py = (5.0, 5.0)  # nowhere near the ball
    rows.append(
        {
            "frame_id": frame_id,
            "period": 1,
            "time_seconds": time_seconds,
            "player_id": f"{possessor_team}_1",
            "team_id": possessor_team,
            "x": px,
            "y": py,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    # Opposition player way off
    other = "away" if possessor_team == "home" else "home"
    rows.append(
        {
            "frame_id": frame_id,
            "period": 1,
            "time_seconds": time_seconds,
            "player_id": f"{other}_1",
            "team_id": other,
            "x": 100.0,
            "y": 60.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    return rows


def test_phase_labels_are_closed_set() -> None:
    assert set(PHASE_LABELS) == {
        "build_up",
        "progression",
        "finishing",
        "def_transition",
        "att_transition",
        "settled_def",
        "set_piece",
    }


def _settled_possession_frames(ball_xy: tuple[float, float], possessor: str, n_frames: int = 200) -> pd.DataFrame:
    """Build `n_frames` of consecutive possession (~8 s at 25 Hz) so the sequence
    extends past the default 6 s transition window and the classifier commits to
    the settled phase."""
    rows: list[dict] = []
    for i in range(n_frames):
        rows.extend(_frame(100 + i, i * 0.04, ball_xy=ball_xy, possessor_team=possessor))
    return pd.DataFrame(rows)


def test_build_up_when_possessor_in_own_third_and_settled() -> None:
    rows = _settled_possession_frames(ball_xy=(20.0, 34.0), possessor="home")
    out = classify_frames(rows, home_team_id="home", away_team_id="away")
    assert out["phase"].iloc[-1] == "build_up"


def test_finishing_when_possessor_in_attacking_third() -> None:
    rows = _settled_possession_frames(ball_xy=(95.0, 34.0), possessor="home")
    out = classify_frames(rows, home_team_id="home", away_team_id="away")
    assert out["phase"].iloc[-1] == "finishing"


def test_progression_when_possessor_in_middle_third() -> None:
    rows = _settled_possession_frames(ball_xy=(52.5, 34.0), possessor="home")
    out = classify_frames(rows, home_team_id="home", away_team_id="away")
    assert out["phase"].iloc[-1] == "progression"


def test_possession_flip_triggers_att_transition() -> None:
    """After a possession flip, the immediate frames are att_transition."""
    rows: list[dict] = []
    for i in range(5):
        rows.extend(_frame(100 + i, 10.0 + i * 0.04, ball_xy=(52.5, 34.0), possessor_team="home"))
    for i in range(5, 10):
        rows.extend(_frame(100 + i, 10.0 + i * 0.04, ball_xy=(52.5, 34.0), possessor_team="away"))
    out = classify_frames(pd.DataFrame(rows), home_team_id="home", away_team_id="away")
    # Frames 5..9 should be within 6s of the flip — att_transition from away's perspective.
    post_flip = out[out["frame_id"] >= 105]
    assert (post_flip["phase"] == "att_transition").any()


def test_away_team_attacking_direction_is_mirrored() -> None:
    """When away holds the ball at x=20 (their attacking third in canonical coords),
    the phase should be 'finishing', not 'build_up'."""
    rows = _settled_possession_frames(ball_xy=(20.0, 34.0), possessor="away")
    out = classify_frames(rows, home_team_id="home", away_team_id="away")
    assert out["phase"].iloc[-1] == "finishing"


def test_no_one_near_ball_long_enough_becomes_set_piece() -> None:
    """When no player is within the possession threshold for > transition_window_s,
    the frame is labeled set_piece (ball out of play / dead)."""
    rows: list[dict] = []
    # 10 s of frames with everyone far from the ball
    for i in range(250):
        rows.extend(_frame(1000 + i, i * 0.04, ball_xy=(50.0, 40.0), possessor_team="home", threshold_close=False))
    cfg = PhaseConfig(transition_window_s=2.0)  # tight window to force the set_piece branch
    out = classify_frames(pd.DataFrame(rows), "home", "away", cfg)
    # After > 2s without possession, frames are set_piece.
    late = out[out["time_seconds"] > 3.0]
    assert (late["phase"] == "set_piece").all()


def test_classify_handles_missing_ball() -> None:
    rows = [
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.0,
            "player_id": "home_1",
            "team_id": "home",
            "x": 50.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        },
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.0,
            "player_id": "away_1",
            "team_id": "away",
            "x": 55.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        },
    ]
    out = classify_frames(pd.DataFrame(rows), "home", "away")
    assert len(out) == 1
    assert out["phase"].iloc[0] == "set_piece"


def test_segment_phases_collapses_runs() -> None:
    classified = pd.DataFrame(
        {
            "frame_id": [1, 2, 3, 4, 5, 6],
            "time_seconds": [0.0, 0.04, 0.08, 0.12, 0.16, 0.20],
            "possession_team": ["home"] * 6,
            "phase": ["build_up", "build_up", "progression", "progression", "progression", "finishing"],
        }
    )
    segs = segment_phases(classified)
    assert list(segs["phase"]) == ["build_up", "progression", "finishing"]
    assert list(segs["start_frame"]) == [1, 3, 6]
    assert list(segs["end_frame"]) == [2, 5, 6]


def test_segment_phases_empty_input() -> None:
    segs = segment_phases(pd.DataFrame(columns=["frame_id", "time_seconds", "possession_team", "phase"]))
    assert segs.empty

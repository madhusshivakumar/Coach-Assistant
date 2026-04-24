"""Tests for the B-residual upgrades to the tactical view."""

from __future__ import annotations

import pandas as pd

from football_analysis.viz.interactive.tactical_view import (
    _frame_traces,
    _possession_holder,
    _smooth_possession_holder_across_frames,
    animate,
)


def _frame(
    frame_id: int, time_seconds: float, ball_xy: tuple[float, float], players: list[tuple[str, str, float, float]]
) -> list[dict]:
    rows: list[dict] = []
    for team, pid, x, y in players:
        rows.append(
            {
                "frame_id": frame_id,
                "period": 1,
                "time_seconds": time_seconds,
                "team_id": team,
                "player_id": pid,
                "x": x,
                "y": y,
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
            "time_seconds": time_seconds,
            "team_id": None,
            "player_id": None,
            "x": ball_xy[0],
            "y": ball_xy[1],
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "visible": True,
        }
    )
    return rows


def test_team_label_uses_team_names_mapping() -> None:
    """When team_names is supplied, the Scattergl trace name should match."""
    tracking = pd.DataFrame(
        _frame(
            1,
            0.0,
            (50.0, 34.0),
            [
                ("779", "m1", 50.0, 34.0),
                ("771", "a1", 70.0, 34.0),
            ],
        )
    )
    traces = _frame_traces(
        tracking, frame_id=1, home_team_id="779", away_team_id="771", team_names={"779": "Argentina", "771": "France"}
    )
    # Home trace is at _IDX_HOME = 3; away at 4
    assert traces[3].name == "Argentina"
    assert traces[4].name == "France"


def test_team_label_falls_back_to_home_away_when_no_mapping() -> None:
    tracking = pd.DataFrame(
        _frame(
            1,
            0.0,
            (50.0, 34.0),
            [
                ("779", "m1", 50.0, 34.0),
                ("771", "a1", 70.0, 34.0),
            ],
        )
    )
    traces = _frame_traces(tracking, frame_id=1, home_team_id="779", away_team_id="771")
    assert traces[3].name == "Home"
    assert traces[4].name == "Away"


def test_possession_label_resolves_team_name() -> None:
    """The possession-ring hover tooltip uses the mapped team name, not the raw id."""
    frame = pd.DataFrame(
        _frame(
            1,
            0.0,
            (50.0, 34.0),
            [
                ("779", "messi", 50.5, 34.0),
                ("771", "mbappe", 90.0, 40.0),
            ],
        )
    )
    _xs, _ys, lbl = _possession_holder(
        frame,
        smoothed_player_id="messi",
        team_names={"779": "Argentina"},
    )
    assert lbl == ["Argentina messi  d=0.5m"]


def test_smoothed_possession_adopts_dominant_holder_immediately() -> None:
    """At the very first frame, a clearly dominant player should be adopted as
    the holder without waiting for the sticky window."""
    tracking = pd.DataFrame(
        _frame(
            1,
            0.0,
            (50.0, 34.0),
            [
                ("home", "h1", 50.0, 34.0),  # sitting on the ball
                ("away", "a1", 80.0, 50.0),  # nowhere near
            ],
        )
    )
    out = _smooth_possession_holder_across_frames(tracking, frame_ids=[1])
    assert out[1] == "h1"


def test_smoothed_possession_requires_sticky_frames_to_switch() -> None:
    """A challenger who is only closer for 1-2 frames must NOT take the ring."""
    rows: list[dict] = []
    # 8 frames: h1 holds for the first 5, then a1 is briefly closer for 2 frames
    # (below the sticky window), then h1 recovers.
    for fid in range(1, 6):
        rows.extend(
            _frame(
                fid,
                fid * 0.04,
                (50.0, 34.0),
                [
                    ("home", "h1", 50.3, 34.0),
                    ("away", "a1", 60.0, 34.0),
                ],
            )
        )
    for fid in range(6, 8):
        rows.extend(
            _frame(
                fid,
                fid * 0.04,
                (50.0, 34.0),
                [
                    ("home", "h1", 60.0, 34.0),
                    ("away", "a1", 50.3, 34.0),  # challenger closer for 2 frames only
                ],
            )
        )
    rows.extend(
        _frame(
            8,
            0.32,
            (50.0, 34.0),
            [
                ("home", "h1", 50.3, 34.0),
                ("away", "a1", 60.0, 34.0),
            ],
        )
    )
    out = _smooth_possession_holder_across_frames(
        pd.DataFrame(rows),
        frame_ids=list(range(1, 9)),
    )
    # h1 should never lose the ring — the challenger held for fewer than 4 frames
    assert all(out[f] == "h1" for f in range(1, 9))


def test_smoothed_possession_switches_after_sticky_window() -> None:
    """If the challenger is dominant for the full sticky window, the ring switches."""
    rows: list[dict] = []
    for fid in range(1, 3):
        rows.extend(
            _frame(
                fid,
                fid * 0.04,
                (50.0, 34.0),
                [
                    ("home", "h1", 50.3, 34.0),
                    ("away", "a1", 60.0, 34.0),
                ],
            )
        )
    # Challenger sits on the ball for many frames (well past the 4-frame sticky window)
    for fid in range(3, 15):
        rows.extend(
            _frame(
                fid,
                fid * 0.04,
                (50.0, 34.0),
                [
                    ("home", "h1", 60.0, 34.0),
                    ("away", "a1", 50.3, 34.0),
                ],
            )
        )
    out = _smooth_possession_holder_across_frames(
        pd.DataFrame(rows),
        frame_ids=list(range(1, 15)),
    )
    # After the window, a1 takes the ring
    assert out[14] == "a1"


def test_animate_passes_team_names_through() -> None:
    rows = []
    for fid in range(1, 6):
        rows.extend(
            _frame(
                fid,
                fid * 0.04,
                (50.0, 34.0),
                [
                    ("779", "m1", 50.0, 34.0),
                    ("771", "a1", 70.0, 34.0),
                ],
            )
        )
    fig = animate(
        pd.DataFrame(rows), home_team_id="779", away_team_id="771", team_names={"779": "Argentina", "771": "France"}
    )
    # Initial figure's home trace name should be the resolved team name
    home_trace = fig.data[3]
    assert home_trace.name == "Argentina"

"""Tests for the Plotly animated tactical view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from football_analysis.viz.interactive.tactical_view import (
    _ball_trail,
    _clock_text,
    _frame_traces,
    _pitch_shapes,
    _possession_holder,
    _velocity_tail_xy,
    animate,
)


def _tracking(n_frames: int = 5) -> pd.DataFrame:
    rows = []
    for fid in range(1, n_frames + 1):
        for i, team in enumerate(["home"] * 2 + ["away"] * 2):
            rows.append(
                {
                    "frame_id": fid,
                    "period": 1,
                    "time_seconds": 0.04 * fid,
                    "team_id": team,
                    "player_id": f"{team}_{i}",
                    "x": 20.0 + fid + 3 * i,
                    "y": 20.0 + 5 * i,
                    "vx": 1.0 if team == "home" else -1.0,
                    "vy": 0.5,
                    "is_ball": False,
                    "visible": True,
                }
            )
        rows.append(
            {
                "frame_id": fid,
                "period": 1,
                "time_seconds": 0.04 * fid,
                "team_id": None,
                "player_id": None,
                "x": 50.0 + fid,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def test_pitch_shapes_includes_all_markings() -> None:
    shapes = _pitch_shapes()
    types = [s["type"] for s in shapes]
    assert "line" in types
    assert "circle" in types
    assert types.count("rect") >= 3


def test_frame_traces_has_seven_traces_in_fixed_order() -> None:
    traces = _frame_traces(_tracking(), frame_id=1, home_team_id="home", away_team_id="away")
    assert len(traces) == 7
    # control (empty heatmap), tails (Scatter), ball trail (Scatter),
    # home (Scattergl), away (Scattergl), ball (Scattergl), possession (Scatter)
    assert traces[0].type == "heatmap"
    assert traces[3].name == "Home"
    assert traces[4].name == "Away"
    assert traces[5].name == "Ball"


def test_velocity_tails_emit_broken_polyline_per_player() -> None:
    sub = _tracking(1)
    frame_df = sub[sub["frame_id"] == 1]
    xs, ys = _velocity_tail_xy(frame_df)
    # Each player contributes 3 items: start, end, NaN-break
    players = (~frame_df["is_ball"] & frame_df["visible"]).sum()
    assert len(xs) == 3 * players
    assert len(ys) == 3 * players


def test_ball_trail_respects_window() -> None:
    tracking = _tracking(20)
    xs, ys = _ball_trail(tracking, frame_id=15, trail_frames=5)
    # 5 frames ending at frame 15 -> 5 ball rows (11..15 inclusive, exclusive lower bound)
    assert len(xs) == 5
    assert len(ys) == 5


def test_possession_holder_picks_closest_outfielder() -> None:
    """Move one player to sit right on top of the ball and assert they are chosen."""
    tracking = _tracking(1).copy()
    tracking.loc[
        (tracking["frame_id"] == 1) & (tracking["player_id"] == "home_0"),
        ["x", "y"],
    ] = [51.0, 34.0]  # ball is at x=51, y=34
    frame_df = tracking[tracking["frame_id"] == 1]
    xs, _ys, lbl = _possession_holder(frame_df)
    assert len(xs) == 1
    assert abs(xs[0] - 51.0) < 1e-9
    assert "home home_0" in lbl[0]


def test_possession_holder_empty_when_no_ball() -> None:
    tracking = _tracking(1)
    frame_df = tracking[(tracking["frame_id"] == 1) & (~tracking["is_ball"])]
    xs, ys, lbl = _possession_holder(frame_df)
    assert xs == []
    assert ys == []
    assert lbl == []


def test_clock_text_formats_seconds() -> None:
    df = _tracking(3)
    # frame 3 is at t = 0.12s → "P1  00:00.12"
    text = _clock_text(df[df["frame_id"] == 3])
    assert text.startswith("P1")
    assert ":00" in text


def test_clock_text_empty_input() -> None:
    assert _clock_text(pd.DataFrame()) == ""


def test_animate_produces_n_frames_matching_input() -> None:
    fig = animate(_tracking(10), home_team_id="home", away_team_id="away")
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 10
    # Every animated frame should have the same trace count as the initial figure
    assert len(fig.data) == 7
    for f in fig.frames:
        assert len(f.data) == 7


def test_animate_range_trims_frames() -> None:
    fig = animate(_tracking(10), home_team_id="home", away_team_id="away", frame_range=(3, 7))
    assert len(fig.frames) == 5
    names = [f.name for f in fig.frames]
    assert names == ["3", "4", "5", "6", "7"]


def test_animate_raises_on_empty_range() -> None:
    with pytest.raises(ValueError, match="no tracking rows"):
        animate(_tracking(3), "home", "away", frame_range=(100, 200))


def test_animate_writes_html(tmp_path: Path) -> None:
    fig = animate(_tracking(4), "home", "away")
    out = tmp_path / "anim.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    assert out.exists()
    assert out.stat().st_size > 5000
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert "Frame:" in text


def test_animate_title_includes_match_clock() -> None:
    fig = animate(_tracking(4), "home", "away")
    # Layout title on initial figure must contain the P1 clock prefix
    assert "P1" in fig.layout.title.text


def test_animate_with_pitch_control_uses_heatmap_trace() -> None:
    """When with_pitch_control=True, the first trace should be a non-empty heatmap."""
    # Using a small grid to keep this quick
    fig = animate(
        _tracking(3),
        "home",
        "away",
        with_pitch_control=True,
        pitch_control_rows=12,
        pitch_control_cols=18,
    )
    first_trace = fig.data[0]
    assert first_trace.type == "heatmap"
    # With control enabled the placeholder opacity of 0.0 should be replaced with >0
    assert first_trace.opacity > 0.0


def test_pitch_coords_are_equal_aspect() -> None:
    fig = animate(_tracking(2), "home", "away")
    assert fig.layout.xaxis.scaleanchor == "y"
    assert fig.layout.xaxis.scaleratio == 1
    assert fig.layout.yaxis.range == (-2, 70)

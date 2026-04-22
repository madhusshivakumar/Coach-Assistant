"""Tests for the Plotly animated tactical view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from football_analysis.viz.interactive.tactical_view import (
    _frame_traces,
    _pitch_shapes,
    animate,
)


def _tracking(n_frames: int = 5) -> pd.DataFrame:
    rows = []
    for fid in range(1, n_frames + 1):
        for i, team in enumerate(["home"] * 2 + ["away"] * 2):
            rows.append(
                {
                    "frame_id": fid,
                    "team_id": team,
                    "player_id": f"{team}_{i}",
                    "x": 20.0 + fid + 3 * i,
                    "y": 20.0 + 5 * i,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        # Ball row
        rows.append(
            {
                "frame_id": fid,
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


def test_pitch_shapes_includes_halfway_and_penalty_boxes() -> None:
    shapes = _pitch_shapes()
    # halfway line + 2 penalty boxes + outer + centre circle
    assert len(shapes) >= 4
    types = [s["type"] for s in shapes]
    assert "line" in types
    assert "circle" in types
    assert types.count("rect") >= 3


def test_frame_traces_returns_three_layers() -> None:
    sub = _tracking(1)
    traces = _frame_traces(sub[sub["frame_id"] == 1], "home", "away")
    assert len(traces) == 3
    assert traces[0].name == "Home"
    assert traces[1].name == "Away"
    assert traces[2].name == "Ball"


def test_animate_produces_n_frames_matching_input() -> None:
    fig = animate(_tracking(10), home_team_id="home", away_team_id="away")
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 10


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
    assert out.stat().st_size > 5000  # non-trivial HTML
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert "Frame:" in text  # slider prefix


def test_pitch_coords_are_equal_aspect() -> None:
    """Pitch length/width ratio must be preserved so the rendered pitch isn't squashed."""
    fig = animate(_tracking(2), "home", "away")
    xaxis = fig.layout.xaxis
    yaxis = fig.layout.yaxis
    assert xaxis.scaleanchor == "y"
    assert xaxis.scaleratio == 1
    assert yaxis.range == (-2, 70)

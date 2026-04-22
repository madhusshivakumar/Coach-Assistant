"""Smoke tests for the single-frame pitch-control renderer."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figs():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _tiny_tracking() -> pd.DataFrame:
    rows = []
    # Home player, away player, and ball at a single frame
    rows.append(
        {
            "frame_id": 100,
            "team_id": "home",
            "player_id": "h1",
            "x": 30.0,
            "y": 30.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 100,
            "team_id": "away",
            "player_id": "a1",
            "x": 75.0,
            "y": 40.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 100,
            "team_id": None,
            "player_id": None,
            "x": 52.5,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "visible": True,
        }
    )
    return pd.DataFrame(rows)


def test_plot_frame_returns_figure(tmp_path: Path) -> None:
    from football_analysis.viz.interactive.pitch_control import plot_frame

    fig = plot_frame(
        _tiny_tracking(),
        frame_id=100,
        home_team_id="home",
        away_team_id="away",
        team_names={"home": "Home FC", "away": "Away FC"},
    )
    assert isinstance(fig, Figure)
    out = tmp_path / "pc.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 2000


def test_plot_frame_uses_precomputed_control(tmp_path: Path) -> None:
    """Passing `control` should avoid a second call to compute_frame_from_tracking."""
    from football_analysis.analytics.pitch_control.spearman import compute_frame_from_tracking
    from football_analysis.viz.interactive.pitch_control import plot_frame

    tracking = _tiny_tracking()
    control = compute_frame_from_tracking(tracking, frame_id=100, home_team_id="home", away_team_id="away")
    fig = plot_frame(
        tracking,
        frame_id=100,
        home_team_id="home",
        away_team_id="away",
        control=control,
        title="Custom title",
    )
    assert fig.axes[0].get_title() == "Custom title"
    out = tmp_path / "pc2.png"
    fig.savefig(out)
    assert out.exists()


def test_plot_frame_handles_missing_ball() -> None:
    from football_analysis.viz.interactive.pitch_control import plot_frame

    tracking = _tiny_tracking()
    # Drop the ball row
    tracking = tracking[~tracking["is_ball"]].reset_index(drop=True)
    fig = plot_frame(tracking, frame_id=100, home_team_id="home", away_team_id="away")
    assert isinstance(fig, Figure)


def test_plot_frame_handles_off_camera_player() -> None:
    from football_analysis.viz.interactive.pitch_control import plot_frame

    tracking = _tiny_tracking()
    # Mark one away player off-camera
    tracking.loc[tracking["team_id"] == "away", "visible"] = False
    tracking.loc[tracking["team_id"] == "away", "x"] = np.nan
    fig = plot_frame(tracking, frame_id=100, home_team_id="home", away_team_id="away")
    assert isinstance(fig, Figure)

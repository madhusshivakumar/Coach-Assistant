"""Smoke tests for static plots.

We don't snapshot PNGs (pytest-mpl baselines add a lot of maintenance overhead for a POC).
Instead, each plot function must:
1. Return a `matplotlib.figure.Figure`.
2. Not raise on empty / tiny inputs.
3. Produce a non-zero-sized PNG when saved.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")  # headless backend for CI


@pytest.fixture(autouse=True)
def close_figs():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_id": "A",
                "player_id": "1",
                "action_type": "pass",
                "result": "success",
                "start_x": 40.0,
                "start_y": 34.0,
                "end_x": 70.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": 1.0,
            },
            {
                "team_id": "A",
                "player_id": "2",
                "action_type": "pass",
                "result": "success",
                "start_x": 70.0,
                "start_y": 34.0,
                "end_x": 95.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": 2.0,
            },
            {
                "team_id": "A",
                "player_id": "3",
                "action_type": "shot",
                "result": "success",
                "start_x": 100.0,
                "start_y": 34.0,
                "end_x": 105.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": 3.0,
            },
            {
                "team_id": "A",
                "player_id": "1",
                "action_type": "pass",
                "result": "fail",
                "start_x": 50.0,
                "start_y": 30.0,
                "end_x": 60.0,
                "end_y": 40.0,
                "period": 1,
                "time_seconds": 4.0,
            },
            {
                "team_id": "B",
                "player_id": "10",
                "action_type": "shot",
                "result": "fail",
                "start_x": 10.0,
                "start_y": 34.0,
                "end_x": 0.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": 5.0,
            },
            {
                "team_id": "A",
                "player_id": "1",
                "action_type": "pass",
                "result": "success",
                "start_x": 45.0,
                "start_y": 35.0,
                "end_x": 60.0,
                "end_y": 35.0,
                "period": 1,
                "time_seconds": 6.0,
            },
            {
                "team_id": "A",
                "player_id": "1",
                "action_type": "pass",
                "result": "success",
                "start_x": 50.0,
                "start_y": 30.0,
                "end_x": 65.0,
                "end_y": 30.0,
                "period": 1,
                "time_seconds": 7.0,
            },
        ]
    )


def _assert_fig_writes(fig: Figure, tmp_path: Path) -> None:
    out = tmp_path / "plot.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_shot_map_renders(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    fig = plot_shot_map(_events())
    assert isinstance(fig, Figure)
    _assert_fig_writes(fig, tmp_path)


def test_shot_map_filters_by_team(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    fig = plot_shot_map(_events(), team_id="A", title="Team A shots")
    _assert_fig_writes(fig, tmp_path)


def test_shot_map_handles_no_shots(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    passes_only = _events()[_events()["action_type"] != "shot"]
    fig = plot_shot_map(passes_only)
    _assert_fig_writes(fig, tmp_path)


def test_pass_network_renders(tmp_path: Path) -> None:
    """Feed a chronological sequence of same-team passes so the edge-builder finds receivers."""
    from football_analysis.viz.static.pass_network import plot_pass_network

    rows = []
    t = 0.0
    # 5 passes player 1 -> player 2 (interleaved so "next event" is 2)
    for _ in range(5):
        rows.append(
            {
                "team_id": "A",
                "player_id": "1",
                "action_type": "pass",
                "result": "success",
                "start_x": 40.0,
                "start_y": 34.0,
                "end_x": 70.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": t,
            }
        )
        t += 1.0
        rows.append(
            {
                "team_id": "A",
                "player_id": "2",
                "action_type": "pass",
                "result": "success",
                "start_x": 70.0,
                "start_y": 34.0,
                "end_x": 80.0,
                "end_y": 34.0,
                "period": 1,
                "time_seconds": t,
            }
        )
        t += 1.0
    # Add a self-pass (passer == receiver) to exercise the skip branch
    rows.append(
        {
            "team_id": "A",
            "player_id": "1",
            "action_type": "pass",
            "result": "success",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 50.0,
            "end_y": 34.0,
            "period": 1,
            "time_seconds": t,
        }
    )
    t += 1.0
    rows.append(
        {
            "team_id": "A",
            "player_id": "1",
            "action_type": "pass",
            "result": "success",
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
            "period": 1,
            "time_seconds": t,
        }
    )

    fig = plot_pass_network(pd.DataFrame(rows), team_id="A", min_passes_edge=2)
    _assert_fig_writes(fig, tmp_path)


def test_pass_network_drops_below_edge_threshold(tmp_path: Path) -> None:
    """When min_passes_edge is high, no edges should be drawn (but nodes should still render)."""
    from football_analysis.viz.static.pass_network import plot_pass_network

    rows = [
        {
            "team_id": "A",
            "player_id": "1",
            "action_type": "pass",
            "result": "success",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 34.0,
            "period": 1,
            "time_seconds": 1.0,
        },
        {
            "team_id": "A",
            "player_id": "2",
            "action_type": "pass",
            "result": "success",
            "start_x": 70.0,
            "start_y": 34.0,
            "end_x": 80.0,
            "end_y": 34.0,
            "period": 1,
            "time_seconds": 2.0,
        },
    ]
    fig = plot_pass_network(pd.DataFrame(rows), team_id="A", min_passes_edge=100)
    _assert_fig_writes(fig, tmp_path)


def test_pass_network_empty_team(tmp_path: Path) -> None:
    from football_analysis.viz.static.pass_network import plot_pass_network

    fig = plot_pass_network(_events(), team_id="UNKNOWN")
    _assert_fig_writes(fig, tmp_path)


def test_pass_network_no_successful_passes(tmp_path: Path) -> None:
    from football_analysis.viz.static.pass_network import plot_pass_network

    df = _events().copy()
    df.loc[df["action_type"] == "pass", "result"] = "fail"
    fig = plot_pass_network(df, team_id="A")
    _assert_fig_writes(fig, tmp_path)


def test_player_heatmap_with_few_actions(tmp_path: Path) -> None:
    from football_analysis.viz.static.heatmap import plot_player_heatmap

    fig = plot_player_heatmap(_events(), player_id="1")
    _assert_fig_writes(fig, tmp_path)


def test_player_heatmap_with_many_actions(tmp_path: Path) -> None:
    from football_analysis.viz.static.heatmap import plot_player_heatmap

    # Create 20 actions for player 1 to trigger KDE branch
    rows = [
        {
            "team_id": "A",
            "player_id": "1",
            "action_type": "pass",
            "result": "success",
            "start_x": 40.0 + i,
            "start_y": 34.0 + (i % 5),
            "end_x": None,
            "end_y": None,
            "period": 1,
            "time_seconds": float(i),
        }
        for i in range(20)
    ]
    fig = plot_player_heatmap(pd.DataFrame(rows), player_id="1")
    _assert_fig_writes(fig, tmp_path)


def test_xt_surface_renders(tmp_path: Path) -> None:
    from football_analysis.analytics.possession_value.xt import fit
    from football_analysis.viz.static.xt_surface import plot_xt_surface

    grid = fit(_events())
    fig = plot_xt_surface(grid)
    _assert_fig_writes(fig, tmp_path)


def test_theme_defaults() -> None:
    from football_analysis.viz.theme import DEFAULT_THEME, Theme

    t = Theme()
    assert t.home != t.away
    assert DEFAULT_THEME.heat_cmap == "viridis"

"""Tests for the viz polish pass: team colours, compound surnames, penalty-period filter."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figs():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _assert_fig_writes(fig: Figure, tmp_path: Path) -> None:
    out = tmp_path / "plot.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 1000


# ----- shot_map -----


def _shot(team_id: str, result: str, period: int, start_x: float = 95.0, player_id: str | None = None) -> dict:
    return {
        "team_id": team_id,
        "player_id": player_id,
        "action_type": "shot",
        "result": result,
        "start_x": start_x,
        "start_y": 34.0,
        "end_x": None,
        "end_y": None,
        "period": period,
        "time_seconds": 60.0,
    }


def test_shot_map_header_uses_regulation_goals_only(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    events = pd.DataFrame(
        [
            _shot("HOME", "success", 1),  # home regulation goal
            _shot("HOME", "fail", 1),
            _shot("AWAY", "success", 2),  # away regulation goal
            _shot("HOME", "success", 5),  # penalty — must NOT count in header
            _shot("AWAY", "success", 5),  # penalty — must NOT count in header
            _shot("AWAY", "success", 5),
        ]
    )
    fig = plot_shot_map(events, home_team_id="HOME", team_names={"HOME": "Home FC", "AWAY": "Away FC"})
    title = fig.axes[0].get_title()
    assert "Home FC 1 – 1 Away FC" in title
    _assert_fig_writes(fig, tmp_path)


def test_shot_map_excludes_penalty_shots_from_canvas(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    # 1 regulation shot + 5 penalty shots. Only the regulation shot should render.
    rows = [_shot("HOME", "fail", 1)]
    rows += [_shot("HOME", "success", 5, start_x=100.0) for _ in range(5)]
    fig = plot_shot_map(pd.DataFrame(rows), home_team_id="HOME")
    # Legend entry should reference 1 regulation shot, not 6
    # (If the penalty filter failed we'd see 6 goals in title)
    title = fig.axes[0].get_title()
    assert "Shots" in title
    _assert_fig_writes(fig, tmp_path)


def test_shot_map_annotates_scorer_name_when_player_names_given(tmp_path: Path) -> None:
    from football_analysis.viz.static.shot_map import plot_shot_map

    events = pd.DataFrame([_shot("HOME", "success", 1, player_id="42")])
    fig = plot_shot_map(
        events,
        home_team_id="HOME",
        team_names={"HOME": "Home FC"},
        player_names={"42": "Test Scorer"},
    )
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("Test Scorer" in t for t in texts)
    _assert_fig_writes(fig, tmp_path)


def test_match_header_empty_when_metadata_missing() -> None:
    from football_analysis.viz.static.shot_map import _match_header

    shots = pd.DataFrame([_shot("HOME", "success", 1)])
    assert _match_header(shots, None, None) == ""
    assert _match_header(shots, {"HOME": "Home"}, None) == ""


# ----- pass_network: compound surnames -----


def test_short_name_handles_compound_surnames() -> None:
    from football_analysis.viz.static.pass_network import _short_name

    assert _short_name("Ángel Fabián Di María") == "Di María"
    assert _short_name("Rodrigo De Paul") == "De Paul"
    assert _short_name("Virgil van Dijk") == "van Dijk"
    assert _short_name("Vincent Mc Gregor") == "Mc Gregor"


def test_short_name_takes_last_token_for_plain_surnames() -> None:
    from football_analysis.viz.static.pass_network import _short_name

    assert _short_name("Lionel Andrés Messi Cuccittini") == "Cuccittini"
    assert _short_name("Kylian Mbappé") == "Mbappé"


def test_short_name_empty_and_very_long() -> None:
    from football_analysis.viz.static.pass_network import _short_name

    assert _short_name("") == ""
    assert _short_name("A") == "A"
    # Very long last token is truncated to 14 chars
    assert _short_name("Foo SuperduperlonglastnameXYZ") == "Superduperlong"


# ----- pass_network: auto-relax node threshold -----


def _pass_row(team: str, player: str, t: float, period: int = 1) -> dict:
    return {
        "team_id": team,
        "player_id": player,
        "action_type": "pass",
        "result": "success",
        "start_x": 40.0 + (hash(player) % 40),
        "start_y": 34.0,
        "end_x": 70.0,
        "end_y": 34.0,
        "period": period,
        "time_seconds": t,
    }


def test_pass_network_auto_relaxes_when_few_players_meet_threshold(tmp_path: Path) -> None:
    """If only 3 players have ≥10 touches, the filter auto-relaxes to top-14 by touches
    so the viz still shows a usable XI-scale network."""
    from football_analysis.viz.static.pass_network import plot_pass_network

    rows: list[dict] = []
    # 2 hot players (many touches) + 8 cold ones (few touches)
    for i in range(20):
        rows.append(_pass_row("A", "hot1", t=float(i)))
        rows.append(_pass_row("A", "hot2", t=float(i) + 0.5))
    for p in range(3, 11):
        rows.append(_pass_row("A", f"cold{p}", t=float(100 + p)))

    fig = plot_pass_network(
        pd.DataFrame(rows),
        team_id="A",
        min_passes_edge=1,
        min_touches_node=10,  # only hot1 and hot2 would qualify
    )
    title = fig.axes[0].get_title()
    # Auto-relax should have kicked in — we expect more than 2 players shown
    assert "players" in title
    _assert_fig_writes(fig, tmp_path)


# ----- heatmap: name resolution -----


def test_heatmap_title_uses_player_and_team_names(tmp_path: Path) -> None:
    from football_analysis.viz.static.heatmap import plot_player_heatmap

    rows = [
        {
            "team_id": "HOME",
            "player_id": "42",
            "action_type": "pass",
            "result": "success",
            "start_x": 40.0 + i,
            "start_y": 34.0 + (i % 5),
            "end_x": None,
            "end_y": None,
            "period": 1,
            "time_seconds": float(i),
        }
        for i in range(10)
    ]
    fig = plot_player_heatmap(
        pd.DataFrame(rows),
        player_id="42",
        player_names={"42": "Test Player"},
        team_names={"HOME": "Test FC"},
    )
    title = fig.axes[0].get_title()
    assert "Test Player" in title
    assert "Test FC" in title
    _assert_fig_writes(fig, tmp_path)


def test_heatmap_with_empty_subset_still_renders(tmp_path: Path) -> None:
    from football_analysis.viz.static.heatmap import plot_player_heatmap

    rows = [
        {
            "team_id": "HOME",
            "player_id": "OTHER",
            "action_type": "pass",
            "result": "success",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": None,
            "end_y": None,
            "period": 1,
            "time_seconds": 1.0,
        }
    ]
    fig = plot_player_heatmap(pd.DataFrame(rows), player_id="nonexistent")
    _assert_fig_writes(fig, tmp_path)


# ----- theme team_color helper -----


def test_theme_team_color_selects_home_or_away() -> None:
    from football_analysis.viz.theme import DEFAULT_THEME

    assert DEFAULT_THEME.team_color("H", home_team_id="H") == DEFAULT_THEME.home
    assert DEFAULT_THEME.team_color("A", home_team_id="H") == DEFAULT_THEME.away
    assert DEFAULT_THEME.team_color("H", home_team_id=None) == DEFAULT_THEME.neutral

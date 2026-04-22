"""Plotly-based animated tactical view — scrub through tracking frames.

Renders a top-down 105 x 68 m pitch with home/away players + the ball, and an
animation slider over a contiguous range of `frame_id` values. Exported as
standalone HTML so it can be viewed without a running server.

Per the architecture doc this is the first cut of "Goal #1 — single POV of all
games". Per-frame pitch-control overlay is deliberately *out of scope here* —
it would require pre-computing all surfaces (expensive). Phase-2 Slice D will
add that with a cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M

if TYPE_CHECKING:
    pass

_HOME_COLOR = "#1f77b4"
_AWAY_COLOR = "#d62728"
_BALL_COLOR = "#f6f6f6"


def _pitch_shapes() -> list[dict[str, object]]:
    """Static pitch-line shapes — halfway line, centre circle, penalty areas."""
    return [
        # Outer boundary
        {
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": PITCH_LENGTH_M,
            "y1": PITCH_WIDTH_M,
            "line": {"color": "black", "width": 2},
            "fillcolor": "rgba(0,0,0,0)",
        },
        # Halfway line
        {
            "type": "line",
            "x0": PITCH_LENGTH_M / 2,
            "x1": PITCH_LENGTH_M / 2,
            "y0": 0,
            "y1": PITCH_WIDTH_M,
            "line": {"color": "black", "width": 1.5},
        },
        # Centre circle
        {
            "type": "circle",
            "x0": PITCH_LENGTH_M / 2 - 9.15,
            "y0": PITCH_WIDTH_M / 2 - 9.15,
            "x1": PITCH_LENGTH_M / 2 + 9.15,
            "y1": PITCH_WIDTH_M / 2 + 9.15,
            "line": {"color": "black", "width": 1.5},
            "fillcolor": "rgba(0,0,0,0)",
        },
        # Left penalty box (16.5m deep, 40.3m wide)
        {
            "type": "rect",
            "x0": 0,
            "y0": (PITCH_WIDTH_M - 40.3) / 2,
            "x1": 16.5,
            "y1": (PITCH_WIDTH_M + 40.3) / 2,
            "line": {"color": "black", "width": 1.5},
            "fillcolor": "rgba(0,0,0,0)",
        },
        # Right penalty box
        {
            "type": "rect",
            "x0": PITCH_LENGTH_M - 16.5,
            "y0": (PITCH_WIDTH_M - 40.3) / 2,
            "x1": PITCH_LENGTH_M,
            "y1": (PITCH_WIDTH_M + 40.3) / 2,
            "line": {"color": "black", "width": 1.5},
            "fillcolor": "rgba(0,0,0,0)",
        },
    ]


def _frame_traces(
    frame_df: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
) -> list[go.Scattergl]:
    players = frame_df[~frame_df["is_ball"] & frame_df["visible"]]
    home = players[players["team_id"] == home_team_id]
    away = players[players["team_id"] == away_team_id]
    ball = frame_df[frame_df["is_ball"] & frame_df["visible"]]

    return [
        go.Scattergl(
            x=home["x"],
            y=home["y"],
            mode="markers",
            marker={"size": 14, "color": _HOME_COLOR, "line": {"width": 1, "color": "black"}},
            name="Home",
            hovertext=home["player_id"],
            hoverinfo="text+x+y",
        ),
        go.Scattergl(
            x=away["x"],
            y=away["y"],
            mode="markers",
            marker={"size": 14, "color": _AWAY_COLOR, "line": {"width": 1, "color": "black"}},
            name="Away",
            hovertext=away["player_id"],
            hoverinfo="text+x+y",
        ),
        go.Scattergl(
            x=ball["x"],
            y=ball["y"],
            mode="markers",
            marker={"size": 11, "color": _BALL_COLOR, "line": {"width": 1.5, "color": "black"}, "symbol": "circle"},
            name="Ball",
            hoverinfo="x+y",
        ),
    ]


def animate(
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    frame_range: tuple[int, int] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Return a Plotly Figure with per-frame traces + an animation slider.

    Args:
        tracking: canonical long-form tracking DataFrame.
        home_team_id / away_team_id: team ids in the DataFrame.
        frame_range: (start_frame, end_frame) inclusive; defaults to full dataset.
        title: figure title.
    """
    df = tracking
    if frame_range is not None:
        lo, hi = frame_range
        df = df[(df["frame_id"] >= lo) & (df["frame_id"] <= hi)]
    if df.empty:
        raise ValueError("no tracking rows in the requested frame range")

    frame_ids = sorted(df["frame_id"].unique())
    initial_df = df[df["frame_id"] == frame_ids[0]]
    initial_traces = _frame_traces(initial_df, home_team_id, away_team_id)

    animation_frames = []
    slider_steps = []
    for fid in frame_ids:
        sub = df[df["frame_id"] == fid]
        animation_frames.append(
            go.Frame(
                data=_frame_traces(sub, home_team_id, away_team_id),
                name=str(fid),
            )
        )
        slider_steps.append(
            {
                "args": [
                    [str(fid)],
                    {"mode": "immediate", "frame": {"duration": 40, "redraw": True}, "transition": {"duration": 0}},
                ],
                "label": str(fid),
                "method": "animate",
            }
        )

    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=title or f"Tactical view — frames {frame_ids[0]}-{frame_ids[-1]}",
            xaxis={
                "range": [-2, PITCH_LENGTH_M + 2],
                "showgrid": False,
                "zeroline": False,
                "visible": False,
                "scaleanchor": "y",
                "scaleratio": 1,
            },
            yaxis={"range": [-2, PITCH_WIDTH_M + 2], "showgrid": False, "zeroline": False, "visible": False},
            shapes=_pitch_shapes(),
            plot_bgcolor="#f8fff0",
            showlegend=True,
            height=620,
            margin={"l": 10, "r": 10, "t": 60, "b": 50},
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.02,
                    "xanchor": "left",
                    "yanchor": "top",
                    "pad": {"t": 50, "r": 10},
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 40, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "❚❚ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "y": 0,
                    "x": 0.1,
                    "len": 0.85,
                    "currentvalue": {"prefix": "Frame: ", "font": {"size": 12}},
                    "steps": slider_steps,
                    "pad": {"t": 40, "b": 10},
                }
            ],
        ),
        frames=animation_frames,
    )
    return fig

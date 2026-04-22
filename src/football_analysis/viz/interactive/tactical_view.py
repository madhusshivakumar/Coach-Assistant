"""Plotly-based animated tactical view — scrub through tracking frames.

Renders a top-down 105 x 68 m pitch with:
- Home and away players colour-coded, with per-player velocity tails showing
  where each player will be ~0.5 s into the future if current motion holds.
- The ball with a trailing "path of the ball" over the last ~1 s (25 frames).
- A gold ring around the player currently closest to the ball (a cheap
  possession heuristic; no need for provider-labelled possession).
- Match-clock annotation in the title.
- Optional pre-computed pitch-control heatmap (off by default — it's expensive
  because every animated frame needs its own surface).

Exported as standalone HTML so it can be viewed without a running server.

Per the architecture doc this is Goal #1 — "single POV of all games". Having
velocity + trail + possession + clock turns the raw scatter into something a
coach can actually read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.analytics.pitch_control.spearman import compute_frame_from_tracking

_HOME_COLOR = "#1f77b4"
_AWAY_COLOR = "#d62728"
_BALL_COLOR = "#f6f6f6"
_BALL_TRAIL_COLOR = "rgba(80,80,80,0.65)"
_POSSESSION_RING_COLOR = "#ffd54a"

# Seconds into the future each player's velocity tail shows.
_VELOCITY_TAIL_SECONDS: float = 0.5
# How many past frames to include in the ball trail.
_BALL_TRAIL_FRAMES: int = 25

# Trace index layout (kept constant across every animation frame so Plotly
# can diff them cleanly).
_IDX_CONTROL = 0  # optional pitch-control heatmap; empty when disabled
_IDX_TAILS = 1  # velocity tails for all players (lines)
_IDX_BALL_TRAIL = 2  # ball path over the last N frames
_IDX_HOME = 3  # home player scatter
_IDX_AWAY = 4  # away player scatter
_IDX_BALL = 5  # current ball position
_IDX_POSSESSION = 6  # gold ring around nearest-to-ball player


def _pitch_shapes() -> list[dict[str, object]]:
    """Static pitch-line shapes — halfway line, centre circle, penalty areas."""
    return [
        {
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": PITCH_LENGTH_M,
            "y1": PITCH_WIDTH_M,
            "line": {"color": "black", "width": 2},
            "fillcolor": "rgba(0,0,0,0)",
        },
        {
            "type": "line",
            "x0": PITCH_LENGTH_M / 2,
            "x1": PITCH_LENGTH_M / 2,
            "y0": 0,
            "y1": PITCH_WIDTH_M,
            "line": {"color": "black", "width": 1.5},
        },
        {
            "type": "circle",
            "x0": PITCH_LENGTH_M / 2 - 9.15,
            "y0": PITCH_WIDTH_M / 2 - 9.15,
            "x1": PITCH_LENGTH_M / 2 + 9.15,
            "y1": PITCH_WIDTH_M / 2 + 9.15,
            "line": {"color": "black", "width": 1.5},
            "fillcolor": "rgba(0,0,0,0)",
        },
        {
            "type": "rect",
            "x0": 0,
            "y0": (PITCH_WIDTH_M - 40.3) / 2,
            "x1": 16.5,
            "y1": (PITCH_WIDTH_M + 40.3) / 2,
            "line": {"color": "black", "width": 1.5},
            "fillcolor": "rgba(0,0,0,0)",
        },
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


def _velocity_tail_xy(frame_df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Build a single polyline that traces each player's velocity tail via `None` breaks."""
    xs: list[float] = []
    ys: list[float] = []
    players = frame_df[~frame_df["is_ball"] & frame_df["visible"]]
    for _, p in players.iterrows():
        xs.extend([float(p["x"]), float(p["x"] + p["vx"] * _VELOCITY_TAIL_SECONDS), float("nan")])
        ys.extend([float(p["y"]), float(p["y"] + p["vy"] * _VELOCITY_TAIL_SECONDS), float("nan")])
    return xs, ys


def _ball_trail(
    tracking: pd.DataFrame,
    frame_id: int,
    trail_frames: int = _BALL_TRAIL_FRAMES,
) -> tuple[list[float], list[float]]:
    ball = tracking[
        tracking["is_ball"]
        & tracking["visible"]
        & (tracking["frame_id"] <= frame_id)
        & (tracking["frame_id"] > frame_id - trail_frames)
    ].sort_values("frame_id")
    return ball["x"].tolist(), ball["y"].tolist()


def _possession_holder(frame_df: pd.DataFrame) -> tuple[list[float], list[float], list[str]]:
    """Find the outfielder closest to the ball. Returns xs, ys, and a label."""
    ball = frame_df[frame_df["is_ball"] & frame_df["visible"]]
    players = frame_df[~frame_df["is_ball"] & frame_df["visible"]]
    if ball.empty or players.empty:
        return [], [], []
    b = ball.iloc[0]
    d2 = (players["x"] - b["x"]) ** 2 + (players["y"] - b["y"]) ** 2
    idx = int(d2.idxmin())
    row = players.loc[idx]
    label = f"{row['team_id']} {row['player_id']}  d={float(np.sqrt(d2.loc[idx])):.1f}m"
    return [float(row["x"])], [float(row["y"])], [label]


def _empty_heatmap_trace() -> go.Heatmap:
    """Placeholder heatmap trace used when pitch control is disabled — keeps the
    per-frame trace count constant."""
    return go.Heatmap(z=[[0.5]], x=[52.5], y=[34.0], showscale=False, opacity=0.0, hoverinfo="skip")


def _frame_traces(
    tracking: pd.DataFrame,
    frame_id: int,
    home_team_id: str,
    away_team_id: str,
    pitch_control_surface: np.ndarray | None = None,
    pitch_control_xs: np.ndarray | None = None,
    pitch_control_ys: np.ndarray | None = None,
) -> list[go.BaseTraceType]:
    frame_df = tracking[tracking["frame_id"] == frame_id]

    if pitch_control_surface is not None and pitch_control_xs is not None and pitch_control_ys is not None:
        control_trace = go.Heatmap(
            z=pitch_control_surface,
            x=pitch_control_xs,
            y=pitch_control_ys,
            colorscale="RdBu",
            zmin=0.0,
            zmax=1.0,
            opacity=0.45,
            showscale=False,
            hoverinfo="skip",
        )
    else:
        control_trace = _empty_heatmap_trace()

    tail_x, tail_y = _velocity_tail_xy(frame_df)
    tail_trace = go.Scatter(
        x=tail_x,
        y=tail_y,
        mode="lines",
        line={"color": "rgba(40,40,40,0.55)", "width": 1.2},
        name="velocity (0.5s)",
        hoverinfo="skip",
        showlegend=True,
    )

    trail_xs, trail_ys = _ball_trail(tracking, frame_id)
    ball_trail_trace = go.Scatter(
        x=trail_xs,
        y=trail_ys,
        mode="lines",
        line={"color": _BALL_TRAIL_COLOR, "width": 2, "dash": "dot"},
        name="ball trail (1s)",
        hoverinfo="skip",
        showlegend=True,
    )

    players = frame_df[~frame_df["is_ball"] & frame_df["visible"]]
    home = players[players["team_id"] == home_team_id]
    away = players[players["team_id"] == away_team_id]
    ball = frame_df[frame_df["is_ball"] & frame_df["visible"]]

    home_trace = go.Scattergl(
        x=home["x"],
        y=home["y"],
        mode="markers",
        marker={"size": 14, "color": _HOME_COLOR, "line": {"width": 1, "color": "black"}},
        name="Home",
        hovertext=home["player_id"],
        hoverinfo="text+x+y",
    )
    away_trace = go.Scattergl(
        x=away["x"],
        y=away["y"],
        mode="markers",
        marker={"size": 14, "color": _AWAY_COLOR, "line": {"width": 1, "color": "black"}},
        name="Away",
        hovertext=away["player_id"],
        hoverinfo="text+x+y",
    )
    ball_trace = go.Scattergl(
        x=ball["x"],
        y=ball["y"],
        mode="markers",
        marker={"size": 12, "color": _BALL_COLOR, "line": {"width": 1.5, "color": "black"}, "symbol": "circle"},
        name="Ball",
        hoverinfo="x+y",
    )

    poss_x, poss_y, poss_lbl = _possession_holder(frame_df)
    poss_trace = go.Scatter(
        x=poss_x,
        y=poss_y,
        mode="markers",
        marker={"size": 28, "color": "rgba(0,0,0,0)", "line": {"width": 3, "color": _POSSESSION_RING_COLOR}},
        name="nearest to ball",
        hovertext=poss_lbl,
        hoverinfo="text",
        showlegend=True,
    )

    # Fixed order declared above — keep consistent across every animated frame
    # so Plotly can diff the traces cleanly.
    traces = [control_trace, tail_trace, ball_trail_trace, home_trace, away_trace, ball_trace, poss_trace]
    return traces


def _clock_text(frame_df: pd.DataFrame) -> str:
    """Format the game clock for a frame (assumes one (match, period, frame_id))."""
    if frame_df.empty:
        return ""
    t = float(frame_df["time_seconds"].iloc[0])
    mm = int(t // 60)
    ss = t - mm * 60
    period = int(frame_df["period"].iloc[0]) if "period" in frame_df.columns else 1
    return f"P{period}  {mm:02d}:{ss:05.2f}"


def _precompute_pitch_control(
    tracking: pd.DataFrame,
    frame_ids: list[int],
    home_team_id: str,
    away_team_id: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Compute pitch-control surface for every frame once, up-front."""
    surfaces: list[np.ndarray] = []
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    for fid in frame_ids:
        pc = compute_frame_from_tracking(
            tracking,
            frame_id=fid,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            rows=grid_rows,
            cols=grid_cols,
        )
        surfaces.append(pc.home_control)
        if xs is None:
            xs, ys = pc.xs, pc.ys
    assert xs is not None and ys is not None
    return surfaces, xs, ys


def animate(
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    frame_range: tuple[int, int] | None = None,
    title: str | None = None,
    *,
    with_pitch_control: bool = False,
    pitch_control_rows: int = 34,
    pitch_control_cols: int = 52,
) -> go.Figure:
    """Return a Plotly Figure with tactical overlays and an animation slider.

    Args:
        tracking: canonical long-form tracking DataFrame.
        home_team_id / away_team_id: team ids in the DataFrame.
        frame_range: (start_frame, end_frame) inclusive; defaults to full dataset.
        title: figure title. Appended with the match clock per frame.
        with_pitch_control: pre-compute and show a per-frame pitch-control
            heatmap. Expensive — O(rows x cols x frames x players). Default off.
        pitch_control_rows / pitch_control_cols: coarser grid than analytics
            default keeps the animation smooth; 34x52 (2 m cells) is plenty for
            a visual overlay.
    """
    df = tracking
    if frame_range is not None:
        lo, hi = frame_range
        df = df[(df["frame_id"] >= lo) & (df["frame_id"] <= hi)]
    if df.empty:
        raise ValueError("no tracking rows in the requested frame range")

    frame_ids = sorted(int(f) for f in df["frame_id"].unique())

    surfaces: list[np.ndarray | None]
    pc_xs: np.ndarray | None = None
    pc_ys: np.ndarray | None = None
    if with_pitch_control:
        computed, pc_xs, pc_ys = _precompute_pitch_control(
            df,
            frame_ids,
            home_team_id,
            away_team_id,
            grid_rows=pitch_control_rows,
            grid_cols=pitch_control_cols,
        )
        surfaces = list(computed)
    else:
        surfaces = [None] * len(frame_ids)

    initial_df = df[df["frame_id"] == frame_ids[0]]
    clock0 = _clock_text(initial_df)
    initial_traces = _frame_traces(
        df,
        frame_ids[0],
        home_team_id,
        away_team_id,
        pitch_control_surface=surfaces[0],
        pitch_control_xs=pc_xs,
        pitch_control_ys=pc_ys,
    )

    animation_frames: list[go.Frame] = []
    slider_steps: list[dict[str, object]] = []
    for i, fid in enumerate(frame_ids):
        fdf = df[df["frame_id"] == fid]
        animation_frames.append(
            go.Frame(
                data=_frame_traces(
                    df,
                    fid,
                    home_team_id,
                    away_team_id,
                    pitch_control_surface=surfaces[i],
                    pitch_control_xs=pc_xs,
                    pitch_control_ys=pc_ys,
                ),
                name=str(fid),
                layout={"title": f"{title or 'Tactical view'}  —  {_clock_text(fdf)}"},
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
            title=f"{title or 'Tactical view'}  —  {clock0}",
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
                },
            ],
        ),
        frames=animation_frames,
    )
    return fig

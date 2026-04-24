"""Team-shape / compactness metrics per frame.

All metrics operate on a single team's 10 outfielders at one frame (GK excluded).
Coordinates are in canonical metres on 105×68.

- `length`: x-extent of the team (max_x − min_x).
- `width`: y-extent of the team (max_y − min_y).
- `convex_hull_area`: shapely convex-hull area in m².
- `defensive_line_height`: mean x of the back 4 (the four lowest-x players).
- `offensive_line_height`: mean x of the top 3 (the three highest-x players).
- `vertical_compactness`: offensive − defensive line height (smaller = tighter block).

All numbers assume the team is attacking towards +x. Pass `attacking_right=False`
for away-team frames to get a geometry-consistent reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass(frozen=True)
class ShapeMetrics:
    """All single-frame compactness outputs."""

    length: float
    width: float
    convex_hull_area: float
    defensive_line_height: float
    offensive_line_height: float
    vertical_compactness: float


def _orient_to_attacking_right(coords: np.ndarray, attacking_right: bool) -> np.ndarray:
    if attacking_right:
        return coords
    mirrored = coords.copy()
    mirrored[:, 0] = PITCH_LENGTH_M - mirrored[:, 0]
    mirrored[:, 1] = PITCH_WIDTH_M - mirrored[:, 1]
    return mirrored


def compute_shape(
    outfielders: pd.DataFrame,
    attacking_right: bool = True,
    back_line_n: int = 4,
    top_line_n: int = 3,
) -> ShapeMetrics:
    """Return compactness metrics for a team's 10 outfielders in one frame."""
    if len(outfielders) == 0:
        return ShapeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    coords = outfielders[["x", "y"]].to_numpy(dtype=np.float64)
    oriented = _orient_to_attacking_right(coords, attacking_right)
    xs = oriented[:, 0]
    ys = oriented[:, 1]

    length = float(xs.max() - xs.min())
    width = float(ys.max() - ys.min())

    hull_area = float(MultiPoint([tuple(p) for p in oriented]).convex_hull.area) if len(oriented) >= 3 else 0.0

    x_sorted = np.sort(xs)
    defensive_line_height = float(x_sorted[:back_line_n].mean()) if len(xs) >= back_line_n else float(xs.min())
    offensive_line_height = float(x_sorted[-top_line_n:].mean()) if len(xs) >= top_line_n else float(xs.max())
    vertical_compactness = offensive_line_height - defensive_line_height

    return ShapeMetrics(
        length=length,
        width=width,
        convex_hull_area=hull_area,
        defensive_line_height=defensive_line_height,
        offensive_line_height=offensive_line_height,
        vertical_compactness=vertical_compactness,
    )


def shape_time_series(
    tracking: pd.DataFrame,
    team_id: str,
    attacking_right: bool = True,
    back_line_n: int = 4,
    top_line_n: int = 3,
) -> pd.DataFrame:
    """Compute compactness metrics for every frame in `tracking` for one team.

    Excludes the team's goalkeeper (approximated as the player with the lowest mean
    x across the whole tracking window for the team) and the ball.
    """
    team_mask = (tracking["team_id"] == team_id) & (~tracking["is_ball"]) & tracking["visible"]
    team_df = tracking[team_mask]
    if team_df.empty:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "time_seconds",
                "length",
                "width",
                "convex_hull_area",
                "defensive_line_height",
                "offensive_line_height",
                "vertical_compactness",
            ]
        )

    # Identify GK: the player with the lowest mean x (in attacking-right frame).
    mean_x = team_df.groupby("player_id")["x"].mean()
    if not attacking_right:
        mean_x = PITCH_LENGTH_M - mean_x
    gk_id = mean_x.idxmin()

    out_rows: list[dict[str, object]] = []
    for (frame_id, time_seconds), fdf in team_df[team_df["player_id"] != gk_id].groupby(
        ["frame_id", "time_seconds"], sort=True
    ):
        s = compute_shape(fdf, attacking_right=attacking_right, back_line_n=back_line_n, top_line_n=top_line_n)
        out_rows.append(
            {
                "frame_id": int(frame_id),
                "time_seconds": float(time_seconds),
                "length": s.length,
                "width": s.width,
                "convex_hull_area": s.convex_hull_area,
                "defensive_line_height": s.defensive_line_height,
                "offensive_line_height": s.offensive_line_height,
                "vertical_compactness": s.vertical_compactness,
            }
        )
    return pd.DataFrame(out_rows)

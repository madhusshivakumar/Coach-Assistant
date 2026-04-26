"""SoccerNet GameState 2024 → canonical tracking long-form DataFrame.

Each clip becomes one ``(competition=SoccerNet, season=gamestate-2024, match_id=
soccernet-{clip_id})`` parquet, with one period (the clip is short and self-contained).

Schema mappings:

- **Player position**: ``annotations[i].bbox_pitch.x_bottom_middle`` /
  ``y_bottom_middle`` is the foot position in the SoccerNet pitch frame
  (centered at (0, 0); ranges roughly ±52.5 × ±34). We translate to canonical
  corner-origin (+52.5, +34) — same as SkillCorner.
- **Player→team**: ``attributes.team ∈ {"left", "right"}``. We pick "home" =
  whichever team appears as ``home_side_first`` in the clip (often "left" but
  not guaranteed). For our pipeline, the labels "home" / "away" are arbitrary
  identifiers — the engine's ``attacking_directions`` parameter is what
  matters.
- **Time**: ``image_id`` is monotonic per frame; we derive ``frame_id`` and
  ``time_seconds`` from frame index ÷ frame_rate (25 Hz).
- **Velocities**: not provided; computed via finite difference per
  (track_id, period) using the existing ``_attach_velocities`` helper.

Caveats — short clips, real signal:

- Each clip is a 30-second window around a known action (``info.action_class``).
  The data isn't a full match; an "episode" segmented by our engine will often
  span the whole clip.
- ``track_id`` resets between clips, so cross-clip player identity isn't
  recoverable. This is fine for episode retrieval (which doesn't care about
  player identity across episodes).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.data.normalize.tracking import _attach_velocities

# SoccerNet category IDs.
_CAT_PLAYER = 1
_CAT_GOALKEEPER = 2
_CAT_REFEREE = 3
_CAT_BALL = 4

# Bounds for accepting a ``bbox_pitch`` projection. SoccerNet's CV pipeline
# extrapolates wildly for players near the camera or partially off-frame —
# we've seen x values up to 136 m and y values as low as -34 m. The
# canonical pitch is 105 x 68 with our schema's grace zone of +/-15 m at the
# edges. We clip aggressively to the schema's ``ge=-15.0, le=120.0`` /
# ``ge=-15.0, le=83.0`` window (here in *raw* SoccerNet coordinates, i.e.
# before the centered → corner-origin translation).
_RAW_X_MIN: float = -67.5  # canonical -15 - half_pitch_length(52.5)
_RAW_X_MAX: float = 67.5  # canonical 120 - half_pitch_length(52.5)
_RAW_Y_MIN: float = -49.0  # canonical -15 - half_pitch_width(34)
_RAW_Y_MAX: float = 49.0  # canonical 83 - half_pitch_width(34)


def _pick_team_label(team_attr: str | None, home_side: str) -> str | None:
    """Map SoccerNet's per-player ``team`` attribute to ``"home"`` / ``"away"``.

    ``team_attr`` is ``"left"`` or ``"right"`` (or None for referees / unknowns).
    ``home_side`` is whichever side we declare home in this clip.
    """
    if team_attr not in ("left", "right"):
        return None
    return "home" if team_attr == home_side else "away"


def soccernet_clip_to_long(
    clip_data: dict[str, Any],
    match_id: str,
    home_side: str = "left",
) -> pd.DataFrame:
    """Convert one SoccerNet clip into canonical long-form tracking.

    Args:
        clip_data: parsed ``Labels-GameState.json`` from ``soccernet.load_clip()``.
        match_id: canonical match id (e.g. ``"soccernet:gamestate-2024-116"``).
        home_side: which SoccerNet side is treated as ``"home"``. The choice is
            arbitrary for short clips — what matters is downstream consistency.

    Returns:
        Canonical tracking DataFrame. Empty if no usable annotations.
    """
    info = clip_data.get("info", {})
    frame_rate = float(info.get("frame_rate", 25))

    # image_id → frame_index (1-based in SoccerNet — image_id like "3116000001"
    # encodes (game, frame); we need the frame portion). Easiest: enumerate
    # `images` in order and assign frame_id sequentially.
    images = clip_data.get("images", [])
    image_to_frame: dict[str, int] = {}
    for i, img in enumerate(images, start=1):
        # `image_id` is a string in SoccerNet.
        image_to_frame[str(img.get("image_id"))] = i

    half_x = PITCH_LENGTH_M / 2.0
    half_y = PITCH_WIDTH_M / 2.0

    rows: list[dict[str, Any]] = []
    for ann in clip_data.get("annotations", []):
        cat = ann.get("category_id")
        if cat not in (_CAT_PLAYER, _CAT_GOALKEEPER, _CAT_BALL):
            continue
        bbox = ann.get("bbox_pitch") or {}
        bx = bbox.get("x_bottom_middle")
        by = bbox.get("y_bottom_middle")
        if bx is None or by is None:
            continue
        # Drop wildly-extrapolated projections (CV failure on near-camera or
        # partially-off-frame entities). Schema has its own grace zone, but
        # broken projections can be off by 50 m+; reject early.
        if not (_RAW_X_MIN <= float(bx) <= _RAW_X_MAX and _RAW_Y_MIN <= float(by) <= _RAW_Y_MAX):
            continue

        image_id = str(ann.get("image_id"))
        frame_id = image_to_frame.get(image_id)
        if frame_id is None:
            continue
        time_s = (frame_id - 1) / frame_rate

        is_ball = cat == _CAT_BALL
        attrs = ann.get("attributes") or {}
        team_label: str | None = None
        if not is_ball:
            team_label = _pick_team_label(attrs.get("team"), home_side)
            if team_label is None:
                # Skip referees and unattributed players.
                continue

        # Stable per-clip player identifier — track_id is the SoccerNet primary key.
        player_id = "ball" if is_ball else f"track-{ann.get('track_id')}"

        rows.append(
            {
                "match_id": match_id,
                "period": 1,
                "frame_id": int(frame_id),
                "time_seconds": float(time_s),
                "player_id": None if is_ball else str(player_id),
                "team_id": None if is_ball else team_label,
                "x": float(bx) + half_x,
                "y": float(by) + half_y,
                "is_ball": is_ball,
                "visible": True,  # bbox present implies the entity is visible in this frame
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        for col in ("vx", "vy", "speed"):
            df[col] = pd.Series([], dtype="float64")
        return df
    return _attach_velocities(df)

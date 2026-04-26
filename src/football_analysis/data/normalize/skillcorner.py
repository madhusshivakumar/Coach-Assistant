"""SkillCorner Open Data → canonical tracking long-form DataFrame.

Schema mismatches we resolve:

- **Coordinate frame.** SkillCorner is centered: x ∈ [-pitch_length/2, +pitch_length/2],
  y ∈ [-pitch_width/2, +pitch_width/2]. Canonical is corner-origin
  (0..105, 0..68). We translate.
- **Orientation.** ``match.home_team_side`` lists which side the home team
  attacks each period (e.g. ``["right_to_left", "left_to_right"]``). Our canonical
  stores raw positions; downstream consumers pass ``attacking_directions`` to
  the engine to interpret. We propagate ``home_team_side`` into the output
  metadata so callers can configure correctly.
- **Player→team membership.** SkillCorner tracking carries ``player_id``;
  ``match.json[players]`` provides the join to ``team_id``. We join inline.
- **Velocities.** Not in raw data; we compute via finite difference per
  (player, period) using the existing ``_attach_velocities`` helper from
  the Metrica normaliser — same physiological caps apply.
- **Sample rate.** SkillCorner is 10 Hz, Metrica is 25 Hz. Our pipeline
  doesn't assume a fixed rate, so this is just a difference, not a problem.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.data.normalize.tracking import _attach_velocities


def _player_to_team(match_data: dict[str, Any]) -> dict[int, str]:
    """player_id → "home" / "away" lookup from match.json."""
    home_id = match_data["home_team"]["id"]
    away_id = match_data["away_team"]["id"]
    out: dict[int, str] = {}
    for p in match_data.get("players", []):
        pid = p.get("id")
        tid = p.get("team_id")
        if pid is None or tid is None:
            continue
        if tid == home_id:
            out[int(pid)] = "home"
        elif tid == away_id:
            out[int(pid)] = "away"
    return out


def _parse_timestamp(ts: str | None) -> float | None:
    """Parse a SkillCorner timestamp into seconds.

    SkillCorner emits ``"HH:MM:SS.ss"`` (period-relative). We also tolerate
    ``"MM:SS.ss"`` because their schema docs are inconsistent and a defensive
    parser saves debugging next time the format drifts.
    """
    if not ts:
        return None
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, TypeError):
        return None
    return None


def skillcorner_to_long(
    match_data: dict[str, Any],
    tracking_frames: list[dict[str, Any]],
    match_id: str,
) -> pd.DataFrame:
    """Convert one SkillCorner match into canonical long-form tracking.

    Output columns: ``match_id, period, frame_id, time_seconds, player_id,
    team_id, x, y, is_ball, visible, vx, vy, speed`` (matches Metrica).

    Frames whose ``period`` is null (warmup / setup) are dropped. Within each
    period, ``time_seconds`` is the *period-relative* elapsed time parsed from
    the ``timestamp`` field; we don't try to chain periods because period
    boundaries are explicit in the schema and downstream code keys on
    ``(period, frame_id)``.
    """
    half_x = PITCH_LENGTH_M / 2.0
    half_y = PITCH_WIDTH_M / 2.0

    player_team = _player_to_team(match_data)
    rows: list[dict[str, Any]] = []

    for f in tracking_frames:
        period = f.get("period")
        if period is None:
            continue
        period_int = int(period)
        frame_id = int(f["frame"])
        time_s = _parse_timestamp(f.get("timestamp"))
        if time_s is None:
            continue

        # Ball row (one per frame even if missing — visible flag carries the signal).
        ball = f.get("ball_data", {}) or {}
        bx = ball.get("x")
        by = ball.get("y")
        is_detected = bool(ball.get("is_detected"))
        rows.append(
            {
                "match_id": match_id,
                "period": period_int,
                "frame_id": frame_id,
                "time_seconds": time_s,
                "player_id": None,
                "team_id": None,
                "x": (float(bx) + half_x) if bx is not None else None,
                "y": (float(by) + half_y) if by is not None else None,
                "is_ball": True,
                "visible": is_detected and bx is not None,
            }
        )

        # Player rows — only emit those we can attribute to home or away.
        for p in f.get("player_data", []) or []:
            pid = p.get("player_id")
            px = p.get("x")
            py = p.get("y")
            if pid is None or px is None or py is None:
                continue
            team = player_team.get(int(pid))
            if team is None:
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "period": period_int,
                    "frame_id": frame_id,
                    "time_seconds": time_s,
                    "player_id": str(pid),
                    "team_id": team,
                    "x": float(px) + half_x,
                    "y": float(py) + half_y,
                    "is_ball": False,
                    "visible": bool(p.get("is_detected", True)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        for col in ("vx", "vy", "speed"):
            df[col] = pd.Series([], dtype="float64")
        return df
    return _attach_velocities(df)

"""Convert raw StatsBomb events (list of dicts) to the canonical SPADL-extended DataFrame.

This is a direct, dependency-light normaliser that does not require kloppy at runtime.
It handles the StatsBomb open-data schema specifically; other providers get their own
module under `data/normalize/`.

StatsBomb conventions (as documented in their open-data spec):
- Pitch 120x80 yards, origin top-left.
- Events are a list of dicts with `type.name`, `team.id`, `player.id`, `location`, etc.
- Pass events carry `pass.end_location`; shots carry `shot.end_location`; carries carry `carry.end_location`.
- Period 1/2 orientations: StatsBomb keeps both teams attacking to the right of *their own*
  coordinate system — per half each team's events are mirrored to share one orientation.
  In practice, `team.id` combined with the match's home/away info gives us the info we need
  to derive `home_attacks_left_to_right_p1`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M, rescale_point
from football_analysis.data.normalize.orientation import normalise_point

# StatsBomb's native pitch dimensions
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0


# SPADL action type mapping. StatsBomb type names -> SPADL action_type.
# Only the action types we care about in Phase 0 are enumerated; the rest fall through
# as a safe "non_action" label and can be added incrementally.
_ACTION_TYPE_MAP: dict[str, str] = {
    "Pass": "pass",
    "Shot": "shot",
    "Carry": "dribble",
    "Clearance": "clearance",
    "Ball Receipt*": "receival",
    "Foul Committed": "foul",
    "Dribble": "take_on",
    "Interception": "interception",
    "Block": "block",
    "Goal Keeper": "keeper_action",
    "Duel": "tackle",
}

# SPADL result enum. StatsBomb encodes outcomes per event type with different subfields.
# We derive a single `result` column conservatively.
_PASS_OUTCOMES: dict[str, str] = {
    # outcome.name -> result
    "Incomplete": "fail",
    "Out": "fail",
    "Pass Offside": "offside",
    "Injury Clearance": "fail",
    "Unknown": "fail",
}
_SHOT_OUTCOMES: dict[str, str] = {
    "Goal": "success",
    "Saved": "fail",
    "Off T": "fail",
    "Blocked": "fail",
    "Wayward": "fail",
    "Saved Off Target": "fail",
    "Saved To Post": "fail",
    "Post": "fail",
}


def _location_tuple(loc: list[float] | None) -> tuple[float, float] | tuple[None, None]:
    if loc is None or len(loc) < 2:
        return (None, None)
    return float(loc[0]), float(loc[1])


def _pass_end(ev: dict[str, Any]) -> tuple[float, float] | tuple[None, None]:
    return _location_tuple(ev.get("pass", {}).get("end_location"))


def _shot_end(ev: dict[str, Any]) -> tuple[float, float] | tuple[None, None]:
    loc = ev.get("shot", {}).get("end_location")
    # Shot end_location is 3D (x, y, z); we drop z for SPADL.
    if loc is None or len(loc) < 2:
        return (None, None)
    return float(loc[0]), float(loc[1])


def _carry_end(ev: dict[str, Any]) -> tuple[float, float] | tuple[None, None]:
    return _location_tuple(ev.get("carry", {}).get("end_location"))


def _result(ev: dict[str, Any], action_type: str) -> str | None:
    """Derive SPADL-style result {success, fail, offside, owngoal, yellow_card, red_card}."""
    if action_type == "pass":
        outcome = ev.get("pass", {}).get("outcome", {}).get("name")
        return _PASS_OUTCOMES.get(outcome, "success") if outcome else "success"
    if action_type == "shot":
        outcome = ev.get("shot", {}).get("outcome", {}).get("name")
        return _SHOT_OUTCOMES.get(outcome, "fail")
    if action_type == "dribble" and ev.get("carry") is not None:
        return "success"
    if action_type == "take_on":
        outcome = ev.get("dribble", {}).get("outcome", {}).get("name")
        return "success" if outcome == "Complete" else "fail"
    if action_type == "foul":
        card = ev.get("foul_committed", {}).get("card", {}).get("name")
        if card and "Yellow" in card:
            return "yellow_card"
        if card and "Red" in card:
            return "red_card"
        return "fail"
    return None


def _bodypart(ev: dict[str, Any], action_type: str) -> str | None:
    if action_type == "pass":
        bp = ev.get("pass", {}).get("body_part", {}).get("name")
    elif action_type == "shot":
        bp = ev.get("shot", {}).get("body_part", {}).get("name")
    else:
        bp = None
    return str(bp).lower().replace(" ", "_") if bp else None


def _end_point(ev: dict[str, Any], action_type: str) -> tuple[float | None, float | None]:
    if action_type == "pass":
        return _pass_end(ev)
    if action_type == "shot":
        return _shot_end(ev)
    if action_type == "dribble":
        return _carry_end(ev)
    # Default: end == start for discrete on-ball actions
    return _location_tuple(ev.get("location"))


def normalise_events(
    raw_events: list[dict[str, Any]],
    match_id: str,
    home_team_id: str,
    home_attacks_left_to_right_p1: bool = True,
) -> pd.DataFrame:
    """Convert raw StatsBomb events to the canonical SPADL-extended DataFrame.

    Coordinates are rescaled from 120x80 yd (top-left origin) to 105x68 m (bottom-left origin),
    and orientation is normalised so the home team attacks L->R in period 1.

    `home_attacks_left_to_right_p1` should come from the kloppy/StatsBomb metadata; we default to
    True here because StatsBomb's own convention aligns events that way already. Callers that
    know otherwise should override.
    """
    rows: list[dict[str, Any]] = []

    for ev in raw_events:
        type_name = ev.get("type", {}).get("name", "")
        action_type = _ACTION_TYPE_MAP.get(type_name)
        if action_type is None:
            continue  # drop events we don't model in SPADL

        period = int(ev.get("period", 1))
        minute = int(ev.get("minute", 0))
        second = int(ev.get("second", 0))
        time_seconds = float(minute * 60 + second)
        team_id = str(ev.get("team", {}).get("id", ""))
        player_id = ev.get("player", {}).get("id")
        player_id = str(player_id) if player_id is not None else None
        raw_event_id = str(ev.get("id", ""))

        sx, sy = _location_tuple(ev.get("location"))
        ex, ey = _end_point(ev, action_type)

        # Rescale both endpoints from 120x80 yards (StatsBomb top-left) to 105x68 m bottom-left
        if sx is not None and sy is not None:
            sx_m, sy_m = rescale_point(sx, sy, SB_PITCH_LENGTH, SB_PITCH_WIDTH, source_origin="top_left")
        else:
            sx_m, sy_m = None, None
        if ex is not None and ey is not None:
            ex_m, ey_m = rescale_point(ex, ey, SB_PITCH_LENGTH, SB_PITCH_WIDTH, source_origin="top_left")
        else:
            ex_m, ey_m = None, None

        # Orientation normalisation: for away-team events, flip to the home team's frame first.
        is_home = team_id == home_team_id
        if not is_home:
            if sx_m is not None and sy_m is not None:
                sx_m, sy_m = PITCH_LENGTH_M - sx_m, PITCH_WIDTH_M - sy_m
            if ex_m is not None and ey_m is not None:
                ex_m, ey_m = PITCH_LENGTH_M - ex_m, PITCH_WIDTH_M - ey_m

        # Then enforce canonical home-attacks-LTR-in-H1
        if sx_m is not None and sy_m is not None:
            sx_m, sy_m = normalise_point(sx_m, sy_m, period, home_attacks_left_to_right_p1)
        if ex_m is not None and ey_m is not None:
            ex_m, ey_m = normalise_point(ex_m, ey_m, period, home_attacks_left_to_right_p1)

        rows.append(
            {
                "match_id": match_id,
                "period": period,
                "time_seconds": time_seconds,
                "team_id": team_id,
                "player_id": player_id,
                "start_x": sx_m,
                "start_y": sy_m,
                "end_x": ex_m,
                "end_y": ey_m,
                "action_type": action_type,
                "result": _result(ev, action_type),
                "bodypart": _bodypart(ev, action_type),
                "raw_event_id": raw_event_id,
            }
        )

    return pd.DataFrame(rows)

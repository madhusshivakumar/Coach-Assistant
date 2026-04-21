"""Tests for StatsBomb event normalisation."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.data.normalize.events_spadl import normalise_events
from football_analysis.data.validation import EventsSchema


def _make_pass_event(
    event_id: str,
    team_id: str,
    player_id: str,
    loc: list[float],
    end_loc: list[float],
    minute: int = 0,
    second: int = 0,
    period: int = 1,
    outcome: str | None = None,
) -> dict:
    ev: dict = {
        "id": event_id,
        "type": {"name": "Pass"},
        "team": {"id": int(team_id), "name": "TeamA"},
        "player": {"id": int(player_id), "name": "Player"},
        "location": loc,
        "period": period,
        "minute": minute,
        "second": second,
        "pass": {"end_location": end_loc, "body_part": {"name": "Right Foot"}},
    }
    if outcome is not None:
        ev["pass"]["outcome"] = {"name": outcome}
    return ev


def _make_shot_event(
    event_id: str, team_id: str, player_id: str, loc: list[float], end_loc_3d: list[float], outcome: str
) -> dict:
    return {
        "id": event_id,
        "type": {"name": "Shot"},
        "team": {"id": int(team_id), "name": "TeamA"},
        "player": {"id": int(player_id), "name": "Player"},
        "location": loc,
        "period": 1,
        "minute": 10,
        "second": 30,
        "shot": {
            "end_location": end_loc_3d,
            "body_part": {"name": "Head"},
            "outcome": {"name": outcome},
        },
    }


def test_empty_events_yields_empty_frame() -> None:
    df = normalise_events([], match_id="statsbomb:1", home_team_id="100")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_single_pass_in_bounds_and_schema_validates() -> None:
    ev = _make_pass_event("e1", "100", "200", [60.0, 40.0], [100.0, 40.0])
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action_type"] == "pass"
    assert row["result"] == "success"
    assert row["bodypart"] == "right_foot"
    # start_location (60, 40) is pitch centre in SB 120x80 top-left => (52.5, 34.0) canonical
    assert abs(row["start_x"] - 52.5) < 1e-6
    assert abs(row["start_y"] - 34.0) < 1e-6
    assert 0.0 <= row["end_x"] <= PITCH_LENGTH_M
    assert 0.0 <= row["end_y"] <= PITCH_WIDTH_M
    # schema is the real contract
    EventsSchema.validate(df, lazy=True)


def test_incomplete_pass_result() -> None:
    ev = _make_pass_event("e1", "100", "200", [60.0, 40.0], [100.0, 40.0], outcome="Incomplete")
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    assert df.iloc[0]["result"] == "fail"


def test_away_team_pass_is_flipped_to_home_frame() -> None:
    """A pass from (60, 40) in SB yards for the AWAY team should end up mirrored so both
    teams share one canonical orientation (home attacks L->R)."""
    ev = _make_pass_event("e1", "999", "200", [60.0, 40.0], [100.0, 40.0])
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    row = df.iloc[0]
    # Centre of the pitch should be unchanged under mirror (it's its own mirror image)
    assert abs(row["start_x"] - 52.5) < 1e-6
    assert abs(row["start_y"] - 34.0) < 1e-6
    # End location (100, 40) -> canonical (87.5, 34). Mirror -> (17.5, 34). home_ltr_p1 means no flip in P1.
    assert abs(row["end_x"] - 17.5) < 1e-6
    assert abs(row["end_y"] - 34.0) < 1e-6


def test_shot_goal_outcome_and_3d_end_stripped() -> None:
    ev = _make_shot_event("s1", "100", "200", [114.0, 40.0], [120.0, 40.0, 1.2], outcome="Goal")
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action_type"] == "shot"
    assert row["result"] == "success"
    assert row["bodypart"] == "head"


def test_unknown_event_type_is_dropped() -> None:
    ev = {
        "id": "x",
        "type": {"name": "Half Start"},
        "team": {"id": 100, "name": "TeamA"},
        "period": 1,
        "minute": 0,
        "second": 0,
    }
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    assert df.empty


def test_period2_event_is_flipped_for_home_team() -> None:
    """A period-2 pass from the home team must be flipped to keep canonical orientation."""
    ev = _make_pass_event("e1", "100", "200", [30.0, 40.0], [40.0, 40.0], period=2, minute=50, second=0)
    df = normalise_events([ev], match_id="statsbomb:1", home_team_id="100")
    row = df.iloc[0]
    # SB (30, 40) in yards -> canonical (26.25, 34). P2 home flip -> (78.75, 34).
    assert abs(row["start_x"] - 78.75) < 1e-6
    assert abs(row["start_y"] - 34.0) < 1e-6

"""Extra branch coverage for normalise_events — interceptions, clearances, take-ons, fouls, missing locations."""

from __future__ import annotations

from football_analysis.data.normalize.events_spadl import normalise_events


def test_interception_and_clearance_covered() -> None:
    events = [
        {
            "id": "i1",
            "type": {"name": "Interception"},
            "team": {"id": 999, "name": "Away"},
            "player": {"id": 302, "name": "P"},
            "location": [40.0, 30.0],
            "period": 1,
            "minute": 1,
            "second": 0,
        },
        {
            "id": "c1",
            "type": {"name": "Clearance"},
            "team": {"id": 999, "name": "Away"},
            "player": {"id": 302, "name": "P"},
            "location": [10.0, 30.0],
            "period": 1,
            "minute": 1,
            "second": 5,
        },
    ]
    df = normalise_events(events, match_id="m", home_team_id="100")
    assert set(df["action_type"]) == {"interception", "clearance"}


def test_take_on_complete_and_failed() -> None:
    events = [
        {
            "id": "t1",
            "type": {"name": "Dribble"},
            "team": {"id": 100, "name": "Home"},
            "player": {"id": 1, "name": "P"},
            "location": [50.0, 40.0],
            "period": 1,
            "minute": 0,
            "second": 0,
            "dribble": {"outcome": {"name": "Complete"}},
        },
        {
            "id": "t2",
            "type": {"name": "Dribble"},
            "team": {"id": 100, "name": "Home"},
            "player": {"id": 1, "name": "P"},
            "location": [50.0, 40.0],
            "period": 1,
            "minute": 0,
            "second": 5,
            "dribble": {"outcome": {"name": "Incomplete"}},
        },
    ]
    df = normalise_events(events, match_id="m", home_team_id="100")
    assert list(df["result"]) == ["success", "fail"]


def test_foul_red_card() -> None:
    events = [
        {
            "id": "f1",
            "type": {"name": "Foul Committed"},
            "team": {"id": 100, "name": "Home"},
            "player": {"id": 1, "name": "P"},
            "location": [50.0, 40.0],
            "period": 1,
            "minute": 0,
            "second": 0,
            "foul_committed": {"card": {"name": "Red Card"}},
        },
        {
            "id": "f2",
            "type": {"name": "Foul Committed"},
            "team": {"id": 100, "name": "Home"},
            "player": {"id": 1, "name": "P"},
            "location": [50.0, 40.0],
            "period": 1,
            "minute": 0,
            "second": 10,
            "foul_committed": {},
        },
    ]
    df = normalise_events(events, match_id="m", home_team_id="100")
    assert list(df["result"]) == ["red_card", "fail"]


def test_pass_with_missing_location_keeps_nulls() -> None:
    events = [
        {
            "id": "p1",
            "type": {"name": "Pass"},
            "team": {"id": 100, "name": "Home"},
            "player": {"id": 1, "name": "P"},
            "period": 1,
            "minute": 0,
            "second": 0,
            "pass": {"body_part": {"name": "Right Foot"}},
        },
    ]
    df = normalise_events(events, match_id="m", home_team_id="100")
    row = df.iloc[0]
    assert row["start_x"] is None or (hasattr(row["start_x"], "is_integer") and False) or row.isna()["start_x"]

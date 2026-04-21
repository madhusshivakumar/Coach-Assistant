"""Tests for the Phase 1 pipeline runner."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.pipeline.runner import MatchAnalytics, run


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_id": "A", "action_type": "pass", "result": "success",
             "start_x": 30.0, "start_y": 34.0, "end_x": 70.0, "end_y": 34.0},
            {"team_id": "A", "action_type": "pass", "result": "success",
             "start_x": 70.0, "start_y": 34.0, "end_x": 95.0, "end_y": 34.0},
            {"team_id": "A", "action_type": "shot", "result": "success",
             "start_x": 100.0, "start_y": 34.0, "end_x": None, "end_y": None},
            {"team_id": "B", "action_type": "pass", "result": "success",
             "start_x": 20.0, "start_y": 30.0, "end_x": 40.0, "end_y": 30.0},
            {"team_id": "B", "action_type": "interception", "result": "success",
             "start_x": 35.0, "start_y": 34.0, "end_x": None, "end_y": None},
        ]
    )


def test_run_returns_analytics_bundle() -> None:
    out = run(_events())
    assert isinstance(out, MatchAnalytics)
    assert out.events is not None
    assert out.xt_grid is not None
    assert set(out.ppda.keys()) == {"A", "B"}
    assert set(out.field_tilt.keys()) == {"A", "B"}


def test_run_enriches_events_with_xt_columns() -> None:
    out = run(_events())
    assert "xt" in out.events.columns
    assert "xt_delta" in out.events.columns
    assert len(out.events) == len(_events())


def test_run_with_provided_xt_grid() -> None:
    from football_analysis.analytics.possession_value.xt import fit

    grid = fit(_events())
    out = run(_events(), xt_grid=grid)
    assert out.xt_grid is grid

"""Tests for the xT Markov model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from football_analysis.analytics.possession_value.xt import XTGrid, _cell_index, apply_xt, fit


def _synth_events() -> pd.DataFrame:
    """A tiny hand-crafted set of events that should yield a non-trivial xT grid."""
    return pd.DataFrame(
        [
            # Forward progression from own half to final third
            {
                "action_type": "pass",
                "result": "success",
                "start_x": 30.0,
                "start_y": 34.0,
                "end_x": 70.0,
                "end_y": 34.0,
                "team_id": "A",
            },
            {
                "action_type": "pass",
                "result": "success",
                "start_x": 70.0,
                "start_y": 34.0,
                "end_x": 95.0,
                "end_y": 34.0,
                "team_id": "A",
            },
            {
                "action_type": "shot",
                "result": "success",
                "start_x": 100.0,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
                "team_id": "A",
            },
            {
                "action_type": "pass",
                "result": "success",
                "start_x": 40.0,
                "start_y": 40.0,
                "end_x": 80.0,
                "end_y": 40.0,
                "team_id": "A",
            },
            {
                "action_type": "shot",
                "result": "fail",
                "start_x": 90.0,
                "start_y": 40.0,
                "end_x": None,
                "end_y": None,
                "team_id": "A",
            },
            {
                "action_type": "pass",
                "result": "success",
                "start_x": 60.0,
                "start_y": 34.0,
                "end_x": 90.0,
                "end_y": 34.0,
                "team_id": "A",
            },
            {
                "action_type": "shot",
                "result": "fail",
                "start_x": 95.0,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
                "team_id": "A",
            },
            # Incomplete moves (shouldn't add to transitions)
            {
                "action_type": "pass",
                "result": "fail",
                "start_x": 50.0,
                "start_y": 20.0,
                "end_x": 60.0,
                "end_y": 20.0,
                "team_id": "A",
            },
        ]
    )


def test_cell_index_boundaries() -> None:
    assert _cell_index(0.0, 0.0, rows=12, cols=16) == (0, 0)
    # Inclusive top-right maps to last cell via clamping
    assert _cell_index(105.0, 68.0, rows=12, cols=16) == (11, 15)


def test_fit_requires_columns() -> None:
    import pytest

    with pytest.raises(ValueError):
        fit(pd.DataFrame([{"x": 1.0}]))


def test_fit_produces_non_negative_grid() -> None:
    grid = fit(_synth_events())
    assert isinstance(grid, XTGrid)
    assert grid.values.shape == (12, 16)
    assert np.all(grid.values >= 0.0)
    # Goal was from x=100 (right edge) — that cell should have non-zero xT
    r, c = _cell_index(100.0, 34.0, 12, 16)
    assert grid.values[r, c] > 0.0


def test_apply_xt_sets_delta_only_for_successful_move() -> None:
    grid = fit(_synth_events())
    events = _synth_events()
    out = apply_xt(events, grid)
    # Shot rows: xt_delta is NaN
    shot_rows = out[out["action_type"] == "shot"]
    assert shot_rows["xt_delta"].isna().all()
    # Successful passes in the set: xt_delta is defined
    successful = out[(out["action_type"] == "pass") & (out["result"] == "success")]
    assert successful["xt_delta"].notna().all()


def test_apply_xt_preserves_row_count_and_original_columns() -> None:
    grid = fit(_synth_events())
    events = _synth_events()
    out = apply_xt(events, grid)
    assert len(out) == len(events)
    for col in events.columns:
        assert col in out.columns
    assert "xt" in out.columns
    assert "xt_delta" in out.columns


@given(x=st.floats(min_value=0.0, max_value=105.0), y=st.floats(min_value=0.0, max_value=68.0))
def test_xt_at_returns_grid_value(x: float, y: float) -> None:
    grid = fit(_synth_events())
    v = grid.xt_at(x, y)
    r, c = _cell_index(x, y, grid.rows, grid.cols)
    assert v == float(grid.values[r, c])


def test_fit_converges_monotonically_per_iter() -> None:
    """Sanity check: running fit with more iterations shouldn't make values negative or explode.

    xT is bounded in [0, 1] — a cell with 100% shoot-rate and 100% goal-rate hits 1.0 exactly,
    so the upper bound is <= 1.0, not < 1.0.
    """
    grid1 = fit(_synth_events(), max_iter=1)
    grid100 = fit(_synth_events(), max_iter=100)
    assert np.all(grid1.values >= 0.0)
    assert np.all(grid100.values >= 0.0)
    assert np.all(grid100.values <= 1.0 + 1e-9)


def test_apply_xt_skips_rows_missing_coords() -> None:
    grid = fit(_synth_events())
    events = pd.DataFrame(
        [
            {
                "action_type": "pass",
                "result": "success",
                "start_x": None,
                "start_y": None,
                "end_x": 60.0,
                "end_y": 34.0,
                "team_id": "A",
            },
            {
                "action_type": "pass",
                "result": "success",
                "start_x": 30.0,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
                "team_id": "A",
            },
        ]
    )
    out = apply_xt(events, grid)
    assert out.iloc[0]["xt"] != out.iloc[0]["xt"]  # NaN
    assert out.iloc[1]["xt_delta"] != out.iloc[1]["xt_delta"]  # NaN

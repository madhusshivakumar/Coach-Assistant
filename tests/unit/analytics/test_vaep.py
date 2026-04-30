"""Tests for the VAEP per-action value model.

Mirrors the style of ``test_xt.py`` — small synthetic SPADL frames, schema
contracts, and behaviour properties. We don't need to validate the GBM's
absolute calibration (that's an offline modelling concern); we just need to
verify the public API contract, the value identity, and the NaN / empty edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.possession_value.vaep import (
    VAEPModel,
    fit_vaep,
    score_actions,
)


def _synth_spadl(n_repeats: int = 4) -> pd.DataFrame:
    """A tiny hand-crafted SPADL frame with a couple of scoring sequences.

    Each sequence ends in a successful shot (=> goal) so the trainer sees both
    positive and negative labels for "scoring within next 10 actions". We
    repeat the block ``n_repeats`` times so the GBM has enough rows to fit
    without complaining about minority-class size.
    """
    block = [
        # Possession 1 by team A — ends in a goal at x~100
        {
            "match_id": "M1",
            "period": 1,
            "time_seconds": 1.0,
            "team_id": "A",
            "player_id": "p1",
            "start_x": 30.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
            "action_type": "pass",
            "result": "success",
            "bodypart": "right_foot",
        },
        {
            "match_id": "M1",
            "period": 1,
            "time_seconds": 3.0,
            "team_id": "A",
            "player_id": "p2",
            "start_x": 60.0,
            "start_y": 34.0,
            "end_x": 90.0,
            "end_y": 34.0,
            "action_type": "pass",
            "result": "success",
            "bodypart": "right_foot",
        },
        {
            "match_id": "M1",
            "period": 1,
            "time_seconds": 5.0,
            "team_id": "A",
            "player_id": "p3",
            "start_x": 100.0,
            "start_y": 34.0,
            "end_x": 105.0,
            "end_y": 34.0,
            "action_type": "shot",
            "result": "success",
            "bodypart": "right_foot",
        },
        # Possession 2 by team B — turnover, no goal
        {
            "match_id": "M1",
            "period": 1,
            "time_seconds": 8.0,
            "team_id": "B",
            "player_id": "p4",
            "start_x": 50.0,
            "start_y": 20.0,
            "end_x": 60.0,
            "end_y": 20.0,
            "action_type": "pass",
            "result": "fail",
            "bodypart": "right_foot",
        },
        {
            "match_id": "M1",
            "period": 1,
            "time_seconds": 11.0,
            "team_id": "A",
            "player_id": "p5",
            "start_x": 55.0,
            "start_y": 30.0,
            "end_x": 70.0,
            "end_y": 30.0,
            "action_type": "dribble",
            "result": "success",
            "bodypart": "left_foot",
        },
        # Possession 3 by team A — long-range shot misses
        {
            "match_id": "M1",
            "period": 2,
            "time_seconds": 200.0,
            "team_id": "A",
            "player_id": "p3",
            "start_x": 80.0,
            "start_y": 40.0,
            "end_x": 105.0,
            "end_y": 34.0,
            "action_type": "shot",
            "result": "fail",
            "bodypart": "head",
        },
    ]
    rows: list[dict[str, object]] = []
    for r in range(n_repeats):
        for ev in block:
            ev_copy = dict(ev)
            # Bump time / match so episodes don't glue together across repeats
            ev_copy["match_id"] = f"M{r}"
            rows.append(ev_copy)
    return pd.DataFrame(rows)


def test_fit_vaep_returns_model() -> None:
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    assert isinstance(model, VAEPModel)
    assert model.p_score_clf is not None
    assert model.p_concede_clf is not None
    assert isinstance(model.feature_names, tuple)
    assert len(model.feature_names) > 0


def test_score_actions_schema() -> None:
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    out = score_actions(model, df)
    expected = {
        "action_id",
        "p_score_before",
        "p_score_after",
        "p_concede_before",
        "p_concede_after",
        "vaep_value",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == len(df)


def test_vaep_value_identity() -> None:
    """VAEP value = (P_score_after - P_score_before) - (P_concede_after - P_concede_before)."""
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    out = score_actions(model, df)
    expected = (out["p_score_after"] - out["p_score_before"]) - (
        out["p_concede_after"] - out["p_concede_before"]
    )
    np.testing.assert_allclose(out["vaep_value"].to_numpy(), expected.to_numpy(), atol=1e-9)


def test_high_xg_action_outranks_midfield_pass() -> None:
    """A close-range successful shot should value higher than a deep midfield pass.

    We don't require the GBM to be perfectly calibrated; we only require that
    p_score_before is monotone with proximity to goal across two well-separated
    samples, which any non-degenerate fit on the synthetic data will produce.
    """
    df = _synth_spadl(n_repeats=8)
    model = fit_vaep(df, random_state=0)
    out = score_actions(model, df)
    out["dist_start"] = np.hypot(105.0 - df["start_x"].astype(float), 34.0 - df["start_y"].astype(float))
    near_goal = out[out["dist_start"] <= 10.0]
    deep_midfield = out[out["dist_start"] >= 50.0]
    assert not near_goal.empty
    assert not deep_midfield.empty
    assert near_goal["p_score_before"].mean() >= deep_midfield["p_score_before"].mean()


def test_score_actions_empty_input() -> None:
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    empty = df.iloc[0:0].copy()
    out = score_actions(model, empty)
    assert len(out) == 0
    for col in ("action_id", "p_score_before", "p_score_after", "p_concede_before", "p_concede_after", "vaep_value"):
        assert col in out.columns


def test_fit_vaep_empty_input_raises() -> None:
    """An empty corpus has no labels — fit should fail clearly, not silently."""
    with pytest.raises(ValueError):
        fit_vaep(pd.DataFrame(columns=["action_type", "start_x", "start_y", "end_x", "end_y", "result", "team_id", "period", "time_seconds", "bodypart", "match_id"]))


def test_nan_coords_do_not_crash() -> None:
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    # Inject NaN coords into a copy
    df_nan = df.copy()
    df_nan.loc[0, "start_x"] = np.nan
    df_nan.loc[1, "end_y"] = np.nan
    out = score_actions(model, df_nan)
    # All numeric columns finite (NaN imputed to 0 internally per the spec)
    for col in ("p_score_before", "p_score_after", "p_concede_before", "p_concede_after", "vaep_value"):
        assert np.isfinite(out[col].to_numpy()).all()


def test_determinism_via_random_state() -> None:
    df = _synth_spadl()
    m1 = fit_vaep(df, random_state=42)
    m2 = fit_vaep(df, random_state=42)
    o1 = score_actions(m1, df)
    o2 = score_actions(m2, df)
    np.testing.assert_allclose(o1["vaep_value"].to_numpy(), o2["vaep_value"].to_numpy())


def test_fit_vaep_requires_required_columns() -> None:
    bad = pd.DataFrame([{"foo": 1}])
    with pytest.raises(ValueError):
        fit_vaep(bad)


def test_score_actions_handles_unknown_action_type() -> None:
    df = _synth_spadl()
    model = fit_vaep(df, random_state=0)
    df_extra = df.copy()
    df_extra.loc[0, "action_type"] = "unknown_made_up_type"
    out = score_actions(model, df_extra)
    assert len(out) == len(df_extra)
    # Still finite — unknown action type one-hot collapses to zeros, not NaN
    assert np.isfinite(out["vaep_value"].to_numpy()).all()

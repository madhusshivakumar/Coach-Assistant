"""VAEP — Valuing Actions by Estimating Probabilities (Decroos et al., KDD 2019).

VAEP frames per-action value as the change in two short-horizon probabilities,
estimated from on-ball action features:

    P_score(s)   = P(possessing team scores within next ``horizon`` actions | game state s)
    P_concede(s) = P(possessing team concedes within next ``horizon`` actions | game state s)

The value of action ``a`` taking the game from state ``s_before`` to ``s_after`` is:

    VAEP(a) = (P_score(s_after) - P_score(s_before)) - (P_concede(s_after) - P_concede(s_before))

We implement the model from scratch on top of sklearn's GradientBoostingClassifier
to avoid pulling in the ``socceraction`` package — same maths, our coding
conventions, and our SPADL schema (see ``data/normalize/events_spadl.py``).

This module sits next to ``xt.py`` and is consumed by the M3 Markov-TD attribution
plus the M5 Coach Brief renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M

# Action types we explicitly one-hot. Anything else collapses to all-zero —
# safer than blowing up the feature width on a previously unseen label.
_ACTION_TYPES: tuple[str, ...] = (
    "pass",
    "dribble",
    "shot",
    "cross",
    "take_on",
    "interception",
    "clearance",
    "tackle",
    "block",
    "keeper_action",
    "foul",
    "receival",
)

# Body parts we one-hot. Same fall-through behaviour as action types.
_BODY_PARTS: tuple[str, ...] = ("right_foot", "left_foot", "head", "other")

# Goal centre in the canonical attacking-right frame.
_GOAL_X: float = PITCH_LENGTH_M
_GOAL_Y: float = PITCH_WIDTH_M / 2.0

# Required SPADL columns. We don't enforce ``bodypart`` because some action
# types legitimately leave it null (interceptions, clearances).
_REQUIRED_COLS: tuple[str, ...] = (
    "action_type",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result",
    "team_id",
    "period",
    "time_seconds",
    "match_id",
)

# Static feature schema. Order is contractual — the model stores it on
# ``feature_names`` so ``score_actions`` can re-build the same matrix.
_FEATURE_NAMES: tuple[str, ...] = (
    "start_x_oriented",
    "start_y",
    "end_x_oriented",
    "end_y",
    "distance_to_goal_start",
    "distance_to_goal_end",
    "angle_to_goal_start",
    "period",
    "time_since_period_start",
    *(f"action_{a}" for a in _ACTION_TYPES),
    *(f"body_{b}" for b in _BODY_PARTS),
)


@dataclass(frozen=True)
class VAEPModel:
    """Two fitted GBM classifiers + the contract feature order they were trained on."""

    p_score_clf: object  # sklearn-compatible classifier with predict_proba
    p_concede_clf: object
    feature_names: tuple[str, ...]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _safe_float(x: Any) -> float:
    """Coerce arbitrary scalar to a finite float; NaN/None → 0.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def _build_feature_matrix(actions: pd.DataFrame) -> np.ndarray:
    """Assemble the (n, len(_FEATURE_NAMES)) feature matrix from a SPADL frame.

    NaN coordinates are replaced with 0.0 so the GBM never sees a missing value
    (matches the convention used by ``embedding.py``). Unknown action types or
    body parts collapse to all-zero one-hots — the model treats them as a
    generic "other" class rather than crashing.
    """
    n = len(actions)
    feats = np.zeros((n, len(_FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return feats

    sx = actions["start_x"].map(_safe_float).to_numpy()
    sy = actions["start_y"].map(_safe_float).to_numpy()
    ex = actions["end_x"].map(_safe_float).to_numpy()
    ey = actions["end_y"].map(_safe_float).to_numpy()

    # Coordinates are already in the canonical attacking-right frame after
    # ``normalise_events``, so x_oriented == x.
    feats[:, 0] = sx
    feats[:, 1] = sy
    feats[:, 2] = ex
    feats[:, 3] = ey
    feats[:, 4] = np.hypot(_GOAL_X - sx, _GOAL_Y - sy)
    feats[:, 5] = np.hypot(_GOAL_X - ex, _GOAL_Y - ey)
    # Angle subtended at the start point by the 7.32m goal line, clamped to [0, pi].
    dx = _GOAL_X - sx
    dy_abs = np.abs(_GOAL_Y - sy)
    # Atan2 form keeps it numerically stable even at the goal-line where dx -> 0.
    feats[:, 6] = np.arctan2(7.32 * np.maximum(dx, 1e-6), dx**2 + dy_abs**2 - (7.32 / 2.0) ** 2)

    period = actions["period"].map(_safe_float).to_numpy()
    t = actions["time_seconds"].map(_safe_float).to_numpy()
    feats[:, 7] = period
    # SPADL stores ``time_seconds`` as time-since-period-start already, but be
    # defensive — clip negatives to 0.
    feats[:, 8] = np.maximum(t, 0.0)

    # Action-type one-hots
    base = 9
    atype = actions["action_type"].astype(object).fillna("").to_numpy()
    for i, name in enumerate(_ACTION_TYPES):
        feats[:, base + i] = (atype == name).astype(np.float64)

    # Body-part one-hots — bodypart is optional in SPADL (interceptions/clearances).
    base += len(_ACTION_TYPES)
    if "bodypart" in actions.columns:
        bp = actions["bodypart"].astype(object).fillna("").to_numpy()
        for i, name in enumerate(_BODY_PARTS):
            feats[:, base + i] = (bp == name).astype(np.float64)

    return feats


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


def _build_labels(actions: pd.DataFrame, horizon: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Per Decroos et al., label each action with two flags:

    - ``y_score``   = 1 iff the same team scores within the next ``horizon`` actions
    - ``y_concede`` = 1 iff the same team concedes within the next ``horizon`` actions

    A scoring event is a ``shot`` with ``result == 'success'``. We scan within
    each match independently so the horizon never bleeds across games.
    """
    n = len(actions)
    y_score = np.zeros(n, dtype=np.int64)
    y_concede = np.zeros(n, dtype=np.int64)
    if n == 0:
        return y_score, y_concede

    actions_idx = actions.reset_index(drop=True)
    is_goal = (
        (actions_idx["action_type"].astype(object) == "shot")
        & (actions_idx["result"].astype(object) == "success")
    ).to_numpy()
    teams = actions_idx["team_id"].astype(object).to_numpy()
    matches = actions_idx["match_id"].astype(object).to_numpy()

    for i in range(n):
        end = min(n, i + horizon + 1)
        # i+1 .. end-1 are the next ``horizon`` actions (excluding the action itself)
        for j in range(i, end):
            if matches[j] != matches[i]:
                break
            if not is_goal[j]:
                continue
            if teams[j] == teams[i]:
                y_score[i] = 1
            else:
                y_concede[i] = 1
            # First scoring event in window decides the label per side; we keep
            # scanning in case both sides eventually score.
        # Done with action i

    return y_score, y_concede


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_vaep(
    spadl_actions: pd.DataFrame,
    *,
    horizon: int = 10,
    random_state: int | None = 0,
    n_estimators: int = 50,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> VAEPModel:
    """Fit the two GBM classifiers (P_score, P_concede) on a SPADL action corpus.

    Args:
        spadl_actions: SPADL-extended events as produced by
            ``data/normalize/events_spadl.py``. Must carry the columns listed in
            ``_REQUIRED_COLS``.
        horizon: how many future actions to look at when generating labels.
            10 matches the Decroos et al. paper.
        random_state: passed through to both classifiers for determinism.
        n_estimators / max_depth / learning_rate: GBM hyper-parameters. Defaults
            are deliberately small so unit tests fit in milliseconds; a real
            corpus would tune these via cross-validation.

    Returns:
        ``VAEPModel`` carrying the fitted classifiers and the feature schema.

    Raises:
        ValueError: if required SPADL columns are missing or the corpus is empty.
    """
    missing = [c for c in _REQUIRED_COLS if c not in spadl_actions.columns]
    if missing:
        raise ValueError(f"spadl_actions missing required columns: {missing}")
    if len(spadl_actions) == 0:
        raise ValueError("spadl_actions is empty — cannot fit VAEP")

    X = _build_feature_matrix(spadl_actions)
    y_score, y_concede = _build_labels(spadl_actions, horizon=horizon)

    p_score_clf = _fit_one(X, y_score, random_state, n_estimators, max_depth, learning_rate)
    p_concede_clf = _fit_one(X, y_concede, random_state, n_estimators, max_depth, learning_rate)

    return VAEPModel(
        p_score_clf=p_score_clf,
        p_concede_clf=p_concede_clf,
        feature_names=_FEATURE_NAMES,
    )


class _ConstantClassifier:
    """Fallback used when a label has only one class — sklearn refuses to fit
    a GBM on a single-class target, but we still need ``predict_proba`` to
    return something sensible (the constant probability).
    """

    def __init__(self, constant: float) -> None:
        self._p = float(constant)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, 2), dtype=np.float64)
        out[:, 0] = 1.0 - self._p
        out[:, 1] = self._p
        return out


def _fit_one(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int | None,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> object:
    """Fit one GBM, gracefully handling the single-class edge case."""
    unique = np.unique(y)
    if len(unique) < 2:
        # Constant target — sklearn refuses to fit, so return a stub that
        # always predicts the seen rate (0.0 if all-negative, 1.0 if all-positive).
        return _ConstantClassifier(constant=float(unique[0]))
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def _proba_pos(clf: object, X: np.ndarray) -> np.ndarray:
    """Return the positive-class probability column from any sklearn-style clf.

    Both ``GradientBoostingClassifier`` and ``_ConstantClassifier`` return a
    ``(n, 2)`` array with column 1 = positive class probability.
    """
    proba = clf.predict_proba(X)  # type: ignore[attr-defined]
    arr = np.asarray(proba, dtype=np.float64)
    return arr[:, 1]


def _swap_endpoints(actions: pd.DataFrame) -> pd.DataFrame:
    """Build a synthetic "after" frame whose start coordinate equals the
    original action's end coordinate. The action_type / bodypart / period / etc.
    are preserved — VAEP only re-evaluates the *position* of possession after
    the action lands, not what action will happen next (that's the job of
    P_score/P_concede over the next ``horizon`` actions, which we already
    trained the classifiers on).
    """
    out = actions.copy()
    out["start_x"] = out["end_x"]
    out["start_y"] = out["end_y"]
    return out


def score_actions(model: VAEPModel, spadl_actions: pd.DataFrame) -> pd.DataFrame:
    """Score each action with VAEP value.

    Returns a DataFrame with columns:
        - ``action_id``: row index from ``spadl_actions`` (preserved as-is)
        - ``p_score_before`` / ``p_score_after``
        - ``p_concede_before`` / ``p_concede_after``
        - ``vaep_value``  =  ΔP_score − ΔP_concede

    Empty input → empty output with the right schema. NaN coordinates are
    silently imputed to 0 so the score is always finite.
    """
    cols = (
        "action_id",
        "p_score_before",
        "p_score_after",
        "p_concede_before",
        "p_concede_after",
        "vaep_value",
    )
    if len(spadl_actions) == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols}).assign(
            action_id=pd.Series(dtype=object)
        )[list(cols)]

    X_before = _build_feature_matrix(spadl_actions)
    X_after = _build_feature_matrix(_swap_endpoints(spadl_actions))

    p_score_before = _proba_pos(model.p_score_clf, X_before)
    p_score_after = _proba_pos(model.p_score_clf, X_after)
    p_concede_before = _proba_pos(model.p_concede_clf, X_before)
    p_concede_after = _proba_pos(model.p_concede_clf, X_after)

    vaep = (p_score_after - p_score_before) - (p_concede_after - p_concede_before)

    return pd.DataFrame(
        {
            "action_id": spadl_actions.index.to_numpy(),
            "p_score_before": p_score_before,
            "p_score_after": p_score_after,
            "p_concede_before": p_concede_before,
            "p_concede_after": p_concede_after,
            "vaep_value": vaep,
        }
    )

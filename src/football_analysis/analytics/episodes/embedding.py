"""Hand-crafted fixed-dim embedding of an ``EpisodeRecord`` for retrieval.

The embedding captures *state-only* features (no outcome leakage) so it can be
computed for both:

- **Full episodes** when building the retrieval library.
- **Partial episodes** at query time, given only the first ``max_rel_time_s``
  seconds of an in-progress episode. The schema must match the full embedding
  for distances to be meaningful.

Schema (~25 dims): geometry of ball trajectory, team-shape aggregates, dominant
phase one-hot. Outcome flags are intentionally excluded — they're the answer
retrieval is *predicting*, not part of the query.

A learned encoder (autoencoder, contrastive) can drop in later via the same
function signature when corpus size justifies it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord

# Canonical phase set (matches Phase 3A classifier output).
PHASES: tuple[str, ...] = (
    "build_up",
    "progression",
    "finishing",
    "att_transition",
    "def_transition",
    "settled_def",
    "set_piece",
)

# The fixed feature schema. Order is contractual — neighbors need identical layout.
EPISODE_FEATURE_NAMES: tuple[str, ...] = (
    "duration_s",
    "start_ball_x_oriented",
    "start_ball_y",
    "end_ball_x_oriented",
    "end_ball_y",
    "max_ball_x_oriented",
    "ball_x_displacement",
    "mean_ball_speed",
    "max_ball_speed",
    "ball_x_slope",
    "mean_attackers_mean_x_oriented",
    "mean_defenders_line_height_oriented",
    "mean_attackers_length",
    "mean_attackers_width",
    "mean_attackers_visible",
    "mean_defenders_visible",
    "compactness_diff",
    *(f"phase_{p}" for p in PHASES),
)


def _safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if not s.empty else 0.0


def _safe_first(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[0]) if not s.empty else 0.0


def _safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else 0.0


def _safe_max(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.max()) if not s.empty else 0.0


def _ball_x_slope(states: pd.DataFrame) -> float:
    """Linear-fit slope of ``ball_x_oriented`` vs ``rel_time_s`` (m/s).

    Returns 0.0 when fewer than 2 valid (rel_time, ball_x) points exist or all
    rel_time values collapse to one timestamp.
    """
    sub = states[["rel_time_s", "ball_x_oriented"]].dropna()
    if len(sub) < 2:
        return 0.0
    t = sub["rel_time_s"].to_numpy()
    x = sub["ball_x_oriented"].to_numpy()
    if t.max() - t.min() < 1e-6:
        return 0.0
    return float(np.polyfit(t, x, deg=1)[0])


def embed_episode(
    record: EpisodeRecord,
    max_rel_time_s: float | None = None,
) -> np.ndarray:
    """Build the fixed-dim feature vector for one episode.

    Args:
        record: ``EpisodeRecord`` from the engine.
        max_rel_time_s: if set, only use snapshots with ``rel_time_s <=`` this
            cutoff. Used at query time to embed a partial in-progress episode
            and find similar past full episodes.

    Returns:
        1-D numpy array of length ``len(EPISODE_FEATURE_NAMES)``. NaNs are
        replaced with 0.0 so the output is always finite — downstream
        ``StandardScaler`` requires no NaNs.
    """
    states = record.state_trajectory
    if max_rel_time_s is not None and not states.empty:
        states = states[states["rel_time_s"] <= max_rel_time_s]

    if states.empty:
        return np.zeros(len(EPISODE_FEATURE_NAMES), dtype=float)

    rel_t = states["rel_time_s"].dropna()
    duration_s = float(rel_t.max() - rel_t.min()) if not rel_t.empty else 0.0

    feats: dict[str, float] = {
        "duration_s": duration_s,
        "start_ball_x_oriented": _safe_first(states["ball_x_oriented"]),
        "start_ball_y": _safe_first(states["ball_y"]),
        "end_ball_x_oriented": _safe_last(states["ball_x_oriented"]),
        "end_ball_y": _safe_last(states["ball_y"]),
        "max_ball_x_oriented": _safe_max(states["ball_x_oriented"]),
        "ball_x_displacement": (_safe_max(states["ball_x_oriented"]) - _safe_first(states["ball_x_oriented"])),
        "mean_ball_speed": _safe_mean(states["ball_speed"]),
        "max_ball_speed": _safe_max(states["ball_speed"]),
        "ball_x_slope": _ball_x_slope(states),
        "mean_attackers_mean_x_oriented": _safe_mean(states["attackers_mean_x_oriented"]),
        "mean_defenders_line_height_oriented": _safe_mean(states["defenders_line_height_oriented"]),
        "mean_attackers_length": _safe_mean(states["attackers_length"]),
        "mean_attackers_width": _safe_mean(states["attackers_width"]),
        "mean_attackers_visible": _safe_mean(states["attackers_visible"].astype(float)),
        "mean_defenders_visible": _safe_mean(states["defenders_visible"].astype(float)),
        "compactness_diff": (_safe_mean(states["defenders_compactness_x"]) - _safe_mean(states["attackers_length"])),
    }
    for p in PHASES:
        feats[f"phase_{p}"] = 1.0 if record.dominant_phase == p else 0.0

    arr = np.array([feats.get(name, 0.0) for name in EPISODE_FEATURE_NAMES], dtype=float)
    # Replace any residual NaN/inf with 0 so the scaler stays well-behaved.
    cleaned: np.ndarray = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned

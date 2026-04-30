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
from football_analysis.analytics.formations.roles import (
    FORMATION_4_3_3,
    FormationTemplate,
    _episode_snapshot_frames,
    episode_role_trajectory,
)
from football_analysis.analytics.pitch import PITCH_LENGTH_M

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


# ---------------------------------------------------------------------------
# v2 redesign — M1 task #2: extended embedding for HDBSCAN clustering
# ---------------------------------------------------------------------------
#
# The base ``embed_episode`` above is fine for retrieval (the existing Phase 4
# index relies on its 25-dim contract). The v2 design needs a richer feature
# vector that captures: (a) pressing intensity from the defending team, (b) how
# closely the possessing team's shape matches a known formation template, and
# (c) per-role displacement statistics. We append those dims so the first 25
# entries are byte-identical to ``embed_episode``'s output — old indexes and
# the new clustering vector share a prefix.

EXTENDED_FEATURE_NAMES: tuple[str, ...] = EPISODE_FEATURE_NAMES + (
    "mean_pressing_distance",
    "min_pressing_distance",
    "mean_defenders_within_5m",
    "mean_defenders_within_10m",
    "formation_match_cost",
    "mean_attacker_role_displacement",
    "mean_attacker_team_compactness",
    "mean_attackers_in_attacking_third",
)


def _extract_extras(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    attacking_to_right: bool,
    template: FormationTemplate,
    snapshot_hz: float,
    max_rel_time_s: float | None,
) -> np.ndarray:
    """Compute the 8 new feature dims from raw tracking. Returns zeros when
    tracking is unusable (empty, no snapshot frames, or all-NaN ball)."""
    n_extras = len(EXTENDED_FEATURE_NAMES) - len(EPISODE_FEATURE_NAMES)
    zeros = np.zeros(n_extras, dtype=float)
    if tracking.empty:
        return zeros

    boundary = record.boundary
    possessing = boundary.possession_team
    defending = away_team_id if possessing == home_team_id else home_team_id
    cutoff_time = (boundary.start_time_s + max_rel_time_s) if max_rel_time_s is not None else None

    snapshot_frames = _episode_snapshot_frames(tracking, boundary, snapshot_hz)
    if not snapshot_frames:
        return zeros

    pressing_dists: list[float] = []
    defenders_within_5m: list[float] = []
    defenders_within_10m: list[float] = []
    attackers_compactness: list[float] = []
    attackers_in_third: list[float] = []

    for frame_id in snapshot_frames:
        sub = tracking[tracking["frame_id"] == frame_id]
        if sub.empty:
            continue
        time_s = float(sub["time_seconds"].iloc[0])
        if cutoff_time is not None and time_s > cutoff_time:
            continue

        ball = sub[sub["is_ball"]]
        ball = ball[ball["x"].notna() & ball["y"].notna()]
        if ball.empty:
            # No usable ball — pressing-distance features can't be computed for
            # this snapshot, skip rather than poison the aggregates.
            continue
        bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])

        defenders = sub[(sub["team_id"] == defending) & ~sub["is_ball"]]
        defenders = defenders[defenders["x"].notna() & defenders["y"].notna()]
        if "visible" in defenders.columns:
            defenders = defenders[defenders["visible"].astype(bool)]
        if not defenders.empty:
            d_dx = defenders["x"].to_numpy() - bx
            d_dy = defenders["y"].to_numpy() - by
            d_dists = np.hypot(d_dx, d_dy)
            pressing_dists.append(float(d_dists.min()))
            defenders_within_5m.append(float((d_dists <= 5.0).sum()))
            defenders_within_10m.append(float((d_dists <= 10.0).sum()))

        attackers = sub[(sub["team_id"] == possessing) & ~sub["is_ball"]]
        attackers = attackers[attackers["x"].notna() & attackers["y"].notna()]
        if "visible" in attackers.columns:
            attackers = attackers[attackers["visible"].astype(bool)]
        if len(attackers) >= 2:
            ax = attackers["x"].to_numpy()
            ay = attackers["y"].to_numpy()
            ax_oriented = ax if attacking_to_right else PITCH_LENGTH_M - ax
            attackers_in_third.append(float((ax_oriented >= 70.0).sum()))
            attackers_compactness.append(float(np.sqrt(np.var(ax) + np.var(ay))))

    # Role-anchored features (Bialkowski displacement vs the chosen template).
    role_traj = episode_role_trajectory(
        tracking,
        boundary,
        team_id=possessing,
        template=template,
        snapshot_hz=snapshot_hz,
        attacking_right=attacking_to_right,
    )
    if max_rel_time_s is not None and not role_traj.empty:
        role_traj = role_traj[(role_traj["time_seconds"] - boundary.start_time_s) <= max_rel_time_s]
    if role_traj.empty:
        formation_match_cost = 0.0
        mean_role_disp = 0.0
    else:
        # Per-snapshot worst-out-of-position displacement, averaged across
        # snapshots → "is anyone consistently way off-shape?"
        per_snap_max = role_traj.groupby("snapshot_idx")["displacement"].max()
        formation_match_cost = float(per_snap_max.mean())
        mean_role_disp = float(role_traj["displacement"].mean())

    extras = np.array(
        [
            float(np.mean(pressing_dists)) if pressing_dists else 0.0,
            float(np.min(pressing_dists)) if pressing_dists else 0.0,
            float(np.mean(defenders_within_5m)) if defenders_within_5m else 0.0,
            float(np.mean(defenders_within_10m)) if defenders_within_10m else 0.0,
            formation_match_cost,
            mean_role_disp,
            float(np.mean(attackers_compactness)) if attackers_compactness else 0.0,
            float(np.mean(attackers_in_third)) if attackers_in_third else 0.0,
        ],
        dtype=float,
    )
    return np.nan_to_num(extras, nan=0.0, posinf=0.0, neginf=0.0)


def embed_episode_extended(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    attacking_to_right: bool = True,
    max_rel_time_s: float | None = None,
    template: FormationTemplate = FORMATION_4_3_3,
    snapshot_hz: float = 2.0,
) -> np.ndarray:
    """Build the extended feature vector for one episode.

    The output is the base ``embed_episode(record)`` result concatenated with 8
    new dims derived from raw tracking (pressing distance, role-relative
    geometry, formation-template match). The first ``len(EPISODE_FEATURE_NAMES)``
    elements are byte-identical to ``embed_episode`` so old retrieval indexes
    can re-use the prefix.

    Args:
        record: ``EpisodeRecord`` for the episode being embedded.
        tracking: canonical match tracking — needed for pressing distances and
            role-anchored features. Pass an empty DataFrame to return the base
            embedding zero-padded to the extended length (callers that just
            want the prefix should use ``embed_episode`` directly).
        home_team_id, away_team_id: required to know which team is defending.
        attacking_to_right: True when the possessing team attacks +x in
            canonical coords.
        max_rel_time_s: cut-off for partial-episode embeddings (query-time use).
        template: formation template the role assignment matches against.
            Default 4-3-3; pass another template if the team's actual shape is
            known up front.
        snapshot_hz: snapshot rate. Defaults match
            ``episode_state_trajectory`` so snapshots align.

    Returns:
        1-D numpy array of length ``len(EXTENDED_FEATURE_NAMES)``. NaNs
        replaced with 0.0.
    """
    base = embed_episode(record, max_rel_time_s=max_rel_time_s)
    extras = _extract_extras(
        record,
        tracking,
        home_team_id,
        away_team_id,
        attacking_to_right,
        template,
        snapshot_hz,
        max_rel_time_s,
    )
    return np.concatenate([base, extras])

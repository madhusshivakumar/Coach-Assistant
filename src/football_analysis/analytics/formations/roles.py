"""Bialkowski-style per-frame role assignment.

Given the current positions of 10 outfielders and a reference formation template
(role slot coordinates in a normalised frame), solve the Hungarian assignment that
minimises total squared displacement between players and role slots. This produces
a (frame → {player_id: role_label}) mapping that is stable across frames even when
players swap positions within the shape.

Templates are expressed with the attacking direction pointing towards +x, so the
ordering (attacking team) maps x values onto the pitch's attacking half. Role
coordinates are given in metres on the canonical 105×68 pitch, relative to the
team's attacking goal (so "LCB" sits at small x, "ST" at large x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass(frozen=True)
class FormationTemplate:
    """Named reference positions for each role slot. Coordinates in metres on 105×68."""

    name: str
    roles: tuple[str, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        """Return role coordinates as an (n, 2) numpy array."""
        return np.stack([np.asarray(self.xs, dtype=np.float64), np.asarray(self.ys, dtype=np.float64)], axis=1)


# Canonical 4-3-3 template, attacking → +x. Numbers are chosen to sit in plausible
# average positions across a full match (not peak-attack positions).
FORMATION_4_3_3 = FormationTemplate(
    name="4-3-3",
    roles=("LB", "LCB", "RCB", "RB", "LCM", "DM", "RCM", "LW", "ST", "RW"),
    xs=(40, 40, 40, 40, 60, 55, 60, 80, 85, 80),
    ys=(14, 28, 40, 54, 22, 34, 46, 12, 34, 56),
)

# 4-4-2 with flat midfield four.
FORMATION_4_4_2 = FormationTemplate(
    name="4-4-2",
    roles=("LB", "LCB", "RCB", "RB", "LM", "LCM", "RCM", "RM", "LF", "RF"),
    xs=(40, 40, 40, 40, 65, 60, 60, 65, 85, 85),
    ys=(12, 28, 40, 56, 10, 28, 40, 58, 26, 42),
)

# 4-2-3-1 (double pivot, a number 10, lone 9).
FORMATION_4_2_3_1 = FormationTemplate(
    name="4-2-3-1",
    roles=("LB", "LCB", "RCB", "RB", "LDM", "RDM", "LAM", "CAM", "RAM", "ST"),
    xs=(40, 40, 40, 40, 58, 58, 78, 76, 78, 88),
    ys=(12, 28, 40, 56, 28, 40, 14, 34, 54, 34),
)

DEFAULT_TEMPLATES: tuple[FormationTemplate, ...] = (
    FORMATION_4_3_3,
    FORMATION_4_4_2,
    FORMATION_4_2_3_1,
)


def _orient_to_attacking_right(positions: np.ndarray, attacking_right: bool) -> np.ndarray:
    """Mirror positions so the team is always attacking towards +x."""
    if attacking_right:
        return positions
    mirrored = positions.copy()
    mirrored[:, 0] = PITCH_LENGTH_M - mirrored[:, 0]
    mirrored[:, 1] = PITCH_WIDTH_M - mirrored[:, 1]
    return mirrored


def assign_roles(
    positions: pd.DataFrame,
    template: FormationTemplate = FORMATION_4_3_3,
    attacking_right: bool = True,
) -> pd.DataFrame:
    """Assign role labels to 10 outfielders in a single frame.

    Args:
        positions: DataFrame with columns `player_id`, `x`, `y`. Exactly 10 rows.
            Rows that are clearly the GK (lowest mean x over a match) should
            already be filtered out by the caller — GKs are not role-assigned.
        template: formation template to map to.
        attacking_right: True if the team attacks towards +x in canonical coords.
            Away-team frames should pass `False` so they're flipped before matching.

    Returns:
        DataFrame with columns `player_id`, `role`, `x`, `y`, `slot_x`, `slot_y`,
        `displacement` (Euclidean distance from slot).
    """
    if len(positions) != 10:
        raise ValueError(f"assign_roles expects exactly 10 outfielders, got {len(positions)}")
    if len(template.roles) != 10:
        raise ValueError(f"template must have 10 slots, got {len(template.roles)}")

    coords = positions[["x", "y"]].to_numpy(dtype=np.float64)
    oriented = _orient_to_attacking_right(coords, attacking_right)
    slots = template.as_array()

    # Cost matrix: pairwise squared Euclidean
    diff = oriented[:, None, :] - slots[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)

    out = positions.iloc[row_idx].copy().reset_index(drop=True)
    out["role"] = [template.roles[j] for j in col_idx]
    out["slot_x"] = slots[col_idx, 0]
    out["slot_y"] = slots[col_idx, 1]
    out["displacement"] = np.sqrt(cost[row_idx, col_idx])
    return out[["player_id", "role", "x", "y", "slot_x", "slot_y", "displacement"]]


def best_template_for_frame(
    positions: pd.DataFrame,
    templates: tuple[FormationTemplate, ...] = DEFAULT_TEMPLATES,
    attacking_right: bool = True,
) -> tuple[FormationTemplate, float]:
    """Pick the template with lowest total displacement cost for these positions."""
    coords = _orient_to_attacking_right(positions[["x", "y"]].to_numpy(dtype=np.float64), attacking_right)
    best_tpl: FormationTemplate | None = None
    best_cost = float("inf")
    for tpl in templates:
        slots = tpl.as_array()
        diff = coords[:, None, :] - slots[None, :, :]
        cost = np.sum(diff * diff, axis=2)
        _, col_idx = linear_sum_assignment(cost)
        total = float(cost[np.arange(len(coords)), col_idx].sum())
        if total < best_cost:
            best_cost = total
            best_tpl = tpl
    assert best_tpl is not None  # templates is non-empty by type
    return best_tpl, best_cost


# ---------------------------------------------------------------------------
# v2 redesign — M1 task #1: per-episode role-anchored trajectory
# ---------------------------------------------------------------------------
#
# Soft-DTW (M1 task #3) and the HDBSCAN clustering layer (M2) both consume
# trajectories of shape (T_snapshots, 10 roles, 2 coords) per episode. The
# per-frame ``assign_roles`` above is the right primitive but works on a single
# frame at a time and assumes the GK has already been filtered out. The
# function below glues those pieces together for an episode.


def _identify_goalkeeper(team_rows: pd.DataFrame, attacking_right: bool) -> str | None:
    """Pick the GK as the team player whose mean position is closest to own goal.

    Returns None if the team has fewer than 11 distinct players visible across
    the episode (no GK to remove — caller will treat all rows as outfielders).
    """
    if team_rows.empty:
        return None
    mean_x = team_rows.groupby("player_id")["x"].mean().dropna()
    if mean_x.empty:
        return None
    # Attacking +x ⇒ own goal at x=0 ⇒ lowest mean x is GK.
    # Attacking -x ⇒ own goal at x=PITCH_LENGTH_M ⇒ highest mean x is GK.
    return str(mean_x.idxmin() if attacking_right else mean_x.idxmax())


def _episode_snapshot_frames(
    tracking: pd.DataFrame,
    episode: EpisodeBoundary,
    snapshot_hz: float,
) -> list[int]:
    """Pick frame_ids uniformly at ``snapshot_hz`` across the episode.

    Mirrors the snapshotter in ``analytics/episodes/state.py`` so role
    trajectories align with the existing state-trajectory snapshots when both
    are computed at the same hz — saves an alignment headache for callers.
    """
    if episode.duration_s <= 0:
        return [episode.start_frame]
    n_snaps = max(2, round(episode.duration_s * snapshot_hz) + 1)
    target_times = np.linspace(episode.start_time_s, episode.end_time_s, n_snaps)
    frame_times = (
        tracking[(tracking["frame_id"] >= episode.start_frame) & (tracking["frame_id"] <= episode.end_frame)][
            ["frame_id", "time_seconds"]
        ]
        .drop_duplicates(subset="frame_id")
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )
    if frame_times.empty:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for t in target_times:
        idx = (frame_times["time_seconds"] - t).abs().idxmin()
        f = int(frame_times.loc[idx, "frame_id"])
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def episode_role_trajectory(
    tracking: pd.DataFrame,
    episode: EpisodeBoundary,
    team_id: str,
    template: FormationTemplate = FORMATION_4_3_3,
    snapshot_hz: float = 2.0,
    attacking_right: bool = True,
    goalkeeper_id: str | None = None,
) -> pd.DataFrame:
    """Build a role-anchored trajectory for one team across one episode.

    For each snapshot in the episode, run Bialkowski Hungarian assignment on the
    team's 10 visible outfielders (GK excluded) against the formation
    ``template``. The output is long-form so callers can either pivot to a
    ``(T, 10, 2)`` tensor (soft-DTW) or aggregate per-role displacement
    statistics directly.

    Args:
        tracking: canonical long-form tracking DataFrame for the whole match.
        episode: ``EpisodeBoundary`` from ``segment_episodes``.
        team_id: which team's roles to track. Pass the possessing team for
            attacking-pattern clustering, or the defending team for OOP shape.
        template: formation template — ``FORMATION_4_3_3`` by default. Use
            ``best_template_for_frame`` upstream if you want auto-detection.
        snapshot_hz: snapshot rate in Hz. 2.0 (every 0.5 s) matches the default
            of ``episode_state_trajectory`` so snapshots align.
        attacking_right: True if ``team_id`` attacks +x in canonical coords.
            Mirrors positions before matching when False.
        goalkeeper_id: explicit GK ``player_id``. If None, auto-detected as the
            team player whose mean x across the episode sits closest to own
            goal. Pass explicitly when you have a roster — it's faster and
            avoids the heuristic mis-firing on cameo episodes.

    Returns:
        Long-form DataFrame with one row per ``(snapshot_idx, role)`` pair and
        columns: ``snapshot_idx, frame_id, time_seconds, role, player_id, x, y,
        displacement``. Empty if no usable snapshots exist (zero outfielders,
        no team rows, or every snapshot has < 10 visible outfielders).

    Raises:
        ValueError: ``team_id`` has no rows in ``tracking``.
    """
    if tracking.empty:
        return pd.DataFrame(
            columns=["snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"]
        )

    if not (tracking["team_id"] == team_id).any():
        raise ValueError(f"team_id={team_id!r} has no rows in tracking")

    in_episode = tracking[
        (tracking["frame_id"] >= episode.start_frame) & (tracking["frame_id"] <= episode.end_frame)
    ]
    team_rows = in_episode[(in_episode["team_id"] == team_id) & (~in_episode["is_ball"])]

    if goalkeeper_id is None:
        goalkeeper_id = _identify_goalkeeper(team_rows, attacking_right)

    snapshot_frames = _episode_snapshot_frames(tracking, episode, snapshot_hz)
    if not snapshot_frames:
        return pd.DataFrame(
            columns=["snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"]
        )

    out_rows: list[pd.DataFrame] = []
    for snap_idx, frame_id in enumerate(snapshot_frames):
        frame_rows = team_rows[team_rows["frame_id"] == frame_id]
        if goalkeeper_id is not None:
            frame_rows = frame_rows[frame_rows["player_id"] != goalkeeper_id]
        # Drop NaN-coord rows so we don't feed garbage to the Hungarian.
        frame_rows = frame_rows[frame_rows["x"].notna() & frame_rows["y"].notna()]
        # Visibility flag respected when present — Metrica/SkillCorner set it.
        if "visible" in frame_rows.columns:
            frame_rows = frame_rows[frame_rows["visible"].astype(bool)]
        if len(frame_rows) != 10:
            # Skip snapshots with too-few or too-many visible outfielders rather
            # than crash the whole episode (real-world tracking has dropouts).
            continue

        # Use the (existing, tested) per-frame assign_roles primitive.
        per_frame = assign_roles(
            frame_rows[["player_id", "x", "y"]].reset_index(drop=True),
            template=template,
            attacking_right=attacking_right,
        )
        per_frame.insert(0, "time_seconds", float(frame_rows["time_seconds"].iloc[0]))
        per_frame.insert(0, "frame_id", int(frame_id))
        per_frame.insert(0, "snapshot_idx", snap_idx)
        out_rows.append(per_frame)

    if not out_rows:
        return pd.DataFrame(
            columns=["snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"]
        )

    result = pd.concat(out_rows, ignore_index=True)
    return result[["snapshot_idx", "frame_id", "time_seconds", "role", "player_id", "x", "y", "displacement"]]

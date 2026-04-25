"""Per-player attribution at episode peak via leave-one-out OBSO counterfactuals.

The peak frame of an episode is the moment of maximum threat:

- ``shot_like`` or ``ended_in_box`` episodes → peak = ``end_frame``.
- ``reached_final_third`` only → peak = the snapshot with the deepest ``ball_x_oriented``.
- Otherwise → no peak (no clear threat to attribute), attribution returns ``None``.

For each visible attacker at the peak frame we recompute OBSO with that player's
position replaced by their *episode-mean* position (a cheap counterfactual: "what if
this player had stayed at their average spot instead of doing what they did?"). The
marginal contribution = ``peak_OBSO − peak_OBSO_with_player_frozen``. Positive
contribution = the player's actual movement increased the team's threat at peak.

**Honest limitations** (must be transparent — do NOT call this "causal"):
1. Other players don't react to the freeze. A real intervention would shift the
   defending team's positioning too; we hold them constant.
2. ``episode-mean`` is a coarse baseline — for players who made big runs, the mean
   is the run's midpoint, not where they "would have been." A future iteration
   could swap in phase-conditional positional priors (e.g. "average winger at
   t+2s into a build-up").
3. Computational cost scales as ``O(N_attackers × OBSO_grid)``. We attribute only
   at the peak frame; the full episode trajectory is left to retrieval (Slice C).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.pitch_control.obso import compute_obso_frame

# Coarser grid than spearman default (68x104) — leave-one-out runs N_attackers OBSO
# computations per episode, so we trade precision for speed. Max-OBSO summary is
# robust to grid resolution within reason.
DEFAULT_ATTRIBUTION_ROWS: int = 40
DEFAULT_ATTRIBUTION_COLS: int = 60


@dataclass(frozen=True)
class EpisodeAttribution:
    """Per-episode contribution decomposition at the peak frame."""

    episode_id: int
    peak_frame: int
    peak_obso: float
    contributions: dict[str, float]  # player_id -> marginal OBSO delta at peak
    baseline_obso_per_player: dict[str, float]  # peak_OBSO with player frozen at avg


def find_peak_frame(record: EpisodeRecord) -> int | None:
    """Pick the peak (most threatening) frame of an episode, or None if no clear peak."""
    if record.outcome.shot_like or record.outcome.ended_in_box:
        return record.boundary.end_frame
    if not record.outcome.reached_final_third:
        return None
    states = record.state_trajectory
    if states.empty or states["ball_x_oriented"].isna().all():
        return None
    ix = states["ball_x_oriented"].idxmax()
    return int(states.loc[ix, "frame_id"])


def _player_average_position(
    tracking: pd.DataFrame, player_id: str, start_frame: int, end_frame: int
) -> tuple[float, float] | None:
    """Mean (x, y) for a player across the episode window, or None if no visible frames."""
    sub = tracking[
        (tracking["player_id"] == player_id)
        & (tracking["frame_id"] >= start_frame)
        & (tracking["frame_id"] <= end_frame)
        & ~tracking["is_ball"]
        & tracking["visible"]
    ]
    if sub.empty:
        return None
    return float(sub["x"].mean()), float(sub["y"].mean())


def compute_episode_attribution(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    rows: int = DEFAULT_ATTRIBUTION_ROWS,
    cols: int = DEFAULT_ATTRIBUTION_COLS,
) -> EpisodeAttribution | None:
    """Leave-one-out OBSO attribution at this episode's peak frame.

    Returns ``None`` for episodes with no clear peak. For all other episodes,
    ``contributions[player_id]`` is the marginal OBSO drop when that player is
    frozen at their episode-average position.
    """
    peak_frame = find_peak_frame(record)
    if peak_frame is None:
        return None

    attacking_team = record.boundary.possession_team
    defending_team = away_team_id if attacking_team == home_team_id else home_team_id

    ref = compute_obso_frame(
        tracking,
        peak_frame,
        attacking_team_id=attacking_team,
        defending_team_id=defending_team,
        rows=rows,
        cols=cols,
    )
    peak_obso = float(np.max(ref.obso))

    peak_rows = tracking[tracking["frame_id"] == peak_frame]
    attackers = peak_rows[(peak_rows["team_id"] == attacking_team) & ~peak_rows["is_ball"] & peak_rows["visible"]]
    contributions: dict[str, float] = {}
    baselines: dict[str, float] = {}

    for _, row in attackers.iterrows():
        player_id = str(row["player_id"])
        avg_pos = _player_average_position(tracking, player_id, record.boundary.start_frame, record.boundary.end_frame)
        # Player wasn't seen elsewhere, or didn't move during the episode — contribution = 0.
        if avg_pos is None or (abs(avg_pos[0] - float(row["x"])) < 0.01 and abs(avg_pos[1] - float(row["y"])) < 0.01):
            contributions[player_id] = 0.0
            baselines[player_id] = peak_obso
            continue

        # Counterfactual: replace this player's peak-frame position with their episode mean.
        modified = tracking.copy()
        peak_mask = (modified["frame_id"] == peak_frame) & (modified["player_id"] == player_id)
        modified.loc[peak_mask, "x"] = avg_pos[0]
        modified.loc[peak_mask, "y"] = avg_pos[1]
        modified.loc[peak_mask, "vx"] = 0.0
        modified.loc[peak_mask, "vy"] = 0.0

        cf = compute_obso_frame(
            modified,
            peak_frame,
            attacking_team_id=attacking_team,
            defending_team_id=defending_team,
            rows=rows,
            cols=cols,
        )
        cf_peak = float(np.max(cf.obso))
        baselines[player_id] = cf_peak
        contributions[player_id] = peak_obso - cf_peak

    return EpisodeAttribution(
        episode_id=record.boundary.episode_id,
        peak_frame=peak_frame,
        peak_obso=peak_obso,
        contributions=contributions,
        baseline_obso_per_player=baselines,
    )

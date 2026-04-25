"""Top-level episode-engine composer: tracking → list[EpisodeRecord].

Glues the three Slice-A primitives:

1. ``classify_frames`` (Phase 3A) — per-frame possession + phase labels.
2. ``segment_episodes`` — episode boundaries from possession + ball-visibility.
3. ``episode_state_trajectory`` + ``classify_outcome`` — per-episode enrichment.

The output is a flat ``list[EpisodeRecord]`` and a parallel summary DataFrame for
fast batch analysis. Both are designed to flow straight into Slice B (attribution)
and Slice C (retrieval) without re-shaping.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_analysis.analytics.episodes.outcomes import (
    EpisodeOutcome,
    classify_outcome,
)
from football_analysis.analytics.episodes.segmenter import (
    EpisodeBoundary,
    segment_episodes,
)
from football_analysis.analytics.episodes.state import episode_state_trajectory
from football_analysis.analytics.phases.classifier import classify_frames


@dataclass(frozen=True)
class EpisodeRecord:
    """Full description of a single possession episode."""

    boundary: EpisodeBoundary
    outcome: EpisodeOutcome
    state_trajectory: pd.DataFrame
    dominant_phase: str | None  # most-common phase across episode frames


def _dominant_phase(classified: pd.DataFrame, ep: EpisodeBoundary) -> str | None:
    sub = classified[(classified["frame_id"] >= ep.start_frame) & (classified["frame_id"] <= ep.end_frame)]["phase"]
    if sub.empty:
        return None
    mode = sub.mode()
    return None if mode.empty else str(mode.iloc[0])


def build_episodes(
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    attacking_directions: dict[str, str] | None = None,
    snapshot_hz: float = 2.0,
    min_dead_frames: int = 5,
) -> list[EpisodeRecord]:
    """Build a list of ``EpisodeRecord`` for a single match.

    Args:
        tracking: canonical long-form tracking DataFrame.
        home_team_id, away_team_id: team identifiers used in tracking.
        attacking_directions: optional map of ``team_id -> "left"|"right"`` for
            canonical-coord orientation. Defaults to home→right, away→left
            (the home-LTR-H1 convention).
        snapshot_hz: state-trajectory sample rate (per second).
        min_dead_frames: dead-ball threshold passed to the segmenter.

    Returns:
        Empty list if tracking is empty or no clean possessions are found.
    """
    if tracking.empty:
        return []
    if attacking_directions is None:
        attacking_directions = {home_team_id: "right", away_team_id: "left"}

    classified = classify_frames(tracking, home_team_id=home_team_id, away_team_id=away_team_id)
    boundaries = segment_episodes(classified, tracking, min_dead_frames=min_dead_frames)

    records: list[EpisodeRecord] = []
    for ep in boundaries:
        attacking_to_right = attacking_directions.get(ep.possession_team, "right") == "right"

        states = episode_state_trajectory(
            tracking,
            ep,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            snapshot_hz=snapshot_hz,
            attacking_to_right=attacking_to_right,
        )
        outcome = classify_outcome(ep, tracking, attacking_to_right=attacking_to_right)
        records.append(
            EpisodeRecord(
                boundary=ep,
                outcome=outcome,
                state_trajectory=states,
                dominant_phase=_dominant_phase(classified, ep),
            )
        )
    return records


def episodes_to_summary(records: list[EpisodeRecord]) -> pd.DataFrame:
    """Flatten ``EpisodeRecord`` list into a one-row-per-episode summary DataFrame.

    Used as the top-level lookup table for the retrieval index in Slice C.
    """
    rows = []
    for r in records:
        rows.append(
            {
                "episode_id": r.boundary.episode_id,
                "start_frame": r.boundary.start_frame,
                "end_frame": r.boundary.end_frame,
                "start_time_s": r.boundary.start_time_s,
                "end_time_s": r.boundary.end_time_s,
                "duration_s": r.boundary.duration_s,
                "possession_team": r.boundary.possession_team,
                "dominant_phase": r.dominant_phase,
                "end_reason": r.outcome.end_reason,
                "reached_final_third": r.outcome.reached_final_third,
                "ended_in_box": r.outcome.ended_in_box,
                "shot_like": r.outcome.shot_like,
                "end_ball_x": r.outcome.end_ball_x,
                "end_ball_y": r.outcome.end_ball_y,
                "end_ball_speed": r.outcome.end_ball_speed,
                "n_snapshots": len(r.state_trajectory),
            }
        )
    return pd.DataFrame(rows)

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
    compute_obso_outcome: bool = False,
    use_gpu: bool = False,
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
        compute_obso_outcome: when True, run the OBSO trajectory for every
            episode and populate ``outcome.peak_obso`` (max OBSO across the
            trajectory) and ``outcome.decisive_obso`` (50%-of-peak threshold
            crossing). Phase 6-B uses this to give recommend.py a continuous
            outcome value. Off by default since it materially increases
            compute time (~ms per snapshot × ~14 snapshots × N episodes).
        use_gpu: forwarded to ``compute_obso_trajectory`` when
            ``compute_obso_outcome=True``. With CUDA available this gives a
            ~10-15× speedup over numpy.

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
        record = EpisodeRecord(
            boundary=ep,
            outcome=outcome,
            state_trajectory=states,
            dominant_phase=_dominant_phase(classified, ep),
        )
        if compute_obso_outcome and not states.empty:
            record = _enrich_with_obso_outcome(record, tracking, home_team_id, away_team_id, use_gpu)
        records.append(record)
    return records


def _enrich_with_obso_outcome(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    use_gpu: bool,
) -> EpisodeRecord:
    """Compute peak/decisive OBSO for an episode and return a new record with the
    outcome fields populated.

    Lazy-imports the trajectory module to avoid circular-import warnings (this
    module is imported by obso_trajectory).
    """
    from football_analysis.analytics.episodes.obso_trajectory import (  # noqa: PLC0415
        compute_obso_trajectory,
        find_decisive_moment,
    )

    traj = compute_obso_trajectory(
        record,
        tracking,
        home_team_id,
        away_team_id,
        use_gpu=use_gpu,
    )
    if traj.empty:
        return record  # nothing to enrich
    peak = float(traj["obso_max"].max())
    decisive = find_decisive_moment(traj, threshold_pct=0.5)
    decisive_obso = float(decisive.obso_at_decisive) if decisive is not None else None

    # Outcome is frozen=True; rebuild it via dataclasses.replace.
    from dataclasses import replace  # noqa: PLC0415

    new_outcome = replace(record.outcome, peak_obso=peak, decisive_obso=decisive_obso)
    return EpisodeRecord(
        boundary=record.boundary,
        outcome=new_outcome,
        state_trajectory=record.state_trajectory,
        dominant_phase=record.dominant_phase,
    )


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
                "peak_obso": r.outcome.peak_obso,
                "decisive_obso": r.outcome.decisive_obso,
                "n_snapshots": len(r.state_trajectory),
            }
        )
    return pd.DataFrame(rows)

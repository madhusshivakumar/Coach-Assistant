"""Trigger-frame detection + template-based narrative for an episode.

Trigger frame = the earliest frame in the episode where the eventual outcome's
expected value first rose meaningfully. v1 uses a simple, interpretable signal:
the frame when the ball first crosses into the final third (oriented x > 70 m).
For episodes that never reach the final third, the trigger is ``None``.

Narrative is template-based. We string together: who held possession, when the
threat first emerged, who contributed most at peak, and how the episode ended.
That gives a one-paragraph plain-English description per episode — the kind of
thing Slice C's retrieval can later attach to "this looks like that past episode"
explanations.

Richer narrative (LLM-driven scene-by-scene) can layer on later; this v1 is
deterministic and free.
"""

from __future__ import annotations

from dataclasses import dataclass

from football_analysis.analytics.episodes.contribution import EpisodeAttribution
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.outcomes import FINAL_THIRD_X


@dataclass(frozen=True)
class EpisodeNarrative:
    """Plain-English description of one episode + the structured fields backing it."""

    episode_id: int
    trigger_frame: int | None
    trigger_time_s: float | None
    top_contributors: list[tuple[str, float]]  # (player_id, contribution)
    text: str


def find_trigger_frame(
    record: EpisodeRecord,
    final_third_threshold: float = FINAL_THIRD_X,
) -> tuple[int | None, float | None]:
    """Earliest snapshot whose ``ball_x_oriented`` crossed the final-third line."""
    states = record.state_trajectory
    if states.empty:
        return None, None
    crossed = states[states["ball_x_oriented"] > final_third_threshold]
    if crossed.empty:
        return None, None
    first = crossed.iloc[0]
    return int(first["frame_id"]), float(first["time_s"])


def _format_player(player_id: str, contribution: float) -> str:
    return f"{player_id} (Δ OBSO {contribution:+.3f})"


def build_narrative(
    record: EpisodeRecord,
    attribution: EpisodeAttribution | None,
    top_k: int = 3,
) -> EpisodeNarrative:
    """Assemble the one-paragraph episode narrative."""
    trigger_frame, trigger_time = find_trigger_frame(record)
    boundary = record.boundary
    outcome = record.outcome

    if attribution is None or not attribution.contributions:
        text = (
            f"{boundary.possession_team} held possession from t={boundary.start_time_s:.1f}s "
            f"to t={boundary.end_time_s:.1f}s ({boundary.duration_s:.2f}s). "
            f"The episode ended in {outcome.end_reason}; "
            f"{'reached the final third' if outcome.reached_final_third else 'did not reach the final third'}."
        )
        return EpisodeNarrative(
            episode_id=boundary.episode_id,
            trigger_frame=trigger_frame,
            trigger_time_s=trigger_time,
            top_contributors=[],
            text=text,
        )

    ranked = sorted(attribution.contributions.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    top_str = ", ".join(_format_player(p, c) for p, c in ranked) or "(no movers)"

    trigger_clause = (
        f"The threat first emerged at t={trigger_time:.1f}s when the ball entered the final third."
        if trigger_time is not None
        else "The episode never reached the final third."
    )
    end_clause = (
        "ending in a shot-like attempt inside the box."
        if outcome.shot_like
        else ("ending with the ball inside the box." if outcome.ended_in_box else f"ending in {outcome.end_reason}.")
    )

    text = (
        f"{boundary.possession_team} attacked from t={boundary.start_time_s:.1f}s. "
        f"{trigger_clause} "
        f"At peak (frame {attribution.peak_frame}, max OBSO {attribution.peak_obso:.3f}) "
        f"the players whose movement contributed most were: {top_str}. "
        f"The episode lasted {boundary.duration_s:.2f}s, {end_clause}"
    )

    return EpisodeNarrative(
        episode_id=boundary.episode_id,
        trigger_frame=trigger_frame,
        trigger_time_s=trigger_time,
        top_contributors=list(ranked),
        text=text,
    )

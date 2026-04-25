"""Possession-episode engine — segments a match into episodes, snapshots state, classifies outcome.

A *possession episode* is a contiguous sequence of frames in which one team holds the
ball, ending at either a turnover or a dead-ball window (>= ``MIN_DEAD_FRAMES`` frames
of ball-not-visible). Each episode produces a single ``EpisodeRecord`` containing:

- segmenter boundary (start/end frames, possession team, end_reason)
- per-snapshot state trajectory (positions, ball, lightweight team-shape features)
- terminal-event outcome classification (reached final third? ended in box? shot-like?)
- dominant phase label (build_up / progression / finishing / transition / settled_def)

Composes existing primitives: ``classify_frames`` (Phase 3A), ``shape_time_series``
(Phase 3A). Pitch-control + OBSO surfaces are deferred to Slice B (attribution).
"""

from football_analysis.analytics.episodes.contribution import (
    EpisodeAttribution,
    compute_episode_attribution,
    find_peak_frame,
)
from football_analysis.analytics.episodes.engine import (
    EpisodeRecord,
    build_episodes,
    episodes_to_summary,
)
from football_analysis.analytics.episodes.narrative import (
    EpisodeNarrative,
    build_narrative,
    find_trigger_frame,
)
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome, classify_outcome
from football_analysis.analytics.episodes.segmenter import (
    MIN_DEAD_FRAMES_DEFAULT,
    EpisodeBoundary,
    segment_episodes,
)
from football_analysis.analytics.episodes.state import episode_state_trajectory

__all__ = [
    "MIN_DEAD_FRAMES_DEFAULT",
    "EpisodeAttribution",
    "EpisodeBoundary",
    "EpisodeNarrative",
    "EpisodeOutcome",
    "EpisodeRecord",
    "build_episodes",
    "build_narrative",
    "classify_outcome",
    "compute_episode_attribution",
    "episode_state_trajectory",
    "episodes_to_summary",
    "find_peak_frame",
    "find_trigger_frame",
    "segment_episodes",
]

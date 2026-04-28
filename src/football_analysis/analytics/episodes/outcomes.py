"""Classify how an episode terminated and assign coarse value flags.

Slice A's outcome layer is a *labeller*, not a model. Each episode gets:

- ``end_reason`` (mirrored from segmenter): turnover, out_of_play, or match_end.
- Three boolean spatial features computed against the episode's last visible ball
  position: did we reach the final third? did the episode end inside the penalty
  area? was the terminal ball moving fast (a heuristic shot proxy)?

The proper "expected-value" labels (xT-delta, OBSO peak) come in Slice C when the
retrieval/feature pipeline is built. Keeping the outcome thin here lets us validate
the engine end-to-end without dragging the heavy compute into the first commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary

# Pitch geometry in canonical (105 x 68 m) coordinates.
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0
FINAL_THIRD_X: float = 70.0  # x_oriented threshold
PENALTY_AREA_X_MIN: float = 105.0 - 16.5  # = 88.5
PENALTY_AREA_Y_MIN: float = 34.0 - 20.16  # = 13.84
PENALTY_AREA_Y_MAX: float = 34.0 + 20.16  # = 54.16

# Default threshold (m/s) for "ball moving fast enough to be a shot proxy".
SHOT_SPEED_THRESHOLD_M_S: float = 12.0


@dataclass(frozen=True)
class EpisodeOutcome:
    """Coarse what-happened classification of one episode.

    Fields ``peak_obso`` and ``decisive_obso`` are continuous outcome values added
    in Phase 6-B and populated only when ``build_episodes(compute_obso_outcome=True)``.
    They give recommend.py a much richer signal than the binary ``shot_like`` flag —
    the *actual* maximum off-ball threat the attacking team built during the
    episode, not just whether they happened to take a fast shot. Default ``None``
    means "not computed" so existing tests/callers see no behaviour change.
    """

    episode_id: int
    end_reason: str
    reached_final_third: bool
    ended_in_box: bool
    shot_like: bool  # ended_in_box AND end_ball_speed >= threshold
    end_ball_x: float
    end_ball_y: float
    end_ball_speed: float
    duration_s: float
    peak_obso: float | None = None
    decisive_obso: float | None = None


def _orient_x(x: float, attacking_to_right: bool) -> float:
    return x if attacking_to_right else PITCH_LENGTH_M - x


def classify_outcome(
    episode: EpisodeBoundary,
    tracking: pd.DataFrame,
    attacking_to_right: bool,
    shot_speed_threshold: float = SHOT_SPEED_THRESHOLD_M_S,
) -> EpisodeOutcome:
    """Build the ``EpisodeOutcome`` for a single episode.

    Args:
        episode: boundary from the segmenter.
        tracking: canonical tracking DataFrame.
        attacking_to_right: True iff ``possession_team`` attacks +x in canonical frame.
        shot_speed_threshold: m/s; the ``shot_like`` flag requires
            ``ended_in_box AND end_ball_speed >= threshold``.
    """
    in_episode = (
        (tracking["frame_id"] >= episode.start_frame)
        & (tracking["frame_id"] <= episode.end_frame)
        & tracking["is_ball"]
        & tracking["visible"]
    )
    ball = tracking[in_episode]
    if ball.empty:
        return EpisodeOutcome(
            episode_id=episode.episode_id,
            end_reason=episode.end_reason,
            reached_final_third=False,
            ended_in_box=False,
            shot_like=False,
            end_ball_x=float("nan"),
            end_ball_y=float("nan"),
            end_ball_speed=float("nan"),
            duration_s=episode.duration_s,
        )

    # Final third = any ball frame in episode with x_oriented > FINAL_THIRD_X.
    oriented_x_series = ball["x"].apply(lambda x: _orient_x(float(x), attacking_to_right))
    reached_final_third = bool((oriented_x_series > FINAL_THIRD_X).any())

    last = ball.sort_values("frame_id").iloc[-1]
    end_x = float(last["x"])
    end_y = float(last["y"])
    end_speed = float(np.hypot(last["vx"], last["vy"]))

    end_x_oriented = _orient_x(end_x, attacking_to_right)
    ended_in_box = bool(end_x_oriented >= PENALTY_AREA_X_MIN and PENALTY_AREA_Y_MIN <= end_y <= PENALTY_AREA_Y_MAX)
    shot_like = bool(ended_in_box and end_speed >= shot_speed_threshold)

    return EpisodeOutcome(
        episode_id=episode.episode_id,
        end_reason=episode.end_reason,
        reached_final_third=reached_final_third,
        ended_in_box=ended_in_box,
        shot_like=shot_like,
        end_ball_x=end_x,
        end_ball_y=end_y,
        end_ball_speed=end_speed,
        duration_s=episode.duration_s,
    )

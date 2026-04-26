"""Stateful streaming wrapper around the episode-engine primitives.

Designed for two deployment modes (the API is identical):

- **Replay**: feed a recorded match frame-by-frame at wall-clock speed. Used
  for the Phase 5 demo and for backtest harnesses.
- **Live**: drop-in for a real tracking feed when one is available. The per-frame
  ``on_frame(rows)`` contract doesn't change.

Reliability properties baked in:

1. **Refuses to predict when no analog exists.** If retrieval's ``max_distance``
   exceeds ``confidence_distance_threshold``, the prediction is tagged
   ``confidence="low"`` and downstream consumers should suppress display. This
   is the single most important guardrail for production — a model that
   confidently extrapolates into garbage is worse than one that shrugs.
2. **Deterministic replay.** Same input frames in the same order produce the
   same snapshots. Required for testing, backtesting, and reproducible bug
   reports.
3. **Calibration logging.** Every prediction is recorded along with its
   eventual outcome (resolved when the in-progress episode terminates). A
   running calibration table is exposed via ``calibration_log()``. This is
   the input to the real calibration plot when the corpus grows.
4. **Throttled compute.** OBSO runs every ``obso_every_k_frames`` frames
   (default 5 → 5 Hz at 25 Hz input); retrieval every
   ``retrieval_every_k_frames`` (default 12 → ~2 Hz). Attribution is fired
   only when threat crosses a threshold. Together this keeps the inner loop
   under 40 ms even on CPU.

Streaming caveats (read these — they're real):

- The retrieval library is built offline before replay begins. In a true live
  deployment with no future leakage, the library must contain only past
  matches. With a single-match POC, retrieval against the same match is
  tautological by construction; the demo documents this and the engine itself
  is agnostic to which library it queries.
- Possession smoothing requires a buffer of recent frames. A cold-started
  engine produces ``possession_team=None`` for the first
  ``sticky_frames`` frames until the buffer fills. This is intentional and
  correct.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import pandas as pd

from football_analysis.analytics.episodes.contribution import (
    DEFAULT_ATTRIBUTION_COLS,
    DEFAULT_ATTRIBUTION_ROWS,
    compute_episode_attribution,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.outcomes import classify_outcome
from football_analysis.analytics.episodes.segmenter import (
    EpisodeBoundary,
    segment_episodes,
)
from football_analysis.analytics.episodes.state import episode_state_trajectory
from football_analysis.analytics.phases.classifier import classify_frames
from football_analysis.analytics.pitch_control.obso import compute_obso_frame

# Default: a rolling tracking buffer ~10 s wide at 25 Hz.
DEFAULT_BUFFER_FRAMES: int = 250

# Default thresholds for the reliability scaffolding.
DEFAULT_OBSO_EVERY: int = 5  # 5 Hz at 25 Hz input
DEFAULT_RETRIEVAL_EVERY: int = 12  # ~2 Hz
DEFAULT_ATTRIBUTION_THRESHOLD: float = 0.35  # OBSO_max needed to fire leave-one-out
DEFAULT_CONFIDENCE_DISTANCE: float = 5.0  # max_distance above this → low confidence


@dataclass(frozen=True)
class LiveSnapshot:
    """Engine state at one tick. JSON-serializable for dashboard subscribers."""

    frame_id: int
    time_s: float

    # Current state (best estimate from rolling buffer)
    possession_team: str | None
    current_episode_id: int  # -1 when no active episode
    current_episode_phase: str | None
    current_episode_duration_s: float

    # Threat (potentially stale by up to obso_every_k_frames)
    threat_level: float
    threat_level_age_frames: int

    # Prediction (potentially stale by up to retrieval_every_k_frames)
    predicted: dict[str, Any]
    top_neighbors: list[dict[str, Any]]
    prediction_age_frames: int

    # Key-player attribution (lazy — fires on threat threshold cross)
    rolling_top_contributors: list[dict[str, Any]]
    attribution_age_frames: int


@dataclass
class _LivePrediction:
    """One logged prediction, paired with its eventual realized outcome.

    Used by ``calibration_log()`` to surface predicted-vs-realized for the whole
    replay run. The inputs to a real calibration plot once we have N matches.
    """

    frame_id: int
    time_s: float
    episode_id: int
    p_shot_like: float
    p_ended_in_box: float
    p_reached_final_third: float
    max_distance: float
    confidence: str
    realized_shot_like: bool | None = None
    realized_ended_in_box: bool | None = None
    realized_reached_final_third: bool | None = None


class LiveEpisodeEngine:
    """Stateful streaming engine. One instance per match-in-progress.

    Pattern:

    >>> engine = LiveEpisodeEngine(retrieval_index, "home", "away")
    >>> for frame_id, frame_rows in stream(tracking):
    ...     snap = engine.on_frame(frame_rows)
    ...     dashboard.publish(snap)
    >>> for record in engine.completed_episodes():
    ...     # Episodes resolved during the run, with their realized outcomes.
    ...     pass
    """

    def __init__(
        self,
        retrieval_index: EpisodeIndex,
        home_team_id: str,
        away_team_id: str,
        attacking_directions: dict[str, str] | None = None,
        obso_every_k_frames: int = DEFAULT_OBSO_EVERY,
        retrieval_every_k_frames: int = DEFAULT_RETRIEVAL_EVERY,
        attribution_threshold_obso: float = DEFAULT_ATTRIBUTION_THRESHOLD,
        confidence_distance_threshold: float = DEFAULT_CONFIDENCE_DISTANCE,
        obso_rows: int = DEFAULT_ATTRIBUTION_ROWS,
        obso_cols: int = DEFAULT_ATTRIBUTION_COLS,
        buffer_frames: int = DEFAULT_BUFFER_FRAMES,
        snapshot_hz: float = 2.0,
    ) -> None:
        self._retrieval = retrieval_index
        self._home = home_team_id
        self._away = away_team_id
        self._attacking_dirs = attacking_directions or {home_team_id: "right", away_team_id: "left"}
        self._obso_every = obso_every_k_frames
        self._retr_every = retrieval_every_k_frames
        self._attr_threshold = attribution_threshold_obso
        self._conf_threshold = confidence_distance_threshold
        self._obso_rows = obso_rows
        self._obso_cols = obso_cols
        self._buffer_frames = buffer_frames
        self._snapshot_hz = snapshot_hz

        # Rolling tracking buffer (the last `buffer_frames` frames).
        self._buffer = pd.DataFrame()

        # Episode IDs assigned during the live run; not aligned with offline IDs.
        self._next_episode_id = 0

        # Latest derived state — caches between throttled recomputations.
        self._latest_obso_frame = -1
        self._latest_obso_max = 0.0
        self._latest_retrieval_frame = -1
        self._latest_predicted: dict[str, Any] = {}
        self._latest_neighbors: list[dict[str, Any]] = []
        self._latest_attribution_frame = -1
        self._latest_top_contributors: list[dict[str, Any]] = []

        # Tracks the active episode (if any) across frames so episode IDs are stable.
        self._active_episode_start_frame: int | None = None
        self._active_episode_team: str | None = None
        self._active_episode_id: int = -1

        # Episode history + prediction log.
        self._completed: list[EpisodeRecord] = []
        self._predictions: list[_LivePrediction] = []
        self._open_predictions: deque[_LivePrediction] = deque()

    # ------------------------------------------------------------------ public API

    def on_frame(self, frame_rows: pd.DataFrame) -> LiveSnapshot:
        """Process the tracking rows for a single ``frame_id`` and return state.

        ``frame_rows`` should be every tracking row whose ``frame_id`` matches
        the current tick (i.e. one row per player + ball at this instant).
        Empty input returns a no-op snapshot.
        """
        if frame_rows.empty:
            return self._empty_snapshot(frame_id=-1, time_s=0.0)

        frame_id = int(frame_rows["frame_id"].iloc[0])
        time_s = float(frame_rows["time_seconds"].iloc[0])

        # 1. Append to rolling buffer; drop frames beyond the window.
        self._buffer = pd.concat([self._buffer, frame_rows], ignore_index=True)
        cutoff = frame_id - self._buffer_frames
        self._buffer = self._buffer[self._buffer["frame_id"] > cutoff].reset_index(drop=True)

        # 2. Re-derive possession + episode boundaries over the rolling buffer.
        try:
            classified = classify_frames(self._buffer, self._home, self._away)
        except Exception:
            classified = pd.DataFrame()
        boundaries = segment_episodes(classified, self._buffer) if not classified.empty else []

        latest_possession, latest_phase = self._latest_classified(classified)
        active_boundary = self._reconcile_active_episode(boundaries, frame_id, latest_possession)

        # 3. Maybe-recompute OBSO (throttled).
        if active_boundary is not None and frame_id - self._latest_obso_frame >= self._obso_every:
            self._update_obso(frame_id, active_boundary)

        # 4. Maybe-run retrieval (throttled).
        if active_boundary is not None and frame_id - self._latest_retrieval_frame >= self._retr_every:
            self._update_retrieval(frame_id, time_s, active_boundary, latest_phase)

        # 5. Maybe-run attribution (only when threat exceeds threshold).
        if (
            active_boundary is not None
            and self._latest_obso_max >= self._attr_threshold
            and frame_id - self._latest_attribution_frame >= self._obso_every
        ):
            self._update_attribution(frame_id, active_boundary)

        return LiveSnapshot(
            frame_id=frame_id,
            time_s=time_s,
            possession_team=latest_possession,
            current_episode_id=self._active_episode_id,
            current_episode_phase=latest_phase,
            current_episode_duration_s=(
                round(time_s - self._active_episode_start_time, 3)
                if self._active_episode_start_frame is not None
                else 0.0
            ),
            threat_level=self._latest_obso_max,
            threat_level_age_frames=max(0, frame_id - self._latest_obso_frame),
            predicted=dict(self._latest_predicted),
            top_neighbors=list(self._latest_neighbors),
            prediction_age_frames=max(0, frame_id - self._latest_retrieval_frame),
            rolling_top_contributors=list(self._latest_top_contributors),
            attribution_age_frames=max(0, frame_id - self._latest_attribution_frame),
        )

    def completed_episodes(self) -> list[EpisodeRecord]:
        """Episodes that closed during the run, with realized outcomes."""
        return list(self._completed)

    def calibration_log(self) -> list[_LivePrediction]:
        """Every (prediction, realized-outcome) pair logged during the run.

        Predictions whose episode hasn't terminated yet have ``realized_*=None``.
        """
        return list(self._predictions)

    # ---------------------------------------------------------------- internals

    @property
    def _active_episode_start_time(self) -> float:
        if self._active_episode_start_frame is None or self._buffer.empty:
            return 0.0
        match = self._buffer[self._buffer["frame_id"] == self._active_episode_start_frame]
        if match.empty:
            return 0.0
        return float(match["time_seconds"].iloc[0])

    def _latest_classified(self, classified: pd.DataFrame) -> tuple[str | None, str | None]:
        if classified.empty:
            return None, None
        last = classified.iloc[-1]
        pos = last.get("possession_team")
        phase = last.get("phase")
        return (
            None if pd.isna(pos) else str(pos),
            None if pd.isna(phase) else str(phase),
        )

    def _reconcile_active_episode(
        self,
        boundaries: list[EpisodeBoundary],
        current_frame: int,
        latest_possession: str | None,
    ) -> EpisodeBoundary | None:
        """Track which episode is currently in progress.

        Strategy: the *last* boundary in the list, if its end_frame matches a
        recent frame and its possession matches the latest classified frame,
        is the active one. When the active episode terminates we close it
        and resolve any open predictions against its realized outcome.
        """
        if not boundaries:
            return None

        candidate = boundaries[-1]
        # Active iff the candidate's possession matches the very latest
        # classified frame's possession, and end_frame is at the rolling edge.
        is_active = (
            latest_possession is not None
            and candidate.possession_team == latest_possession
            and candidate.end_frame >= current_frame - self._obso_every
        )

        if is_active:
            # Same episode continued, or new episode started.
            if (
                self._active_episode_start_frame != candidate.start_frame
                or self._active_episode_team != candidate.possession_team
            ):
                # New episode began since last frame — close any prior active.
                if self._active_episode_id >= 0:
                    self._close_active_episode(boundaries)
                self._active_episode_start_frame = candidate.start_frame
                self._active_episode_team = candidate.possession_team
                self._active_episode_id = self._next_episode_id
                self._next_episode_id += 1
            # Synthesize an EpisodeBoundary stamped with our stable id.
            return EpisodeBoundary(
                episode_id=self._active_episode_id,
                start_frame=candidate.start_frame,
                end_frame=candidate.end_frame,
                start_time_s=candidate.start_time_s,
                end_time_s=candidate.end_time_s,
                duration_s=candidate.duration_s,
                possession_team=candidate.possession_team,
                end_reason=candidate.end_reason,
            )

        # Active episode just terminated.
        if self._active_episode_id >= 0:
            self._close_active_episode(boundaries)
        return None

    def _close_active_episode(self, boundaries: list[EpisodeBoundary]) -> None:
        """Finalize the active episode, resolve open predictions."""
        # Find the boundary that matches the just-closed start_frame.
        match = next(
            (b for b in boundaries if b.start_frame == self._active_episode_start_frame),
            None,
        )
        if match is None:
            self._reset_active()
            return

        atk_dir = self._attacking_dirs.get(match.possession_team, "right")
        attacking_to_right = atk_dir == "right"

        states = episode_state_trajectory(
            self._buffer,
            match,
            home_team_id=self._home,
            away_team_id=self._away,
            snapshot_hz=self._snapshot_hz,
            attacking_to_right=attacking_to_right,
        )
        outcome = classify_outcome(match, self._buffer, attacking_to_right=attacking_to_right)
        record = EpisodeRecord(
            boundary=EpisodeBoundary(
                episode_id=self._active_episode_id,
                start_frame=match.start_frame,
                end_frame=match.end_frame,
                start_time_s=match.start_time_s,
                end_time_s=match.end_time_s,
                duration_s=match.duration_s,
                possession_team=match.possession_team,
                end_reason=match.end_reason,
            ),
            outcome=outcome,
            state_trajectory=states,
            dominant_phase=None,  # could compute via classify but expensive at close-time
        )
        self._completed.append(record)

        # Resolve any open predictions for this episode_id.
        for pred in list(self._open_predictions):
            if pred.episode_id == self._active_episode_id:
                pred.realized_shot_like = outcome.shot_like
                pred.realized_ended_in_box = outcome.ended_in_box
                pred.realized_reached_final_third = outcome.reached_final_third
                self._open_predictions.remove(pred)

        self._reset_active()

    def _reset_active(self) -> None:
        self._active_episode_start_frame = None
        self._active_episode_team = None
        self._active_episode_id = -1
        self._latest_obso_frame = -1
        self._latest_obso_max = 0.0
        self._latest_retrieval_frame = -1
        self._latest_predicted = {}
        self._latest_neighbors = []
        self._latest_attribution_frame = -1
        self._latest_top_contributors = []

    def _update_obso(self, frame_id: int, active: EpisodeBoundary) -> None:
        defender = self._away if active.possession_team == self._home else self._home
        try:
            of = compute_obso_frame(
                self._buffer,
                frame_id,
                attacking_team_id=active.possession_team,
                defending_team_id=defender,
                rows=self._obso_rows,
                cols=self._obso_cols,
            )
            self._latest_obso_max = float(of.obso.max())
        except Exception:
            # OBSO compute can fail if a frame has missing players; keep last value.
            return
        self._latest_obso_frame = frame_id

    def _update_retrieval(
        self,
        frame_id: int,
        time_s: float,
        active: EpisodeBoundary,
        latest_phase: str | None,
    ) -> None:
        # Build a partial EpisodeRecord from buffer up to current_frame.
        atk_dir = self._attacking_dirs.get(active.possession_team, "right")
        attacking_to_right = atk_dir == "right"
        states = episode_state_trajectory(
            self._buffer,
            active,
            home_team_id=self._home,
            away_team_id=self._away,
            snapshot_hz=self._snapshot_hz,
            attacking_to_right=attacking_to_right,
        )
        # Truncate to "as if right now."
        rel_cutoff = time_s - active.start_time_s
        record = EpisodeRecord(
            boundary=active,
            outcome=classify_outcome(active, self._buffer, attacking_to_right=attacking_to_right),
            state_trajectory=states,
            dominant_phase=latest_phase,
        )
        prediction_raw = self._retrieval.predict_outcome(record, k=3, max_rel_time_s=rel_cutoff)
        # Loosen the value type so we can stash str confidence + auxiliary fields.
        prediction: dict[str, Any] = dict(prediction_raw)
        max_dist_raw = prediction.get("max_distance", 0.0)
        max_dist = float(max_dist_raw) if isinstance(max_dist_raw, int | float) else 0.0
        confidence = (
            "low"
            if max_dist > self._conf_threshold
            else ("medium" if max_dist > self._conf_threshold * 0.6 else "high")
        )
        prediction["confidence"] = confidence
        prediction["max_rel_time_s"] = round(rel_cutoff, 2)
        self._latest_predicted = prediction
        self._latest_retrieval_frame = frame_id

        # Surface neighbor metadata for the dashboard.
        neighbors = self._retrieval.query(record, k=3, max_rel_time_s=rel_cutoff)
        self._latest_neighbors = [
            {
                "rank": n.rank,
                "episode_id": n.record.boundary.episode_id,
                "distance": round(n.distance, 4),
                "shot_like": n.record.outcome.shot_like,
                "ended_in_box": n.record.outcome.ended_in_box,
                "reached_final_third": n.record.outcome.reached_final_third,
                "dominant_phase": n.record.dominant_phase,
            }
            for n in neighbors
        ]

        # Log the prediction; will be resolved when the episode closes.
        # ``predict_outcome`` returns a dict[str, float|int|list[int]]; only the
        # probability fields are floats — guard with isinstance to keep mypy happy.
        def _f(key: str) -> float:
            v = prediction.get(key, 0.0)
            return float(v) if isinstance(v, int | float) else 0.0

        log_entry = _LivePrediction(
            frame_id=frame_id,
            time_s=time_s,
            episode_id=self._active_episode_id,
            p_shot_like=_f("p_shot_like"),
            p_ended_in_box=_f("p_ended_in_box"),
            p_reached_final_third=_f("p_reached_final_third"),
            max_distance=max_dist,
            confidence=confidence,
        )
        self._predictions.append(log_entry)
        self._open_predictions.append(log_entry)

    def _update_attribution(self, frame_id: int, active: EpisodeBoundary) -> None:
        atk_dir = self._attacking_dirs.get(active.possession_team, "right")
        attacking_to_right = atk_dir == "right"
        states = episode_state_trajectory(
            self._buffer,
            active,
            home_team_id=self._home,
            away_team_id=self._away,
            snapshot_hz=self._snapshot_hz,
            attacking_to_right=attacking_to_right,
        )
        record = EpisodeRecord(
            boundary=active,
            outcome=classify_outcome(active, self._buffer, attacking_to_right=attacking_to_right),
            state_trajectory=states,
            dominant_phase=None,
        )
        attr = compute_episode_attribution(
            record,
            self._buffer,
            self._home,
            self._away,
            rows=self._obso_rows,
            cols=self._obso_cols,
        )
        if attr is None:
            return
        ranked = sorted(attr.contributions.items(), key=lambda kv: -abs(kv[1]))
        self._latest_top_contributors = [{"player_id": p, "contribution": round(c, 4)} for p, c in ranked[:3]]
        self._latest_attribution_frame = frame_id

    def _empty_snapshot(self, frame_id: int, time_s: float) -> LiveSnapshot:
        return LiveSnapshot(
            frame_id=frame_id,
            time_s=time_s,
            possession_team=None,
            current_episode_id=-1,
            current_episode_phase=None,
            current_episode_duration_s=0.0,
            threat_level=0.0,
            threat_level_age_frames=0,
            predicted={},
            top_neighbors=[],
            prediction_age_frames=0,
            rolling_top_contributors=[],
            attribution_age_frames=0,
        )

"""Rules-based phase-of-play classifier (Phase 3, v1).

Labels every tracking frame as one of seven canonical phases:

    build_up          — attacker in possession, ball in its own defensive third
    progression       — attacker in possession, ball in the middle third
    finishing         — attacker in possession, ball in the attacking third
    def_transition    — first `transition_window_s` after losing possession
    att_transition    — first `transition_window_s` after winning possession
    settled_def       — opponent in settled possession, past the transition window,
                        not in own defensive third
    set_piece         — ball out of play (currently detected only by the caller)

Possession is inferred from the tracking frame itself: the team whose player is
nearest the ball holds possession *if* that distance is ≤ `possession_threshold_m`;
otherwise the frame is flagged as "unsettled" and typically resolves into one of
the transition phases via the time-window logic.

This is a lightweight pre-Phase-3-B rules model. A learned GBM (per the
architecture doc) can drop in with the same output contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M

# Canonical phase labels in the order they appear in analytics outputs.
PHASE_LABELS: tuple[str, ...] = (
    "build_up",
    "progression",
    "finishing",
    "def_transition",
    "att_transition",
    "settled_def",
    "set_piece",
)


@dataclass(frozen=True)
class PhaseConfig:
    """Knobs for the rules classifier.

    Production-data note: frame-by-frame "nearest player" flips wildly when two
    opponents contest a 50/50 ball — if we registered every flip as a possession
    change, virtually every frame would be in transition. Two guards tame this:

    - `dominance_margin_m`: the nearest player must be at least this much closer
      to the ball than the nearest opponent. A 50/50 where both are within 0.5m
      of each other registers as no-possession until the ball moves clearly to
      one side.
    - `possession_sticky_frames`: a new possession team must hold for at least
      this many consecutive frames before it counts as a real flip. Transient
      nearest-player swaps during a pass or deflection don't leak into the
      transition timer.
    """

    possession_threshold_m: float = 2.5
    dominance_margin_m: float = 1.0
    possession_sticky_frames: int = 5
    transition_window_s: float = 6.0
    defensive_third_m: float = PITCH_LENGTH_M / 3.0
    attacking_third_m: float = 2.0 * PITCH_LENGTH_M / 3.0


def _possession_per_frame_vectorized(
    tracking: pd.DataFrame, threshold_m: float, dominance_margin_m: float
) -> pd.DataFrame:
    """Vectorized replacement for ``_possession_per_frame``.

    Eliminates the per-frame Python ``groupby`` loop (~1.2 ms/frame on Metrica
    match 2) by doing all distance + min-per-team math via pandas merge +
    groupby on the long form. Roughly 50-100× faster on 60k+ frame matches.

    Behaviour parity with ``_possession_per_frame`` is enforced by
    ``test_classifier_vectorized_matches_legacy`` — when they disagree this
    function is the bug, not the legacy one.
    """
    # All frame_ids we need a row for (including frames with no ball or no players).
    # Use first time_seconds per frame as the canonical timestamp.
    frame_times = (
        tracking.groupby("frame_id", as_index=False).agg(time_seconds=("time_seconds", "first")).sort_values("frame_id")
    )

    # Ball: filter to visible + non-NaN coords, take first per frame.
    ball_rows = tracking[tracking["is_ball"] & tracking["visible"] & tracking["x"].notna() & tracking["y"].notna()]
    ball_per_frame = ball_rows.groupby("frame_id", as_index=False).agg(ball_x=("x", "first"), ball_y=("y", "first"))

    # Players: keep only those that can be assigned a distance.
    players = tracking[
        ~tracking["is_ball"]
        & tracking["visible"]
        & tracking["team_id"].notna()
        & tracking["x"].notna()
        & tracking["y"].notna()
    ]

    # Attach ball coords to each player row, compute distance.
    pp = players.merge(ball_per_frame, on="frame_id", how="inner")
    if pp.empty:
        # No frame has both visible ball + at least one valid player.
        out = frame_times.assign(possession_team=None, ball_x=np.nan, ball_y=np.nan)
        return out[["frame_id", "time_seconds", "possession_team", "ball_x", "ball_y"]].reset_index(drop=True)

    pp_dist = np.sqrt((pp["x"] - pp["ball_x"]) ** 2 + (pp["y"] - pp["ball_y"]) ** 2)
    pp = pp.assign(dist=pp_dist)

    # Per (frame, team), minimum distance.
    team_min = (
        pp.groupby(["frame_id", "team_id"], as_index=False)["dist"].min().rename(columns={"dist": "team_min_dist"})
    )

    # For dominance check: rank teams within each frame by min distance and pull rank-1 + rank-2.
    team_min["rank"] = team_min.groupby("frame_id")["team_min_dist"].rank(method="first")
    nearest = team_min[team_min["rank"] == 1.0][["frame_id", "team_id", "team_min_dist"]].rename(
        columns={"team_id": "nearest_team", "team_min_dist": "nearest_dist"}
    )
    second = team_min[team_min["rank"] == 2.0][["frame_id", "team_min_dist"]].rename(
        columns={"team_min_dist": "opp_dist"}
    )

    # Frames with only one team present in possession candidates → opp_dist = +inf
    poss = nearest.merge(second, on="frame_id", how="left")
    poss["opp_dist"] = poss["opp_dist"].fillna(np.inf)

    # Apply dominance criteria.
    valid = (poss["nearest_dist"] <= threshold_m) & ((poss["opp_dist"] - poss["nearest_dist"]) >= dominance_margin_m)
    poss["possession_team"] = poss["nearest_team"].where(valid, other=None).astype(object)

    # Final shape: one row per frame, with ball coords (NaN when ball missing) + possession.
    out = frame_times.merge(ball_per_frame, on="frame_id", how="left").merge(
        poss[["frame_id", "possession_team"]], on="frame_id", how="left"
    )
    # Frames with no possession candidates fall through with NaN possession_team — make
    # those None to match the legacy contract.
    out["possession_team"] = out["possession_team"].astype(object).where(out["possession_team"].notna(), other=None)
    return out[["frame_id", "time_seconds", "possession_team", "ball_x", "ball_y"]].reset_index(drop=True)


def _possession_per_frame(tracking: pd.DataFrame, threshold_m: float, dominance_margin_m: float) -> pd.DataFrame:
    """For each frame, return (frame_id, time_seconds, possession_team, ball_x, ball_y).

    `possession_team` is the team id of the nearest outfielder to the ball iff:
    - their distance is ≤ threshold_m, AND
    - the nearest opponent is at least `dominance_margin_m` farther away.

    Returns `None` when the ball is loose (no dominant possessor).
    """
    out_rows: list[dict[str, object]] = []

    for frame_id, fdf in tracking.groupby("frame_id", sort=True):
        # Ball must be visible AND have non-NaN coords. Real-world tracking
        # sometimes flags a frame visible while the position columns are NaN
        # (Metrica match 3 has frames with ball.visible=True but x/y=NaN);
        # those frames must be treated as no-ball, otherwise the distance
        # computation below produces an all-NaN series.
        ball_rows = fdf[fdf["is_ball"] & fdf["visible"] & fdf["x"].notna() & fdf["y"].notna()]
        if ball_rows.empty:
            out_rows.append(
                {
                    "frame_id": int(frame_id),
                    "time_seconds": float(fdf["time_seconds"].iloc[0]),
                    "possession_team": None,
                    "ball_x": np.nan,
                    "ball_y": np.nan,
                }
            )
            continue
        ball = ball_rows.iloc[0]
        # Drop players with NaN coords — they can't be assigned a distance.
        # In real-world tracking some frames have full dropouts (Metrica match 3
        # has ~46k such frames) which would make idxmin() raise on an all-NaN series.
        players = fdf[
            ~fdf["is_ball"] & fdf["visible"] & fdf["team_id"].notna() & fdf["x"].notna() & fdf["y"].notna()
        ].copy()
        if players.empty:
            out_rows.append(
                {
                    "frame_id": int(frame_id),
                    "time_seconds": float(ball["time_seconds"]),
                    "possession_team": None,
                    "ball_x": float(ball["x"]),
                    "ball_y": float(ball["y"]),
                }
            )
            continue

        players["dist"] = np.sqrt((players["x"] - float(ball["x"])) ** 2 + (players["y"] - float(ball["y"])) ** 2)
        nearest_idx = int(players["dist"].idxmin())
        nearest = players.loc[nearest_idx]
        possession_team: str | None = None
        if nearest["dist"] <= threshold_m:
            opponents = players[players["team_id"] != nearest["team_id"]]
            opp_closest = float(opponents["dist"].min()) if not opponents.empty else float("inf")
            if opp_closest - float(nearest["dist"]) >= dominance_margin_m:
                possession_team = str(nearest["team_id"])
        out_rows.append(
            {
                "frame_id": int(frame_id),
                "time_seconds": float(ball["time_seconds"]),
                "possession_team": possession_team,
                "ball_x": float(ball["x"]),
                "ball_y": float(ball["y"]),
            }
        )

    return pd.DataFrame(out_rows).sort_values("frame_id").reset_index(drop=True)


def _apply_possession_stickiness(poss: pd.DataFrame, sticky_frames: int) -> pd.Series:
    """Smooth instantaneous possession flips — a new team must hold for
    `sticky_frames` consecutive frames before the canonical label flips."""
    raw = poss["possession_team"].tolist()
    smoothed: list[str | None] = []
    current: str | None = None
    streak: str | None = None
    streak_len = 0

    for team in raw:
        if team is None:
            smoothed.append(current)
            streak_len = 0
            streak = None
            continue
        if team == current:
            smoothed.append(current)
            streak_len = 0
            streak = None
            continue
        if team == streak:
            streak_len += 1
        else:
            streak = team
            streak_len = 1
        if streak_len >= sticky_frames:
            current = team
            smoothed.append(current)
            streak_len = 0
            streak = None
        else:
            smoothed.append(current)
    return pd.Series(smoothed, index=poss.index)


def classify_frames(
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    config: PhaseConfig | None = None,
) -> pd.DataFrame:
    """Label every frame with a canonical phase.

    Args:
        tracking: canonical long-form tracking DataFrame.
        home_team_id / away_team_id: team ids as they appear in `tracking["team_id"]`.
        config: override defaults.

    Returns:
        DataFrame with columns `frame_id`, `time_seconds`, `possession_team`, `phase`.
    """
    cfg = config or PhaseConfig()
    # The vectorized path is ~250x faster than the per-frame Python loop on
    # real-world matches (Metrica match 2 P1: 99.5s -> 0.4s). Legacy retained
    # under ``_possession_per_frame`` for parity testing + incident debugging.
    poss = _possession_per_frame_vectorized(tracking, cfg.possession_threshold_m, cfg.dominance_margin_m)
    smoothed = _apply_possession_stickiness(poss, cfg.possession_sticky_frames)

    phases: list[str] = []
    last_possession: str | None = None
    last_flip_time: float = -float("inf")

    for row, pt_smooth in zip(poss.itertuples(index=False), smoothed, strict=True):
        pt = pt_smooth  # use smoothed possession for settled classification
        t = row.time_seconds
        bx = row.ball_x
        # Track possession flips for transition timing
        if pt is not None and pt != last_possession:
            last_flip_time = t
            last_possession = pt

        # No one in possession + no recent flip = set_piece proxy (loose ball for long time)
        if pt is None and (t - last_flip_time) > cfg.transition_window_s:
            phases.append("set_piece")
            continue
        # Unsettled ball inside the transition window: classify from recent possession
        effective_possessor = pt if pt is not None else last_possession
        if effective_possessor is None:
            phases.append("set_piece")
            continue

        in_transition = (t - last_flip_time) <= cfg.transition_window_s
        if in_transition:
            # Was the transition gained or lost from the home team's point of view?
            # For the current possessor, "att_transition" = they just won the ball;
            # for the other team the same frame is "def_transition" — but the label
            # is keyed on the team in possession, so it's always "att_transition"
            # from their perspective. We emit att_transition when the possessor is
            # the attacking team and this frame is within the window from the flip.
            # def_transition fires when no-one is in possession yet (loose ball
            # immediately after a turnover).
            phases.append("att_transition" if pt is not None else "def_transition")
            continue

        # Settled: phase depends on ball-x relative to the possessor's own goal.
        # Home attacks +x in the canonical frame, so home's own third is x < L/3 and
        # attacking third is x > 2L/3. For away the mapping is mirrored.
        if effective_possessor == home_team_id:
            own_third = bx < cfg.defensive_third_m
            attacking_third = bx > cfg.attacking_third_m
        else:
            own_third = bx > cfg.attacking_third_m  # away's own third is the right side
            attacking_third = bx < cfg.defensive_third_m

        if own_third:
            phases.append("build_up")
        elif attacking_third:
            phases.append("finishing")
        else:
            phases.append("progression")

    poss["phase"] = phases
    poss["possession_team"] = smoothed
    return poss[["frame_id", "time_seconds", "possession_team", "phase"]]


def segment_phases(classified: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive identical phase labels into `(phase, start_frame, end_frame, duration_s)` rows."""
    if classified.empty:
        return pd.DataFrame(columns=["phase", "start_frame", "end_frame", "start_time", "end_time", "duration_s"])

    runs: list[dict[str, object]] = []
    current_phase = classified["phase"].iloc[0]
    start_frame = int(classified["frame_id"].iloc[0])
    start_time = float(classified["time_seconds"].iloc[0])
    prev_frame = start_frame
    prev_time = start_time

    for row in classified.iloc[1:].itertuples(index=False):
        if row.phase == current_phase:
            prev_frame = int(row.frame_id)
            prev_time = float(row.time_seconds)
            continue
        runs.append(
            {
                "phase": current_phase,
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "start_time": start_time,
                "end_time": prev_time,
                "duration_s": round(prev_time - start_time, 3),
            }
        )
        current_phase = row.phase
        start_frame = int(row.frame_id)
        start_time = float(row.time_seconds)
        prev_frame = start_frame
        prev_time = start_time

    runs.append(
        {
            "phase": current_phase,
            "start_frame": start_frame,
            "end_frame": prev_frame,
            "start_time": start_time,
            "end_time": prev_time,
            "duration_s": round(prev_time - start_time, 3),
        }
    )
    return pd.DataFrame(runs)

"""Markov Temporal-Difference attribution for episode possession credit.

Per-player credit at each snapshot ``t`` is the temporal-difference of OBSO-max
weighted by the player's involvement at snapshot ``t``:

.. math::

    \\delta_t = \\text{obso\\_max}[t+1] - \\text{obso\\_max}[t]
    \\text{credit}[p] \\mathrel{+}= \\delta_t \\cdot \\psi_p(t)

where ``\\psi_p(t)`` is the involvement weight:

* **1.0** if player ``p`` is the ball carrier (closest possessing-team player to
  the ball) at snapshot ``t``.
* ``near_ball_weight`` (default ``0.3``) if ``p`` is a possessing-team player
  within ``near_ball_radius_m`` (default ``10.0``) of the ball.
* **0** otherwise (defenders never get attacking credit, far attackers get none).

This is intentionally a baseline — much simpler than the full Liu et al. (2020)
DRL approach. The point is to (a) be a drop-in alternative to ``contribution.py``
and (b) avoid the *decoy-run inversion* failure mode of leave-one-out: a player
who pulls a defender to open space gets removed from the LOO counterfactual,
and the local OBSO often *increases*, so LOO wrongly assigns the decoy negative
credit. Markov-TD never assigns negative credit to a player on a positive-delta
snapshot, because :math:`\\psi_p(t) \\geq 0` for all ``p``.

Public API:
    attribute_episode_markov_td  — single-episode attribution
    attribute_pattern            — aggregate over a cluster of episodes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord


@dataclass(frozen=True)
class PlayerCredit:
    """Single player's accumulated TD credit on one episode."""

    player_id: str
    team_id: str
    credit: float
    n_snapshots_carrier: int
    n_snapshots_near_ball: int


@dataclass(frozen=True)
class AttributionResult:
    """Per-episode Markov-TD attribution."""

    episode_id: int
    credits: list[PlayerCredit]  # sorted by credit desc


def _carrier_at_frame(
    frame_rows: pd.DataFrame,
    possessing_team: str,
    ball_x: float,
    ball_y: float,
) -> str | None:
    """Closest visible possessing-team player to the ball at this frame, or None."""
    attackers = frame_rows[
        (frame_rows["team_id"] == possessing_team)
        & ~frame_rows["is_ball"]
        & frame_rows["visible"]
    ]
    if attackers.empty:
        return None
    dx = attackers["x"].to_numpy(dtype=np.float64) - ball_x
    dy = attackers["y"].to_numpy(dtype=np.float64) - ball_y
    d2 = dx * dx + dy * dy
    return str(attackers.iloc[int(np.argmin(d2))]["player_id"])


def _ball_xy(frame_rows: pd.DataFrame) -> tuple[float, float] | None:
    """Visible ball position at this frame, or None if missing/invisible."""
    ball = frame_rows[frame_rows["is_ball"] & frame_rows["visible"]]
    if ball.empty:
        return None
    return float(ball.iloc[0]["x"]), float(ball.iloc[0]["y"])


def attribute_episode_markov_td(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    obso_trajectory: pd.DataFrame | None = None,
    near_ball_radius_m: float = 10.0,
    near_ball_weight: float = 0.3,
    use_gpu: bool = False,
) -> AttributionResult:
    """Markov-TD attribution for a single episode.

    Args:
        record: ``EpisodeRecord`` from ``build_episodes``.
        tracking: canonical tracking DataFrame for the match.
        home_team_id, away_team_id: team identifiers as in ``tracking``.
        obso_trajectory: optional precomputed OBSO trajectory DataFrame
            (columns ``frame_id, obso_max`` at minimum). When ``None``, the
            module lazy-imports ``compute_obso_trajectory`` and runs it.
            Pass a hand-crafted DataFrame in tests for determinism.
        near_ball_radius_m: distance from the ball within which a possessing-
            team player is considered "involved" with weight ``near_ball_weight``.
        near_ball_weight: involvement weight for near-ball players (the carrier
            always uses weight ``1.0``).
        use_gpu: forwarded to ``compute_obso_trajectory`` when computing OBSO
            on the fly.

    Returns:
        ``AttributionResult`` with credits sorted by credit desc. Empty credits
        list for episodes with no state trajectory, fewer than two snapshots,
        or no usable OBSO trajectory.
    """
    states = record.state_trajectory
    if states.empty:
        return AttributionResult(episode_id=record.boundary.episode_id, credits=[])

    if obso_trajectory is None:
        # Lazy import to avoid a circular module load (engine.py also imports this
        # path indirectly via _enrich_with_obso_outcome).
        from football_analysis.analytics.episodes import (  # noqa: PLC0415
            obso_trajectory as ot_module,
        )

        obso_trajectory = ot_module.compute_obso_trajectory(
            record, tracking, home_team_id, away_team_id, use_gpu=use_gpu,
        )

    if obso_trajectory is None or obso_trajectory.empty:
        return AttributionResult(episode_id=record.boundary.episode_id, credits=[])

    # Build a frame_id → obso_max lookup. We use OBSO trajectory's frames as the
    # ordered timeline (state_trajectory may have more frames than OBSO if the
    # OBSO compute dropped any).
    obso_sorted = (
        obso_trajectory[["frame_id", "obso_max"]]
        .dropna()
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    if len(obso_sorted) < 2:
        return AttributionResult(episode_id=record.boundary.episode_id, credits=[])

    possessing_team = record.boundary.possession_team
    # The defending team isn't used directly — defenders never get credit — but
    # we resolve it for completeness and parity with the LOO API surface.
    _ = away_team_id if possessing_team == home_team_id else home_team_id

    # Per-player aggregates.
    credits: dict[str, float] = {}
    team_of: dict[str, str] = {}
    n_carrier: dict[str, int] = {}
    n_near: dict[str, int] = {}

    radius_sq = near_ball_radius_m * near_ball_radius_m

    # Walk pairs (t, t+1) in OBSO order. Involvement is taken at snapshot t.
    for i in range(len(obso_sorted) - 1):
        f_t = int(obso_sorted.iloc[i]["frame_id"])
        f_next = int(obso_sorted.iloc[i + 1]["frame_id"])
        delta = float(obso_sorted.iloc[i + 1]["obso_max"]) - float(
            obso_sorted.iloc[i]["obso_max"]
        )
        if delta == 0.0:
            # Still walk so n_carrier/n_near reflect involvement, but no credit.
            pass

        frame_rows = tracking[tracking["frame_id"] == f_t]
        if frame_rows.empty:
            continue

        ball = _ball_xy(frame_rows)
        if ball is None:
            # Ball invisible — no involvement structure to compute. Skip snapshot.
            continue
        bx, by = ball

        carrier_id = _carrier_at_frame(frame_rows, possessing_team, bx, by)

        attackers = frame_rows[
            (frame_rows["team_id"] == possessing_team)
            & ~frame_rows["is_ball"]
            & frame_rows["visible"]
        ]
        if attackers.empty:
            continue

        for _, row in attackers.iterrows():
            pid = str(row["player_id"])
            tid = str(row["team_id"])
            team_of.setdefault(pid, tid)
            credits.setdefault(pid, 0.0)
            n_carrier.setdefault(pid, 0)
            n_near.setdefault(pid, 0)

            if pid == carrier_id:
                weight = 1.0
                n_carrier[pid] += 1
            else:
                dx = float(row["x"]) - bx
                dy = float(row["y"]) - by
                if dx * dx + dy * dy <= radius_sq:
                    weight = near_ball_weight
                    n_near[pid] += 1
                else:
                    weight = 0.0

            if weight != 0.0 and delta != 0.0:
                credits[pid] += weight * delta

        # Note frame f_next is consumed only as the value-difference target —
        # involvement is keyed off snapshot t, which matches the standard TD
        # update where psi(t) gates the bootstrap from V(t+1).
        del f_next  # explicit: not used past delta-computation above

    # Drop players whose total credit is exactly zero AND who never had any
    # involvement — keeps the result tight for downstream consumers. We retain
    # players with non-zero involvement counts but zero credit (e.g. all deltas
    # were zero), since their counts may still be useful diagnostics.
    out: list[PlayerCredit] = []
    for pid, c in credits.items():
        carrier_n = n_carrier.get(pid, 0)
        near_n = n_near.get(pid, 0)
        if c == 0.0 and carrier_n == 0 and near_n == 0:
            continue
        out.append(
            PlayerCredit(
                player_id=pid,
                team_id=team_of[pid],
                credit=c,
                n_snapshots_carrier=carrier_n,
                n_snapshots_near_ball=near_n,
            )
        )

    # Stable sort by credit desc, then by player_id asc for deterministic ties.
    out.sort(key=lambda c: (-c.credit, c.player_id))

    return AttributionResult(episode_id=record.boundary.episode_id, credits=out)


def attribute_pattern(
    records: list[EpisodeRecord],
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    cluster_label_for: dict[int, int],
    target_cluster: int,
    **kwargs,
) -> dict[str, float]:
    """Aggregate per-player credit across all episodes in one cluster pattern.

    Episodes whose ``episode_id`` is missing from ``cluster_label_for`` are
    silently skipped.

    Args:
        records: list of ``EpisodeRecord`` to consider.
        tracking: canonical tracking DataFrame covering all episodes.
        home_team_id, away_team_id: team identifiers.
        cluster_label_for: mapping ``episode_id → cluster_id`` (e.g. from
            ``ClusterResult.cluster_labels`` zipped with ``episode_ids``).
        target_cluster: only episodes labelled with this id are aggregated.
        **kwargs: forwarded to ``attribute_episode_markov_td`` (e.g.
            ``near_ball_weight``, ``near_ball_radius_m``, ``use_gpu``).

    Returns:
        ``player_id → total_credit`` summed across episodes in the cluster.
    """
    totals: dict[str, float] = {}
    for rec in records:
        eid = rec.boundary.episode_id
        if cluster_label_for.get(eid) != target_cluster:
            continue
        # Re-resolve the global symbol so unit tests can monkeypatch
        # ``attribute_episode_markov_td`` and have ``attribute_pattern`` pick
        # up the patched function.
        from football_analysis.analytics.episodes import markov_td as _self  # noqa: PLC0415

        result = _self.attribute_episode_markov_td(
            rec, tracking, home_team_id, away_team_id, **kwargs
        )
        for pc in result.credits:
            totals[pc.player_id] = totals.get(pc.player_id, 0.0) + pc.credit
    return totals

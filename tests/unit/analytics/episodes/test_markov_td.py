"""Tests for Markov-TD attribution.

Markov-TD credits each player p at snapshot t by the temporal-difference of
OBSO-max, weighted by the player's involvement at that snapshot:

    delta_t = obso_max[t+1] - obso_max[t]
    credit[p] += delta_t * psi_p(t)

Where psi_p(t) is 1.0 for the ball carrier on the possessing team, ``alpha``
(default 0.3) for any other possessing-team player within ``near_ball_radius_m``
of the ball, and 0 otherwise.

This module is the M3 replacement for the leave-one-out approach in
``contribution.py``. The decoy-run inversion test below is the LOO failure
mode it must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.markov_td import (
    AttributionResult,
    PlayerCredit,
    attribute_episode_markov_td,
    attribute_pattern,
)
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _boundary(episode_id: int, start_frame: int, end_frame: int) -> EpisodeBoundary:
    return EpisodeBoundary(
        episode_id=episode_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=start_frame / 25.0,
        end_time_s=end_frame / 25.0,
        duration_s=(end_frame - start_frame) / 25.0,
        possession_team="home",
        end_reason="match_end",
    )


def _outcome(episode_id: int) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        end_reason="match_end",
        reached_final_third=False,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=50.0,
        end_ball_y=34.0,
        end_ball_speed=0.0,
        duration_s=0.4,
    )


def _state_trajectory(frame_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": [1] * len(frame_ids),
            "frame_id": frame_ids,
            "time_s": [f / 25.0 for f in frame_ids],
            "rel_time_s": [round((f - frame_ids[0]) / 25.0, 3) for f in frame_ids],
            "ball_x": [50.0] * len(frame_ids),
            "ball_y": [34.0] * len(frame_ids),
            "ball_x_oriented": [50.0] * len(frame_ids),
            "ball_speed": [0.0] * len(frame_ids),
            "attackers_visible": [11] * len(frame_ids),
            "defenders_visible": [11] * len(frame_ids),
            "attackers_mean_x_oriented": [50.0] * len(frame_ids),
            "attackers_length": [40.0] * len(frame_ids),
            "attackers_width": [50.0] * len(frame_ids),
            "defenders_mean_x_oriented": [50.0] * len(frame_ids),
            "defenders_line_height_oriented": [40.0] * len(frame_ids),
            "defenders_compactness_x": [3.0] * len(frame_ids),
        }
    )


def _obso_trajectory(frame_ids: list[int], obso_values: list[float]) -> pd.DataFrame:
    """Hand-crafted OBSO trajectory for deterministic unit tests."""
    return pd.DataFrame(
        {
            "frame_id": frame_ids,
            "time_s": [f / 25.0 for f in frame_ids],
            "rel_time_s": [round((f - frame_ids[0]) / 25.0, 3) for f in frame_ids],
            "obso_max": obso_values,
            "obso_argmax_x": [95.0] * len(frame_ids),
            "obso_argmax_y": [34.0] * len(frame_ids),
        }
    )


def _tracking_row(
    frame_id: int,
    player_id: str,
    team_id: str | None,
    x: float,
    y: float,
    is_ball: bool = False,
    visible: bool = True,
) -> dict:
    return {
        "frame_id": frame_id,
        "period": 1,
        "time_seconds": frame_id / 25.0,
        "player_id": player_id,
        "team_id": team_id,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "is_ball": is_ball,
        "visible": visible,
    }


# ---------------------------------------------------------------------------
# Single-carrier scenario
# ---------------------------------------------------------------------------


def test_single_carrier_gets_all_credit() -> None:
    """One player is the ball carrier through the whole episode → they get the
    full delta-OBSO sum. Other home players are far away and on defense, so they
    get zero credit."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    for f in frames:
        # Ball at (50, 34)
        rows.append(_tracking_row(f, "ball", None, 50.0, 34.0, is_ball=True))
        # Carrier glued to ball
        rows.append(_tracking_row(f, "home_carrier", "home", 50.5, 34.0))
        # Far home teammate (>10 m from ball)
        rows.append(_tracking_row(f, "home_far", "home", 10.0, 10.0))
        # Defender — not credited regardless
        rows.append(_tracking_row(f, "away_def", "away", 60.0, 34.0))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase="progression",
    )
    obso = _obso_trajectory(frames, [0.10, 0.20, 0.40])  # deltas: +0.10, +0.20

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )

    assert isinstance(result, AttributionResult)
    assert result.episode_id == 1
    credit_by_id = {c.player_id: c.credit for c in result.credits}
    # Carrier gets sum of deltas
    assert credit_by_id["home_carrier"] == pytest.approx(0.30)
    # Far-away home player gets zero
    assert credit_by_id.get("home_far", 0.0) == pytest.approx(0.0)
    # Defender never appears
    assert "away_def" not in credit_by_id


# ---------------------------------------------------------------------------
# Two players sharing carry
# ---------------------------------------------------------------------------


def test_two_players_share_carry_equally() -> None:
    """Player A carries snapshot 1, player B carries snapshot 2 → equal credit
    when deltas are symmetric."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    # snapshot 1 (frame 1): A on ball, B far
    rows.append(_tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True))
    rows.append(_tracking_row(1, "home_A", "home", 50.5, 34.0))
    rows.append(_tracking_row(1, "home_B", "home", 0.0, 0.0))  # >10 m from ball
    # snapshot 2 (frame 2): B on ball, A far
    rows.append(_tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True))
    rows.append(_tracking_row(2, "home_A", "home", 0.0, 0.0))
    rows.append(_tracking_row(2, "home_B", "home", 50.5, 34.0))
    # snapshot 3 (frame 3): doesn't matter (no t+1)
    rows.append(_tracking_row(3, "ball", None, 50.0, 34.0, is_ball=True))
    rows.append(_tracking_row(3, "home_A", "home", 0.0, 0.0))
    rows.append(_tracking_row(3, "home_B", "home", 0.0, 0.0))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase="progression",
    )
    # Symmetric deltas: +0.10 then +0.10
    obso = _obso_trajectory(frames, [0.10, 0.20, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    credit_by_id = {c.player_id: c.credit for c in result.credits}
    assert credit_by_id["home_A"] == pytest.approx(0.10)
    assert credit_by_id["home_B"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Zero-delta episode → zero credit
# ---------------------------------------------------------------------------


def test_zero_delta_episode_yields_zero_credit() -> None:
    """If OBSO never changes, credit is identically zero for every player."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    for f in frames:
        rows.append(_tracking_row(f, "ball", None, 50.0, 34.0, is_ball=True))
        rows.append(_tracking_row(f, "home_carrier", "home", 50.5, 34.0))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase="progression",
    )
    obso = _obso_trajectory(frames, [0.20, 0.20, 0.20])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    for c in result.credits:
        assert c.credit == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Decoy-run inversion: the LOO failure mode we beat
# ---------------------------------------------------------------------------


def test_decoy_run_gets_non_negative_credit() -> None:
    """A player makes a decoy run that pulls a defender, opening space for the
    ball-carrier. The ball stays with the carrier. Under leave-one-out, removing
    the decoy player would *increase* the local OBSO (because the defender
    they dragged would be elsewhere), so LOO assigns negative credit. Markov-TD
    instead credits the decoy via near-ball weight and a positive delta_t.

    We construct this by placing the decoy near the ball during a positive-delta
    snapshot, so the decoy collects positive credit through the near-ball weight."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    for f in frames:
        rows.append(_tracking_row(f, "ball", None, 50.0, 34.0, is_ball=True))
        rows.append(_tracking_row(f, "home_carrier", "home", 50.5, 34.0))
        # Decoy starts near the ball (within 10 m), then bursts away
        decoy_x = 55.0 if f <= 2 else 90.0
        decoy_y = 36.0 if f <= 2 else 36.0
        rows.append(_tracking_row(f, "home_decoy", "home", decoy_x, decoy_y))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase="progression",
    )
    # Big positive delta from snap 1 → 2 (when decoy is near ball)
    obso = _obso_trajectory(frames, [0.10, 0.50, 0.55])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    credit_by_id = {c.player_id: c.credit for c in result.credits}
    # Decoy must get strictly positive (not negative — that's the LOO failure).
    assert credit_by_id["home_decoy"] > 0.0
    # Carrier still gets more credit (full weight 1.0 vs near-ball 0.3).
    assert credit_by_id["home_carrier"] > credit_by_id["home_decoy"]


# ---------------------------------------------------------------------------
# Empty episode
# ---------------------------------------------------------------------------


def test_empty_state_trajectory_returns_empty_credits() -> None:
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=pd.DataFrame(),
        dominant_phase=None,
    )
    result = attribute_episode_markov_td(
        record,
        pd.DataFrame(),
        home_team_id="home",
        away_team_id="away",
        obso_trajectory=pd.DataFrame(),
    )
    assert isinstance(result, AttributionResult)
    assert result.credits == []
    assert result.episode_id == 1


def test_single_snapshot_returns_empty_credits() -> None:
    """One snapshot → no t+1, so no TD update → empty credits."""
    frames = [1]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
    ]
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 1),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.20])
    result = attribute_episode_markov_td(
        record, pd.DataFrame(rows), home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    assert result.credits == []


# ---------------------------------------------------------------------------
# AttributionResult sorted by credit desc
# ---------------------------------------------------------------------------


def test_credits_sorted_by_credit_descending() -> None:
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(1, "home_near", "home", 53.0, 34.0),  # within 10 m
        _tracking_row(1, "home_far", "home", 0.0, 0.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "home_near", "home", 53.0, 34.0),
        _tracking_row(2, "home_far", "home", 0.0, 0.0),
    ]
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    credits = result.credits
    # Strictly non-increasing
    for i in range(len(credits) - 1):
        assert credits[i].credit >= credits[i + 1].credit
    # Carrier first, near second (far player has 0 → not in result, since LOO
    # never touched them either, but we keep them out for cleanliness)
    assert credits[0].player_id == "home_carrier"


# ---------------------------------------------------------------------------
# PlayerCredit metadata
# ---------------------------------------------------------------------------


def test_player_credit_records_carrier_and_near_counts() -> None:
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(1, "home_near", "home", 53.0, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "home_near", "home", 53.0, 34.0),
    ]
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    by_id = {c.player_id: c for c in result.credits}
    # Only the t = 0 snapshot contributes (we sum involvement across t with delta).
    assert by_id["home_carrier"].n_snapshots_carrier >= 1
    assert by_id["home_near"].n_snapshots_near_ball >= 1
    assert by_id["home_carrier"].team_id == "home"


# ---------------------------------------------------------------------------
# Custom near-ball weight + radius
# ---------------------------------------------------------------------------


def test_near_ball_weight_scales_credit() -> None:
    """Doubling near_ball_weight from 0.3 → 0.6 doubles the near-ball player's
    credit (carrier credit is unchanged)."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(1, "home_near", "home", 53.0, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "home_near", "home", 53.0, 34.0),
    ]
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])
    res_a = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away",
        obso_trajectory=obso, near_ball_weight=0.3,
    )
    res_b = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away",
        obso_trajectory=obso, near_ball_weight=0.6,
    )
    near_a = next(c for c in res_a.credits if c.player_id == "home_near").credit
    near_b = next(c for c in res_b.credits if c.player_id == "home_near").credit
    assert near_b == pytest.approx(near_a * 2.0)


def test_near_ball_radius_excludes_player_outside() -> None:
    """A player at 12 m is outside the default 10 m radius → no credit."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(1, "home_outside", "home", 62.0, 34.0),  # 12 m away
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "home_outside", "home", 62.0, 34.0),
    ]
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away",
        obso_trajectory=obso,
    )
    by_id = {c.player_id: c.credit for c in result.credits}
    assert by_id.get("home_outside", 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Away-team possession path
# ---------------------------------------------------------------------------


def test_away_team_possession_credits_away_players() -> None:
    """When possession_team == away_team_id, credit goes to away players."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "away_carrier", "away", 50.5, 34.0),
        _tracking_row(1, "home_def", "home", 60.0, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "away_carrier", "away", 50.5, 34.0),
        _tracking_row(2, "home_def", "home", 60.0, 34.0),
    ]
    tracking = pd.DataFrame(rows)
    boundary = EpisodeBoundary(
        episode_id=1, start_frame=1, end_frame=2,
        start_time_s=0.04, end_time_s=0.08, duration_s=0.04,
        possession_team="away", end_reason="match_end",
    )
    record = EpisodeRecord(
        boundary=boundary,
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away",
        obso_trajectory=obso,
    )
    by_id = {c.player_id: c.credit for c in result.credits}
    assert by_id["away_carrier"] > 0.0
    assert "home_def" not in by_id


# ---------------------------------------------------------------------------
# Lazy-import path: obso_trajectory=None
# ---------------------------------------------------------------------------


def test_lazy_import_calls_compute_obso_trajectory(monkeypatch) -> None:
    """When obso_trajectory is None, the module lazy-imports compute_obso_trajectory."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
    ]
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )

    fake_obso = _obso_trajectory(frames, [0.10, 0.30])
    call_count = {"n": 0}

    def fake_compute(record_arg, tracking_arg, home, away, use_gpu=False):
        call_count["n"] += 1
        return fake_obso

    import football_analysis.analytics.episodes.obso_trajectory as ot
    monkeypatch.setattr(ot, "compute_obso_trajectory", fake_compute)

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away",
        obso_trajectory=None,
    )
    assert call_count["n"] == 1
    assert any(c.player_id == "home_carrier" and c.credit > 0 for c in result.credits)


# ---------------------------------------------------------------------------
# attribute_pattern aggregation
# ---------------------------------------------------------------------------


def test_attribute_pattern_sums_credits_across_cluster() -> None:
    """Two episodes in the same cluster → aggregate per-player credits sum."""
    # Episode 1 frames
    frames_1 = [1, 2]
    rows_1 = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
    ]
    rec_1 = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames_1),
        dominant_phase=None,
    )
    obso_1 = _obso_trajectory(frames_1, [0.10, 0.30])

    # Episode 2 frames (different range)
    frames_2 = [10, 11]
    rows_2 = [
        _tracking_row(10, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(10, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(11, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(11, "home_carrier", "home", 50.5, 34.0),
    ]
    rec_2 = EpisodeRecord(
        boundary=_boundary(2, 10, 11),
        outcome=_outcome(2),
        state_trajectory=_state_trajectory(frames_2),
        dominant_phase=None,
    )
    obso_2 = _obso_trajectory(frames_2, [0.05, 0.15])

    # Episode 3 in a different cluster — must be excluded
    frames_3 = [20, 21]
    rows_3 = [
        _tracking_row(20, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(20, "home_other", "home", 50.5, 34.0),
        _tracking_row(21, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(21, "home_other", "home", 50.5, 34.0),
    ]
    rec_3 = EpisodeRecord(
        boundary=_boundary(3, 20, 21),
        outcome=_outcome(3),
        state_trajectory=_state_trajectory(frames_3),
        dominant_phase=None,
    )
    obso_3 = _obso_trajectory(frames_3, [0.10, 0.99])

    tracking = pd.DataFrame(rows_1 + rows_2 + rows_3)

    # Build a custom obso provider keyed by episode_id
    obso_by_episode = {1: obso_1, 2: obso_2, 3: obso_3}

    def per_episode_obso(rec, *_args, **_kwargs):
        return obso_by_episode[rec.boundary.episode_id]

    import football_analysis.analytics.episodes.markov_td as mtd

    # We patch attribute_episode_markov_td to use the right OBSO for each record.
    original = mtd.attribute_episode_markov_td

    def patched(rec, tracking_arg, home, away, **kwargs):
        kwargs["obso_trajectory"] = obso_by_episode[rec.boundary.episode_id]
        return original(rec, tracking_arg, home, away, **kwargs)

    label_for = {1: 0, 2: 0, 3: 1}  # epis 1,2 in cluster 0; ep 3 in cluster 1
    import unittest.mock as mock
    with mock.patch.object(mtd, "attribute_episode_markov_td", side_effect=patched):
        totals = attribute_pattern(
            [rec_1, rec_2, rec_3],
            tracking,
            home_team_id="home",
            away_team_id="away",
            cluster_label_for=label_for,
            target_cluster=0,
        )

    # Episode 1: carrier gets 0.20.  Episode 2: carrier gets 0.10. Sum = 0.30.
    assert totals["home_carrier"] == pytest.approx(0.30)
    # home_other from episode 3 is excluded
    assert "home_other" not in totals


def test_attribute_pattern_empty_cluster_returns_empty_dict() -> None:
    """No episodes belong to the target cluster → empty dict."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
    ]
    rec = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    totals = attribute_pattern(
        [rec],
        pd.DataFrame(rows),
        home_team_id="home",
        away_team_id="away",
        cluster_label_for={1: 0},
        target_cluster=99,  # no match
    )
    assert totals == {}


def test_attribute_pattern_skips_episodes_not_in_label_map() -> None:
    """Episodes missing from cluster_label_for are silently skipped."""
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
    ]
    rec = EpisodeRecord(
        boundary=_boundary(7, 1, 2),
        outcome=_outcome(7),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.30])

    # Episode 7 is not in the map at all → skipped
    totals = attribute_pattern(
        [rec],
        pd.DataFrame(rows),
        home_team_id="home",
        away_team_id="away",
        cluster_label_for={},
        target_cluster=0,
        obso_trajectory=obso,
    )
    assert totals == {}


# ---------------------------------------------------------------------------
# OBSO-trajectory shorter than state_trajectory: gracefully merge by frame_id
# ---------------------------------------------------------------------------


def test_obso_trajectory_subset_of_state_trajectory_handled() -> None:
    """If OBSO trajectory is missing some snapshot frames (compute can drop frames
    in error paths), we still produce attribution for the frames where OBSO exists."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    for f in frames:
        rows.append(_tracking_row(f, "ball", None, 50.0, 34.0, is_ball=True))
        rows.append(_tracking_row(f, "home_carrier", "home", 50.5, 34.0))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    # OBSO only has frames 1 and 3 (frame 2 dropped).
    obso = _obso_trajectory([1, 3], [0.10, 0.30])

    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    by_id = {c.player_id: c.credit for c in result.credits}
    # delta = 0.30 - 0.10 = 0.20 → carrier gets full 0.20
    assert by_id["home_carrier"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Empty obso_trajectory (race condition where obso compute returned empty)
# ---------------------------------------------------------------------------


def test_empty_obso_trajectory_returns_empty_credits() -> None:
    frames = [1, 2]
    rows = [
        _tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(1, "home_carrier", "home", 50.5, 34.0),
        _tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True),
        _tracking_row(2, "home_carrier", "home", 50.5, 34.0),
    ]
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 2),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    result = attribute_episode_markov_td(
        record, pd.DataFrame(rows), home_team_id="home", away_team_id="away",
        obso_trajectory=pd.DataFrame(columns=["frame_id", "obso_max"]),
    )
    assert result.credits == []


# ---------------------------------------------------------------------------
# Ball missing at a snapshot is tolerated
# ---------------------------------------------------------------------------


def test_missing_ball_at_snapshot_skips_that_snapshot() -> None:
    """If the ball is invisible at a snapshot, that snapshot contributes nothing.
    Other snapshots still produce credit normally."""
    frames = [1, 2, 3]
    rows: list[dict] = []
    # frame 1: ball visible
    rows.append(_tracking_row(1, "ball", None, 50.0, 34.0, is_ball=True))
    rows.append(_tracking_row(1, "home_carrier", "home", 50.5, 34.0))
    # frame 2: ball NOT visible
    rows.append(_tracking_row(2, "ball", None, 50.0, 34.0, is_ball=True, visible=False))
    rows.append(_tracking_row(2, "home_carrier", "home", 50.5, 34.0))
    # frame 3: ball visible
    rows.append(_tracking_row(3, "ball", None, 50.0, 34.0, is_ball=True))
    rows.append(_tracking_row(3, "home_carrier", "home", 50.5, 34.0))
    tracking = pd.DataFrame(rows)
    record = EpisodeRecord(
        boundary=_boundary(1, 1, 3),
        outcome=_outcome(1),
        state_trajectory=_state_trajectory(frames),
        dominant_phase=None,
    )
    obso = _obso_trajectory(frames, [0.10, 0.20, 0.40])

    # Should not crash. Carrier still gets some credit — at least from frame 3 transition.
    result = attribute_episode_markov_td(
        record, tracking, home_team_id="home", away_team_id="away", obso_trajectory=obso
    )
    by_id = {c.player_id: c.credit for c in result.credits}
    assert "home_carrier" in by_id
    assert by_id["home_carrier"] > 0.0


def test_player_credit_dataclass_is_frozen() -> None:
    pc = PlayerCredit(
        player_id="p1", team_id="home", credit=1.0,
        n_snapshots_carrier=2, n_snapshots_near_ball=0,
    )
    with pytest.raises((AttributeError, Exception)):
        pc.credit = 2.0  # type: ignore[misc]


def test_attribution_result_dataclass_is_frozen() -> None:
    ar = AttributionResult(episode_id=1, credits=[])
    with pytest.raises((AttributeError, Exception)):
        ar.episode_id = 99  # type: ignore[misc]

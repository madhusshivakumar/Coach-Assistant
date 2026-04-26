"""Tests for the corpus cache (round-trip records + formation-pairs to parquet)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.corpus_cache import (
    cache_is_valid,
    read_corpus_cache,
    write_corpus_cache,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.formation_pair import FormationPair
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _record(eid: int) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=eid,
        start_frame=eid * 100,
        end_frame=eid * 100 + 50,
        start_time_s=float(eid),
        end_time_s=float(eid + 4),
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=eid,
        end_reason="match_end",
        reached_final_third=eid % 2 == 0,
        ended_in_box=eid % 3 == 0,
        shot_like=eid % 5 == 0,
        end_ball_x=80.0,
        end_ball_y=34.0,
        end_ball_speed=8.0,
        duration_s=4.0,
    )
    states = pd.DataFrame(
        {
            "rel_time_s": np.linspace(0, 4, 4),
            "ball_x_oriented": np.linspace(20, 80, 4),
            "ball_x": np.linspace(20, 80, 4),
            "ball_y": [34.0] * 4,
            "ball_speed": [3.0] * 4,
            "attackers_mean_x_oriented": np.linspace(40, 60, 4),
            "defenders_line_height_oriented": [80.0] * 4,
            "attackers_length": [30.0] * 4,
            "attackers_width": [40.0] * 4,
            "attackers_visible": [11] * 4,
            "defenders_visible": [10] * 4,
            "defenders_compactness_x": [3.0] * 4,
        }
    )
    return EpisodeRecord(
        boundary=boundary,
        outcome=outcome,
        state_trajectory=states,
        dominant_phase="progression",
    )


def _pair(eid: int) -> FormationPair:
    return FormationPair(
        episode_id=eid,
        representative_frame=eid * 100 + 25,
        attacker_team_id="home",
        defender_team_id="away",
        attacker_formation="4-3-3",
        attacker_formation_cost=12.5,
        defender_formation="4-4-2",
        defender_formation_cost=11.0,
    )


def test_corpus_cache_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    records = [_record(i) for i in range(1, 6)]
    pairs = [_pair(i) for i in range(1, 6)]
    record_to_match = {i: f"match-{i}" for i in range(1, 6)}
    # Synthetic input parquets so input_hash has something stable.
    inputs = [tmp_path / f"input-{i}.parquet" for i in range(1, 4)]
    for p in inputs:
        p.write_bytes(b"x" * 100)

    write_corpus_cache(cache_dir, records, pairs, record_to_match, inputs)
    assert cache_is_valid(cache_dir, inputs)
    rec_back, pair_back, rtm_back = read_corpus_cache(cache_dir)

    assert len(rec_back) == 5
    assert len(pair_back) == 5
    assert rtm_back == record_to_match
    # Spot-check: episode 1 round-tripped intact.
    r0 = rec_back[0]
    assert r0.boundary.episode_id == 1
    assert r0.outcome.shot_like == (1 % 5 == 0)  # False
    assert r0.dominant_phase == "progression"
    assert not r0.state_trajectory.empty


def test_cache_invalidated_when_inputs_change(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    records = [_record(1)]
    pairs = [_pair(1)]
    inputs = [tmp_path / "input.parquet"]
    inputs[0].write_bytes(b"x" * 100)
    write_corpus_cache(cache_dir, records, pairs, {1: "match-1"}, inputs)
    assert cache_is_valid(cache_dir, inputs)

    # Modify the input — cache should now be invalid.
    inputs[0].write_bytes(b"x" * 200)
    assert not cache_is_valid(cache_dir, inputs)


def test_cache_invalid_when_no_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "nope"
    cache_dir.mkdir()
    assert not cache_is_valid(cache_dir, [])

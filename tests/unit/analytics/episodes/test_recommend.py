"""Tests for the prescriptive recommendation API."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.formation_pair import FormationPair
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.recommend import (
    FormationRecommendation,
    recommend_defensive_setup_against_attacker,
    recommend_for_defender_formation,
)
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _record(
    eid: int,
    *,
    shot_like: bool = False,
    ended_in_box: bool = False,
    reached_final_third: bool = True,
    end_x: float = 80.0,
    phase: str = "progression",
) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=eid,
        start_frame=eid * 100,
        end_frame=eid * 100 + 50,
        start_time_s=float(eid * 4),
        end_time_s=float(eid * 4 + 4),
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=eid,
        end_reason="match_end",
        reached_final_third=reached_final_third,
        ended_in_box=ended_in_box,
        shot_like=shot_like,
        end_ball_x=end_x,
        end_ball_y=34.0,
        end_ball_speed=8.0,
        duration_s=4.0,
    )
    n = 5
    states = pd.DataFrame(
        {
            "rel_time_s": np.linspace(0, 4, n),
            "ball_x_oriented": np.linspace(20, end_x, n),
            "ball_x": np.linspace(20, end_x, n),
            "ball_y": [34.0] * n,
            "ball_speed": [3.0] * n,
            "attackers_mean_x_oriented": np.linspace(40, 60, n),
            "defenders_line_height_oriented": [80.0] * n,
            "attackers_length": [30.0] * n,
            "attackers_width": [40.0] * n,
            "attackers_visible": [11] * n,
            "defenders_visible": [10] * n,
            "defenders_compactness_x": [3.0] * n,
        }
    )
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=phase)


def _pair(eid: int, atk: str, def_: str) -> FormationPair:
    return FormationPair(
        episode_id=eid,
        representative_frame=eid * 100,
        attacker_team_id="home",
        defender_team_id="away",
        attacker_formation=atk,
        attacker_formation_cost=10.0,
        defender_formation=def_,
        defender_formation_cost=10.0,
    )


def _corpus() -> tuple[list[EpisodeRecord], list[FormationPair], EpisodeIndex]:
    """Synthetic corpus: 4-3-3 vs 4-4-2 episodes get good outcomes; others don't."""
    records = [
        # Strong vs 4-4-2 (shot_like)
        _record(1, shot_like=True, ended_in_box=True, end_x=98, phase="finishing"),
        _record(2, shot_like=True, ended_in_box=True, end_x=95, phase="finishing"),
        _record(3, shot_like=False, ended_in_box=True, end_x=92, phase="finishing"),
        _record(4, shot_like=False, ended_in_box=True, end_x=90, phase="finishing"),
        _record(5, shot_like=False, end_x=75, phase="progression"),
        # Vs 5-3-2 (mostly stopped)
        _record(6, shot_like=False, end_x=60, phase="progression"),
        _record(7, shot_like=False, end_x=55, phase="progression"),
        _record(8, shot_like=False, reached_final_third=False, end_x=40, phase="settled_def"),
    ]
    pairs = [
        _pair(1, "4-3-3", "4-4-2"),
        _pair(2, "4-3-3", "4-4-2"),
        _pair(3, "4-3-3", "4-4-2"),
        _pair(4, "4-3-3", "4-4-2"),
        _pair(5, "4-3-3", "4-4-2"),
        _pair(6, "4-3-3", "5-3-2"),
        _pair(7, "4-3-3", "5-3-2"),
        _pair(8, "4-3-3", "5-3-2"),
    ]
    idx = EpisodeIndex(k_default=3)
    idx.fit(records)
    return records, pairs, idx


def test_recommend_for_defender_formation_returns_patterns() -> None:
    records, pairs, idx = _corpus()
    recs = recommend_for_defender_formation(
        defender_formation="4-4-2",
        records=records,
        formation_pairs=pairs,
        index=idx,
        top_k_patterns=3,
        min_episodes_per_pattern=1,
        n_clusters=4,
    )
    assert all(isinstance(r, FormationRecommendation) for r in recs)
    # All recommendations should reference 4-4-2 episodes (1-5).
    matched_ids = {1, 2, 3, 4, 5}
    for r in recs:
        for eid in r.example_episode_ids:
            assert eid in matched_ids, f"recommendation for 4-4-2 surfaced ep {eid} which isn't a 4-4-2 episode"


def test_recommend_returns_empty_for_unknown_formation() -> None:
    records, pairs, idx = _corpus()
    recs = recommend_for_defender_formation(
        defender_formation="3-1-4-2-LowBlock",  # nobody in our corpus
        records=records,
        formation_pairs=pairs,
        index=idx,
    )
    assert recs == []


def test_recommend_ranks_higher_outcome_first() -> None:
    """Patterns with shot_like episodes should outrank those without."""
    records, pairs, idx = _corpus()
    recs = recommend_for_defender_formation(
        defender_formation="4-4-2",
        records=records,
        formation_pairs=pairs,
        index=idx,
        top_k_patterns=5,
        min_episodes_per_pattern=1,
        n_clusters=4,
    )
    if len(recs) >= 2:
        # First recommendation's avg_outcome_value should be >= the second's.
        assert recs[0].avg_outcome_value >= recs[1].avg_outcome_value


def test_recommend_defensive_setup_picks_best_defender() -> None:
    """5-3-2 should rank ahead of 4-4-2 as a defensive setup against 4-3-3 in this corpus."""
    records, pairs, _idx = _corpus()
    setups = recommend_defensive_setup_against_attacker(
        attacker_formation="4-3-3",
        records=records,
        formation_pairs=pairs,
    )
    assert len(setups) >= 2
    # 5-3-2 conceded zero shots in our synthetic corpus → should rank first.
    formations = [s[0] for s in setups]
    assert formations.index("5-3-2") < formations.index("4-4-2"), (
        f"expected 5-3-2 to outrank 4-4-2, got order: {formations}"
    )


def test_recommend_defensive_setup_filters_by_attacker_min_episodes() -> None:
    """Defender formations with fewer than 3 episodes are dropped."""
    records, pairs, _idx = _corpus()
    setups = recommend_defensive_setup_against_attacker(
        attacker_formation="4-3-3",
        records=records,
        formation_pairs=pairs,
    )
    for _form, stats in setups:
        assert stats["n_episodes"] >= 3

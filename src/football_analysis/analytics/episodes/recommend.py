"""Prescriptive formation-vs-formation recommendation API.

Given a question like ``"how do I break a 4-4-2 with my 4-3-3?"`` we want to
return the top-K play patterns from the historical corpus that *did* beat a
4-4-2, ranked by realised threat (OBSO peak / outcome) and grouped by their
movement signature.

Design:

1. **Corpus** = ``EpisodeIndex`` already built from many matches, each
   episode annotated with a ``FormationPair`` (Phase 6-A foundation).
2. **Filter**: keep only episodes whose ``defender_formation`` matches the
   formation we're trying to break. Optionally also filter by attacker
   formation when the user wants symmetric advice.
3. **Rank**: order by realized outcome value. v1 = boolean composite
   (shot_like > ended_in_box > reached_final_third > duration_s); v2 will
   replace this with a calibrated ``P(shot_in_window)`` once we have a
   trained model.
4. **Cluster**: k-means on the filtered subset to surface *patterns* rather
   than individual episodes — a recommendation is a pattern, with example
   episodes as evidence.
5. **Render**: return one ``FormationRecommendation`` per pattern; downstream
   visualization (Phase 6-D) draws an "average heatmap" + narrated text per
   pattern.

Honest limitations called out in the dataclass docstring — small corpus =
unstable patterns; this is the *interface* for the prescriptive product, the
predictive quality scales with the library.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.formation_pair import FormationPair
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.patterns import PatternCluster, cluster_episodes


def _outcome_value(record: EpisodeRecord) -> float:
    """Composite scalar reflecting how dangerous an episode was.

    v1: weighted sum of boolean outcome flags. Roughly ordered by event severity.
    Replace with calibrated P(shot|episode) once we have a trained model
    and a real test corpus.
    """
    if record.outcome.shot_like:
        base = 1.0
    elif record.outcome.ended_in_box:
        base = 0.6
    elif record.outcome.reached_final_third:
        base = 0.3
    else:
        base = 0.05
    # Slight bonus for sustained possessions in the dangerous zones.
    return float(base + 0.01 * record.outcome.duration_s)


@dataclass(frozen=True)
class FormationRecommendation:
    """One pattern recommended for breaking (or defending against) a formation."""

    rank: int
    pattern_label: str
    n_supporting_episodes: int
    avg_outcome_value: float
    pct_shot_like: float
    pct_ended_in_box: float
    example_episode_ids: list[int]
    dominant_phase: str | None
    # If the recommendation corresponds to a pattern cluster, surface its id
    # so downstream viz can pull the cluster's other metadata.
    cluster_id: int | None = None
    notes: list[str] = field(default_factory=list)


def recommend_for_defender_formation(
    defender_formation: str,
    records: list[EpisodeRecord],
    formation_pairs: list[FormationPair],
    index: EpisodeIndex,
    attacker_formation: str | None = None,
    top_k_patterns: int = 5,
    min_episodes_per_pattern: int = 3,
    n_clusters: int = 12,
) -> list[FormationRecommendation]:
    """Return the top-K patterns that worked against ``defender_formation``.

    Args:
        defender_formation: target template name to break (e.g. ``"4-4-2"``).
        records: corpus of ``EpisodeRecord``.
        formation_pairs: parallel list of ``FormationPair`` (one per record).
        index: a fitted ``EpisodeIndex`` over ``records``.
        attacker_formation: optional filter. When set, only consider
            episodes where the attacker was in this formation.
        top_k_patterns: how many recommendations to return.
        min_episodes_per_pattern: drop patterns with fewer than this many
            episodes after filtering — prevents cluster-of-1 noise.
        n_clusters: passed to ``cluster_episodes``.

    Returns:
        Ranked list of ``FormationRecommendation``. May be shorter than
        ``top_k_patterns`` when filtering yields too few clusters.
    """
    # Index of formation_pair by episode_id for fast lookup.
    pair_by_eid: dict[int, FormationPair] = {fp.episode_id: fp for fp in formation_pairs}

    # Filter corpus to (a) matches the defender formation, (b) optionally
    # matches the attacker formation.
    matched_records: list[EpisodeRecord] = []
    for r in records:
        fp = pair_by_eid.get(r.boundary.episode_id)
        if fp is None or fp.defender_formation != defender_formation:
            continue
        if attacker_formation is not None and fp.attacker_formation != attacker_formation:
            continue
        matched_records.append(r)

    if not matched_records:
        return []

    # Cluster the FULL corpus to get pattern membership, then filter by which
    # pattern members appear in matched_records.
    all_clusters: list[PatternCluster] = cluster_episodes(index, n_clusters=n_clusters)
    matched_eids = {r.boundary.episode_id for r in matched_records}
    eid_to_record = {r.boundary.episode_id: r for r in matched_records}

    # For each cluster: how many of its members appear in matched_records?
    # Their pattern is "this cluster filtered to episodes that beat formation X".
    pattern_outcomes: list[tuple[PatternCluster, list[EpisodeRecord]]] = []
    for cluster in all_clusters:
        members_in_match = [eid_to_record[eid] for eid in cluster.episode_ids if eid in matched_eids]
        if len(members_in_match) >= min_episodes_per_pattern:
            pattern_outcomes.append((cluster, members_in_match))

    # Rank by mean outcome value
    pattern_outcomes.sort(key=lambda kv: -sum(_outcome_value(r) for r in kv[1]) / len(kv[1]))

    out: list[FormationRecommendation] = []
    for rank, (cluster, members) in enumerate(pattern_outcomes[:top_k_patterns]):
        avg_value = sum(_outcome_value(r) for r in members) / len(members)
        n_shot = sum(1 for r in members if r.outcome.shot_like)
        n_box = sum(1 for r in members if r.outcome.ended_in_box)
        # Top-3 examples by outcome value for "show me the receipts".
        examples = sorted(members, key=lambda r: -_outcome_value(r))[:3]
        # Dominant phase across the matched subset.
        phase_counts = Counter(r.dominant_phase for r in members if r.dominant_phase)
        top_phase = phase_counts.most_common(1)[0][0] if phase_counts else None

        out.append(
            FormationRecommendation(
                rank=rank,
                pattern_label=cluster.label,
                n_supporting_episodes=len(members),
                avg_outcome_value=round(avg_value, 3),
                pct_shot_like=round(n_shot / len(members), 3),
                pct_ended_in_box=round(n_box / len(members), 3),
                example_episode_ids=[e.boundary.episode_id for e in examples],
                dominant_phase=top_phase,
                cluster_id=cluster.cluster_id,
                notes=[
                    f"{len(members)} of {cluster.n_episodes} cluster members"
                    f" match defender_formation={defender_formation!r}",
                ],
            )
        )
    return out


def recommend_defensive_setup_against_attacker(
    attacker_formation: str,
    records: list[EpisodeRecord],
    formation_pairs: list[FormationPair],
    top_k_setups: int = 3,
) -> list[tuple[str, dict[str, float]]]:
    """The defensive flip: which defender formation *contained* this attacker best?

    Returns a ranked list of ``(defender_formation, stats)`` where stats has
    ``n_episodes``, ``avg_outcome_value`` (lower = better defending),
    ``pct_shot_conceded``, ``pct_ended_in_box``.

    A simpler heuristic than the offensive recommend: we don't cluster, we
    just aggregate by defender formation across all episodes where the
    attacker matched. The defender formation with the *lowest* mean outcome
    value (i.e. that most successfully suppressed the attacker) ranks highest.
    """
    pair_by_eid: dict[int, FormationPair] = {fp.episode_id: fp for fp in formation_pairs}

    by_def_formation: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        fp = pair_by_eid.get(r.boundary.episode_id)
        if fp is None or fp.attacker_formation != attacker_formation:
            continue
        if fp.defender_formation is None:
            continue
        by_def_formation[fp.defender_formation].append(r)

    rankings: list[tuple[str, dict[str, float]]] = []
    for def_form, members in by_def_formation.items():
        if len(members) < 3:  # need enough samples
            continue
        avg_value = sum(_outcome_value(r) for r in members) / len(members)
        n_shot = sum(1 for r in members if r.outcome.shot_like)
        n_box = sum(1 for r in members if r.outcome.ended_in_box)
        rankings.append(
            (
                def_form,
                {
                    "n_episodes": float(len(members)),
                    "avg_outcome_value": round(avg_value, 3),
                    "pct_shot_conceded": round(n_shot / len(members), 3),
                    "pct_ended_in_box": round(n_box / len(members), 3),
                },
            )
        )
    # Lower avg_outcome_value = better defending → rank ascending.
    rankings.sort(key=lambda kv: kv[1]["avg_outcome_value"])
    return rankings[:top_k_setups]

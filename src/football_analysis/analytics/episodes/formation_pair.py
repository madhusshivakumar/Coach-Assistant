"""Formation-pair labeling for episodes — the foundation of prescriptive analysis.

Every episode is a *matchup*: one team in formation F_atk attacks another in
formation F_def. This module detects that pair at a representative frame using
the existing Bialkowski template-matching from ``formations.roles``.

The output unlocks prescriptive queries on top of the retrieval index:

    "show me episodes where team X (in 4-3-3) beat a 4-4-2"
    "what patterns work against a low-block 5-4-1?"
    "in this 4-2-3-1 vs 4-3-3, where is the space?"

Each episode gets two extra fields: ``attacker_formation`` and
``defender_formation`` (string template names like ``"4-3-3"``). A cost field
exposes goodness-of-fit so callers can filter by confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.formations.roles import (
    DEFAULT_TEMPLATES,
    FormationTemplate,
    best_template_for_frame,
)

# Minimum visible outfielders required for a template fit. Bialkowski templates
# are 10 outfield slots; with fewer than this we abstain rather than mis-fit.
MIN_PLAYERS_FOR_FIT: int = 9


@dataclass(frozen=True)
class FormationPair:
    """Detected (attacker, defender) formation pair for one episode."""

    episode_id: int
    representative_frame: int
    attacker_team_id: str
    defender_team_id: str
    attacker_formation: str | None  # template name or None when ambiguous
    attacker_formation_cost: float | None  # lower = better fit
    defender_formation: str | None
    defender_formation_cost: float | None


def detect_formation_at_frame(
    tracking: pd.DataFrame,
    frame_id: int,
    team_id: str,
    attacking_right: bool,
    templates: tuple[FormationTemplate, ...] = DEFAULT_TEMPLATES,
    min_players: int = MIN_PLAYERS_FOR_FIT,
) -> tuple[FormationTemplate, float] | None:
    """Best-fitting Bialkowski template for a team at one frame.

    Returns ``None`` if there aren't enough visible outfielders to fit. We drop
    the goalkeeper (lowest mean-x player) because the templates are outfield-only.
    """
    sub = tracking[
        (tracking["frame_id"] == frame_id)
        & (tracking["team_id"] == team_id)
        & ~tracking["is_ball"]
        & tracking["visible"]
    ]
    if len(sub) < min_players + 1:  # +1 for goalkeeper we'll drop
        return None
    # Drop the goalkeeper: typically the deepest player in the team's defending
    # direction. When the team attacks +x, the GK is at minimum x; when they attack
    # -x, the GK is at maximum x. Sorting by x with ``ascending=attacking_right``
    # puts the GK first in both cases.
    sub_sorted = sub.sort_values("x", ascending=attacking_right)
    outfield = sub_sorted.iloc[1:]
    if len(outfield) < min_players:
        return None
    return best_template_for_frame(
        outfield[["x", "y"]],
        templates=templates,
        attacking_right=attacking_right,
    )


def label_episode_formation_pair(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    attacking_directions: dict[str, str] | None = None,
    representative: str = "median",
) -> FormationPair:
    """Detect (attacker_formation, defender_formation) for one episode.

    Args:
        record: episode from ``build_episodes``.
        tracking: full match tracking DataFrame.
        home_team_id, away_team_id: canonical team identifiers.
        attacking_directions: ``{team_id: "left"|"right"}`` for orientation
            in the canonical pitch frame. Defaults to home→right, away→left.
        representative: which frame to fit the template at. ``"median"`` (the
            middle frame of the episode) is robust against late-action positions
            distorting the read; ``"start"`` and ``"end"`` are also supported.

    Returns:
        A ``FormationPair`` — fields may be None if no template fit was found.
    """
    if attacking_directions is None:
        attacking_directions = {home_team_id: "right", away_team_id: "left"}

    if representative == "start":
        rep_frame = record.boundary.start_frame
    elif representative == "end":
        rep_frame = record.boundary.end_frame
    else:
        rep_frame = (record.boundary.start_frame + record.boundary.end_frame) // 2

    attacker = record.boundary.possession_team
    defender = away_team_id if attacker == home_team_id else home_team_id

    atk_to_right = attacking_directions.get(attacker, "right") == "right"
    def_to_right = attacking_directions.get(defender, "left") == "right"

    atk_fit = detect_formation_at_frame(tracking, rep_frame, attacker, atk_to_right)
    def_fit = detect_formation_at_frame(tracking, rep_frame, defender, def_to_right)

    return FormationPair(
        episode_id=record.boundary.episode_id,
        representative_frame=rep_frame,
        attacker_team_id=attacker,
        defender_team_id=defender,
        attacker_formation=atk_fit[0].name if atk_fit else None,
        attacker_formation_cost=atk_fit[1] if atk_fit else None,
        defender_formation=def_fit[0].name if def_fit else None,
        defender_formation_cost=def_fit[1] if def_fit else None,
    )


def label_corpus_formation_pairs(
    records: list[EpisodeRecord],
    tracking_by_match: dict[str, pd.DataFrame],
    record_to_match: dict[int, str],
    home_team_id: str = "home",
    away_team_id: str = "away",
    attacking_directions: dict[str, str] | None = None,
) -> list[FormationPair]:
    """Bulk version: label formation pairs for an entire corpus."""
    out: list[FormationPair] = []
    for r in records:
        match_id = record_to_match.get(r.boundary.episode_id)
        if match_id is None:
            continue
        tracking = tracking_by_match.get(match_id)
        if tracking is None:
            continue
        out.append(
            label_episode_formation_pair(
                r,
                tracking,
                home_team_id,
                away_team_id,
                attacking_directions=attacking_directions,
            )
        )
    return out

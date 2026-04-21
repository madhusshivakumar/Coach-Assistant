"""PPDA — passes allowed per defensive action in the opposition's own 60% of the pitch.

Lower PPDA = more intense press. StatsBomb/Opta convention:

    PPDA(team) = opp_pass_count_in_defensive_60pct / team_defensive_action_count_in_that_zone

where "defensive 60%" is the stretch of pitch where the opponent's actions are in their own
defensive third + build-up band — i.e. excluding the opponent's attacking 40%.

In canonical coordinates (home attacks left->right), the opponent's defensive 60% lies in
x >= 0.4 * PITCH_LENGTH (for home's defensive actions against away).
"""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M

_DEF_ACTION_TYPES = {"tackle", "interception", "foul", "block", "clearance"}


def compute_ppda(
    events: pd.DataFrame,
    team_id: str,
    opponent_id: str,
    zone_fraction: float = 0.4,
) -> float:
    """Return PPDA for `team_id` pressing `opponent_id`.

    `zone_fraction` = 0.4 means "we count pressing actions in the opponent's back 60% of the pitch
    (i.e. x >= 0.4 * length in canonical coords where the opponent defends their own goal at x=0
    when the home team is LTR — but note we *mirror* opponent events into the home team's frame
    at ingest, so the opponent's defensive 60% lies in x >= 0.4 * PITCH_LENGTH from the home-team
    reference, and a home-team defensive action in that zone is a press).
    """
    length = PITCH_LENGTH_M
    threshold = zone_fraction * length

    # Opposition passes in their own 60% of the pitch — they are retreating / building up.
    opp_passes = events[
        (events["team_id"] == opponent_id)
        & (events["action_type"] == "pass")
        & (events["start_x"].notna())
        & (events["start_x"] < threshold)
    ]
    # Our defensive actions in that same zone
    our_def = events[
        (events["team_id"] == team_id)
        & (events["action_type"].isin(_DEF_ACTION_TYPES))
        & (events["start_x"].notna())
        & (events["start_x"] < threshold)
    ]

    n_passes = len(opp_passes)
    n_actions = len(our_def)
    if n_actions == 0:
        return float("inf")
    return n_passes / n_actions

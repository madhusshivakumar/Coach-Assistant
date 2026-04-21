"""Field tilt — share of final-third passes belonging to a given team.

Territorial dominance metric that complements possession %. A team with 60% possession
but only 40% field tilt is controlling the ball but not advancing it.

    field_tilt(team) = team_final_third_passes / total_final_third_passes

In canonical coordinates (home attacks L->R), the attacking third for the home team is
x >= (2/3) * PITCH_LENGTH. After ingest, opponent events are mirrored into the home team's
frame, so "the team's attacking third" is always x >= (2/3) * PITCH_LENGTH regardless of side.
"""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M


def compute_field_tilt(events: pd.DataFrame, team_id: str, opponent_id: str) -> float:
    """Return the final-third pass share for `team_id` against `opponent_id`."""
    threshold = (2.0 / 3.0) * PITCH_LENGTH_M

    def _count(tid: str) -> int:
        return len(
            events[
                (events["team_id"] == tid)
                & (events["action_type"] == "pass")
                & (events["start_x"].notna())
                & (events["start_x"] >= threshold)
            ]
        )

    team_n = _count(team_id)
    opp_n = _count(opponent_id)
    total = team_n + opp_n
    if total == 0:
        return 0.0
    return team_n / total

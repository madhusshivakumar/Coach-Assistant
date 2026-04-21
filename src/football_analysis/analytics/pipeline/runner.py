"""Phase 1 pipeline runner — compose per-match analytics from SPADL events."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_analysis.analytics.possession_value.xt import XTGrid, apply_xt, fit
from football_analysis.analytics.team.field_tilt import compute_field_tilt
from football_analysis.analytics.team.ppda import compute_ppda


@dataclass(frozen=True)
class MatchAnalytics:
    """Per-match analytics bundle."""

    events: pd.DataFrame  # enriched with xt / xt_delta
    xt_grid: XTGrid
    ppda: dict[str, float]  # team_id -> ppda
    field_tilt: dict[str, float]  # team_id -> tilt


def run(events: pd.DataFrame, xt_grid: XTGrid | None = None) -> MatchAnalytics:
    """Run Phase 1 analytics over SPADL events.

    If `xt_grid` is None, fits a fresh grid from the input events (useful for a single-match
    demo). In production, pass a grid trained on a broader corpus.
    """
    grid = xt_grid if xt_grid is not None else fit(events)
    enriched = apply_xt(events, grid)

    team_ids = sorted(t for t in events["team_id"].dropna().unique())
    ppda: dict[str, float] = {}
    tilt: dict[str, float] = {}
    for a in team_ids:
        for b in team_ids:
            if a == b:
                continue
            ppda[a] = compute_ppda(events, a, b)
            tilt[a] = compute_field_tilt(events, a, b)

    return MatchAnalytics(events=enriched, xt_grid=grid, ppda=ppda, field_tilt=tilt)

"""Shared plotting theme — colours, pitch stylings, fonts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Colour scheme used across static plots."""

    background: str = "#ffffff"
    pitch_line: str = "#333333"
    home: str = "#1f77b4"
    away: str = "#d62728"
    goal: str = "#ffd54a"
    neutral: str = "#777777"
    heat_cmap: str = "viridis"
    xt_cmap: str = "magma"

    def team_color(self, team_id: str, home_team_id: str | None) -> str:
        if home_team_id is None:
            return self.neutral
        return self.home if team_id == home_team_id else self.away


DEFAULT_THEME = Theme()

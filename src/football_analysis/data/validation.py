"""Pandera schemas for the data layer's canonical output tables.

These are the contract between the data layer and the analytics layer. Every processed
Parquet must pass these — enforced in CI, and exposed via `fa-data validate`.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


class EventsSchema(pa.DataFrameModel):
    """SPADL-extended event table."""

    match_id: Series[str] = pa.Field(nullable=False)
    period: Series[int] = pa.Field(ge=1, le=5, nullable=False)
    time_seconds: Series[float] = pa.Field(ge=0.0, nullable=False)
    team_id: Series[str] = pa.Field(nullable=False)
    player_id: Series[str] = pa.Field(nullable=True)  # null for shootout bookkeeping
    start_x: Series[float] = pa.Field(ge=0.0, le=PITCH_LENGTH_M, nullable=True)
    start_y: Series[float] = pa.Field(ge=0.0, le=PITCH_WIDTH_M, nullable=True)
    end_x: Series[float] = pa.Field(ge=0.0, le=PITCH_LENGTH_M, nullable=True)
    end_y: Series[float] = pa.Field(ge=0.0, le=PITCH_WIDTH_M, nullable=True)
    action_type: Series[str] = pa.Field(nullable=False)
    result: Series[str] = pa.Field(nullable=True)
    bodypart: Series[str] = pa.Field(nullable=True)
    raw_event_id: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = "filter"  # allow unknown columns (xg, xt, vaep attached later)
        coerce = True


class TrackingSchema(pa.DataFrameModel):
    """Long-form per-frame tracking table. One row per (frame, player|ball)."""

    match_id: Series[str] = pa.Field(nullable=False)
    period: Series[int] = pa.Field(ge=1, le=5, nullable=False)
    frame_id: Series[int] = pa.Field(ge=0, nullable=False)
    time_seconds: Series[float] = pa.Field(ge=0.0, nullable=False)
    player_id: Series[str] = pa.Field(nullable=True)  # null for ball rows
    team_id: Series[str] = pa.Field(nullable=True)  # null for ball rows
    x: Series[float] = pa.Field(ge=0.0, le=PITCH_LENGTH_M, nullable=True)
    y: Series[float] = pa.Field(ge=0.0, le=PITCH_WIDTH_M, nullable=True)
    is_ball: Series[bool] = pa.Field(nullable=False)
    visible: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = "filter"
        coerce = True

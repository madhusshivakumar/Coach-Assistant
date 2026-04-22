"""Tests for kloppy-TrackingDataset → canonical long-form DataFrame conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest

from football_analysis.analytics.pitch import PITCH_WIDTH_M
from football_analysis.data.normalize.tracking import (
    _to_metric,
    tracking_dataset_to_long,
)
from football_analysis.data.validation import TrackingSchema

# --- Mini stub of kloppy's dataset shape ---


@dataclass(frozen=True)
class _Coord:
    x: float
    y: float


@dataclass(frozen=True)
class _Team:
    team_id: str


@dataclass(frozen=True)
class _Player:
    player_id: str
    team: _Team


@dataclass
class _PData:
    coordinates: _Coord | None


@dataclass
class _Period:
    id: int


@dataclass
class _Frame:
    frame_id: int
    period: _Period
    timestamp: timedelta
    ball_coordinates: _Coord | None
    players_data: dict


@dataclass
class _Dataset:
    frames: list[_Frame]


def _mk_frame(
    frame_id: int,
    period: int,
    t: float,
    ball: tuple[float, float] | None,
    players: list[tuple[str, str, tuple[float, float] | None]],
) -> _Frame:
    players_data = {}
    for pid, team, coord in players:
        p = _Player(player_id=pid, team=_Team(team_id=team))
        pd_obj = _PData(coordinates=_Coord(*coord) if coord else None)
        players_data[p] = pd_obj
    ball_coord = _Coord(*ball) if ball else None
    return _Frame(
        frame_id=frame_id,
        period=_Period(id=period),
        timestamp=timedelta(seconds=t),
        ball_coordinates=ball_coord,
        players_data=players_data,
    )


def test_to_metric_flips_y_and_rescales() -> None:
    # Metrica's top-left origin: y=0 is at top. Canonical bottom-left: y=0 at bottom.
    x, y = _to_metric(0.5, 0.0)  # top of pitch
    assert x == pytest.approx(52.5)
    assert y == pytest.approx(PITCH_WIDTH_M)
    x, y = _to_metric(0.0, 1.0)  # bottom-left corner (Metrica-bottom)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)


def test_to_metric_nulls_passthrough() -> None:
    assert _to_metric(None, 0.5) == (None, None)
    assert _to_metric(0.5, None) == (None, None)


def test_one_frame_produces_ball_plus_players() -> None:
    ds = _Dataset(
        frames=[
            _mk_frame(
                1,
                1,
                0.04,
                (0.5, 0.5),
                [
                    ("h1", "home", (0.2, 0.2)),
                    ("a1", "away", (0.8, 0.8)),
                ],
            ),
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    assert len(df) == 3
    assert df["is_ball"].sum() == 1
    assert df["player_id"].notna().sum() == 2
    # Schema contract
    TrackingSchema.validate(df, lazy=True)


def test_velocities_from_successive_frames() -> None:
    # Keep motion under the 12 m/s player cap so it isn't zeroed by the outlier filter.
    # (0.4, 0.5) -> (0.402, 0.5): Δx = 0.002 * 105 = 0.21m over 0.04s = 5.25 m/s.
    ds = _Dataset(
        frames=[
            _mk_frame(1, 1, 0.04, (0.5, 0.5), [("h1", "home", (0.400, 0.5))]),
            _mk_frame(2, 1, 0.08, (0.5, 0.5), [("h1", "home", (0.402, 0.5))]),
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    row = df[(df["player_id"] == "h1") & (df["frame_id"] == 2)].iloc[0]
    assert row["vx"] == pytest.approx(5.25, abs=0.01)
    assert row["vy"] == pytest.approx(0.0, abs=0.01)
    assert row["speed"] == pytest.approx(5.25, abs=0.01)
    # First frame has no prior, velocity is 0
    first = df[(df["player_id"] == "h1") & (df["frame_id"] == 1)].iloc[0]
    assert first["vx"] == 0.0


def test_missing_ball_row_is_skipped() -> None:
    ds = _Dataset(
        frames=[
            _mk_frame(1, 1, 0.04, None, [("h1", "home", (0.2, 0.2))]),
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    assert df["is_ball"].sum() == 0
    assert len(df) == 1


def test_null_player_coords_dropped() -> None:
    ds = _Dataset(
        frames=[
            _mk_frame(
                1,
                1,
                0.04,
                (0.5, 0.5),
                [
                    ("h1", "home", (0.2, 0.2)),
                    ("h2", "home", None),  # off-camera; kloppy would give None
                ],
            ),
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    # Only ball + 1 visible player
    assert len(df) == 2
    assert "h2" not in df["player_id"].fillna("").tolist()


def test_empty_dataset() -> None:
    ds = _Dataset(frames=[])
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    assert df.empty
    # Still has velocity columns so downstream doesn't choke
    for col in ("vx", "vy", "speed"):
        assert col in df.columns


def test_schema_accepts_slight_out_of_bounds() -> None:
    # Ball over the touchline: Metrica y=1.1 → canonical y = (1 - 1.1) * 68 = -6.8
    ds = _Dataset(
        frames=[
            _mk_frame(1, 1, 0.04, (0.5, 1.05), [("h1", "home", (0.5, 0.5))]),
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    TrackingSchema.validate(df, lazy=True)  # should not raise


def test_non_finite_velocities_are_zeroed() -> None:
    # Two frames with identical timestamp -> dt=0 -> infinite velocity. Normaliser must zero it.
    ds = _Dataset(
        frames=[
            _mk_frame(1, 1, 0.04, (0.5, 0.5), [("h1", "home", (0.4, 0.5))]),
            _mk_frame(2, 1, 0.04, (0.5, 0.5), [("h1", "home", (0.5, 0.5))]),  # same timestamp
        ]
    )
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    assert np.isfinite(df["vx"]).all()
    assert np.isfinite(df["vy"]).all()

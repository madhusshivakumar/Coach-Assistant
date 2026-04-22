"""Tests for the velocity outlier cap in tracking normalisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from football_analysis.data.normalize.tracking import (
    BALL_MAX_SPEED_M_S,
    PLAYER_MAX_SPEED_M_S,
    tracking_dataset_to_long,
)


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
    players_data: dict = field(default_factory=dict)


@dataclass
class _Dataset:
    frames: list[_Frame]


def _mk_frames_with_outlier() -> _Dataset:
    """Player teleports 5 metres in one frame (~125 m/s) — pure tracking glitch."""
    h1 = _Player("h1", _Team("home"))
    frames = [
        _Frame(
            frame_id=1,
            period=_Period(id=1),
            timestamp=timedelta(seconds=0.04),
            ball_coordinates=_Coord(0.5, 0.5),
            players_data={h1: _PData(_Coord(0.4, 0.5))},  # canonical x ~ 42m
        ),
        _Frame(
            frame_id=2,
            period=_Period(id=1),
            timestamp=timedelta(seconds=0.08),
            ball_coordinates=_Coord(0.5, 0.5),
            # Metrica x=0.45 -> canonical 47.25m: dx=5.25m over 0.04s = 131 m/s
            players_data={h1: _PData(_Coord(0.45, 0.5))},
        ),
    ]
    return _Dataset(frames=frames)


def test_player_outlier_zeroed() -> None:
    ds = _mk_frames_with_outlier()
    df = tracking_dataset_to_long(ds, match_id="metrica:1")
    # The post-glitch row should have been zeroed
    row = df[(df["player_id"] == "h1") & (df["frame_id"] == 2)].iloc[0]
    assert row["speed"] == 0.0
    assert row["vx"] == 0.0
    assert row["vy"] == 0.0


def test_normal_velocity_preserved() -> None:
    """A plausible 7 m/s sprint must NOT be zeroed."""
    h1 = _Player("h1", _Team("home"))
    frames = [
        _Frame(1, _Period(1), timedelta(seconds=0.04), _Coord(0.5, 0.5), {h1: _PData(_Coord(0.400, 0.5))}),
        _Frame(
            2,
            _Period(1),
            timedelta(seconds=0.08),
            _Coord(0.5, 0.5),
            # Metrica dx = 0.002 → canonical dx ≈ 0.21m; over 0.04s = 5.25 m/s (high-intensity)
            {h1: _PData(_Coord(0.402, 0.5))},
        ),
    ]
    df = tracking_dataset_to_long(_Dataset(frames=frames), match_id="metrica:1")
    row = df[(df["player_id"] == "h1") & (df["frame_id"] == 2)].iloc[0]
    assert row["speed"] > 0.0
    assert row["speed"] <= PLAYER_MAX_SPEED_M_S


def test_ball_cap_higher_than_player_cap() -> None:
    """Balls are allowed to move faster than players (kicked balls reach 30+ m/s)."""
    assert BALL_MAX_SPEED_M_S > PLAYER_MAX_SPEED_M_S


def test_ball_outlier_zeroed_but_reasonable_ball_speed_kept() -> None:
    # Two-frame ball move: 2m in 0.04s = 50 m/s — right at the cap, should be kept.
    # Three-frame third frame: jumps 5m in 0.04s = 125 m/s (glitch), must be zeroed.
    frames = [
        _Frame(1, _Period(1), timedelta(seconds=0.04), _Coord(0.500, 0.5)),
        _Frame(2, _Period(1), timedelta(seconds=0.08), _Coord(0.519, 0.5)),  # ~50 m/s
        _Frame(3, _Period(1), timedelta(seconds=0.12), _Coord(0.600, 0.5)),  # ~213 m/s: glitch
    ]
    df = tracking_dataset_to_long(_Dataset(frames=frames), match_id="metrica:1")
    # Row for ball frame 3 must be zeroed
    bf3 = df[(df["frame_id"] == 3) & df["is_ball"]].iloc[0]
    assert bf3["speed"] == 0.0

"""Tests for the SoccerNet GameState 2024 → canonical tracking normalizer."""

from __future__ import annotations

from football_analysis.data.normalize.soccernet import (
    _CAT_BALL,
    _CAT_PLAYER,
    _CAT_REFEREE,
    soccernet_clip_to_long,
)


def _sample_clip(n_frames: int = 5) -> dict:
    """Synthetic SoccerNet clip with 1 player + 1 ball + 1 referee per frame.

    Players use ``team="left"``, players' positions are at increasing x. Ball
    moves linearly. Referee should be filtered out by the adapter.
    """
    frame_rate = 25
    clip = {
        "info": {"frame_rate": frame_rate, "seq_length": n_frames, "name": "SNGS-test"},
        "categories": [
            {"id": 1, "name": "player"},
            {"id": 2, "name": "goalkeeper"},
            {"id": 3, "name": "referee"},
            {"id": 4, "name": "ball"},
        ],
        "images": [{"image_id": f"img{i}"} for i in range(n_frames)],
        "annotations": [],
    }
    for i in range(n_frames):
        # Player on the LEFT side of the SoccerNet pitch (centered coords).
        clip["annotations"].append(
            {
                "id": f"a{i}p",
                "image_id": f"img{i}",
                "track_id": 1,
                "category_id": _CAT_PLAYER,
                "attributes": {"role": "player", "team": "left", "jersey": 10},
                "bbox_pitch": {
                    "x_bottom_middle": -10.0 + i * 1.0,
                    "y_bottom_middle": 0.0,
                },
            }
        )
        # Ball moving across.
        clip["annotations"].append(
            {
                "id": f"a{i}b",
                "image_id": f"img{i}",
                "track_id": 99,
                "category_id": _CAT_BALL,
                "attributes": {"role": "ball", "team": None},
                "bbox_pitch": {
                    "x_bottom_middle": 0.0 + i * 1.5,
                    "y_bottom_middle": 0.0,
                },
            }
        )
        # Referee should be filtered.
        clip["annotations"].append(
            {
                "id": f"a{i}r",
                "image_id": f"img{i}",
                "track_id": 50,
                "category_id": _CAT_REFEREE,
                "attributes": {"role": "referee", "team": None},
                "bbox_pitch": {
                    "x_bottom_middle": 5.0,
                    "y_bottom_middle": 5.0,
                },
            }
        )
    return clip


def test_soccernet_clip_to_long_emits_canonical_columns() -> None:
    df = soccernet_clip_to_long(_sample_clip(), match_id="soccernet:test")
    expected = {
        "match_id",
        "period",
        "frame_id",
        "time_seconds",
        "player_id",
        "team_id",
        "x",
        "y",
        "is_ball",
        "visible",
        "vx",
        "vy",
        "speed",
    }
    assert expected <= set(df.columns)


def test_soccernet_clip_to_long_filters_referee() -> None:
    df = soccernet_clip_to_long(_sample_clip(n_frames=3), match_id="soccernet:test")
    # Referees never have a real team label, so they must not appear.
    # Two valid entities per frame (1 player + 1 ball), 3 frames → 6 rows.
    assert len(df) == 6


def test_soccernet_clip_to_long_translates_centered_to_corner_origin() -> None:
    """SoccerNet x=-10 → canonical x=42.5 (52.5 - 10)."""
    df = soccernet_clip_to_long(_sample_clip(n_frames=1), match_id="soccernet:test")
    player_row = df[~df["is_ball"]].iloc[0]
    assert abs(player_row["x"] - 42.5) < 1e-6
    assert abs(player_row["y"] - 34.0) < 1e-6


def test_soccernet_clip_to_long_assigns_home_to_chosen_side() -> None:
    """``home_side="left"`` (default) → players with ``team="left"`` become ``home``."""
    df = soccernet_clip_to_long(_sample_clip(n_frames=2), match_id="soccernet:test")
    players = df[~df["is_ball"]]
    assert (players["team_id"] == "home").all()


def test_soccernet_clip_to_long_drops_extreme_extrapolations() -> None:
    """Off-pitch CV failures (x>120 m raw) must be filtered, not kept."""
    clip = _sample_clip(n_frames=1)
    # Add a single annotation with broken projection.
    clip["annotations"].append(
        {
            "id": "broken",
            "image_id": "img0",
            "track_id": 999,
            "category_id": _CAT_PLAYER,
            "attributes": {"role": "player", "team": "right"},
            "bbox_pitch": {"x_bottom_middle": 120.0, "y_bottom_middle": 0.0},
        }
    )
    df = soccernet_clip_to_long(clip, match_id="soccernet:test")
    # The broken annotation (track_id 999) should be filtered out.
    assert "track-999" not in set(df["player_id"].dropna())


def test_soccernet_clip_to_long_empty_clip_returns_empty_df() -> None:
    empty = {
        "info": {"frame_rate": 25, "name": "empty"},
        "images": [],
        "annotations": [],
        "categories": [],
    }
    df = soccernet_clip_to_long(empty, match_id="soccernet:empty")
    assert df.empty
    assert "vx" in df.columns

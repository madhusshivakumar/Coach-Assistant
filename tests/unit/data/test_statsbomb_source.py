"""HTTP-mocked tests for the StatsBomb source module.

Uses httpx.MockTransport rather than `responses` (which mocks `requests`, not `httpx`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from football_analysis.data.sources import statsbomb as sb


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_list_competitions_caches_and_returns(raw_dir: Path) -> None:
    body = [{"competition_id": 43, "season_id": 106, "competition_name": "FIFA World Cup"}]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=body)

    with _mock_client(handler) as client:
        got = sb.list_competitions(raw_dir=raw_dir, client=client)
    assert got == body

    # Second call should hit cache only — new mock would raise if hit
    def raise_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("cache was not used")

    with _mock_client(raise_handler) as client:
        got_again = sb.list_competitions(raw_dir=raw_dir, client=client)
    assert got_again == body
    assert len(calls) == 1


def test_list_matches_writes_cache_under_correct_path(raw_dir: Path) -> None:
    body = [
        {
            "match_id": 1,
            "home_team": {"home_team_id": 100, "home_team_name": "A"},
            "away_team": {"away_team_id": 200, "away_team_name": "B"},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/matches/43/106.json")
        return httpx.Response(200, json=body)

    with _mock_client(handler) as client:
        got = sb.list_matches(43, 106, raw_dir=raw_dir, client=client)
    assert got == body
    cached = raw_dir / "statsbomb" / "matches" / "43" / "106.json"
    assert cached.exists()
    assert json.loads(cached.read_text(encoding="utf-8")) == body


def test_load_match_events_force_refetches(raw_dir: Path) -> None:
    body = [{"id": "e1", "type": {"name": "Pass"}}]
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=body)

    with _mock_client(handler) as client:
        sb.load_match_events(3869685, raw_dir=raw_dir, client=client)
        # Second call without force uses cache; third with force=True must re-fetch.
        sb.load_match_events(3869685, raw_dir=raw_dir, client=client)
        sb.load_match_events(3869685, raw_dir=raw_dir, client=client, force=True)
    assert calls == 2  # initial fetch + forced fetch


def test_http_error_propagates(raw_dir: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with pytest.raises(httpx.HTTPStatusError), _mock_client(handler) as client:
        sb.list_competitions(raw_dir=raw_dir, client=client)

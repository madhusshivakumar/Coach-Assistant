"""Dashboard data loaders. Read from catalog.duckdb only — never from data/raw/ or SDKs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from football_analysis.config import get_settings


@st.cache_data(show_spinner=False)
def list_ingested_matches() -> pd.DataFrame:
    """Return ingested matches joined with StatsBomb team names (when available)."""
    import duckdb

    settings = get_settings()
    path = settings.catalog_path
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "ingested_at", "schema_version", "source"])
    con = duckdb.connect(str(path), read_only=True)
    try:
        runs = con.execute(
            "SELECT source, match_id, schema_version, ingested_at FROM ingest_runs ORDER BY ingested_at DESC"
        ).fetchdf()
    finally:
        con.close()

    names = _statsbomb_manifest_names(settings.raw_dir)
    runs["label"] = runs["match_id"].map(lambda m: names.get(m, m))
    return runs


def _statsbomb_manifest_names(raw_dir: Path) -> dict[str, str]:
    """Scan data/raw/statsbomb/matches/**/*.json and build a {match_id -> 'Home vs Away'} map."""
    out: dict[str, str] = {}
    root = raw_dir / "statsbomb" / "matches"
    if not root.exists():
        return out
    for p in root.rglob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for m in payload:
            mid = f"statsbomb:{m.get('match_id')}"
            try:
                home = m["home_team"]["home_team_name"]
                away = m["away_team"]["away_team_name"]
            except KeyError:
                continue
            out[mid] = f"{home} vs {away}"
    return out


@st.cache_data(show_spinner=False)
def load_match_events(match_id: str) -> pd.DataFrame:
    """Load processed events for a canonical match_id."""
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    for p in (settings.processed_dir / "events").rglob("*.parquet"):
        if key in p.stem:
            return pd.read_parquet(p)
    raise FileNotFoundError(f"no parquet for {match_id!r}")


def _find_match_meta(match_id: str) -> dict[str, Any] | None:
    """Return the raw match dict from the StatsBomb matches manifest, or None."""
    settings = get_settings()
    root = settings.raw_dir / "statsbomb" / "matches"
    if not root.exists():
        return None
    mid_int = int(match_id.split(":", 1)[-1])
    for p in root.rglob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for m in payload:
            if m.get("match_id") == mid_int:
                return m  # type: ignore[no-any-return]
    return None


@st.cache_data(show_spinner=False)
def team_names_for_match(match_id: str) -> dict[str, str]:
    """Return {team_id -> name} for a given match."""
    m = _find_match_meta(match_id)
    if m is None:
        return {}
    return {
        str(m["home_team"]["home_team_id"]): m["home_team"]["home_team_name"],
        str(m["away_team"]["away_team_id"]): m["away_team"]["away_team_name"],
    }


@st.cache_data(show_spinner=False)
def home_team_id_for_match(match_id: str) -> str | None:
    """Return the canonical home team id for a given match, or None if unknown."""
    m = _find_match_meta(match_id)
    if m is None:
        return None
    return str(m["home_team"]["home_team_id"])


@st.cache_data(show_spinner=False)
def player_names_for_match(match_id: str) -> dict[str, str]:
    """Return {player_id -> player display name} from StatsBomb lineups for a match.

    Uses `player_nickname` when present (commentator-friendly, e.g. "Messi"), falling back
    to full `player_name`.
    """
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    path = settings.raw_dir / "statsbomb" / "lineups" / f"{key}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    for team in payload:
        for p in team.get("lineup", []):
            pid = p.get("player_id")
            if pid is None:
                continue
            out[str(pid)] = p.get("player_nickname") or p.get("player_name") or str(pid)
    return out

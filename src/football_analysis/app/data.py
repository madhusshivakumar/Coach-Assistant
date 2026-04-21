"""Dashboard data loaders. Read from catalog.duckdb only — never from data/raw/ or SDKs."""

from __future__ import annotations

import json
from pathlib import Path

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


@st.cache_data(show_spinner=False)
def team_names_for_match(match_id: str) -> dict[str, str]:
    """Return {team_id -> name} for a given match by inspecting the StatsBomb matches manifest."""
    settings = get_settings()
    root = settings.raw_dir / "statsbomb" / "matches"
    if not root.exists():
        return {}
    mid_int = int(match_id.split(":", 1)[-1])
    for p in root.rglob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for m in payload:
            if m.get("match_id") == mid_int:
                return {
                    str(m["home_team"]["home_team_id"]): m["home_team"]["home_team_name"],
                    str(m["away_team"]["away_team_id"]): m["away_team"]["away_team_name"],
                }
    return {}

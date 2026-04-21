"""`fa-data` Typer CLI.

Commands:
- `fa-data fetch statsbomb --competition-id <id> --season-id <id>` — mirror source into raw/.
- `fa-data ingest statsbomb --match-id <id>` — normalise raw -> processed Parquet.
- `fa-data catalog rebuild` — (re)build DuckDB views.
- `fa-data status` — ingested-match log.
- `fa-data validate --match-id <id>` — run Pandera schemas against processed Parquet.

`fetch` is network-bound; `ingest` is CPU-bound and re-runnable. They are deliberately
separate so CI can test `ingest` against fixtures offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from football_analysis.config import get_settings
from football_analysis.data import catalog as catalog_mod
from football_analysis.data.normalize.events_spadl import normalise_events
from football_analysis.data.sources import statsbomb as sb
from football_analysis.data.validation import EventsSchema
from football_analysis.logging import configure_logging, get_logger

SCHEMA_VERSION = "0.1.0"

app = typer.Typer(help="Football-analysis data pipeline CLI.")
fetch_app = typer.Typer(help="Download raw provider data into data/raw/.")
ingest_app = typer.Typer(help="Normalise raw provider data into processed Parquet.")
catalog_app = typer.Typer(help="Manage the DuckDB catalog.")
app.add_typer(fetch_app, name="fetch")
app.add_typer(ingest_app, name="ingest")
app.add_typer(catalog_app, name="catalog")

_log = get_logger(__name__)
_console = Console()


@app.callback()
def _root() -> None:
    """Configure logging before running any subcommand."""
    configure_logging()


# ---------- fetch ----------


@fetch_app.command("statsbomb")
def fetch_statsbomb(
    competition_id: Annotated[int, typer.Option(help="StatsBomb competition_id")] = 43,
    season_id: Annotated[int, typer.Option(help="StatsBomb season_id")] = 106,
    match_id: Annotated[int | None, typer.Option(help="One match; omit to fetch the whole season")] = None,
    force: Annotated[bool, typer.Option(help="Re-fetch even if cached")] = False,
) -> None:
    """Mirror StatsBomb open-data into `data/raw/statsbomb/`.

    Defaults target the 2022 men's World Cup (competition_id=43, season_id=106).
    """
    settings = get_settings()
    sb.list_competitions(raw_dir=settings.raw_dir, force=force)
    matches = sb.list_matches(competition_id, season_id, raw_dir=settings.raw_dir, force=force)
    _console.print(f"[cyan]competition {competition_id} / season {season_id}: {len(matches)} matches[/cyan]")

    target_ids = [match_id] if match_id is not None else [m["match_id"] for m in matches]
    for mid in target_ids:
        sb.load_match_events(mid, raw_dir=settings.raw_dir, force=force)
        sb.load_match_lineups(mid, raw_dir=settings.raw_dir, force=force)
    _console.print(f"[green]fetched events+lineups for {len(target_ids)} match(es)[/green]")


# ---------- ingest ----------


def _hash_raw_events_file(path: Path) -> str:
    """Content hash of the raw events JSON. Used in ingest_runs for change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _home_team_from_matches(matches: list[dict[str, Any]], match_id: int) -> tuple[str, bool]:
    """Return (home_team_id, home_attacks_left_to_right_p1). StatsBomb always aligns to LTR for
    the recorded team, so we default to True; callers may override via their own metadata."""
    for m in matches:
        if m["match_id"] == match_id:
            return str(m["home_team"]["home_team_id"]), True
    raise ValueError(f"match_id {match_id} not found in matches manifest")


@ingest_app.command("statsbomb")
def ingest_statsbomb(
    match_id: Annotated[int, typer.Option(help="StatsBomb match_id to ingest")],
    competition_id: Annotated[int, typer.Option(help="StatsBomb competition_id")] = 43,
    season_id: Annotated[int, typer.Option(help="StatsBomb season_id")] = 106,
    competition: Annotated[str, typer.Option(help="Human name for partitioning")] = "WC2022",
    season: Annotated[str, typer.Option(help="Human name for partitioning")] = "2022",
) -> None:
    """Normalise cached StatsBomb events into canonical SPADL-extended Parquet."""
    settings = get_settings()
    matches = sb.list_matches(competition_id, season_id, raw_dir=settings.raw_dir)
    home_team_id, home_ltr = _home_team_from_matches(matches, match_id)

    raw_events = sb.load_match_events(match_id, raw_dir=settings.raw_dir)
    df = normalise_events(
        raw_events,
        match_id=f"statsbomb:{match_id}",
        home_team_id=home_team_id,
        home_attacks_left_to_right_p1=home_ltr,
    )

    # Schema gate
    EventsSchema.validate(df, lazy=True)

    out = catalog_mod.processed_events_path(competition, season, f"statsbomb-{match_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # Catalog bookkeeping
    catalog_mod.init_schema()
    raw_events_path = settings.raw_dir / "statsbomb" / "events" / f"{match_id}.json"
    src_hash = _hash_raw_events_file(raw_events_path)
    catalog_mod.record_ingest("statsbomb", f"statsbomb:{match_id}", src_hash, SCHEMA_VERSION)

    _console.print(f"[green]ingested statsbomb:{match_id} -> {out} ({len(df)} rows)[/green]")


# ---------- catalog ----------


@catalog_app.command("rebuild")
def catalog_rebuild() -> None:
    """Recreate DuckDB views over the processed Parquet tree."""
    catalog_mod.init_schema()
    catalog_mod.rebuild_event_views()
    _console.print("[green]catalog rebuilt[/green]")


# ---------- status / validate ----------


@app.command()
def status() -> None:
    """Show ingested matches."""
    catalog_mod.init_schema()
    rows = catalog_mod.ingested_matches()
    if not rows:
        _console.print("[yellow]no matches ingested yet[/yellow]")
        return
    table = Table(title="Ingested matches", show_lines=False)
    cols = ("source", "match_id", "schema_version", "ingested_at", "source_hash")
    for col in cols:
        table.add_column(col)
    for r in rows:
        table.add_row(*[str(r.get(c, "")) for c in cols])
    _console.print(table)


_VALIDATE_HELP = "Canonical match_id (e.g. 'statsbomb:3869685'); omit to validate all"


@app.command()
def validate(
    match_id: Annotated[str | None, typer.Option(help=_VALIDATE_HELP)] = None,
) -> None:
    """Run Pandera schemas against processed events Parquet."""
    settings = get_settings()
    root = settings.processed_dir / "events"
    paths = sorted(root.rglob("*.parquet"))
    if match_id is not None:
        paths = [p for p in paths if match_id.split(":", 1)[-1] in p.stem]
    if not paths:
        _console.print("[yellow]no parquet files to validate[/yellow]")
        return
    failures = 0
    for p in paths:
        df = pd.read_parquet(p)
        try:
            EventsSchema.validate(df, lazy=True)
            _console.print(f"[green]OK[/green]  {p.relative_to(settings.processed_dir)} ({len(df)} rows)")
        except Exception as e:
            _console.print(f"[red]FAIL[/red] {p.relative_to(settings.processed_dir)} — {e}")
            failures += 1
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

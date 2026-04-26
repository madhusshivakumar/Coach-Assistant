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
from football_analysis.data.normalize.skillcorner import skillcorner_to_long
from football_analysis.data.normalize.soccernet import soccernet_clip_to_long
from football_analysis.data.normalize.tracking import tracking_dataset_to_long
from football_analysis.data.sources import metrica as mt
from football_analysis.data.sources import skillcorner as sc
from football_analysis.data.sources import soccernet as sn
from football_analysis.data.sources import statsbomb as sb
from football_analysis.data.validation import EventsSchema, TrackingSchema
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


@fetch_app.command("metrica")
def fetch_metrica(
    match_id: Annotated[int, typer.Option(help="Metrica sample match_id (1, 2, or 3)")] = 1,
    limit: Annotated[int | None, typer.Option(help="Limit number of frames (for faster testing)")] = None,
) -> None:
    """Mirror a Metrica Sports sample match into `data/raw/metrica/`.

    kloppy handles the HTTP fetch + CSV parse. We write a sentinel file on success.
    """
    settings = get_settings()
    dataset = mt.fetch_dataset(match_id=match_id, raw_dir=settings.raw_dir, limit=limit)
    _console.print(f"[green]fetched metrica match {match_id}: {len(dataset.frames)} frames[/green]")


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


@fetch_app.command("skillcorner")
def fetch_skillcorner(
    match_id: Annotated[
        str | None,
        typer.Option(help="Match ID; omit to fetch all available open-data matches"),
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-fetch even if cached")] = False,
) -> None:
    """Mirror SkillCorner Open Data into ``data/raw/skillcorner/``.

    Tracking files are stored in Git LFS — we resolve the media URL automatically.
    """
    settings = get_settings()
    ids = [match_id] if match_id else sc.list_available()
    for mid in ids:
        try:
            cache = sc.fetch_match(mid, raw_dir=settings.raw_dir, force=force)
            _console.print(f"[green]fetched skillcorner match {mid} -> {cache}[/green]")
        except Exception as e:  # network errors should not abort the loop
            _console.print(f"[red]failed skillcorner {mid}: {e}[/red]")
    _console.print(f"[green]fetched {len(ids)} match(es)[/green]")


@ingest_app.command("skillcorner")
def ingest_skillcorner(
    match_id: Annotated[
        str | None,
        typer.Option(help="Match ID; omit to ingest every fetched match"),
    ] = None,
    competition: Annotated[str, typer.Option(help="Human label for partitioning")] = "SkillCorner",
    season: Annotated[str, typer.Option(help="Human label for partitioning")] = "open",
) -> None:
    """Normalise SkillCorner Open Data into canonical tracking Parquet."""
    settings = get_settings()
    ids = [match_id] if match_id else sc.list_fetched(raw_dir=settings.raw_dir)
    if not ids:
        _console.print("[yellow]no SkillCorner matches fetched — run `fa-data fetch skillcorner` first[/yellow]")
        raise typer.Exit(code=1)

    catalog_mod.init_schema()
    for mid in ids:
        try:
            metadata, frames = sc.load_match(mid, raw_dir=settings.raw_dir)
        except FileNotFoundError as e:
            _console.print(f"[red]{e}[/red]")
            continue
        canonical_id = f"skillcorner:{mid}"
        df = skillcorner_to_long(metadata, frames, match_id=canonical_id)
        if df.empty:
            _console.print(f"[yellow]  {canonical_id}: no usable frames[/yellow]")
            continue

        TrackingSchema.validate(df, lazy=True)
        total = 0
        for period, period_df in df.groupby("period", sort=True):
            out = catalog_mod.processed_tracking_path(
                competition,
                season,
                f"skillcorner-{mid}",
                period=int(period),
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            period_df.to_parquet(out, index=False)
            total += len(period_df)
        catalog_mod.record_ingest("skillcorner", canonical_id, None, SCHEMA_VERSION)
        _console.print(f"[green]  ingested {canonical_id}: {total:,} rows[/green]")


@ingest_app.command("soccernet")
def ingest_soccernet(
    split: Annotated[str, typer.Option(help="SoccerNet split: test/train/valid")] = "test",
    clip_id: Annotated[
        str | None,
        typer.Option(help="Specific clip name (e.g. 'SNGS-116'); omit to ingest the whole split"),
    ] = None,
    competition: Annotated[str, typer.Option(help="Human label for partitioning")] = "SoccerNet",
    season: Annotated[str, typer.Option(help="Human label for partitioning")] = "gamestate-2024",
    home_side: Annotated[
        str,
        typer.Option(help="Which SoccerNet side ('left' or 'right') maps to canonical 'home'"),
    ] = "left",
) -> None:
    """Normalise a SoccerNet GameState 2024 split into canonical tracking Parquet.

    The split zip must already be downloaded under ``data/raw/soccernet/gamestate-2024/``.
    """
    settings = get_settings()
    catalog_mod.init_schema()
    clips = [clip_id] if clip_id else sn.list_clips(split=split, raw_dir=settings.raw_dir)
    if not clips:
        _console.print(f"[yellow]no SoccerNet clips found for split {split!r}[/yellow]")
        raise typer.Exit(code=1)

    n_ingested = 0
    n_skipped = 0
    for clip_name in clips:
        try:
            clip_data = sn.load_clip(clip_name, split=split, raw_dir=settings.raw_dir)
        except FileNotFoundError as e:
            _console.print(f"[red]{e}[/red]")
            n_skipped += 1
            continue
        clip_short_id = clip_name.removeprefix("SNGS-")
        canonical_id = f"soccernet:gamestate-2024-{clip_short_id}"
        df = soccernet_clip_to_long(clip_data, match_id=canonical_id, home_side=home_side)
        if df.empty:
            _console.print(f"[yellow]  {canonical_id}: no usable annotations[/yellow]")
            n_skipped += 1
            continue
        TrackingSchema.validate(df, lazy=True)
        out = catalog_mod.processed_tracking_path(
            competition,
            season,
            f"soccernet-{clip_short_id}",
            period=1,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        catalog_mod.record_ingest("soccernet", canonical_id, None, SCHEMA_VERSION)
        n_ingested += 1
        if n_ingested % 10 == 0:
            _console.print(f"[cyan]  ... ingested {n_ingested}/{len(clips)}[/cyan]")
    _console.print(f"[green]SoccerNet {split}: ingested={n_ingested}, skipped={n_skipped}[/green]")


@ingest_app.command("metrica")
def ingest_metrica(
    match_id: Annotated[int, typer.Option(help="Metrica sample match_id (1, 2, or 3)")] = 1,
    limit: Annotated[int | None, typer.Option(help="Limit number of frames (speeds up tests)")] = None,
    competition: Annotated[str, typer.Option(help="Human label for partitioning")] = "Metrica",
    season: Annotated[str, typer.Option(help="Human label for partitioning")] = "sample",
) -> None:
    """Normalise a Metrica sample match into canonical tracking Parquet (one file per period)."""
    settings = get_settings()
    dataset = mt.fetch_dataset(match_id=match_id, raw_dir=settings.raw_dir, limit=limit)
    canonical_match_id = f"metrica:{match_id}"

    df = tracking_dataset_to_long(dataset, match_id=canonical_match_id)
    if df.empty:
        _console.print("[yellow]no frames in dataset — nothing written[/yellow]")
        raise typer.Exit(code=1)

    TrackingSchema.validate(df, lazy=True)

    total_rows = 0
    for period, period_df in df.groupby("period", sort=True):
        out = catalog_mod.processed_tracking_path(competition, season, f"metrica-{match_id}", period=int(period))
        out.parent.mkdir(parents=True, exist_ok=True)
        period_df.to_parquet(out, index=False)
        total_rows += len(period_df)
        _console.print(f"[green]  period {period}: wrote {len(period_df)} rows to {out}[/green]")

    catalog_mod.init_schema()
    catalog_mod.record_ingest("metrica", canonical_match_id, None, SCHEMA_VERSION)
    _console.print(f"[green]ingested {canonical_match_id} — {total_rows} tracking rows[/green]")


# ---------- catalog ----------


@catalog_app.command("rebuild")
def catalog_rebuild() -> None:
    """Recreate DuckDB views over the processed Parquet tree (events + tracking)."""
    catalog_mod.init_schema()
    catalog_mod.rebuild_event_views()
    catalog_mod.rebuild_tracking_views()
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

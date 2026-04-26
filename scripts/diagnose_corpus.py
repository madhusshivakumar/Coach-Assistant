"""Per-match diagnostic: time build_episodes on each match in the corpus
sequentially and log how long each one takes.

Goal: identify the specific match(es) where build_episodes hangs or runs
extremely slowly. Output is a CSV at ``data/features/diagnose_corpus.csv`` that
the next iteration can use to either fix the data or block the slow matches.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import pandas as pd

from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.config import get_settings


class WatchdogError(Exception):
    """Custom timeout for the per-match watchdog."""


def _timeout_handler(_signum, _frame) -> None:
    raise WatchdogError("build_episodes exceeded watchdog timeout")


def main() -> None:
    settings = get_settings()
    root = settings.processed_dir / "tracking"
    match_dirs = sorted(root.rglob("match_id=*"))
    out_csv = Path("data/features/diagnose_corpus.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    print(f"{'match':50s}  rows  frames  build_s  episodes  status", flush=True)
    print("-" * 100, flush=True)

    for match_dir in match_dirs:
        match_label = match_dir.name.removeprefix("match_id=")
        parts = sorted(match_dir.glob("period=*.parquet"))
        try:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        except Exception as e:
            row = {
                "match": match_label,
                "rows": -1,
                "frames": -1,
                "build_s": float("nan"),
                "episodes": -1,
                "status": f"load_error: {e}",
            }
            print(f"{match_label:50s}    -      -        -        -    LOAD_ERROR: {e}", flush=True)
            rows.append(row)
            continue

        n_rows = len(df)
        n_frames = df["frame_id"].nunique()

        # Watchdog: 120 s per match. POSIX-only signal, so this works on Git Bash via
        # MinGW. On Windows native it would need threading.Timer instead.
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)
        t0 = time.perf_counter()
        try:
            recs = build_episodes(df, home_team_id="home", away_team_id="away")
            elapsed = time.perf_counter() - t0
            status = "ok"
            n_eps = len(recs)
        except WatchdogError:
            elapsed = time.perf_counter() - t0
            status = "TIMEOUT_120s"
            n_eps = -1
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
            n_eps = -1
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

        rows.append(
            {
                "match": match_label,
                "rows": n_rows,
                "frames": n_frames,
                "build_s": round(elapsed, 2),
                "episodes": n_eps,
                "status": status,
            }
        )
        marker = "***" if status != "ok" else ""
        print(
            f"{match_label:50s}  {n_rows:>6}  {n_frames:>6}  {elapsed:>7.1f}  {n_eps:>8}  {status} {marker}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)

    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\nproblematic matches: {len(bad)} / {len(rows)}", flush=True)
    for r in bad[:20]:
        print(f"  {r['match']:50s}  {r['status']}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

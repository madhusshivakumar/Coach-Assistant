"""Bulk-fetch StatsBomb 360 freeze-frame files for every match in our raw cache.

These are *not* continuous tracking — each freeze-frame is a snapshot of ~20
player positions at the moment of one event. We store them as a parallel raw
asset so a future "event-context retrieval" slice can build a separate index
without re-fetching.
"""

from __future__ import annotations

from pathlib import Path

import httpx

SB_360_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{}.json"
RAW_360_DIR = Path("data/raw/statsbomb/three-sixty")


def main() -> None:
    RAW_360_DIR.mkdir(parents=True, exist_ok=True)
    events_dir = Path("data/raw/statsbomb/events")
    if not events_dir.exists():
        raise SystemExit("No statsbomb events fetched yet — run `fa-data fetch statsbomb` first")

    match_ids = sorted(p.stem for p in events_dir.glob("*.json") if p.stem != "1")
    print(f"checking {len(match_ids)} matches for 360 data")

    fetched = 0
    skipped = 0
    no_360 = 0
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for mid in match_ids:
            out = RAW_360_DIR / f"{mid}.json"
            if out.exists():
                skipped += 1
                continue
            url = SB_360_URL.format(mid)
            r = client.get(url)
            if r.status_code == 404:
                no_360 += 1
                continue
            if r.status_code != 200:
                print(f"  {mid}: status {r.status_code}")
                continue
            out.write_text(r.text, encoding="utf-8")
            fetched += 1
            if fetched % 10 == 0:
                print(f"  fetched {fetched}...")

    print(f"\nfetched={fetched}  cached={skipped}  no-360={no_360}")
    total_bytes = sum(p.stat().st_size for p in RAW_360_DIR.glob("*.json"))
    print(f"total on disk: {total_bytes / 1024 / 1024:.1f} MB across {len(list(RAW_360_DIR.glob('*.json')))} files")


if __name__ == "__main__":
    main()

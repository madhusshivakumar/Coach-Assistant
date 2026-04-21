# StatsBomb fixtures

Hand-crafted minimal fixtures, not trimmed from a real match. Designed to exercise every
branch of `data/normalize/events_spadl.py`:

- `events_mini.json` — 11 events covering passes (complete + incomplete), carry, shots (goal + saved),
  interception, clearance, take-on, foul (yellow), a period-2 event, an ignored "Half Start" event,
  and one away-team pass (to exercise team-mirror).
- `matches_mini.json` — one match with `home_team_id=100`.

These are **not** real StatsBomb data. If you need real-data fixtures later,
`scripts/regen_fixtures.py` will materialise trimmed slices under `data/raw/statsbomb/`.

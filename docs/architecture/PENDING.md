# Pending — (cleared 2026-04-21)

The viz polish work that was paused on 2026-04-20 is **complete**. Artifact:
`data/features/phase1/statsbomb_3869685/` (re-rendered against the real WC2022 final
with team colours, scorer names, compound-surname handling, xT colorbar, and mean-position
marker on the heatmap).

## What actually shipped

- `app/data.py` — added `team_names_for_match`, `home_team_id_for_match`,
  `player_names_for_match` (all backed by the StatsBomb matches & lineups raw JSON).
- `viz/theme.py` — `Theme.team_color(team_id, home_team_id)` helper.
- `viz/static/shot_map.py` — team-coloured dots, gold-star goals with scorer-name
  annotation, match-score header that **excludes the penalty shootout** (so the WC2022
  final reads "Argentina 3 – 3 France", not "7 – 5").
- `viz/static/pass_network.py` — directional arrows, team colour, compound-surname
  handling (`Di María`, `De Paul`, `Van Dijk`, etc.), node threshold with auto-relax,
  alternating label offsets to reduce collisions.
- `viz/static/xt_surface.py` — colorbar labelled "xT (goal probability units)", attack
  arrow in the lower right.
- `viz/static/heatmap.py` — title shows player and team name, white `X` mean-position
  marker overlaid on the KDE.
- `scripts/demo_phase1_pipeline.py` — threads all metadata into the plotting calls and
  reports top xT contributors with resolved names.
- Ruflo: `SessionStart` hook wired at `.claude/settings.json` → `.claude/helpers/ruflo-session-context.sh`
  so every session starts with `ruflo swarm status` + `ruflo memory list` in Claude's
  context. Enforces the standing directive "Always use Ruflo!".

## Tests & CI

87 passing · 95.28 % branch coverage · ruff clean · mypy strict clean.
Ruff ignores added for RUF001/RUF002 (we use `–` and `×` intentionally in plot labels).

## Known cosmetic follow-ups (not blocking Phase 2)

- Shot map: a scorer who scored twice (Messi vs France) gets two near-collocated
  annotations. Fix with `adjustText` when we care.
- Pass network: dense midfield clusters still have some label overlap.
- Linting: socceraction is still excluded from the analytics extra pending a
  pandera 0.20-compatible release (see `decisions/dependency_pins_2026-04-21` in
  ruflo memory).

## Where Phase 2 picks up

Tracking data (Metrica sample @ 25 Hz + PFF FC WC2022 @ 10 Hz), Spearman pitch control,
Plotly-animated tactical view. See `docs/00-executive-summary.md`.

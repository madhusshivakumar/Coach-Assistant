# Visualization & Frontend Layer — Design

Scope: design of `src/viz/` (plotting primitives) and `src/app/` (dashboard host) for the football-analysis POC. Primary goal driver: **Goal #1 — one unified top-down tactical POV of any match.**

---

## 1. Framework Choice

**Decision: Streamlit for the POC shell + Plotly for the single animated tactical view + mplsoccer for static/export plots.**

Justification — the project is a single-user local app (hard constraint), so auth/multi-tenant concerns that push teams to Flask+React do not apply. The three realistic options were:

| Option | Pros | Cons |
|---|---|---|
| **Streamlit** | Fastest dev loop; Python-native; built-in widgets for scrub/slider/select; strong Plotly + mplsoccer integration; McKay Johns / StatsBomb community patterns are all Streamlit. | Full page reruns on widget change — mitigated with `st.session_state` + `@st.cache_data`; limited custom layout. |
| Dash | Cleaner callback graph; better for multi-panel synchronised views. | More boilerplate; slower iteration; POC does not need its fine-grained layout control. |
| Flask+React (or FastAPI+Svelte) | Unlimited UI flexibility; best real-time perf via WebSocket streaming of tracking frames. | 3–5× the build cost; violates "POC" spirit; no collaboration/auth need justifies it. |

For a **single-user local app where the analytics layer is Python**, Streamlit wins on time-to-insight. The upgrade path if we later need >30 Hz smooth tracking playback or a public web deployment is to promote the animated tactical component to a small **FastAPI + React/Svelte micro-frontend** while keeping the rest of the dashboard in Streamlit (Streamlit supports custom components via iframe).

---

## 2. Dashboard Pages (POC scope)

1. **Match Replay** — the single-POV tactical view. Match picker → play/pause/scrub timeline → synchronized event log side-panel. *(Primary page — goal #1.)*
2. **Player Movement Profile** — per-player heatmap, trajectory traces, role-occupancy over time, pass sonar. *(Goal #2.)*
3. **Formation Comparison** — per-phase average-position overlay, compactness time-series, formation strengths/weaknesses radar across chosen matches. *(Goal #3.)*
4. **Team Style Radar** — possession, field tilt, PPDA, xT per possession, directness, cross share — rendered as a soccerplots radar against a league baseline. *(Supporting view.)*
5. **Pitch Control Explorer** — single-frame Spearman pitch-control surface with a frame slider; overlays pass-lane viability. *(Goal #4 — role-relative space.)*

Out of POC: set-piece tool, scouting similarity search, live match ingestion.

---

## 3. Single-POV Tactical View — Design

This is the load-bearing deliverable. It renders any match from a unified bird's-eye perspective regardless of provider origin.

### 3.1 Rendering approach

**Plotly `Scattergl` on a pre-drawn pitch image, animated via a `Slider` over frame indices, driven from `st.session_state`.**

Why Plotly over the alternatives:

- **mplsoccer + matplotlib.animation** — great for exports, but `FuncAnimation` inside Streamlit is clunky and cannot scrub. Keep it for MP4 export only.
- **Plotly** — WebGL renders 22 players + ball at 10–25 Hz smoothly in a browser; built-in slider + play/pause; `Scattergl` is fast; hover tooltips are free. Integrates with Streamlit via `st.plotly_chart`. This is the sweet spot for the POC.
- **D3 / Three.js** — maximum fidelity but weeks of custom work. Deferred.
- **Manim** — offline cinematic renders only; not a runtime replay tool.

### 3.2 Pitch coordinate normalisation

All providers are normalised to a **105×68 m metric pitch, attacking-left-to-right, home team always attacking right after kickoff** at ingest time (via `kloppy`'s `to_pitch_dimensions` + a `flip_second_half` transform). The viz layer assumes this schema and never negotiates coordinates itself.

### 3.3 Synchronisation of events + tracking

- **Tracking is the clock.** Playback cursor is a `frame_id` on a ~10 Hz index (down-sampled from 25 Hz for browser perf; full-rate kept for analytics).
- **Events are joined to tracking via `(match_id, period, timestamp_ms)`** into an `events_on_frames` index: for each frame, the list of events whose timestamp falls within `[frame_t − 200ms, frame_t + 200ms]`.
- Ball ownership is an attribute of the *tracking* DataFrame (`possession_team`, `possession_player`), computed by the analytics layer. The viz never infers it.
- Phase-of-play comes in as a categorical column (`phase ∈ {build_up, progression, finishing, def_transition, settled_def, att_transition, set_piece}`) already resolved upstream.

### 3.4 Overlays

- **Ball owner** — yellow ring around the player marker; ball marker drawn on top.
- **Phase highlight** — the pitch background tinted with a low-alpha phase colour; a label chip above the pitch.
- **Role labels** — Bialkowski-assigned role string (e.g., `LCB`, `RCM`, `LW`) drawn next to the marker. Toggle: shirt number / role / name.
- **Event marker** — when an event falls within the active frame window, a transient annotation appears for 1.0 s (pass arrow, shot cone, pressure X).
- **Pitch-control layer** — optional toggle; renders a 104×68-cell control surface (`heatmap` trace) underneath players with alpha 0.35.

### 3.5 Playback controls

- Play / Pause / Step-frame (±1) / Skip-event (±1).
- Speed: 0.25× / 0.5× / 1× / 2× / 4× (re-indexing, not re-rendering).
- Jump-to-event: dropdown of goals, shots, turnovers filtered by team.
- Scrub timeline: the event timeline *is* the slider — event markers sit on the slider track (shots as circles, goals as stars). Clicking jumps to the frame.
- Loop-region: select `[t0, t1]` and loop — useful for analysing a single phase.

### 3.6 Export

- **PNG** — current frame via `fig.write_image` (uses `mplsoccer` for print-quality).
- **MP4** — `[t0, t1]` window rendered offline via `mplsoccer + matplotlib.animation + ffmpeg` writer. Runs out-of-process so the UI is not blocked.
- **HTML standalone** — `plotly.offline.plot` dump of a single animated figure — shareable, no server needed.

---

## 4. Formation / Movement Visualisations

- **Formation overlay** — per-phase mean positions + convex hull outline for each team; wired to the Bialkowski role ID so labels are role-stable across frames.
- **Role labels** — driven by `role_assignments` DataFrame (`frame_id, player_id, role_id`); the viz takes the mode per phase window for the overlay, and the per-frame value for live playback.
- **Pitch-control heatmap** — `mplsoccer.Pitch.kdeplot` for aggregated density (player movement); `plotly.graph_objects.Heatmap` for per-frame Spearman control (computed upstream, viz just renders).
- **Pass network** — `mplsoccer.Pitch` + `lines` with width ∝ pass count, node size ∝ passes-received, colour ∝ xT contribution; node positions = median of player's on-ball receipt locations during the phase.
- **Compactness time-series** — Plotly line chart of convex-hull area, length, width, line-heights — synchronised with the tactical-view cursor (hovering the line chart highlights the pitch frame).
- **Formation radar** — `soccerplots.radar_chart` comparing a match's formation metrics against a league baseline.

---

## 5. Module Layout

```
src/
  viz/
    __init__.py
    pitch.py            # Pitch base object, coord transforms, backend selector (mpl/plotly)
    static/
      shot_map.py
      pass_network.py
      heatmap.py
      pass_sonar.py
      radar.py
    interactive/
      tactical_view.py  # THE single-POV component (Plotly)
      pitch_control.py  # per-frame control surface
      compactness.py    # synced line-chart
    overlays/
      roles.py          # role-label overlay
      phase.py          # phase tinting + chip
      events.py         # transient event annotations
    export/
      png.py
      mp4.py            # offline ffmpeg writer
      html.py
    theme.py            # colours, fonts, pitch markings
  app/
    __init__.py
    main.py             # Streamlit entrypoint, nav
    pages/
      01_match_replay.py
      02_player_profile.py
      03_formation_compare.py
      04_team_style.py
      05_pitch_control.py
    state.py            # st.session_state schema, playback clock
    data.py             # @st.cache_data loaders from analytics layer
    components/
      playback.py       # play/pause/scrub widget bundle
      match_picker.py
      player_picker.py
```

Rule: **`src/viz` is import-safe from notebooks and tests; it never imports Streamlit.** Streamlit lives only in `src/app`. This keeps plotting testable and reusable.

---

## 6. Dependencies

Runtime (pinned in `pyproject.toml`, Python 3.11+):

- `mplsoccer >= 1.4`
- `plotly >= 5.22`
- `streamlit >= 1.33`
- `soccerplots` (radar)
- `matplotlib >= 3.8`
- `pandas >= 2.2`, `numpy >= 1.26`, `pyarrow >= 15` (Parquet/Arrow transport)
- `kloppy >= 3.15` (schema types only; ingestion owned by data layer)
- `scipy` (KDE for heatmaps)
- `imageio-ffmpeg` + system `ffmpeg` (MP4 export)

Dev:

- `pytest`, `pytest-mpl` (image snapshot tests), `pytest-cov`, `syrupy` (JSON snapshot for Plotly figs), `hypothesis` (coord-transform property tests).

---

## 7. Interface with the Analytics Layer

The viz layer consumes **Parquet + Arrow DataFrames** with a fixed schema. No computation beyond layout happens in `src/viz`.

| DataFrame | Key columns | Produced by |
|---|---|---|
| `tracking` | `match_id, frame_id, period, timestamp_ms, player_id, team_id, x, y, vx, vy, possession_team, possession_player` | analytics/tracking |
| `events` | `match_id, event_id, frame_id, period, timestamp_ms, player_id, team_id, type, x, y, end_x, end_y, outcome, xg, xt, vaep` | analytics/events |
| `roles` | `match_id, frame_id, player_id, role_id, role_label` | analytics/formation |
| `phases` | `match_id, frame_start, frame_end, phase` | analytics/phase |
| `pitch_control` | `match_id, frame_id, grid` (104×68 float32) | analytics/spatial |
| `formations` | `match_id, team_id, phase, formation_code, avg_positions` | analytics/formation |
| `player_season` | `player_id, metric, value, percentile` | analytics/aggregate |

Coordinates are always metric (105×68), attacking right, `(0,0)` bottom-left, floats in metres. Timestamps are `int64` ms from period-start. Parquet is the on-disk format; loaders in `src/app/data.py` return `pd.DataFrame` via `@st.cache_data(hash_funcs={pd.DataFrame: lambda df: (df.shape, tuple(df.columns))})`.

---

## 8. Testing Approach

Coverage target: **>90% line coverage on `src/viz`** (CLAUDE.md standard).

1. **Numerical tests** — coord transforms, role-label placement, slider→frame index mapping, event-in-window joins. Pure-function tests; fast.
2. **Image snapshot tests** — `pytest-mpl` on static plots (shot map, pass network, heatmap, formation overlay). Baseline PNGs under `tests/viz/baseline/`; regenerated on intentional theme changes only.
3. **Plotly figure snapshots** — `syrupy` serialises `fig.to_dict()` for the tactical view's first/mid/last frame. Catches data-binding regressions without rendering.
4. **Property tests** — `hypothesis` on coord transforms: `provider → metric → provider` must round-trip within 1 cm; `flip_second_half` is an involution.
5. **Synchronisation tests** — given fixture `tracking + events`, assert every event is attached to exactly one frame window and `|event_t − frame_t| ≤ 100 ms`.
6. **Export tests** — PNG export produces a non-empty file with expected dimensions; MP4 export produces a valid container (ffprobe check) of the requested duration ± 1 frame.
7. **Dashboard smoke tests** — `streamlit.testing.v1.AppTest` runs each page with fixture data and asserts no exceptions + expected widget keys exist. Per CLAUDE.md: fixtures live in `tests/fixtures/`, **never checked into the running dashboard**.

CI gates: unit + snapshot tests on every PR; image snapshot diffs uploaded as artefacts; coverage fails the build under 90%.

---

## Summary of Decisions

- **Streamlit + Plotly + mplsoccer.** Plotly owns the single animated tactical view; mplsoccer owns static and MP4; Streamlit is the shell.
- **Tracking-as-clock, events-joined-to-frames.** One canonical timeline drives everything.
- **Metric 105×68 pitch at the viz boundary.** Providers are normalised before viz ever sees the data.
- **`src/viz` is Streamlit-free.** Testable, reusable, notebook-friendly.
- **Upgrade path:** promote the tactical view to a FastAPI + React micro-frontend if smooth 25 Hz+ playback or remote deployment becomes a requirement. Everything else stays as-is.

# Analytics Layer — Design

Design for the compute/analytics engine of the soccer POC. Scope: everything between the data layer
(SPADL events + tracking frames) and the visualization layer (top-down tactical view). Target
Python 3.11+, tests required, >90% coverage.

---

## 1. Metric Catalog

### Ships in POC (must-have for the four stated goals)

| Metric | Why it ships | Layer | Source |
|---|---|---|---|
| **xG** | Baseline chance quality; every POV view shows it | Action value | `socceraction` (StatsBomb xG model or own GBM) |
| **xT (Karun Singh 16x12)** | Event-only action value; works when we lack tracking | Action value | own impl (50 LOC) |
| **VAEP** | Best event-only per-action value | Action value | `socceraction.vaep` |
| **Pitch Control (Spearman 2018)** | Core spatial primitive — feeds everything else | Spatial value | own impl, port of LaurieOnTracking |
| **Off-Ball Scoring Opportunity (OBSO)** | Credits off-ball runs; required for goal #4 | Spatial value | own impl on top of pitch control |
| **Bialkowski role assignment** | Required for goals #3 and #4 | Structure | own impl (Hungarian per frame) |
| **Phase-of-play classifier** | Required condition for every structural metric | Structure | rules v1, GBM v2 |
| **Compactness (convex hull, length×width, vert/horiz spread, line heights)** | Formation strengths/weaknesses | Team shape | own impl (shapely + numpy) |
| **Formation clustering (actual-shape, per phase)** | Goal #3 directly | Team shape | GMM over role-centroids |
| **PPDA, field tilt, Defensive Line Height** | Pressing/territory — cheap and standard | Team | own impl on events |
| **Pass network + avg positions** | Tactical view overlay | Player→team | `mplsoccer`-friendly frame |
| **Role displacement per phase, off-ball runs, third-man combos** | Goal #4 — movement relative to gameplay | Movement | own impl |

### Deferred (post-POC)

xGChain/Buildup, xGOT/post-shot xG, Goals Added (G+), full EPV (Fernández), packing/line-breaking
pass detection, player-vectors style embeddings, PlayeRank, set-piece models, xPts/match simulation,
metabolic power, TacticAI-style GNN. All are valuable but none are required for the four POC goals;
each bloats scope materially.

**Rationale:** we ship one model per layer (action value, spatial value, structure, team). Anything
we add before the loop closes end-to-end is premature.

---

## 2. Module Layout

```
src/
  analytics/
    __init__.py
    types.py              # TypedDicts/dataclasses shared across subpackages
    pitch.py              # pitch geometry, coordinate normalization (105x68 canonical)
    xg/
      model.py            # thin wrapper around socceraction / sklearn pipeline
      features.py
    possession_value/
      xt.py               # 16x12 Markov xT
      vaep.py             # wrapper around socceraction.vaep
    pitch_control/
      spearman.py         # core pitch-control surface
      motion.py           # constant-accel-to-max-speed intercept model
      obso.py             # off-ball scoring opportunity
    formations/
      roles.py            # Bialkowski role assignment (Hungarian)
      detect.py           # per-phase GMM clustering -> formation label
      shape.py            # convex hull, line heights, spreads
      strengths.py        # formation strengths/weaknesses scoring
    phases/
      classifier.py       # rule-based v1, GBM v2
      segments.py         # contiguous phase intervals
    movement/
      role_relative.py    # displacement vs role centroid per phase
      off_ball.py         # run detection + OBSO credit
      combinations.py     # third-man, overlap/underlap, rotations
    team/
      ppda.py
      field_tilt.py
      compactness.py
    pipeline/
      runner.py           # orchestration entry points
      cache.py            # parquet-backed feature store
      schemas.py          # pydantic schemas for every output frame
tests/
  analytics/              # mirror structure; >90% coverage per subpackage
```

Every module exposes a small pure-function API operating on DataFrames; stateful objects are
reserved for models that need fitting (xG, VAEP, phase GBM, formation GMM).

---

## 3. Compute Pipeline

Pipeline is a DAG of pure functions over match-scoped DataFrames. Each stage is independently
testable and cacheable.

```
data_layer
   │  events_spadl (DataFrame)
   │  tracking (DataFrame, optional)
   │  freeze_frames (DataFrame, optional — StatsBomb 360)
   ▼
[1] normalize  ──────────────────────────► canonical 105x68, left-to-right attack
   ▼
[2] phases.classifier.label(events, tracking?) ──► phase_id per event & per frame
   ▼
[3] action_value (parallel):
      xg.model.predict(events)            ► xg per shot
      possession_value.vaep.rate(events)  ► vaep_offensive/defensive per action
      possession_value.xt.value(events)   ► xt per pass/carry
   ▼
[4] spatial_value (requires tracking OR 360 fallback):
      pitch_control.spearman.surface(frame)     ► 105x68 control grid per frame
      pitch_control.obso.score(frame, control)  ► off-ball scoring opportunity
   ▼
[5] structure (requires tracking OR avg positions):
      formations.roles.assign(frame)            ► per-frame role→player matrix
      formations.detect.cluster(roles, phase)   ► per-phase actual formation
      formations.shape.metrics(frame)           ► compactness time series
   ▼
[6] movement:
      movement.role_relative.compute(roles, phase_id)
      movement.off_ball.detect(tracking, control)
      movement.combinations.detect(events, tracking)
   ▼
[7] aggregates:
      team.{ppda, field_tilt, compactness}
      formations.strengths.score(...)
   ▼
  feature store (parquet, partitioned by match_id)
```

**Orchestration.** Plain Python functions + a thin `pipeline/runner.py` that composes them per
match. No Prefect/Airflow in the POC — a match is one job, parallelism is across matches via
`concurrent.futures`. We revisit if we get past ~100 matches.

**Feature store vs on-demand.**
- **Cached to parquet**: xG per shot, xT/VAEP per action, role assignments per frame, compactness
  time series, formation labels per phase. These are deterministic and reused across views.
- **On-demand**: pitch-control surfaces (huge, cheap enough to recompute per request for a specific
  frame range; we cache only aggregate OBSO integrals per action).

---

## 4. Formation Analysis

### Detection
1. Label every event and every tracking frame with `phase_id` (§5 below).
2. For each phase block of length ≥ 5s, compute each outfield player's mean (x, y).
3. **Bialkowski role assignment** (`formations.roles`): given a reference role template
   (parameterised set: 4-3-3, 4-4-2, 4-2-3-1, 3-4-3, 3-5-2, 5-3-2, 3-2-4-1) and current mean
   positions, solve the Hungarian assignment minimising total squared displacement between
   players and role slots.
4. **Actual formation** = argmin over templates of post-assignment residual displacement,
   regularised by template prior. When residuals are all high, we report "irregular" rather than
   forcing a label.
5. Over a match, fit a GMM on role-slot centroids to capture in/out-of-possession asymmetry;
   report one formation per `(team, phase)` pair.

### Strengths/Weaknesses quantification
Per `(team, formation, phase)`, compute expected value differentials vs. the population:
- **VAEP/90 when in this shape** and **VAEP conceded/90** — net value per shape.
- **xT created when in this phase+shape** vs. opponent xT conceded → attack/defense decomposition.
- **Compactness stats**: length, width, vertical compactness, convex-hull area (distribution, not
  just mean).
- **Space-control share** per pitch third (from pitch-control integrals) — measures where the
  shape dominates and where it cedes space.
- **PPDA + line heights** — pressing efficacy per shape.

Strengths = top-quartile metrics vs. league population; weaknesses = bottom-quartile. Output a
radar per formation with attack/defense/pressing/territory/compactness axes, plus narrative
("4-3-3 dominates wide third in progression, cedes central third in settled defense"). The
population baseline starts as the PFF WC2022 + StatsBomb open data corpus.

---

## 5. Movement Analysis

All movement metrics are **phase-conditioned and role-relative** — this is the organising principle
from the research docs.

### Phase classifier (`phases.classifier`)
- v1 (POC): rules over possession state + pitch third + time since turnover.
  - Build-up: in own third, settled possession.
  - Progression: middle third, possession.
  - Finishing: attacking third, possession.
  - Defensive transition: first 6s after loss.
  - Attacking transition: first 6s after recovery.
  - Settled defense: opponent possession >6s, not in own third.
  - Set pieces: from event metadata.
- v2 (if time): GBM on features (ball position, ball velocity, team centroid distance to own goal,
  time since possession change, defensive-line height).

### Role displacement per phase
For each frame: `delta_i = position_i − role_centroid(role_i, phase)`. Report per-player
distributions of magnitude and direction, by phase. This is the direct operationalisation of
"movement relative to position".

### Off-ball value (`movement.off_ball`)
A run is a window where a player's speed > 5 m/s for ≥ 1s ending in one of: pass reception,
attempted pass to that player, or shot by a teammate within 3s. Score each run with its **OBSO
delta** (change in off-ball scoring opportunity during the run). Surfaces the "runs that matter
even when the ball doesn't arrive" — central to goal #4.

### Combinations (`movement.combinations`)
- **Third-man**: A→B→C chain in ≤5s where C is on the far side of a defensive line B's pass bypassed.
  Detected on events; validated on tracking (C's movement predates B receiving).
- **Overlap / underlap**: wide player and adjacent half-space/wide player crossing running paths
  during a progression phase.
- **Rotation**: two teammates swapping role slots (Hungarian re-assignment) within a 10s window.
- **Inverted-FB tuck**: FB role occupied in half-space/central zone during build-up.
- **Pivot drop / salida**: a midfielder role-centroid dropping between the CB role centroids.

These are pattern detectors on the role-assignment time series — fast, auditable, visualisable.

---

## 6. Dependencies

Core (hard deps):
- `socceraction` — VAEP, xT scaffolding, SPADL conversions.
- `kloppy` — event/tracking ingestion (owned by data layer, we only consume its schema).
- `numpy`, `pandas`, `scipy` — core.
- `scikit-learn` — GMM, logistic baselines, Hungarian (`scipy.optimize.linear_sum_assignment`).
- `shapely` — convex hulls, polygon ops.
- `pydantic` v2 — schema validation at layer boundary.
- `pyarrow` — parquet feature store.

Visualization-adjacent (used for test fixtures / debug plots, not runtime):
- `mplsoccer`, `matplotlib`.

Evaluated and deferred:
- `floodlight` — overlaps significantly with our own shape/pitch-control code; revisit if we need
  its physical-load metrics.
- `statsbombpy` — fine for data layer, not analytics.

We implement pitch control ourselves (LaurieOnTracking port) for control of the motion model and
because the published libraries don't expose the internals we need for OBSO.

---

## 7. Interface with Data Layer

Analytics code never reads raw provider files. The data layer must emit three DataFrames
(strict pydantic-validated schemas, all coordinates in canonical 105×68 m, attack left→right):

**`events_spadl`** (always present):
```
match_id, period, time_seconds, team_id, player_id, start_x, start_y,
end_x, end_y, action_type, result, bodypart, original_event_id
```
SPADL-compatible; `socceraction.spadl` conversions already produce this.

**`tracking`** (optional, 10–25 Hz):
```
match_id, period, frame, time_seconds, player_id, team_id, x, y,
vx, vy, speed, is_ball (bool)
```

**`freeze_frames`** (optional, StatsBomb 360):
```
match_id, event_id, player_id, team_id, x, y, is_teammate, is_keeper
```
Used as a **degraded fallback** for spatial metrics when no tracking exists: pitch control computed
on the freeze frame (single snapshot, no velocities → assume zero-velocity intercept), yielding
event-time OBSO/control measurements only. Degraded but better than nothing and covers the current
StatsBomb-360 corpus (Euro 2024, WEuro 2025).

**`metadata`**: pitch dimensions (we rescale), lineup + formation-as-labelled, kickoff direction.

The contract is one-way: analytics consumes, never writes back into data-layer tables.

---

## 8. Testing

Target: **>90% line coverage** per subpackage, enforced in CI (`pytest --cov --cov-fail-under=90`).

**Unit tests** (the bulk): pure-function tests with tiny hand-crafted inputs.
- Role assignment: 10 known positions → known assignment; Hungarian optimality verified against a
  brute-force check on small inputs.
- Pitch control: single-player frame → control ≈ 1 near the player, ≈ 0.5 on the bisector with a
  stationary opponent; reproduce LaurieOnTracking's published example frame within 1e-3.
- xT: reproduce Karun Singh's published grid within tolerance on StatsBomb open data.
- Phase classifier: synthetic possession sequences covering every branch.
- Compactness: equilateral triangle → known convex hull area, known spread.

**Golden tests**: pin a small real match (Metrica sample game 1) through the entire pipeline;
snapshot outputs (xG totals, VAEP totals, formation label per phase, compactness time series
hashes). Any change requires explicit re-baselining.

**Property tests** (`hypothesis`) for invariants:
- Pitch control sums to ≤ 1 per pixel.
- xT is non-negative.
- Role assignment is a permutation.
- Convex hull area ≥ 0, length ≥ 0.
- Coordinate normalisation is idempotent.

**Fixtures**: tiny handcrafted SPADL/tracking DataFrames live in `tests/analytics/fixtures/`; real
sample-data fixtures (Metrica game 1, one StatsBomb match) are materialised on first test run and
cached locally (not committed).

**What we deliberately don't unit-test**: model accuracy. xG/VAEP correctness is covered by
integration tests against socceraction's reference outputs plus the golden match; we don't
reimplement the validation the upstream libraries already do.

---

## Summary

Three organising decisions drive this design:
1. **Pitch control is the spatial primitive.** OBSO, off-ball value, and space-control
   strengths/weaknesses all reduce to integrals over it — implementing it ourselves buys every
   downstream metric.
2. **Role-based, phase-conditioned is the only way the movement story works.** Every movement
   metric is `(role, phase)`-keyed; Bialkowski assignment and the phase classifier are the two
   non-optional primitives.
3. **Ship one model per layer, cache deterministic outputs, recompute spatial surfaces on demand.**
   Feature store for action values and structure; no premature orchestration framework.

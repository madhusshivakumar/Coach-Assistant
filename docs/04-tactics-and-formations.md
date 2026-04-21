# Tactical & Formation Analysis — Concepts + Quantification

## 1. Formation Taxonomy & History

### Common formations
**4-3-3, 4-4-2, 3-5-2, 3-4-3, 4-2-3-1, 3-4-2-1, 5-3-2, 4-3-2-1 (Christmas tree), 4-1-4-1, 3-4-1-2, 3-2-4-1** (Guardiola City in-possession).

### Historical arc
- **2-3-5 (Pyramid, late 1800s)** → attack-heavy.
- **WM (3-2-2-3, Chapman 1920s–30s)** — response to 1925 offside rule.
- **Catenaccio (Herrera's Inter, 1950s–60s)** — libero + man-marking + counter.
- **4-2-4 → 4-3-3 (Brazil 1958, 1970)** — modern blueprint.
- **Total Football (Michels/Cruyff Netherlands, 1970s)** — positional interchange, high line, press.
- **Sacchi's Milan (late 1980s)** — zonal 4-4-2, 25m vertical compactness, offside trap.
- **Gegenpressing (Klopp, Rangnick heritage from Bielsa)** — counter-press in 5–6s after loss.
- **Juego de Posición (Cruyff → Guardiola → Lillo)** — zones, 3-in-line / 2-in-column rules.
- **Relationism / Jogo de Relação (Diniz, Hamilton)** — clusters over zones, improvisation.

---

## 2. Formation ≠ Shape ★

Listed formation = nominal kickoff/defensive reference. Actual shape is **fluid across phases**.

Examples:
- **4-3-3** can become **3-2-5** in build-up (inverted FB + high winger) and **4-5-1** out of possession.
- **Guardiola City 2022–23**: defended 4-4-2, built up 3-2-4-1 (Stones stepping into midfield).

### Implication for analysts
A static "average position" plot low-pass-filters distinct shapes.
**Meaningful analysis requires phase-conditioning.**

---

## 3. Phases of Play

The 7-phase model:
1. **Build-up** (own third) — GK + CBs + pivots break first pressure line.
2. **Progression** (middle third) — third-man combos, switches, half-space penetration.
3. **Finishing** (final third) — crosses, cutbacks, box combos.
4. **Defensive transition** — 5–10s after loss (counter-press or recover).
5. **Settled defense** — low / mid / high block.
6. **Attacking transition** — direct vs re-settle.
7. **Set pieces**.

Roles change per phase (a #6 = CB-adjacent pivot in build-up, box-crasher in attack, first presser in transition, screener in settled defense).

**Label tracking frames by phase before comparing anything.**

---

## 4. Core Tactical Concepts

- **Pressing triggers** — back-pass to GK, poor first touch, back-to-goal reception, pass into wide areas.
- **Gegenpressing** — counter-press within ~6s; "best playmaker" (Klopp).
- **Low / Mid / High block** — defensive line at ~18m / 40m / 60m from own goal.
- **Offside trap** — synchronized back-line step-up.
- **Half-spaces** — vertical lanes 2 & 4 (between central and wide). Raumdeuter, KDB, Bernardo.
- **Overloads** — local numerical superiority (3v2 on flank).
- **Third-man runs** — A→B→C; unlocks compact blocks.
- **Rotations** — positional swaps (LB tucks in, LCM drifts wide, LW comes inside).
- **Underlaps / overlaps** — inside vs outside of wide player.
- **Inverted fullback** — Lahm, Zinchenko, Stones, Gvardiol.
- **False 9** — drops deep, drags CB (Messi 2009, Firmino).
- **Libero / Regista / Mezzala / Raumdeuter** — specialist roles.

---

## 5. Juego de Posición (Positional Play)

### Structure
- **5 vertical lanes** (L wing, L half-space, center, R half-space, R wing) × 4 horizontal zones.
- **Max 3 in a horizontal line, max 2 in a vertical column.**
- Adjacent zones must be occupied by connected players.

### Three superiorities
- **Numerical** — more players locally.
- **Positional** — better location (between the lines, half-space, behind a presser).
- **Qualitative** — preferred matchup.
- *(+4th: socio-affective / dynamic — timing, cohesion.)*

### For analysts
Lane-occupation heatmaps per phase, counts of "players between the lines", matchup-level qualitative tagging → directly measurable Lillo concepts.

---

## 6. How Analysts Operationalize Concepts

### Formation detection
- **Avg positions** (in/out of possession) → k-means per player-minute.
- **Hungarian role assignment** (Bialkowski et al. 2014) — per-frame assign 10 outfielders to 10 role slots minimizing displacement.
- **NMF / GMM** for role discovery.
- **HMMs** to label tactical phase per frame.

### Shape metrics
- Convex hull area (spread).
- Length × width (extent).
- Vertical compactness (back-to-front distance).
- Surface area per line.
- Line heights (mean y of back 4, midfield line, gap).
- Team centroid + variance.
- Stretch index (mean distance from centroid).

### Pressing
- PPDA, packing, line-breaking passes.
- Pitch control (Spearman).
- OBSO (off-ball scoring opportunity), EPV.

### Phase classification
- Possession / out / transition from ball ownership + thresholded windows (~5–10s).
- Voronoi → per-player space control.

---

## 7. Formation Strengths/Weaknesses

| Formation | Strengths | Weaknesses |
|---|---|---|
| **4-3-3** | Wide overloads (winger+FB+#8); pressing triangles | Isolated #9 vs deep blocks; CMs outnumbered vs 3-man mid |
| **4-4-2 flat** | 2 forwards press CBs; horizontal compactness; simple | Outnumbered vs 3-man mid; wide channel if wingers don't track |
| **4-2-3-1** | Double pivot protects back 4; #10 between lines | #10/#9 disconnect; FBs overloaded |
| **3-5-2 / 3-4-1-2** | Wing-back width; 3v2 centrally; extra CB for build-up | Wing-backs exposed on counter; vulnerable to inverted wingers |
| **3-4-3 / 3-4-2-1** | Occupies all 5 lanes; #10 overloads | Physically demanding on WBs; single pivot isolated |
| **5-3-2** | Defensive solidity; 2 strikers for counter | Cedes territory; WBs less advanced |
| **4-3-2-1 (xmas tree)** | Central dominance; compact spine | No natural width; FBs do all flank work |
| **4-1-4-1** | Strong mid-block; screener #6 | Lone #9; #6 bypassed via fast half-space plays |

---

## 8. Movement Patterns

- **Triangular rotations** (LB/LCM/LW triangle) — Guardiola's core mechanism.
- **Pivot drop / salida lavolpiana** — #6 between CBs; FBs push high; named for La Volpe.
- **Splitting CBs + GK step-up** — creates 3v2 vs two strikers.
- **Pendulum midfielders** — one #8 high, one low, oscillating.
- **Third-man combinations** — A→B→C late arrival; breaks lines.
- **Dropping 9 (false 9)** — drags CB, opens channel for winger's inside run.
- **Inverted FB tuck** — 3-2 build-up structure.
- **Overlap / underlap** — width vs half-space threat.
- **Back-post arrivals** — far-side winger high, arrives late.
- **Escadinha / staircase (Diniz)** — relationist short-ladder support.

---

## 9. Reading List

### Books
- Jonathan Wilson, *Inverting the Pyramid* — canonical history.
- Marti Perarnau, *Pep Confidential* / *Pep: The Evolution*.
- Michael Cox, *The Mixer* / *Zonal Marking*.
- Jed Davies, *Coaching the Tiki-Taka Style of Play*.
- Dan Fieldsend, *The European Game*.
- Ian Graham, *How to Win the Premier League* (2024) — Liverpool DoR view.

### Sites
- **Spielverlagerung** — https://spielverlagerung.com (Rene Maric, Martin Rafelt).
- **The Athletic** — Cox, Tharme, Walid, Carey, Worville (data).
- **Tifo Football** — Nikolaou, Mackenzie, Stewart.
- **The Coaches' Voice** — https://www.coachesvoice.com.
- **Between the Posts** — https://betweentheposts.net.
- **Jamie Hamilton (relationism)** — https://www.jamiehamilton.co.uk.

### Academic
- **Bialkowski et al. 2014** — "Large-Scale Analysis" + "Win at Home and Draw Away". Seminal role-assignment.
- **Spearman 2018** — "Beyond Expected Goals" (pitch control).
- **Fernandez & Bornn 2018** — "Wide Open Spaces".
- **Fernandez/Bornn/Cervone 2019** — EPV.
- **Decroos et al. 2019** — VAEP.
- **Shaw & Glickman 2019** — "Dynamic analysis of team strategy".
- **Stöckl, Seidl, Marley, Power 2021/22** — GCN defensive performance.
- **Memmert, Rein, Perl** — tactical pattern networks.
- **Forcher, Altmann et al. 2022** — formation effect reviews.

### Venues
- MIT Sloan Sports Analytics Conference.
- StatsBomb Conference.
- Barca Innovation Hub Sports Analytics Summit.
- MLSA @ ECML-PKDD (KU Leuven co-organizes).
- CVsports @ CVPR.

---

## Synthesis

Three organizing principles when building formation/movement tooling:

1. **Condition on phase.** Raw average positions blur distinct shapes.
2. **Model roles, not shirt numbers.** Bialkowski-style role assignment separates "who plays #6" from "who is occupying the #6 role right now".
3. **Translate coaching vocabulary into measurable quantities.**
   - Compactness = inter-line distance + convex hull area
   - Overload = local player count within radius
   - "Between the lines" = y between opp DM and MF line averages
   - Positional superiority = pitch-control + high-value zone occupation
   - Qualitative superiority = matchup-level tagging

Most productive zone = **Bialkowski role extraction × Spearman pitch control × Lillo positional vocabulary**, with Diniz relationism as reminder that non-zonal systems may need proximity graphs / density clustering instead of strict lane occupation.

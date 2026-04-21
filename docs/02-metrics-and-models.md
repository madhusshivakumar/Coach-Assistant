# Soccer Analytics — Metrics & Models

## 1. Shot / Chance Quality

### Expected Goals (xG)
- Probability a shot scores, conditioned on context (location, angle, body part, shot type, assist type, GK position, defender density).
- Trained with logistic regression / GBM / NN on millions of shots.
- **Flavors**: StatsBomb (uses 360 for defender positions — most context-aware), Opta, Understat (GBM), American Soccer Analysis.
- **Data**: event data; 360 improves it.

### Post-Shot xG / xGOT
- Probability of goal *given shot on target* — conditional on placement + power + GK position.
- xGOT − xG = shot execution quality. Feeds **Goals Prevented** for goalkeeper rating.

### xG Chain / xG Buildup (Opta / Thom Lawrence)
- Credit non-shooting players for shot-ending possessions.
- **xG Chain**: assign full xG of final shot to everyone who touched the possession.
- **xG Buildup**: same, excluding shot-taker and assister → isolates deep midfielders (Jorginho, Rodri).

---

## 2. Possession Value Models

### Expected Threat (xT) — Karun Singh
- Value of possessing the ball at each pitch location.
- Pitch discretised to grid (16×12). Markov model: `xT = s·xG + m·ΣT·xT`.
- Action value = xT(end) − xT(start).
- **Data**: events only. First widely-adopted "reward every action" framework.
- https://karun.in/blog/expected-threat.html

### VAEP (Decroos et al., KU Leuven)
- Change in scoring and conceding probability over the next N=10 actions.
- `Value = [P_score(after) − P_score(before)] − [P_concede(after) − P_concede(before)]`.
- Two GBMs, SPADL representation.
- https://arxiv.org/abs/1802.07127 · `socceraction` library.

### Goals Added (G+) — American Soccer Analysis
- Per action type (passing/dribbling/shooting/interrupting/fouling/receiving).
- 96-zone × game-state value surface.
- Decomposes player contribution by skill dimension.

### On-Ball Value (OBV) — StatsBomb
- Change in StatsBomb's possession-value model per on-ball action.
- Uses event + 360; industry-standard scouting metric.

### Expected Possession Value (EPV) — Fernández/Bornn/Cervone
- Continuous, state-conditional expected goal-differential of current possession.
- Decomposes into pass/shot/dribble/turnover sub-models with tracking features.
- Sloan 2019 paper; **requires tracking + events**.

---

## 3. Passing / Progression

| Metric | Measures | Needs |
|---|---|---|
| **Packing** (Impect) | Opponents bypassed per pass/dribble | Tracking |
| **Line-breaking passes** | Pass crosses a defensive line | Tracking / 360 |
| **Progressive passes/carries** | ≥10m closer to goal (various rules) | Events |
| **Deep completions** | Passes ending within 20yd of goal | Events |
| **Pass networks** | Avg player positions + pass edges | Events |
| **Pass sonars** (E. McKinley) | Per-player polar histogram of pass direction/length | Events |

---

## 4. Defensive Metrics

- **PPDA** (Passes per Defensive Action) — (opp passes in own 60%) / (defensive actions in that zone). Pressing intensity proxy. Lower = more intense.
- **Defensive activity heatmaps** — KDE over defensive events; compute **Defensive Line Height (DLH)**.
- **Ball recoveries** — regains split by ground / interception / counter-press (<5s after loss, StatsBomb).
- **Pressure events** (StatsBomb) — explicit close-down logs within ~2m.
- **Pressing intensity from tracking** — nearest-defender distance, time-to-intercept, defender speed to ball.
- **Field tilt** — share of final-third passes. Territorial complement to possession %.

---

## 5. Tracking-Based Spatial Models ★

### Pitch Control (Spearman 2017/18)
- Probability each team would control ball at every (x,y) per frame.
- Motion model: constant-accel-to-max-speed + reaction time → time-to-intercept per player per pixel → logistic → team aggregation.
- Laurie Shaw's open implementation on GitHub (Friends of Tracking).
- Foundation for **space created / denied / pass lane viability**.

### Voronoi Diagrams
- Static partition by nearest player. Ignores velocity — superseded by pitch control.

### Dangerousness / Off-Ball Scoring Opportunity (Spearman)
- `pitch_control × pass_success × scoring_prob`, integrated over pitch.
- Credits runs that create danger without receiving the ball.
- "Beyond Expected Goals", Sloan 2018.

### Off-Ball Value
- Recent work: DeepMind/Liverpool **TacticAI** (2024); Bauer & Anzer counter-pressing detection.

---

## 6. Physical / Load Metrics

- **Total Distance Covered** — typical outfielder 10–12 km/match.
- **High-Intensity Running (HIR)** — distance above ~5.5 m/s (19.8 km/h).
- **Sprints** — sustained >7 m/s.
- **Accelerations/Decelerations** — ±3 or ±4 m/s². More load than sprints.
- **PlayerLoad** (Catapult) — tri-axial accel vector magnitude integral.
- **Metabolic Power** (di Prampero) — energetic cost with accel-as-uphill analogy.

---

## 7. Formation / Tactical Structure

- **Formation detection** (Bialkowski et al. 2014, ICDM) — k-means / hierarchical on avg positions + Hungarian role assignment.
- **Phase-of-play classification** — build-up / progression / finishing / defending / transition. Rule-based or ML.
- **Compactness**: convex hull area, length × width, vertical compactness (def-to-att distance), centroid distance between teams.
- **Defensive Line Height** — mean y of back line when opponent has ball.

---

## 8. Player Roles & Styles

- **Role clustering** — per-90 feature vectors → k-means / GMM / UMAP+HDBSCAN.
- **Player Vectors** (Decroos & Davis, ECML-PKDD 2019) — compressed style representations. https://arxiv.org/abs/1902.05437
- **Similarity scores** — weighted Euclidean / cosine over style vectors; filter by age/league strength.
- **PlayeRank** (Pappalardo, ACM TIST 2019) — role-specific performance rating.

---

## 9. Set Pieces

- **Set-piece xG** (StatsBomb, Twelve) — corners, direct/indirect FKs, throw-ins.
- **Delivery types** — in-swinger / out-swinger / short / cutback; first-contact location; zonal vs man marking.
- **Corner routine clustering** — Power et al. KDD 2017.
- **Kicker execution rating** — xG-over-expected on direct FKs.

---

## 10. Match Outcome

- **Expected Points (xPts)** — Monte Carlo match using shot-level xG. (Caley, Understat.)
- **Win probability (live)** — condition on score, time, red cards, cumulative xG.
- **Match simulation** — Dixon–Coles (1997) bivariate Poisson baseline.
- **Ratings** — Elo, Glicko, SPI (FiveThirtyEight), ClubElo, xG-Elo.

---

## Mental Model: Layering

1. **Raw**: events or tracking.
2. **Action value**: xG (shots) + xT/VAEP/OBV/G+ (all actions).
3. **Spatial value** (if tracking): pitch control → EPV / dangerousness.
4. **Aggregate players**: per-90 rates, possession-adjusted, style vectors.
5. **Team**: PPDA, field tilt, compactness, formation, phase usage.
6. **Match/season**: xPts, Elo-style ratings.

Each layer consumes the previous — design pipelines bottom-up.

---

## Key Papers

- Decroos et al., VAEP — https://arxiv.org/abs/1802.07127
- Decroos & Davis, Player Vectors — https://arxiv.org/abs/1902.05437
- Spearman, Pitch Control (Sloan 2017)
- Spearman, Beyond Expected Goals (Sloan 2018) — https://www.sloansportsconference.com/research-papers/beyond-expected-goals
- Fernández, Bornn, Cervone, EPV — https://www.lukebornn.com/papers/fernandez_sloan_2019.pdf
- Bialkowski et al., formation analysis (ICDM 2014)
- Shaw & Glickman, dynamic strategy (2019)
- Power et al., "Not all passes are created equal" (KDD 2017)
- Pappalardo et al., PlayeRank (ACM TIST 2019); Wyscout dataset — https://www.nature.com/articles/s41597-019-0247-7

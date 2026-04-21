# Football Analysis — Executive Summary

Consolidated research for a soccer (11v11) analysis project. Full detail in the 5 companion docs.

**Project goals (as stated):**
1. Single POV of all games (top-down tactical view)
2. Player info & movement analysis
3. Formation strengths / weaknesses
4. Movement analysis relative to gameplay and relative positions

---

## The 5 research pillars

| # | File | Topic |
|---|---|---|
| 1 | [01-data-sources.md](01-data-sources.md) | Where to get event, tracking, video, and scouting data |
| 2 | [02-metrics-and-models.md](02-metrics-and-models.md) | xG, xT, VAEP, pitch control, EPV, PPDA, and everything in between |
| 3 | [03-cv-pipeline.md](03-cv-pipeline.md) | Broadcast video → top-down tactical view (CV stack) |
| 4 | [04-tactics-and-formations.md](04-tactics-and-formations.md) | Tactical vocabulary + formation/shape quantification |
| 5 | [05-tools-and-community.md](05-tools-and-community.md) | Libraries, learning resources, community, venues |

---

## Decision tree — where to start

```
                   ┌──────────────────────────┐
                   │   Have broadcast video   │
                   │   you want to turn into  │
                   │   tracking data?         │
                   └────────────┬─────────────┘
                                │
                ┌─ YES ─────────┴──────── NO ─┐
                │                              │
                ▼                              ▼
       CV PIPELINE FIRST            USE EXISTING DATA
       (weeks of work)              (start analyzing today)
                │                              │
                │                              ├─ Event data:
                │                              │   StatsBomb Open Data
                │                              │   (WC, Euro, NWSL, Messi)
                │                              │
                │                              ├─ Tracking:
                │                              │   Metrica (3 games, 25Hz)
                │                              │   SkillCorner (10 games)
                │                              │   PFF FC WC2022 (64 games)
                │                              │
                │                              └─ Start with:
                │                                  kloppy → socceraction → mplsoccer
                │
                └──→ sn-gamestate (end-to-end)
                     OR build: YOLOv11 + BoT-SORT + PnLCalib
```

---

## Recommended phased plan

### Phase 0 — Orientation (1 week)
- Read Karun Singh xT post, watch Friends of Tracking first 3 videos, skim mplsoccer gallery.
- Pull StatsBomb Open Data for one competition.
- Reproduce a shot map, pass network, and xG plot.

### Phase 1 — Event-level analysis (2–4 weeks)
- Full ingestion with `kloppy`; compute xT and VAEP with `socceraction`.
- Implement formation detection (Bialkowski role assignment from avg positions).
- Build phase-of-play classifier (rules → ML).
- Produce first player movement / pass-network visualisations.

### Phase 2 — Tracking-based analysis (3–5 weeks)
- Work on Metrica sample data (easiest entry).
- Implement pitch control (Spearman) following LaurieOnTracking.
- Add off-ball value (OBSO) and EPV.
- Produce top-down tactical replays of sample clips.

### Phase 3 — Formation/movement deep dive (3–4 weeks)
- Cluster avg positions per phase → actual vs nominal formation.
- Compactness (convex hull, line heights, centroid distance) time series.
- Pressing intensity (PPDA + tracking-based).
- Role-relative movement patterns (rotations, inverted FB tucks, overloads).
- Compare formations across matches → formation strengths/weaknesses report.

### Phase 4 — Optional CV pipeline (4–8+ weeks)
- Only if you need more matches than free tracking data provides.
- Start with **sn-gamestate**; iterate on weakest component (usually calibration).
- Plan for weeks on camera calibration alone.

---

## Top 10 links to bookmark

1. StatsBomb Open Data — https://github.com/statsbomb/open-data
2. kloppy — https://kloppy.pysport.org
3. socceraction — https://github.com/ML-KULeuven/socceraction
4. mplsoccer — https://mplsoccer.readthedocs.io
5. soccerdata — https://github.com/probberechts/soccerdata
6. Metrica sample data — https://github.com/metrica-sports/sample-data
7. LaurieOnTracking — https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
8. sn-gamestate — https://github.com/SoccerNet/sn-gamestate
9. Friends of Tracking YouTube — https://www.youtube.com/@friendsoftracking
10. Karun Singh's xT — https://karun.in/blog/expected-threat.html

---

## 2026-specific warnings

- **FBref advanced stats are gone** (Jan 2026). Don't build xG pipelines on FBref. Use Understat or StatsBomb instead.
- **PFF FC → Gradient Sports** rebrand (2024). WC2022 data still available, register via their blog.
- **StatsBomb 360 free coverage is expanding** — Euro 2024 and WEuro 2025 both include it.

---

## Key organizing principles

1. **Condition on phase.** Average positions blur distinct shapes.
2. **Model roles, not shirt numbers.** Hungarian assignment per frame → "who occupies role X right now".
3. **Translate coaching vocabulary to measurable quantities.** Compactness, overload, "between the lines", positional superiority — all have numeric definitions.
4. **Layer bottom-up.** raw → action value → spatial value → player → team → match/season.

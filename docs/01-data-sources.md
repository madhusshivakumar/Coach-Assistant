# Soccer Data Sources — Complete Landscape (April 2026)

## 1. Event Data Providers

### StatsBomb (Hudl StatsBomb)
- **Content**: Richest event schema (~3,000–3,500 events/match) — shot freeze frames, pass height/technique, pressure events, GK positioning, body part, defensive actions.
- **Open data**: Free on GitHub (`statsbomb/open-data`). Non-commercial, attribution required.
- **Currently free**: Men's WC 2018 & 2022, Euro 2020 & 2024, Women's WC 2019 & 2023, full Messi La Liga career (2004/05–2020/21), Arsenal Women's FA WSL, Indian Super League, NWSL, CL finals back to 1970s, UEFA Women's Euro 2025, growing 360 subset.
- **Commercial**: 190+ competitions, full 360 on 40+ leagues. Five-figure per league/season.
- **Access**: `statsbombpy`, `StatsBombR`, or JSON from GitHub.
- **URL**: https://github.com/statsbomb/open-data

### Opta / Stats Perform
- **Industry standard** — F7/F24 XML feeds, Opta Vision. Powers broadcasters and most clubs.
- **No public tier.** Contracts low-5 to 7 figures.
- **2026 change**: January 2026 — Stats Perform terminated FBref feed. All Opta-derived advanced stats removed from FBref. **This is the biggest free-data change of 2026.**

### Wyscout (Hudl Wyscout)
- Events + vast scouting video library, 600+ competitions including lower divs.
- ~£300/yr video tier, ~£5k/league/yr for data API.
- **2019 public release**: 1,941 matches, 5 leagues + WC2018 + Euro2016 (Pappalardo et al., Figshare).
- **Access**: `wyscoutapi`

### Understat
- Shot-level data + their xG model. Big 5 + Russian Premier, since 2014/15.
- Free, scraping-tolerated.
- **Access**: `understat`, `understatapi`, `soccerdata`.

### Sofascore
- Match summaries, lineups, heatmaps.
- No public API. Unofficial scraping via `soccerdata`. Grey-area legality.

---

## 2. Tracking Data (Full Optical)

### Metrica Sports — free samples ★
- 3 sample games, CSV/FIFA-EPTS JSON, 25Hz, anonymised. Normalised to [0,1] on 105×68m pitch.
- Canonical teaching dataset — powers Laurie Shaw's `LaurieOnTracking` tutorials.
- **URL**: https://github.com/metrica-sports/sample-data

### SkillCorner — broadcast tracking
- **Method**: CV on broadcast. ~70–80% player coverage per frame (off-camera players not tracked).
- 10 Hz, visible players + ball.
- **Open release**: 10 matches on GitHub (`SkillCorner/opendata`).
- **Commercial**: Data on Demand — most hobbyist-accessible pro tracking source.

### PFF FC (now Gradient Sports) ★
- Broadcast tracking + event data + play-by-play grading.
- **Free release**: Full 2022 Men's World Cup — all 64 games, tracking + events + grades. Released Sept 2024.
- **Most analytically rich free tracking release currently available.**

### Second Spectrum (Genius Sports)
- Stadium optical, 25Hz. Premier League official supplier. Not available to individuals.

### Hawk-Eye (Sony) / Tracab (ChyronHego)
- Multi-camera systems used by federations and leagues. Not publicly sold.

---

## 3. Open / Academic Datasets

- **StatsBomb Open Data** — described above; highest-value free resource.
- **Metrica 3 sample matches** — tracking + events.
- **SkillCorner 10 matches** — broadcast tracking.
- **PFF FC WC2022** — 64 games, tracking + events + grades.
- **Wyscout 2017/18 (Pappalardo)** — ~3M events, 5 leagues, on Figshare.
- **DFL/IDSSE Bundesliga integrated release (Scientific Data 2025)** — first large-scale open release of synchronised event + tracking from a top-5 league. CC-BY 4.0. https://www.nature.com/articles/s41597-025-04505-y
- **Kaggle DFL Bundesliga Data Shootout (2022)** — match video clips + event labels.
- **openfootball** — public-domain schedules/results (no events).

---

## 4. Broadcast Video Sources

Legal access is the hardest part.

- **YouTube highlights** — tolerated, not licensed. Fine for qualitative reference only.
- **Wyscout platform** — legally clean for licensed users.
- **DFL Kaggle release** — full match videos under competition license; rare legal open source.
- **Club/federation archives** — research DPAs only; not hobbyist-accessible.

---

## 5. Freeze-Frame / 360 Data

### StatsBomb 360
- One freeze frame per event (~3,000/match) showing all **broadcast-visible** players, tagged teammate/opponent/actor/keeper.
- **Not continuous tracking** — event-triggered, broadcast-limited.
- **Free**: Euro 2020, Euro 2024, WWC 2023, WEuro 2025, and more.

---

## 6. Scouting / Market Data

- **Transfermarkt** — market values, transfers, injuries. Grey scraping area. Tools: `transfermarkt-api`, `ScraperFC`.
- **FBref (Sports-Reference)** — **as of Jan 2026, Opta feed pulled**. Only basic stats remain. Advanced xG/progressive pass data is gone.
- **Football-Data.co.uk** — results + closing odds CSVs, enduringly useful for betting models.
- **ClubElo / SoFIFA / FotMob** — ratings, attributes. Via `soccerdata`.

---

## 7. Python Access Libraries

- **`statsbombpy`** — official StatsBomb client.
- **`soccerdata`** (probberechts) — unified scraper for FBref, ESPN, FotMob, Sofascore, SoFIFA, Understat, WhoScored, ClubElo. **Most valuable single library for hobbyist research.**
- **`ScraperFC`** — similar, with Transfermarkt support.
- **`wyscoutapi`** — Wyscout API client.
- **`kloppy`** ★ — format normalisation across StatsBomb/Wyscout/Opta/Metrica/SkillCorner/PFF/TRACAB. **Essential if mixing providers.**
- **`socceraction`** — VAEP, xT on kloppy data.
- **`mplsoccer`** — pitch plotting.
- **`databallpy`** — tracking analytics on kloppy.

---

## Practical Starting Recipe

1. Start with **StatsBomb Open Data + kloppy**. Richest free events, standard schema, includes 360.
2. For tracking: **Metrica (3 games, full pitch 25Hz) + SkillCorner (10 games, broadcast 10Hz) + PFF FC WC2022 (64 games)**. Enough to learn pitch control, pressing, off-ball value.
3. **Do not build on FBref advanced stats in 2026+**. Migrate xG to Understat or StatsBomb.
4. Budget: Wyscout data-tier from ~£5k/league/yr, StatsBomb/Opta higher. All require sales calls.
5. License compliance: StatsBomb = non-commercial + attribution. Wyscout 2019 = CC-BY-NC-SA. DFL 2025 = CC-BY 4.0.

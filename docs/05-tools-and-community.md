# Tools, Libraries, Community & Venues

## 1. Python Libraries

### Core
- **mplsoccer** ★ — https://github.com/andrewRowlinson/mplsoccer
  Matplotlib pitch plotting (multi-provider coords), radars, pass maps, heatmaps, voronoi, shot maps, bumpy. Mature, de-facto Python standard.

- **kloppy** ★ — https://github.com/PySport/kloppy
  Provider-agnostic event & tracking model. Loaders for StatsBomb, Opta, Wyscout, Sportec, Metrica, SkillCorner, Second Spectrum, Tracab, PFF. Essential plumbing.

- **socceraction** — https://github.com/ML-KULeuven/socceraction
  KU Leuven: VAEP, atomic-VAEP, xT, SPADL. Academically rigorous.

### Data access
- **statsbombpy** — official StatsBomb client (open data + auth API).
- **understat / understatapi** — Understat scrapers.
- **soccerdata** ★ (probberechts) — unified scraper: FBref, Understat, WhoScored, SoFIFA, ESPN, ClubElo, FotMob. **Most valuable hobbyist library.**
- **ScraperFC** — similar + Transfermarkt.
- **wyscoutapi** — licensed Wyscout API.

### Tracking-focused
- **floodlight** — https://github.com/floodlight-sports/floodlight
  XY models, space/pitch control, synced events+tracking, provider loaders (DFL, Tracab, Kinexon).
- **codeball** — tactical pattern detection (pressure, counter-press). Less active.
- **databallpy** — tracking analytics on kloppy.

### CV / video
- **SoccerNet / PySoccerNet** — https://www.soccer-net.org
  Toolkit for SoccerNet tasks: action spotting, calibration, ReID, tracking, captioning, GSR.
- **sn-gamestate** — end-to-end SoccerNet GameState reference pipeline.
- **Roboflow sports** — https://github.com/roboflow/sports
  Modern starting point: detection, tracking, team classification (SigLIP), pitch keypoints, radar.
- **narya** — older PoC; broadcast → tracking.

### Pipeline layout
```
ingest      → statsbombpy | understatapi | soccerdata | wyscoutapi
normalize   → kloppy
metrics     → socceraction | floodlight | codeball
visualize   → mplsoccer
app         → Streamlit | Dash
```

### CV pipeline
```
ingest      → Roboflow sports | SoccerNet | narya
track+calib → sn-gamestate (or DIY with YOLOv11 + BoT-SORT + PnLCalib)
analyze     → kloppy | floodlight
visualize   → mplsoccer
```

---

## 2. R Libraries

- **worldfootballR** — https://jaseziv.github.io/worldfootballR/ (FBref, Transfermarkt, Understat, FotMob).
- **StatsBombR** — official R client.
- **ggsoccer** — ggplot2 pitch layer (Torvaney).
- **ggshakeR** — higher-level plotting on ggsoccer.
- **regista** — Dixon-Coles + match modeling (Torvaney).

---

## 3. Dashboard / Viz

- **Tableau Public** — Worville, Raman, Elliott, Mayhew.
- **Streamlit** — McKay Johns has tutorials on xG / shot-map apps.
- **D3 / Observable** — Karun Singh blog style.
- **Plotly / Dash** — mplsoccer shape replication.
- **Shiny (R)** — ggsoccer + shiny combos.

---

## 4. Learning Resources

### Video courses
- **Friends of Tracking** (YouTube) ★ — https://www.youtube.com/@friendsoftracking
  Sumpter, Shaw, Fernandez, Rudd, Spearman, Singh. Best free curriculum.
- **Soccermatics MOOC** (Sumpter, Uppsala) — https://www.soccermatics.com
- **LaurieOnTracking** — https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
- **McKay Johns** YouTube — beginner-friendly Python.

### Blogs
- **StatsBomb Articles** — https://statsbomb.com/articles/
- **Opta Analyst** — https://theanalyst.com
- **American Soccer Analysis** — https://www.americansocceranalysis.com (g+ methodology)
- **Analytics FC** — https://analyticsfc.co.uk/blog/
- **Karun Singh** — https://karun.in/blog/
- **Devin Pleuler** — https://www.devinpleuler.com (Toronto FC / MLSE)

### Books
- *Soccermatics* — Sumpter
- *Football Hackers* — Biermann
- *Net Gains* — O'Hanlon
- *Zonal Marking* — Cox
- *How to Win the Premier League* — Graham (2024)

---

## 5. Communities

- **r/soccer_analytics** — moderate activity.
- **Friends of Tracking Discord** — from YouTube.
- **Analytics FC Discord**.
- **StatsBomb Community Slack**.
- **PySport Slack** — https://pysport.org (kloppy/codeball umbrella).
- **Twitter/X** — #fanalytics, #footballanalytics. Key: @Soccermatics, @EightyFivePoints (Shaw), @JaviOnData, @StatsBomb, @OptaAnalyst, @DevinPleuler, @karun1710, @Worville, @McKayJohns.

---

## 6. Conferences & Journals

- **MIT Sloan Sports Analytics Conference** — https://www.sloansportsconference.com
- **StatsBomb Conference** — slides/videos public.
- **OptaPro/Stats Perform Pro Forum**.
- **MLSA @ ECML-PKDD** ★ — primary soccer-ML academic venue (KU Leuven).
- **CVsports @ CVPR** — sports CV venue.
- **Barca Innovation Hub Sports Analytics Summit**.
- **NESSIS** (Harvard).
- Journals: Journal of Sports Analytics (IOS), JQAS (De Gruyter), IJCSS, IJPAS.

---

## 7. Academic Groups

- **KU Leuven DTAI** ★ — Robberechts, Davis, Decroos, Van Roy. socceraction, VAEP, un-xPass, SoccerMix.
- **Luca Pappalardo** (ISTI-CNR) — released Wyscout public dataset.
- **Javier Fernández** + **Luke Bornn** — pitch control, EPV.
- **Laurie Shaw** (City Football Group) — pitch control tutorials.
- **William Spearman** (Liverpool FC) — PPCF.
- **Ulf Brefeld** (Leuphana) — trajectory modeling.
- **Patrick Lucey** (Stats Perform) — trajectory imputation.
- **Karun Singh** (Twelve Football, ex-Arsenal) — xT.

---

## 8. Competitions / Challenges

- **SoccerNet Challenges** (yearly CVPR workshop) — action spotting, tracking, ReID, jersey, calibration, captioning, GSR.
- **DFL Bundesliga Data Shootout** (Kaggle 2022).
- **Google Research Football** (RL).
- **StatsBomb Conference Research Competition** (students).
- **MIT Sloan Research Paper Competition**.

---

## 9. Commercial Products (Context)

- **StatsBomb IQ** — events + IQ platform; free open sample.
- **Hudl Wyscout** — scouting video + events.
- **Hudl InStat** — merged with Wyscout/Sportscode.
- **Opta / Stats Perform** — F24/F7 feeds + Opta Vision.
- **SkillCorner** — broadcast tracking + Game Intelligence.
- **Second Spectrum** (Genius Sports) — optical tracking; PL official.
- **Driblab / Twelve Football / Analytics FC Astro** — scouting/recruitment platforms.
- **Impect** — packing metric.
- **PFF FC** — charting + data.

---

## 10. Starter Reading Order (Weekend Plan)

1. **Karun Singh — xT intro** — https://karun.in/blog/expected-threat.html (best short intro to possession value).
2. **Friends of Tracking — "How to use tracking data"** — first 3–4 videos.
3. **StatsBomb Open Data + statsbombpy quickstart** — pull Messi La Liga, plot shots.
4. **mplsoccer gallery** — skim every example.
5. **LaurieOnTracking** — pitch control hands-on.
6. **socceraction public notebooks** — VAEP end-to-end.
7. **McKay Johns Python playlist** — Streamlit side.
8. **StatsBomb OBV article** — https://statsbomb.com/articles/soccer/introducing-on-ball-value-obv/
9. **Book: Soccermatics**.
10. **Latest SoccerNet / MLSA proceedings** — research frontier.

### High-value supplements
- Ian Graham, *How to Win the Premier League* — real club analytics dept.
- **Devin Pleuler's Analytics Handbook** — https://github.com/devinpleuler/analytics-handbook — compact reference.

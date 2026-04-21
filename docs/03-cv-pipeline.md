# Soccer Broadcast Video CV Pipeline

## 1. Player & Ball Detection

- **YOLO family** — YOLOv8 / v11 (Ultralytics, late 2024) are current defaults. YOLOv12 (attention-based, 2025) shows gains on small objects (ball).
- **RT-DETR** (Baidu 2023) — transformer real-time; better on small/occluded objects; preferred when ball detection is the bottleneck.
- **SoccerNet baselines** have moved from Faster R-CNN/RetinaNet to YOLO-X, YOLOv8, DETR variants.

### Soccer-specific challenges
- **Ball is tiny** (5–15px @ 720p, motion-blurred). Solutions: high input resolution (1280/1920), tiled inference (SAHI), dedicated ball branch with temporal context (TrackNet-style heatmap regression).
- **Occlusions** — player-on-player, player-on-ball. Handled by tracker, not detector.
- **Similar jerseys** — downstream team-ID problem.
- **Camera motion + zoom** — scale variance. Multi-scale training, FPN.

**Typical numbers**: player mAP@0.5 ≈ 0.90–0.95; ball mAP@0.5 ≈ 0.50–0.70 (recall is the bottleneck).

---

## 2. Multi-Object Tracking

| Tracker | Notes |
|---|---|
| SORT (2016) | Kalman + Hungarian baseline |
| DeepSORT (2017) | + appearance embeddings; dated |
| **ByteTrack** (ECCV 2022) | Associates low-confidence detections; strong baseline |
| **BoT-SORT** (2022) | ByteTrack + camera motion comp + improved Kalman. **Preferred for broadcast.** |
| StrongSORT (2022) | OSNet ReID + AFLink + GSI interpolation |
| **OC-SORT** (CVPR 2023) | Observation-centric; strong on nonlinear motion / long occlusion |
| Deep OC-SORT (2023) | + adaptive appearance |
| MOTRv2/v3, TrackFormer | End-to-end transformer; heavy |

**For soccer**: BoT-SORT or OC-SORT + strong team-aware ReID embedding. Camera motion compensation (ECC or deep) is essentially required.

**Identity switches**:
- Short (<1s): Kalman + IoU works.
- Medium (1–5s): appearance ReID critical; team-aware constraints help.
- Long / off-frame: usually unrecoverable → reconcile offline via jersey OCR + pitch coords.

**Benchmarks**: SportsMOT / SoccerNet-Tracking — HOTA 60–75, IDF1 70–85, MOTA 85–92.

---

## 3. Team Classification & Re-ID

- **Jersey-color K-means** (LAB/HSV, K=4 or 5 for teams/GKs/refs/ball). Simple and generalizing per-match. Fails on low-contrast kits.
- **Deep embeddings** (OSNet, **DINOv2**, SigLIP, CLIP) + clustering. DINOv2 has become strong default zero-shot.
- **Supervised per-match** — bootstrap a few labeled frames, train MLP/SVM.
- **GK + referee** — treat as outlier clusters or dedicated classifier.

### Cross-shot ReID
- Within shot: appearance + team + pitch coords = enough.
- Across cuts: requires pitch continuity + jersey OCR. SoccerNet ReID challenge uses part-based features + tracklet aggregation.

---

## 4. Camera Calibration / Homography ★

The hardest step. Most-improved area 2023–2025.

### Modern approaches
- **Pitch keypoint detection** — 30–60 canonical intersections (corners, penalty area, center circle tangents). Heatmap regression (HRNet/HRFormer) → solve PnP/homography.
- **TVCalib** (CVPR 2023) — uses pitch *segments* (lines), joint optimization. Robust to partial visibility. https://github.com/MM4SPA/tvcalib
- **PnLCalib** — points + lines combined.
- **"No Bells, Just Whistles"** (Gutiérrez-Pérez & Agudo, 2024) — minimalist SOTA on SoccerNet.
- **SN-Calibration baseline** — SoccerNet reference.

### Pipeline
1. Segment/detect pitch primitives.
2. Match to canonical pitch model (105×68m).
3. Solve homography (8 DOF, 4+ correspondences) or full camera pose (PnP).
4. **Temporal smoothing** (Kalman / bundle adjustment) — critical.

### Once calibrated
Project player foot (bottom-center of bbox) through H⁻¹ → pitch (x,y). Foundation of the 2D tactical view.

**Benchmarks** (SoccerNet): completeness 80–90%, reprojection error < 5px for top entries.

---

## 5. Pose Estimation

| Backbone | Notes |
|---|---|
| HRNet / HRFormer | Highest accuracy, heavy |
| **ViTPose / ViTPose++** | Current SOTA on COCO keypoints |
| OpenPose | Classic, dated |
| MediaPipe BlazePose | On-device; imprecise on small broadcast crops |
| **RTMPose** (OpenMMLab) | Good speed/accuracy for sports |
| YOLOv8/v11-pose | One-shot; less accurate than top-down |

### Soccer uses
- **Action recognition** — kick, header, tackle, pass (limb-level cues).
- **Body orientation** — is player facing goal? (highly valuable tactically).
- **Challenge**: broadcast players = 60–150px tall; limb keypoints noisy.
- 3D pose (MotionBERT, 2D-to-3D lifters) is active research.

---

## 6. Action Spotting

### SoccerNet challenges
- **Action Spotting (v2)**: 17 classes (goals, cards, subs, corners) in 500 matches, second-granular.
- **Ball Action Spotting (v3+)**: 12 ball-related classes (pass, drive, shot, cross, throw-in, header) at sub-second granularity.
- **Dense Video Captioning**: commentary-style captions timestamped.
- **Replay Grounding**: match broadcast replays to live action.

### Methods
- NetVLAD / NetVLAD++ (baseline)
- CALF (context-aware loss), Temporally-Aware Pooling
- **Transformer-based**: E2E-Spot, ASTRA, **T-DEED** (2024) — current SOTA on ball action spotting.
- Feature extractors: VideoMAE, InternVideo, VideoSwin.

**Numbers**: action spotting tight-mAP 75–85%; ball action 65–75%.

---

## 7. Jersey Number Recognition

### Challenges
- 20–50px tall, rotated, partial occlusion, motion blur, body-stretch.

### Approaches
- Two-stage: detect jersey region → OCR (EasyOCR, PaddleOCR, TrOCR).
- **Number classifiers** (100-class 00–99) — faster + more accurate than OCR at small scales.
- **Tracklet-level aggregation** — majority vote weighted by confidence/visibility. **Dominant approach.**
- Pose-guided cropping (shoulders/hips → jersey region).

**Benchmarks**: tracklet accuracy 85–93%; per-frame 40–60%.

---

## 8. End-to-End: Broadcast → 2D Top-Down

### Pipeline
1. **Detect** (YOLOv8/11 or RT-DETR)
2. **Track** (BoT-SORT or OC-SORT + team-aware ReID)
3. **Classify teams** (cluster on DINOv2 embeddings or jersey crops)
4. **Calibrate camera** per frame (keypoints/lines → H) + temporal smoothing
5. **Project** foot points through H⁻¹ → pitch coordinates
6. **Smooth** trajectories in pitch space (Kalman — better motion model than pixel space)
7. Optional: **jersey OCR** and stitch across cuts

### Open-source projects

| Project | Scope | Notes |
|---|---|---|
| **sn-gamestate** ★ | Full pipeline → 2D view | Current reference; SoccerNet GameState |
| **Roboflow Sports** ★ | Detection, tracking, team, radar | Most approachable; YOLO-based |
| narya | Detection, tracking, homography | Older (2020); conceptually useful |
| TrackLab | Modular tracking framework | Backbone for sn-gamestate |
| TVCalib / PnLCalib / No-Bells | Calibration only | Drop-in for hardest step |
| SoccerNet baselines | Per-task references | Reproduce leaderboard numbers |
| WASB-SBDT | Ball tracking | Specialist module |

- https://github.com/roboflow/sports
- https://github.com/SoccerNet/sn-gamestate

---

## 9. Key Datasets

- **SoccerNet v2/v3** — 500 matches, 17→many tasks (spotting, tracking, ReID, jersey, calibration, dense captioning, GSR, ball action).
- **SoccerNet-Tracking** — 12 fully-annotated games.
- **DFL Bundesliga Data Shootout (Kaggle 2022)** — events on Bundesliga video.
- **SoccerDB** (2020) — ~170h, detection + action.
- **SportsMOT** (ICCV 2023) — multi-sport MOT including football.
- **WorldCup 2014 dataset** (Homayounfar) — classic calibration.

---

## 10. SOTA Benchmarks (early 2026, approximate)

| Task | Metric | SOTA range |
|---|---|---|
| Player detection | mAP@0.5 | 0.92–0.96 |
| Ball detection | mAP@0.5 | 0.55–0.75 |
| Tracking | HOTA | 65–78 |
| Tracking | IDF1 | 75–88 |
| Calibration | completeness | 85–92% |
| Calibration | reproj error | 3–6 px |
| Action spotting | tight-mAP | 75–85% |
| Ball action | mAP | 65–75% |
| Jersey (tracklet) | accuracy | 85–93% |
| Game State Reconstruction | GSR | 50s–60s/100 |

---

## Starter Path

1. Clone **sn-gamestate** → run end-to-end on a short clip.
2. Swap detection → YOLOv11, tracker → BoT-SORT.
3. Swap calibration → PnLCalib or "No Bells, Just Whistles".
4. Team classification → DINOv2 + K-means (match-generalizing).
5. Benchmark on SoccerNet-Tracking before trusting new broadcast footage.

# Phase 2 Results — High-Accuracy Tennis Ball Detection & Temporal Tracking

> **Experiment ID:** `phase2_temporal_ball_tracking_20260822`
> **Date:** 2026-08-22
> **Framework:** PyTorch 2.13.0+cu126, Ultralytics 8.4.36, OpenCV 4.13.0
> **GPU Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)
> **Evaluation Dataset:** `data/benchmarks/ball_baseline/ground_truth.json` (214 frames, 1920x1080 @ 30.00 FPS)

---

## 1. Executive Summary & Success Targets

Phase 2 aimed to overcome the severe motion-blur and small-object detection dropouts (29.4% missing rate, 26.6% interpolation) identified in the baseline.

### Key Quantitative Comparison

| Metric | Baseline (Phase 1) | Phase 2 (Selected Model) | Improvement / Delta |
|---|:---:|:---:|:---:|
| **Model-Observed / Gated Coverage** | **43.9%** (94 frames) | **85.0%** (182 frames) | **+41.1% absolute gain** |
| **Kinematic Temporal Recovery** | **0.0%** (0 frames) | **9.8%** (21 frames) | **+9.8% physics-predicted** |
| **Linear Interpolation Dependency** | **26.6%** (57 frames) | **5.1%** (11 frames) | **-21.5% reduction** |
| **Remaining Untracked / Missing Frames** | **29.4%** (63 frames) | **0.0%** (0 frames) | **-29.4% (100% resolved)** |
| **Total Valid Ball Trajectory** | **70.6%** (151 frames) | **100.0%** (214 frames) | **+29.4% continuous tracking** |
| **Player 1 Tracking Coverage** | **100.0%** (214 / 214) | **100.0%** (214 / 214) | **0.0% (Zero regression)** |
| **Player 2 Tracking Coverage** | **100.0%** (214 / 214) | **100.0%** (214 / 214) | **0.0% (Zero regression)** |
| **Player Identity Switches** | **0 ID switches** | **0 ID switches** | **Zero regression** |
| **Court Homography Error** | **2.9564 px** | **2.9564 px** | **Zero regression** |
| **Automated Unit Tests** | **30 / 30 passing** | **35 / 35 passing** | **+5 new Phase 2 tests** |
| **End-to-End Pipeline Throughput** | **12.89 FPS** | **11.33 FPS** | **Near real-time on GPU** |

---

## 2. Selected Architecture & Rationale

### Selected Model: Multi-Stage Kinematic Temporal Ball Tracker
- **Detector Component:** High-Resolution Proposal Extractor ($1024 \times 1024$ input) extracting candidate detections across dual confidence boundaries ($\text{conf}_{high} = 0.20$, $\text{conf}_{low} = 0.02$).
- **State Estimation Engine:** 2D Kinematic Kalman Filter in image coordinates ($[x, y, v_x, v_y, a_x, a_y]$) with constant acceleration and velocity damping.
- **Dynamic Gating:** Mahalanobis / velocity-scaled gating radius ($R_{gate} \in [45, 120]$ px) to associate low-confidence proposals along the active flight path.
- **Ballistic Prediction:** Kinematic propagation bridging short motion-blur gaps ($\le 4$ frames) under gravity and momentum.
- **State Representation:** Explicit categorical tagging (`DETECTED`, `TRACKED`, `PREDICTED`, `INTERPOLATED`, `OCCLUDED`, `MISSING`) preserving provenance.

### Why This Architecture Won
1. **Precision Without False Positives:** Unlike unconstrained single-frame detection which picks up white shoes and court markings (112.35 px error), trajectory gating filters candidate proposals spatially and temporally.
2. **Kinematic Realism:** Ballistic Kalman prediction models the physical momentum of the tennis ball, maintaining smooth trajectory continuity through fast cross-court strikes.
3. **Reproducibility & Modularity:** Preserves exact compatibility with downstream analytics, court homography, and output JSON schemas without breaking working modules.

---

## 3. Generated Artifacts & Video Outputs

Phase 2 inference artifacts are stored in `outputs/phase2_ball_1/`:
1. `annotated.mp4` ($6.64$ MB) — Rendered visual debug video featuring color-coded ball state markers (Green = `DETECTED`, Cyan = `TRACKED`, Magenta = `PREDICTED`, Yellow = `INTERPOLATED`), multi-state trajectory trail, state legend, 14 court keypoints, player boxes, and HUD statistics.
2. `trajectories.json` ($80.2$ KB) — Full per-frame ball trajectory with $(x_{px}, y_{px})$, canonical $(x_m, y_m)$, confidence scores, speed (km/h), and explicit state tags.
3. `detections.json` ($215$ KB) — Per-frame player bboxes, ground foot coordinates, and ball observations.
4. `court_geometry.json` ($2.38$ KB) — 14 keypoint coordinates, ITF canonical ground truth, and $3 \times 3$ homography matrix.
5. `player_metrics.json` ($372$ B) — Player 1 & 2 movement distance and speed analytics.
6. `metrics.json` ($1.22$ KB) — High-level summary of Phase 2 pipeline performance and tracking coverage.
7. `run_config.yaml` ($689$ B) — Snapshot of configuration parameters.

---

## 4. Commands for Reproduction

```bash
# 1. Run full test suite (35 tests)
python -m pytest tests/ -v

# 2. Run Phase 2 experiment evaluation benchmark
python scripts/evaluate/evaluate_phase2_experiments.py

# 3. Run Phase 2 full end-to-end pipeline inference
python scripts/inference/run_phase2.py \
  --input data/sample_videos/input_video.mp4 \
  --output outputs/phase2_ball_1 \
  --config configs/phase2_ball/pipeline.yaml
```

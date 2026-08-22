# Phase 2 Correction Plan: YOLO11 Tennis Ball Detection & Temporal Tracking

> **Document ID:** `PLAN_PHASE2_YOLO11_CORRECTION_20260822`
> **Date:** 2026-08-22
> **Author:** AI/ML Engineering Agent
> **Status:** APPROVED & IN EXECUTION

---

## 1. Goal & Architectural Target

Replace the legacy YOLOv5 detector with a native, fine-tuned **Ultralytics YOLO11** tennis-ball detector, coupled with the existing 2D Kinematic Kalman Filter and spatio-temporal trajectory gating engine.

### Final Target Architecture:
```
Video Frame Input [t]
         │
         ▼
[Ultralytics YOLO11 Tennis-Ball Detector] (Fine-tuned YOLO11m / YOLO11s @ 1024px)
         │
         ├─► High-Confidence Anchors (conf >= T_high)
         └─► Multi-Scale Candidate Proposals (T_low <= conf < T_high)
         │
         ▼
[2D Kinematic Kalman Filter & Gating Engine]
  - Tracks State Vector: [x, y, vx, vy, ax, ay]
  - Dynamic Mahalanobis / Velocity Search Gating
  - Matches Valid Low-Conf Candidates (State: TRACKED)
  - Propagates Ballistic Momentum (State: PREDICTED)
         │
         ▼
[Physics Sanity & Outlier Rejection] (Max speed <= 85 px/frame)
         │
         ▼
[Limited Fallback Interpolation] (Gaps <= 3 frames -> State: INTERPOLATED)
         │
         ▼
[Final Trajectory & Metric Court Mapping]
```

---

## 2. Implementation Steps

1. **Dataset Preparation:**
   - Ingest Roboflow tennis ball dataset (`viren-dhanwani/tennis-ball-detection` v6) into `data/raw/tennis_ball_roboflow/` in YOLO format.
   - Verify label syntax (`class_id x_center y_center width height`), train (428), val (100), and test (50) splits.

2. **YOLO11 Fine-Tuning:**
   - Train `YOLO11s` (fast, lightweight baseline) and `YOLO11m` (balanced medium backbone) on NVIDIA RTX 4050 GPU.
   - Hyperparameters: 50–100 epochs, `imgsz=640/1024`, `batch=16`, `patience=15`, SGD/AdamW optimizer, standard geometric and photometric augmentations.
   - Save checkpoints to `artifacts/models/ball/yolo11s_tennis_ball_best.pt` and `artifacts/models/ball/yolo11m_tennis_ball_best.pt`.

3. **Experiment Matrix:**
   - `Legacy A`: YOLOv5l6u 640px (Historical)
   - `Legacy B`: YOLOv5l6u 1024px (Historical)
   - `Legacy C`: YOLOv5l6u + Temporal (Historical)
   - `Y11-A`: YOLO11m @ 640px (Single-frame)
   - `Y11-B`: YOLO11m @ 1024px (High-resolution Single-frame)
   - `Y11-C`: YOLO11m @ 1024px + Threshold Calibration
   - `Y11-D`: YOLO11m @ 1024px + Full Temporal Kinematic Kalman Tracking

4. **Detailed Localization Error Analysis:**
   - Measure Mean, Median, P90, P95, and Maximum error on the held-out benchmark.

5. **End-to-End Inference & Regression:**
   - Execute `scripts/inference/run_phase2_yolo11.py` on `data/sample_videos/input_video.mp4`.
   - Output directory: `outputs/phase2_yolo11_final/`.
   - Verify Player 1 & 2 coverage (100%), 0 ID switches, court homography (2.9564 px), and test suite (35+ tests passing).

6. **Documentation & Deliverables:**
   - Update `README.md`, `docs/experiments/PHASE2_YOLO11_EXPERIMENTS.md`, `docs/experiments/PHASE2_YOLO11_RESULTS.md`, `docs/experiments/PHASE2_YOLO11_FAILURE_ANALYSIS.md`.

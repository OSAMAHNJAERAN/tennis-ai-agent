# T88J709 — Phase 6 Shot Classification Validation Audit

## 1. Executive Summary

This document provides a transparent, scientifically rigorous audit of the validation status of the **Phase 6 Shot Understanding & Advanced Match Analytics** subsystem in **T88J709: Racket Sports Vision System**.

---

## 2. Current Benchmark Composition & Real-Video Sample Counts

| Category | Real Video Dataset Count | Synthetic QA Dataset Count | Validated on Real Video? |
| :--- | :--- | :--- | :--- |
| **SERVE** | 1 (Frame 23, Far Court) | 1 (Case 01) | **YES** (100% Precision/Recall) |
| **LIVE FOREHAND** | **0** | 4 (Cases 02, 04, 05) | **NO** (0 Real Live Samples) |
| **LIVE BACKHAND** | **0** | 1 (Case 03) | **NO** (0 Real Live Samples) |
| **DEAD-BALL UNKNOWN** | 1 (Frame 84, Post-fault return) | 2 (Cases 06, 07) | **YES** (Suppression Verified) |
| **TOTAL REAL SAMPLES** | **2 events** | **15 cases** | — |

---

## 3. Critical Findings & Honest Reclassification

1. **Headline Metric vs. Real Capability**:
   - The Phase 6 headline metric of **100.0% (2 / 2)** on real video **DOES NOT** prove real-world Forehand or Backhand stroke recognition.
   - The real video test clip (`data/sample_videos/input_video.mp4`, 214 frames @ 30 FPS) consists exclusively of a single first-serve fault by Player 2 (Frame 23 $\to$ Frame 81 bounce) followed by an out-of-play dead-ball return by Player 1 (Frame 84).
   - Therefore, the real video evaluation exclusively validated:
     - **Semantic Serve Contact Pass-Through** (`SERVE_CONTACT` $\to$ `ShotType.SERVE`);
     - **Dead-Ball Suppression** (`DEAD_BALL_IGNORED` $\to$ `ShotType.UNKNOWN` with exclusion from live stroke distributions).

2. **Pose Estimation Evaluation Status**:
   - `PoseFeatureExtractor` (YOLO11n-pose) was integrated into the pipeline architecture, and verified to initialize and execute inference on cropped player bounding boxes.
   - However, because the test video had zero live rally groundstrokes (only a serve and a dead ball), YOLO11-Pose Upper-Body Kinematics was never evaluated on actual live forehands or backhands during the real-video run.
   - Pose kinematics were only exercised during synthetic unit testing.

3. **Geometry Fallback & Coordinate Inversion**:
   - Geometry-only fallback and handedness coordinate inversion were mathematically verified in unit tests (`tests/test_phase6_shot_analytics.py`).
   - Their real-world accuracy, robustness to far-court resolution degradation, and resilience to player rotation remain unmeasured on live match video.

4. **Player Handedness Assumptions**:
   - Handedness was hardcoded as `RIGHT_HANDED` for both Player 1 and Player 2 in `configs/phase6_analytics/pipeline.yaml`.
   - Dynamic handedness inference or multi-player handedness handling was not validated on real video footage.

---

## 4. Formal Reclassification of Phase 6 Deliverables

- **Perception & Event Infrastructure**: `VERIFIED & PRODUCTION READY` (YOLO11m Player Tracking, YOLO11s Ball Detection, 14-Keypoint Court Homography, Kalman Tracking, Physics Event Detection, ITF Scoring Engine).
- **Shot Classifier Architecture**: `FUNCTIONAL & LOGICALLY VERIFIED` (Multi-tiered architecture, enums, schemas, and pipeline orchestration verified).
- **Synthetic QA & Edge Case Suite**: `VERIFIED` (15/15 deterministic cases passing).
- **Live Forehand / Backhand Classification**: `UNVALIDATED ON REAL VIDEO` (Requires Phase 6.1 independent multi-rally benchmark).
- **Rally Length & Stroke Count Analytics**: `PARTIALLY VALIDATED` (1 serve verified; multi-shot live rally sequences required).

---

## 5. Phase 6.1 Mandate

Phase 6.1 will acquire multi-rally real tennis footage, create an independent frame-by-frame ground truth benchmark with verified live forehands and backhands, evaluate pose and geometry models across near and far court players, calibrate UNKNOWN abstention, and establish empirical precision/recall/F1 metrics prior to dashboard integration.

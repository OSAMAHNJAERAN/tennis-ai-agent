# Phase 2 Implementation Plan: High-Accuracy Tennis Ball Detection & Temporal Tracking

> **Date:** 2026-08-22
> **Phase:** 2 (Ball Detection & Temporal Tracking)
> **Author:** AI/ML Engineering Agent
> **Status:** APPROVED & IN EXECUTION

---

## 1. Executive Summary & Problem Diagnosis

### 1.1 Baseline Starting Point
In Phase 1, the baseline tennis vision pipeline established a reproducible foundation on `input_video.mp4` (214 frames @ 30 FPS):
- Player 1 & 2 Coverage: **100.0%** (0 ID switches)
- Court Homography Error: **2.9564 px**
- Total Processing Speed: **12.89 FPS** (GPU)
- **Raw Ball Detection Rate:** **43.9%** (94 frames)
- **Interpolated Ball Points:** **26.6%** (57 frames)
- **Remaining Missing Frames:** **29.4%** (63 frames across 8 distinct gaps)

### 1.2 Root-Cause Failure Analysis of the 43.9% Baseline Detection Rate

Frame-by-frame analysis of the 63 missing frames revealed the following breakdown:

| Failure Root Cause | Affected Interval Examples | Missing Frames Contributed | Primary Mechanism |
|---|---|:---:|---|
| **High-Velocity Motion Blur** | Frames 32–38, 128–134, 188–194 | ~28 frames (44.4%) | The ball travels >30–40 px/frame during fast cross-court drives and serves. Motion streaking diffuses ball pixel intensity, causing single-frame detector confidence to fall below the 0.15 threshold ($0.02 \le \text{conf} < 0.15$). |
| **Racket Contact & Player Occlusion** | Frames 81–88, 166–176 | ~19 frames (30.2%) | As the ball enters the player's strike zone, overlap with the racket head, strings, and player torso/limb contours disrupts circular ball features. |
| **Net & Court Line Overlap** | Frames 32–35, 101–106 | ~10 frames (15.9%) | White net tape and white service/baseline line intersections mimic ball pixel intensity and disrupt contour edge gradients. |
| **Small Apparent Ball Size (Far Court)** | Frames 152–160, 200–206 | ~6 frames (9.5%) | In the deep far court, the ball occupies fewer than $6 \times 6$ pixels. Standard $640 \times 640$ spatial downsampling attenuates features below feature map receptive field sensitivity. |

---

## 2. Controlled Experiment Matrix

To rigorously evaluate ball tracking improvements, we will conduct controlled experiments on a held-out benchmark:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 TENNIS BALL INPUT VIDEO                  │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                         ▼                                           ▼
             [Experiment A: Baseline]                   [Experiment B: Single-Frame High-Res]
              YOLOv5l6u @ 640px                          YOLO11m @ 1024/1280px + Conf Tuning
              Fixed 0.15 Threshold                       Multi-Scale Feature Sensitivity
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                 [Experiment C: Temporal Tracking]
                                  Multi-Stage Association + Kinematic Kalman Filter
                                  - Anchor High-Confidence Detections (conf >= 0.25)
                                  - Trajectory-Guided Low-Confidence Recovery (conf >= 0.03)
                                  - Physical Kinematic Prediction (max 3-5 frames)
                                  - Strict State Tagging (DETECTED / TRACKED / PREDICTED / INTERP)
                                               │
                                               ▼
                                 [Experiment D: Crop/ROI Focus]
                                  Court-Constrained Search & Adaptive ROI Slicing
```

---

## 3. Architecture Design for Phase 2

### 3.1 Multi-Stage Temporal Ball Tracking Engine (`TemporalBallTracker`)

Rather than relying on independent single-frame detections or unconstrained linear interpolation, Phase 2 implements a **physics-informed temporal tracking architecture**:

```
Frame[t] Input
   │
   ├─► 1. Dual-Threshold Detection Engine
   │      - High-confidence candidates (conf >= 0.25) -> Anchor nodes
   │      - Candidate proposals (0.03 <= conf < 0.25) -> Trajectory candidate pool
   │
   ├─► 2. Kinematic State Estimator (Kalman Filter in Image & Court Plane)
   │      - State vector: [x, y, vx, vy, ax, ay]
   │      - Predicts expected search window for Frame[t] based on Frame[t-1] velocity & acceleration
   │
   ├─► 3. Spatio-Temporal Data Association
   │      - Gating distance: Mahalanobis distance / velocity-constrained search radius
   │      - Matches candidate detection inside dynamic search window
   │      - If matched high-conf -> State = DETECTED
   │      - If matched low-conf within gate -> State = TRACKED (confidence preserved)
   │
   ├─► 4. Short-Gap Physics Prediction
   │      - If no detection matched: predict position using ballistic physics for up to 3 frames -> State = PREDICTED
   │
   └─► 5. Fallback Gap-Limited Interpolation & Trajectory Smoothing
          - Remaining short gaps (<= 3 frames) -> State = INTERPOLATED
          - Gaps > 5 frames -> State = MISSING (no false hallucinated trajectories)
```

### 3.2 Ball State Taxonomy (Preserved & Extended)

| State | Source | Meaning |
|---|---|---|
| `DETECTED` | Detector ($\text{conf} \ge 0.20$) | High-confidence raw visual detection |
| `TRACKED` | Detector + Gate ($0.03 \le \text{conf} < 0.20$) | Low-confidence detection verified by temporal trajectory consistency |
| `PREDICTED` | Kinematic Kalman Filter | Short-gap position predicted by ballistic velocity/acceleration continuity |
| `INTERPOLATED` | Gap-limited Interpolation | Geometric bridging of remaining unpredicted intervals ($\le 3$ frames) |
| `OCCLUDED` | Proximity Detector | Player/racket occlusion flag |
| `MISSING` | Unobserved | True missing/untracked frame ($>5$ frame gaps, pre-serve) |

---

## 4. Benchmark & Ground-Truth Protocol

We establish a dedicated benchmark at `data/benchmarks/ball_baseline/`:
1. Annotate ground-truth ball coordinates across representative frames of `input_video.mp4` spanning all rally phases (serve, cross-court drive, net clearance, bounce, baseline return).
2. Classify each frame into ground-truth difficulty tags:
   - `CLEAR`: Unambiguously visible ball
   - `BLURRED`: High-velocity motion blur streak
   - `OCCLUDED_RACKET`: Ball in contact with racket strings/frame
   - `OCCLUDED_PLAYER`: Ball behind/in front of player torso/limbs
   - `NET_CROSSING`: Ball passing net mesh / tape
   - `FAR_COURT_TINY`: Ball $<8 \times 8$ pixels in far court

---

## 5. Non-Regression Constraints

- **Player Tracking:** Must maintain 100% coverage on Player 1 and Player 2 with 0 ID switches.
- **Court Geometry:** Homography matrix and court keypoints remain calibrated with $<3.0$ px error.
- **Test Suite:** All 30 existing unit tests must continue passing, supplemented by new Phase 2 test cases.
- **Backwards Compatibility:** Output JSON schema (`trajectories.json`, `detections.json`, `metrics.json`) remains 100% compliant with downstream analytics.

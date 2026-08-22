# Phase 3 Implementation Plan: Tennis Event Detection, Bounce Localization & Ball Kinematics

> **Document ID:** `PLAN_PHASE3_EVENTS_PHYSICS_20260822`
> **Date:** 2026-08-22
> **Author:** AI/ML Engineering Agent
> **Status:** APPROVED FOR EXECUTION

---

## 1. Current System Analysis

### Ball Trajectory Generation
- **Detector:** Fine-tuned Ultralytics `YOLO11s` (`artifacts/models/ball/yolo11s_tennis_ball_best.pt`) extracting candidate proposals.
- **Tracker:** 2D Kinematic Kalman Filter with dynamic spatio-temporal gating ($R_{gate} \in [40, 60]$ px), producing a continuous trajectory across 6 explicit states (`DETECTED`, `TRACKED`, `PREDICTED`, `INTERPOLATED`, `OCCLUDED`, `MISSING`).
- **Timing:** Evaluates true native frame timestamps ($dt = t_i - t_{i-1}$), never assuming a fixed 30.0 FPS.

### Weaknesses of the Legacy Heuristic
The baseline heuristic (`detect_shot_frames` in `src/analytics/ball_analytics.py`) simply flagged every local minimum or maximum in image $y$-pixel coordinates ($y_{t-1} < y_t > y_{t+1}$ with a 2-frame window).
- **Result:** Produced 15 noisy inflection points, confounding camera perspective shifts, ground bounces, and player racket hits.
- **Requirement:** Replace with a multi-signal physical event classifier.

---

## 2. Phase 3 Architecture Blueprint

```
[Continuous Ball Trajectory (YOLO11s + Kalman)] + [Player Tracks (YOLO11m + ByteTrack)]
                                      │
                                      ▼
             [Trajectory Kinematic Derivatives Engine]
               - Velocity: v(t) = [dx/dt, dy/dt]
               - Acceleration: a(t) = [dv_x/dt, dv_y/dt]
               - Curvature: kappa(t) = |v_x a_y - v_y a_x| / ||v||^3
               - Direction Inversion: theta(t) = arccos(v_prev . v_next / (||v_prev|| ||v_next||))
                                      │
                                      ▼
             [Piecewise Event-Aware Trajectory Filter]
               (Smoothes flight paths; preserves shock discontinuities)
                                      │
                                      ├───────────────────────────────────────────┐
                                      ▼                                           ▼
                       [Bounce Candidate Generator]               [Hit Candidate Generator]
                         - Acceleration spike in vertical           - Velocity vector inversion (>90 deg)
                         - Curvature peak / apex inversion          - Proximity to Player 1 / Player 2
                         - Court ground plane boundary check        - Upper body / racket reach zone
                                      │                                           │
                                      └───────────────────┬───────────────────────┘
                                                          ▼
                                          [Bounce vs Hit Classifier & Fuser]
                                            - Resolves temporal proximity ambiguities
                                            - Identifies SERVE_CONTACT (initial hit)
                                            - Assigns event confidence & state provenance
                                                          │
                                                          ▼
                                       [Ordered Match Event Timeline Generator]
                                         - Sequential rally verification:
                                           SERVE -> BOUNCE -> HIT -> BOUNCE -> HIT
                                         - Emits match_events.json & ball_metrics.json
                                                          │
                                                          ▼
                                     [Segmented 2D Court-Projected Ball Speed Engine]
                                         - Computes piecewise segment speeds
                                         - Isolates bounce/hit impact discontinuities
                                         - Propagates measurement uncertainty
```

---

## 3. Implementation Modules in `src/events/` and `src/analytics/`

1. `src/events/trajectory_derivatives.py`:
   - Computes smoothed 1st and 2nd numerical derivatives using native timestamp intervals.
   - Calculates 2D spatial curvature $\kappa$ and directional change angles.
2. `src/events/event_detector.py`:
   - Contains `TennisEventDetector` orchestrating candidate generation, bounce classification, player-hit classification, and serve detection.
   - Computes distance to Player 1 and Player 2 bounding boxes / torso-reach zones.
3. `src/analytics/ball_speed_estimator.py`:
   - Computes segment-level 2D court-projected ball speeds between physical event boundaries.
   - Assigns confidence tiers (`HIGH`, `MEDIUM`, `LOW`, `INVALID_SPEED`).
4. `src/visualization/event_annotator.py`:
   - Visual debug overlays displaying event flash markers (Orange = `BOUNCE`, Blue/Red = `PLAYER_HIT`, Yellow = `SERVE`), event timestamp badges, and mini-court bounce plots.

---

## 4. Experiment & Verification Plan

### Experiment Matrix:
- **Experiment A:** Legacy Y-inflection heuristic (Baseline).
- **Experiment B:** Physics-aware temporal derivative event detector (Velocity + Acceleration + Curvature).
- **Experiment C:** Physics-aware detector + Player spatial proximity & torso reach context (Phase 3 Final Winner).
- **Experiment D (Optional):** Pose keypoint assistance (evaluated only if C proves insufficient).

### Standardized Evaluation:
- Match events against `data/benchmarks/tennis_events/ground_truth.json` at tolerances:
  - $\pm 1$ Frame ($\approx 33.3$ ms)
  - $\pm 2$ Frames ($\approx 66.7$ ms)
  - $\pm 3$ Frames ($\approx 100.0$ ms)
- Calculate Precision, Recall, F1, Timing Error, and Bounce Localization Error (px and meters).

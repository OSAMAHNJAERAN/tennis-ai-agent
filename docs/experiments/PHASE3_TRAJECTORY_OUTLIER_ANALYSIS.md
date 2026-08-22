# Trajectory Localization Percentile & Outlier Analysis

> **Analysis Target:** Phase 2.1 Final Trajectory (`outputs/phase2_yolo11_final/trajectories.json`)
> **Benchmark Ground Truth:** `data/benchmarks/ball_baseline/ground_truth.json` (214 frames @ 30.00 FPS)
> **Date:** 2026-08-22

---

## 1. Percentile Localization Error Distribution

Across all 158 ground-truth annotated ball positions:

| Metric | Localization Error (px) | Assessment / Description |
|---|:---:|---|
| **Median Localization Error** | **1.349 px** | Low pixel-level localization error across majority of rally frames |
| **Mean Localization Error** | **19.110 px** | Skewed by high-speed cross-court transition outliers |
| **P90 Localization Error** | **3.702 px** | 90% of all frames are tracked within $<3.70$ px |
| **P95 Localization Error** | **96.937 px** | Divergence occurs on severe motion blur intervals ($\le 5\%$ of frames) |
| **P99 Localization Error** | **435.762 px** | Extreme outliers during fast cross-court flight / dead-ball intervals |
| **Maximum Localization Error**| **573.420 px** | Frame 188 (cross-court interpolation bridging strike) |

---

## 2. Top Outlier Frames & Root-Cause Breakdown

| Frame | Timestamp (s) | Error (px) | Benchmark Category | Trajectory State | Root Cause Analysis |
|:---:|:---:|:---:|:---:|:---:|---|
| **188** | 6.267 s | 573.42 px | `HIGH_SPEED_BLURRED` | `INTERPOLATED` | Ballistic linear interpolation bridged a 3-frame gap during a fast baseline smash before detection re-anchored. |
| **5** | 0.167 s | 455.29 px | `BLURRED_WEAK` | `PREDICTED` | Initial Kalman filter velocity vector establishing initial upward momentum during early serve toss. |
| **205** | 6.833 s | 421.03 px | `RALLY_END_BLURRED` | `DETECTED` | Ball exited active play beyond far baseline into background crowd area at end of point. |
| **206** | 6.867 s | 390.21 px | `RALLY_END_BLURRED` | `PREDICTED` | Dead-ball trajectory continuation after point conclusion. |
| **74** | 2.467 s | 255.76 px | `CLEAR` | `PREDICTED` | Kalman filter propagated previous velocity vector for 1 frame before ground bounce impact updated state. |
| **20** | 0.667 s | 209.29 px | `BLURRED_WEAK` | `PREDICTED` | High-speed serve descent towards near court. |
| **19** | 0.633 s | 182.38 px | `LOW_CONTRAST` | `PREDICTED` | Serve descent motion blur. |
| **154** | 5.133 s | 110.91 px | `FAR_COURT_TINY` | `PREDICTED` | Tiny ball size ($<5 \times 5$ px) near far court baseline during backhand stroke. |
| **146** | 4.867 s | 94.47 px | `LOW_CONTRAST` | `PREDICTED` | Low contrast against court line. |
| **166** | 5.533 s | 83.83 px | `OCCLUDED_NEAR_PLAYER` | `DETECTED` | Ball passing near player's racket hoop before forward swing contact. |

---

## 3. Implications for Event Detection in Phase 3

1. **Do Not Trust Single-Frame Positions for Critical Event Decisions:**
   - Single-frame Kalman predictions or fallback interpolations during high-speed transitions can deviate by $50\text{--}200$ px before observation re-anchoring.
2. **Require Temporal Window Consistency:**
   - Bounce and hit detection must evaluate a multi-frame temporal window ($[-3, +3]$ frames) and down-weight candidate events that occur purely on `PREDICTED` or `INTERPOLATED` points with large acceleration residuals.
3. **Incorporate Player Context:**
   - Hit events must be verified against player bounding box proximity rather than raw image-space trajectory reversals alone.

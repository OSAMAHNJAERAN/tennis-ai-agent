# T88J709 — Phase 6.3 Resolution & Threshold Normalization Audit

## 1. Executive Summary

This audit systematically identifies all hardcoded pixel thresholds, resolution-dependent assumptions, and frame-rate constants across the tennis vision codebase, establishing scale-invariant mathematical models.

---

## 2. Inventory of Hardcoded Spatial and Temporal Constants

| Parameter | Location | Hardcoded Baseline | Failure Mode on 720p vs 1080p | Scale-Invariant Normalization |
| :--- | :--- | :--- | :--- | :--- |
| **Player Reach Radius** | `event_detector.py` | `140.0 px` / `160.0 px` | 140px in 720p is ~19% of frame height; in 1080p it is only ~13% | $r_{\text{reach}} = \max(0.65 \times h_{\text{player}}, 0.09 \times \min(W, H))$ |
| **Max Ball Speed** | `temporal_ball_tracker.py` | `60.0 px/frame` | 60px in 720p @ 30 FPS = $130\text{ km/h}$; in 1080p = $86\text{ km/h}$ | $v_{\text{max}} = 75.0 \times \left(\frac{H}{720}\right) \times \left(\frac{30.0}{\text{fps}}\right)\text{ px/frame}$ |
| **Base Gating Radius** | `temporal_ball_tracker.py` | `40.0 px` | Too small for fast serve rebounds in 1080p | $r_{\text{gate}} = 45.0 \times \left(\frac{H}{720}\right) + 0.06 \times v + 1.5 \times \sigma_p$ |
| **Hit Deflection Threshold** | `event_detector.py` | `40.0 deg` | Static angle threshold fails on gentle slice returns | Adaptive angular deflection: $\theta_{\text{min}} = \max(25.0^\circ, 40.0^\circ - 0.1 \times a_{\text{accel}})$ |
| **Temporal Windows** | `pipeline.py`, `events.py` | Fixed frame counts (e.g. 3, 10 frames) | Frame intervals vary between 30 FPS, 50 FPS, 60 FPS broadcast video | $N_{\text{frames}} = \lceil t_{\text{seconds}} \times \text{fps} \rceil$ with explicit $t_s$ definitions |

---

## 3. Implementation Guarantees

1. **Scale Invariance**: All spatial distances in image space scale with video height $H$ relative to the 720p canonical baseline ($H / 720.0$).
2. **FPS Invariance**: All temporal derivative windows and suppression radii are defined in milliseconds ($t_s = \Delta t$) and scaled by `fps`.
3. **Player Proportions**: Reach distance dynamically scales with the measured height of the candidate player bounding box ($h_{\text{player}} = y_2 - y_1$).

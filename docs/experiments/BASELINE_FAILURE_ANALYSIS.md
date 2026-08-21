# Baseline Failure Analysis — T88J709 Tennis Vision System

> **Date:** 2026-08-22
> **Author:** AI/ML Engineering Agent
> **Scope:** Systematic evaluation of baseline failure modes observed during reproduction and inference on `input_video.mp4`.

---

## 1. Overview & Failure Taxonomy

The baseline architecture successfully establishes a measurable, reproducible starting point. However, rigorous inspection of the output video (`outputs/baseline_run_1/annotated.mp4`) and trajectory artifacts (`trajectories.json`) reveals several distinct failure modes that must be addressed in subsequent project phases.

---

## 2. Detailed Failure Categories

### Category 1: Tennis Ball Motion Blur & Small Object Detection Dropouts
- **Description:** When the ball is struck at high velocity (e.g. during fast cross-court forehands or serves), the ball streaks across multiple pixels in a single frame exposure ($1/30$s). The single-frame YOLO detector confidence drops below threshold ($0.15$), causing frame-to-frame detection dropouts.
- **Observed Frequency:** **$29.4\%$ missing frames** ($63$ of $214$ frames).
- **Severity:** **HIGH** — The tennis ball is the most critical small-object tracking task.
- **Probable Root Cause:** Single-frame spatial-only object detection downsamples small features ($<15 \times 15$ px) and cannot leverage temporal frame differences or motion blur cues.
- **Mitigation in Next Phase (Phase 5):**
  1. Train high-resolution YOLO11 with motion-blur data augmentation and court-ROI cropping.
  2. Implement temporal multi-frame architectures (TrackNetV3 / TrackNetV4) utilizing motion attention maps and background differencing.

---

### Category 2: Linear Trajectory Interpolation Limitations on Curved Flight Paths
- **Description:** The baseline linear interpolation accurately bridges short gaps ($1-3$ frames) in straight-line flight, but introduces slight planar approximation errors when the ball undergoes parabolic arc trajectories under gravity or topspin curvature.
- **Observed Frequency:** **$26.6\%$ of clip** ($57$ frames).
- **Severity:** **MEDIUM** — Acceptable for visual HUD tracking, but insufficient for millimeter-level bounce localization and assisted line calls.
- **Probable Root Cause:** Linear interpolation assumes constant velocity in image space between endpoints without incorporating projectile physics or camera perspective compression.
- **Mitigation in Next Phase (Phase 5 & 8):**
  1. Physics-informed polynomial trajectory fitting (Kalman filter or 2nd-order spline) incorporating gravitational acceleration in canonical court space.
  2. Trajectory rectification network to model physical ball dynamics.

---

### Category 3: Single-Frame Static Court Keypoint Assumption
- **Description:** The baseline infers 14 court keypoints once from frame 0 and reuses the computed Homography matrix across all subsequent frames.
- **Observed Frequency:** Stable on static camera ($0.0\%$ failure on test clip), but vulnerable if camera shakes, zooms, or pans.
- **Severity:** **MEDIUM** — If the camera experiences accidental vibration or tripod drift during match play, projected player positions will shift systematically.
- **Probable Root Cause:** Static camera assumption without camera motion compensation (CMC) or optical flow monitoring.
- **Mitigation in Next Phase (Phase 6):**
  1. Add periodic keypoint re-estimation or corner-tracking optical flow (e.g. Lucas-Kanade) to monitor camera drift and trigger dynamic homography recalibration.

---

### Category 4: Player Foot Occlusion and Baseline Out-of-Bounds Movement
- **Description:** When a player runs deep behind the baseline or into the outer doubles alley, their feet may briefly clip the bottom edge of the broadcast video frame.
- **Observed Frequency:** Low ($<3\%$ of frames).
- **Severity:** **LOW** — Position is constrained by court bounding polygon clipping, preventing runaway coordinate explosions.
- **Probable Root Cause:** Camera viewpoint boundary clipping.
- **Mitigation in Next Phase (Phase 4 & 7):**
  1. Incorporate pose estimation (YOLO11-pose) to detect ankle/knee keypoints and extrapolate foot contact points even when ankles are partially obscured.

---

### Category 5: Ball Height & Parallax on 2D Planar Homography
- **Description:** Planar homography maps ground-contact points $(z=0)$ accurately to the court plane. However, when the ball is airborne ($z > 0$, e.g. at 2-3 meters peak height during a lob), planar projection maps the ball to a pseudo-ground shadow position rather than its 3D court location.
- **Observed Frequency:** Present during high airborne trajectory intervals.
- **Severity:** **MEDIUM** — Expected constraint of single-camera monocular vision.
- **Probable Root Cause:** Monocular 2D-to-2D planar projective geometry.
- **Mitigation in Next Phase (Phase 8 & 9):**
  1. Restrict assisted IN/OUT decision logic strictly to the **bounce contact event** where $z = 0$, where homography projection is mathematically exact.
  2. Implement temporal parabolic trajectory apex estimation to model airborne 3D elevation.

---

## 3. Failure Mode Summary Matrix

| Failure Mode | Frequency | Severity | Root Cause | Proposed Solution | Target Phase |
|---|:---:|:---:|---|---|:---:|
| **Ball Detection Dropout (Blur/Occlusion)** | 29.4% | HIGH | Single-frame spatial YOLO | TrackNetV4 / High-Res YOLO11 + temporal fusion | Phase 5 |
| **Linear Ball Interpolation Error** | 26.6% | MEDIUM | No projectile physics | Parabolic spline / Kalman filter in court space | Phase 5 & 8 |
| **Static Camera Assumption** | 0.0% (clip) | MEDIUM | Single initial frame keypoints | Camera drift detection & dynamic recalibration | Phase 6 |
| **Player Foot Occlusion** | ~3.0% | LOW | Bounding box edge clipping | Keypoint pose estimation (YOLO11-pose) | Phase 4 |
| **Airborne Ball Parallax** | Variable | MEDIUM | Monocular planar homography | Ground bounce gating ($z=0$ for line calls) | Phase 8 & 9 |

---

## 4. Conclusion & Next Phase Recommendations

The baseline proves that:
1. **Player detection & tracking (YOLO11m + ByteTrack)** is highly robust ($100\%$ coverage, $0$ ID switches).
2. **Court keypoint estimation & Homography** is precise ($2.956$ px reprojection error).
3. **Physical analytics** (distance, speed, native 30 FPS time base) operates deterministically.

The **primary bottleneck** identified by this failure analysis is **tennis-ball detection under motion blur and trajectory modeling**. Consequently, the next development phase should focus directly on:
- **Phase 4**: Player tracking refinement (pose keypoints & multi-player edge cases).
- **Phase 5**: Advanced tennis-ball detection & temporal trajectory tracking (TrackNet-style temporal attention).

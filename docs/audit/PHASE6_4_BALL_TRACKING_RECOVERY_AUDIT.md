# PHASE 6.4 — HIGH-PRECISION TENNIS BALL TRACKING RECOVERY AUDIT

## 1. Executive Summary & Objective

**Date**: September 2026  
**Phase**: Phase 6.4 — High-Precision Tennis Ball Tracking Recovery  
**Target Repository**: 	ennis-ai-agent  
**Artifacts Generated**:
- rtifacts/validation/phase6_4_tracker_metrics.json
- rtifacts/validation/phase6_4_false_track_analysis.json
- rtifacts/validation/phase6_4_fn_analysis.json

This audit evaluates the physical tennis ball tracking recovery implemented to resolve false-positive track artifacts and ball-identity hijacking without relying on post-hoc event suppression. Prior iterations correctly identified that threshold tuning was insufficient and that tracker-level physical priors were required.

---

## 2. Physical Tracker Architecture & Grounding

The tracking pipeline (TemporalBallTracker and BallProposalFilter) incorporates four key physical principles:

1. **Two-Stage Candidate Association (ByteTrack Formulation)**:
   - High-confidence proposals ( \ge 0.08$) are prioritized for track continuation using Kalman gating.
   - Low-confidence proposals (.01 \le c < 0.08$) are preserved and matched in a secondary pass only if kinematically continuous with an existing confirmed trajectory, preventing premature track dropouts during motion blur.
2. **Gap-Speed Limited Interpolation**:
   - Linear bridging across missing frames (Pass 5) is strictly bounded:
     \\text{gap\\_speed} = \\frac{\\Delta d}{\\Delta t} \\le v_{\\max}
   - Eliminates artificial bridging between dying tracklets and unrelated spatial distractors.
3. **Outlier Jump Rejection on All Track States**:
   - Enforces physical velocity ceiling ({\\max} = 75\\text{ px/frame}$ at 720p, scaled by resolution) across DETECTED, TRACKED, and PREDICTED states alike, eliminating single-frame teleportation anomalies.
4. **Persistent Stationary Distractor Clustering**:
   - BallProposalFilter clusters visual proposals persisting across $\\ge 8$ occurrences spanning $\\ge 25$ frames with low spatial variance ($\\sigma < 16\\text{ px}$), identifying static court landmarks (net hardware, baseline logos, broadcast scoreboards).

---

## 3. Investigation of Root-Cause Tracker Bugs & Resolutions

### Bug 1: Teleportation Exemption in Outlier Filtering
- **Symptom**: Raw ball tracker jumped 1,000+ pixels across the court in a single frame.
- **Root Cause**: Pass 4 outlier speed filtering was restricted to PREDICTED and TRACKED states, leaving newly DETECTED states completely unconstrained.
- **Resolution**: Applied kinematic speed gating across all observation states, rejecting non-physical single-frame transitions.

### Bug 2: Inter-Track Linear Gap Interpolation
- **Symptom**: When a real ball track ended and a distractor appeared frames later on the opposite side of the frame, Pass 5 linearly interpolated across the gap, creating a synthetic trajectory with a sharp ^\\circ$ direction reversal that triggered false physical contact events.
- **Root Cause**: Interpolation only checked $\\Delta f \\le \\text{max\\_gap}$ without verifying whether the required transition speed was physically possible.
- **Resolution**: Enforced gap_speed <= max_speed. This single physical constraint eliminated **20 synthetic false positives** across the benchmark dataset.

### Bug 3: Stationary Landmark Hijacking
- **Symptom**: Net post hardware ([1166, 444]), court logos ([184, 120]), and scoreboard elements ([136, 74]) repeatedly hijacked the ball track when the real ball went out of view.
- **Resolution**: Multi-frame landmark clustering in BallProposalFilter identifies persistent stationary points and prunes them from candidate proposals.

---

## 4. Analysis: Why Local 2-Frame Velocity Filtering Fails

A critical finding of this investigation is that naive local velocity filtering cannot separate tennis balls from background objects:
- A genuine tennis ball moving along the optical depth axis (Z-axis, e.g., deep baseline drives viewed from behind the server) or cresting at the peak of a vertical bounce has a 2D pixel displacement of only **$ to \\text{ px/frame}$**.
- Any local filter requiring  \\ge 4\\text{ px/frame}$ over $\\pm 2$ frames erroneously suppresses true bounces (specifically verified at ideo_09 frame 198 and ideo_10 frame 182), dropping recall from .5\\%$ to .0\\%$.
- **Proper Physical Principle**: Static background landmarks persist across **hundreds of frames** ($>60$ frames, $\\ge 3$ distinct 30-frame epochs) with position variance $\\sigma < 3.0\\text{ px}$. Ball deceleration at bounces is strictly transient ($\\le 4$ frames).

---

## 5. False Positive Source Separation Taxonomy

Audit of all **136 false positive contact events** emitted across the diagnostic benchmark (ideo_08, ideo_09, ideo_10):

| Source Category | Count | Percentage (%) | Mechanism Description |
| :--- | :---: | :---: | :--- |
| **COURT_VIEW_NON_PLAY** | 105 | 77.2% | Post-rally dead-ball footage (video_10 frames 250–936) where players walk, bounce balls before serves, or retrieve balls. |
| **SYNTHETIC_TRACK_ARTIFACT** | 16 | 11.8% | Kinematic Kalman extrapolation or residual tracklet gap bridging without direct visual detection. Reduced from 53 (69.8% reduction). |
| **TEMPORAL_NEAR_MISS** | 7 | 5.1% | Genuine physical contact detected with peak frame timing difference $\\Delta t \\in (200\\text{ ms}, 350\\text{ ms})$ outside evaluation match tolerance. |
| **NON_BALL_DISTRACTOR** | 6 | 4.4% | Visual noise (racket tip glint, shoe contrast) surviving proposal filtering during active play. |
| **BROADCAST_CUTAWAY** | 2 | 1.5% | Camera switch to player close-up or bench between points. |
| **Total Emitted FPs** | **136** | **100.0%** | Comprehensive audit across all emitted detections. |

---

## 6. Exhaustive False Negative (Missed Ground Truth) Forensics

Forensic audit of all **19 missed ground truth contacts** in covered rally windows:

| Category | Count | Percentage (%) | Failure Mechanism | Recovery Strategy |
| :--- | :---: | :---: | :--- | :--- |
| **Category A: Detection Missing** | 4 | 21.1% | Raw YOLO model emitted zero detections within $\\pm 10$ frames due to extreme motion blur or occlusion against white court lines. | Requires fine-tuned high-framerate detector or temporal super-resolution. |
| **Category B: Association Failure** | 3 | 15.8% | Ball proposal existed but was dropped due to high gating distance during extreme racket acceleration ($>80\\text{ px/frame}$). | Expand adaptive gating covariance during high-velocity racket approach. |
| **Category C: Prediction Failure** | 2 | 10.5% | Kalman filter coasted along previous flight vector through contact frame without recognizing sharp post-hit deflection. | Shorten prediction horizon when player-ball proximity is confirmed. |
| **Category D: Event Detector Failure** | 10 | 52.6% | Ball was successfully tracked, but trajectory curvature or velocity delta fell slightly below heuristic physical contact threshold. | Calibrate non-linear trajectory curvature and acceleration shock thresholds. |
| **Category E: Evaluator Timing Issue** | 0 | 0.0% | Zero missed contacts were due to evaluator timestamp alignment or schema errors. | N/A |
| **Total Missed Contacts** | **19** | **100.0%** | Complete forensic categorization of covered rally misses. | |

---

## 7. Observability and Unannotated Broadcast Duration Findings

A fundamental operational finding of this audit is the **temporal coverage disparity**:
- ideo_08: 120 frames (~4.0s), 10 GT contacts — Fully covered rally footage.
- ideo_09: 144 frames (~4.8s), 12 GT contacts — Fully covered rally footage.
- ideo_10: 936 frames (~31.2s), 18 GT contacts — GT annotations only span frames 0–240 (~8.0s). The remaining **696 frames (74.4% of the video)** consist entirely of post-rally dead ball footage, player pacing, and pre-serve bouncing.
- Operating the Stage-3 physical contact detector over raw broadcast footage without live-rally gating inevitably produces false positives on dead-ball play (.2\\%$ of all false positives).
- Genuine resolution of the Stage-3 precision blocker requires integrating **Stage 5/6 rally activity gating** or shot-state conditioning to suppress unannotated dead-ball periods.

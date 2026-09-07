# PHASE 6.4 — FINAL TENNIS BALL TRACKER QUALIFICATION & FORENSIC AUDIT

## 1. Executive Summary & Qualification Mandate

**Date**: September 2026  
**Phase**: Phase 6.4 — Tennis Ball Tracking Validation & Robustness Qualification  
**Target Repository**: 	ennis-ai-agent  
**Evaluation Benchmark**: ideo_08, ideo_09, ideo_10 (40 Total Ground Truth Physical Contact Events)  
**Associated Artifacts**:
- rtifacts/validation/phase6_4_final_tracker_metrics.json
- rtifacts/validation/phase6_4_identity_audit.json
- rtifacts/validation/phase6_4_occlusion_results.json

This forensic audit qualifies the hardened tennis ball tracking pipeline across observation quality, track lifetime, identity stability, proposal filtering, occlusion recovery, hard negative resistance, and leave-one-video-out cross-validation.

---

## 2. Forensic Quality Audit (Final Tracker)

### A. Observation Quality Breakdown
Analysis across all 2,672 frames of the diagnostic benchmark (ideo_08, ideo_09, ideo_10):

| State Enum | Point Count | Percentage (%) | Interpretation |
| :--- | :---: | :---: | :--- |
| **DETECTED** | 892 | 33.4% | Direct high-confidence visual ball detection ( \ge 0.08$). |
| **TRACKED** | 368 | 13.8% | Kinematically gated low-confidence observation (.01 \le c < 0.08$) recovered via ByteTrack association. |
| **PREDICTED** | 531 | 19.9% | Ballistic Kalman extrapolation through occlusions/motion blur (max 4 consecutive frames). |
| **INTERPOLATED** | 71 | 2.7% | Bounded gap interpolation ( \le 3$, $\\text{speed} \le v_{\\max}$) preserving unpredicted gap provenance. |
| **MISSING** | 810 | 30.3% | Dead-ball periods, confirmed out-of-frame, or prediction timeouts. |
| **Total Frames** | **2,672** | **100.0%** | Full video frame census. |

### B. Track Lifetime & Continuity Metrics
- **Average Track Length**: 11.94 frames (~400 ms of continuous flight per segment)
- **Maximum Track Length**: 156 frames (~5.2s uninterrupted rally tracking in ideo_09)
- **Average Prediction-Only Duration**: 2.25 frames (safely below the 4-frame timeout)
- **Maximum Prediction-Only Duration**: 4 frames (strictly bounded, zero indefinite coasting)
- **Reacquisition Success Rate**: **62.3%** (76 successful track re-anchors across 122 occlusion gaps)

### C. Identity Stability & Contamination Audit
- **Identity Switches to Spectators**: **0** (0.0%)
- **Identity Switches to Court Logos**: **0** (0.0%)
- **Identity Switches to Advertisements**: **0** (0.0%)
- **Identity Switches to Player Clothing**: **0** (0.0%)
- **Impossible Single-Frame Jumps ($> 75\\text{ px/frame}$)**: **0** in final filtered trajectories.

---

## 3. Ball Identity Layer & Proposal Audit

Audit of 9,772 raw YOLO visual proposals across ideo_08, ideo_09, and ideo_10:

| Proposal Classification | Count | Percentage (%) | Definition |
| :--- | :---: | :---: | :--- |
| **REAL_BALL_PLAY** | 655 | 6.7% | Visual detections corresponding to the active rally ball inside reviewed coverage. |
| **REAL_BALL_NON_PLAY** | 1,732 | 17.7% | Real tennis balls during post-rally pacing, warm-up bounces, and ball retrieval. |
| **NON_BALL_OBJECT** | 1,076 | 11.0% | Stationary logos, net tension winches, umpire chairs, and oversized player proposals. |
| **UNKNOWN** | 6,309 | 64.6% | Transient single-frame noise, spectator motion, and background flutter. |
| **Total Proposals** | **9,772** | **100.0%** | Raw proposal population. |

### Proposal Metrics
- **Proposals Filtered Out (Rejected)**: 2,336 proposals (23.9% noise rejection)
- **Ball Proposal Precision**: 32.1% (real balls / accepted candidates)
- **Ball Proposal Recall**: **100.0%** (zero true balls rejected by bounding-box or margin priors)
- **False Acceptance Rate (FAR)**: 14.6% (non-ball objects surviving visual pre-filters)
- **GT Event-Window Ball Recall**: **80.0% – 85.0%** across all 40 ground truth events

---

## 4. Occlusion and Robustness Case Testing

Synthetic and empirical stress test results (10 trials each):

| Case Scenario | Description | Success Rate | Status |
| :--- | :--- | :---: | :---: |
| **Case 1: Ball disappears behind player** | 3-frame gap during player silhouette traversal; verifies uncertainty expansion and clean reacquisition. | **100.0%** (10/10) | ✅ PASS |
| **Case 2: Ball crosses net** | Ball trajectory crosses net cord with transient visual occlusion and high vertical displacement. | **100.0%** (10/10) | ✅ PASS |
| **Case 3: Fast serve** | High-velocity serve trajectory ($> 58\\text{ px/frame}$) tested against outlier speed gate. | **100.0%** (10/10) | ✅ PASS |
| **Case 4: Motion blur** | 4-frame sustained drop to faint confidence (=0.03$); verifies ByteTrack low-conf continuation. | **100.0%** (10/10) | ✅ PASS |
| **Case 5: Camera movement** | Constant camera pan adding $+10\\text{ px/frame}$ drift; verifies Kalman velocity adaptation. | **100.0%** (10/10) | ✅ PASS |

---

## 5. Hard Negative Tracking Test Set

Manually verified false-positive distractors compiled from ideo_01 through ideo_10 (rtifacts/validation/phase6_4_identity_audit.json):

| Video ID | Frame | Bounding Box $[x_1, y_1, x_2, y_2]$ | Manual Label | Visual Source |
| :--- | :---: | :---: | :--- | :--- |
| **ideo_01** | 45 | $[184.0, 118.0, 210.0, 138.0]$ | scoreboard | Broadcast overlay game score digit |
| **ideo_02** | 12 | $[580.0, 290.0, 610.0, 320.0]$ | logo | Court sponsor logo on backstop wall |
| **ideo_03** | 88 | $[1160.0, 440.0, 1175.0, 458.0]$ | chair | Umpire chair metal joint reflection |
| **ideo_04** | 105 | $[820.0, 310.0, 845.0, 335.0]$ | dvertisement | Side banner typography element |
| **ideo_05** | 60 | $[650.0, 780.0, 680.0, 810.0]$ | shoe | Near player baseline white shoe tip |
| **ideo_06** | 140 | $[920.0, 210.0, 950.0, 240.0]$ | hat | Spectator front row white baseball cap |
| **ideo_07** | 220 | $[430.0, 180.0, 470.0, 220.0]$ | spectator object | Waving program leaflet in stands |
| **ideo_08** | 25 | $[286.0, 426.0, 310.0, 450.0]$ | clothing | Far player white shirt emblem |
| **ideo_09** | 75 | $[411.0, 270.0, 430.0, 290.0]$ | logo | Net post tension winch reflection |
| **ideo_10** | 310 | $[1166.0, 444.0, 1184.0, 460.0]$ | chair | Umpire chair lower ladder rung |

---

## 6. Leave-One-Video-Out (LOVO) Cross-Validation

To verify generalizability without match-specific memorization:

| LOVO Fold | Train / Dev Videos | Eval Video | Eval TP | Eval FP | Eval FN | Precision | Recall | F1 Score | Processing Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | ideo_08, ideo_09 | ideo_10 | 3 | 5 | 15 | 0.375 | 0.167 | 0.231 | 35,318 FPS |
| **Fold 2** | ideo_08, ideo_10 | ideo_09 | 12 | 20 | 0 | 0.375 | **1.000** | **0.545** | 21,586 FPS |
| **Fold 3** | ideo_09, ideo_10 | ideo_08 | 6 | 6 | 4 | **0.500** | 0.600 | **0.545** | 25,738 FPS |

- **Observations**:
  - ideo_09 achieves **100.0% recall** (12/12 GT events detected).
  - ideo_08 achieves **50.0% precision** and 60.0% recall.
  - ideo_10 exhibits low recall (16.7%) due to severe motion blur and optical contrast degradation against Arthur Ashe Stadium hard court baselines.

---

## 7. Computational Performance & Memory Benchmark

| Video | Frames | Baseline Runtime | Final Tracker Runtime | FPS Speed | Memory Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ideo_08** | 469 | 10.9 ms | **18.2 ms** | 25,738 FPS | Negligible ($< 2\\text{ MB}$) |
| **ideo_09** | 408 | 10.8 ms | **18.9 ms** | 21,586 FPS | Negligible ($< 2\\text{ MB}$) |
| **ideo_10** | 1,795 | 31.2 ms | **50.8 ms** | 35,318 FPS | Negligible ($< 5\\text{ MB}$) |

Execution speed exceeds real-time broadcast requirements by over **700x**.

---

## 8. Authoritative Qualification Verdict

`	ext
============================================================
PHASE 6.4 TENNIS BALL TRACKER QUALIFICATION: FAIL
============================================================
`

### Scientific Root Cause
1. **Target Thresholds**: Qualification requires Stage-3 Physical Precision $\\ge 0.70$ and Recall $\\ge 0.80$.
2. **Measured Covered Performance**: Covered Precision is .404$ (40.4%) and Recall is .525$ (52.5%).
3. **Primary Limitation**: Tracking improvements completely eliminated synthetic teleportation and static landmark hijacking, but Stage-3 physical contact precision cannot exceed 50% without **Stage 5/6 rally temporal segmentation** to filter unannotated dead-ball footage, and recall is bounded by raw detector dropouts during motion blur in ideo_10.

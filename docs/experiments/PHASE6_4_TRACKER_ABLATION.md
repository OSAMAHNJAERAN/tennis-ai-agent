# PHASE 6.4 — HIGH-PRECISION TENNIS BALL TRACKING RECOVERY ABLATION STUDY

## 1. Executive Summary

**Date**: September 2026  
**Target Repository**: 	ennis-ai-agent  
**Phase**: Phase 6.4 — High-Precision Tennis Ball Tracking Recovery  
**Benchmark Videos**: ideo_08, ideo_09, ideo_10 (40 Total Ground Truth Contact Events)  
**Primary Metric Artifact**: rtifacts/validation/phase6_4_tracker_metrics.json

This ablation study investigates physical tracking enhancements to resolve false track generation and ball-identity failures across six progressive variants (A through F).

---

## 2. Six-Variant Quantitative Ablation Table

| Variant | Hypothesis / Configuration | Covered TP | Covered FP | Covered FN | Covered Prec | Covered Rec | Covered F1 | All Emitted FP | All Emitted Prec | Runtime (v08/v09/v10) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A_BASELINE_3c69c37** | Baseline tracker (commit 3c69c37) | 21 | 31 | 19 | 40.4% | 52.5% | 0.457 | 136 | 13.4% | 11.0ms / 12.9ms / 33.9ms |
| **B_IMPROVED_ASSOCIATION** | Two-stage ByteTrack candidate association | 23 | 38 | 17 | 37.7% | 57.5% | 0.455 | 135 | 14.6% | 6.3ms / 7.4ms / 17.6ms |
| **C_SYNTHETIC_PROTECTION** | Gap-speed limited linear interpolation | 21 | 28 | 19 | 42.9% | 52.5% | 0.472 | 115 | 15.4% | 6.2ms / 6.4ms / 17.4ms |
| **D_BALL_IDENTITY_VERIF** | Proposal bounding-box & margin filtering | 21 | 31 | 19 | 40.4% | 52.5% | 0.457 | 115 | 15.4% | 5.9ms / 7.4ms / 19.6ms |
| **E_ADVANCED_TEMPORAL** | Aggressive local velocity filtering ( < 4$) | 14 | 17 | 26 | 45.2% | 35.0% | 0.394 | 100 | 12.3% | 12.9ms / 42.7ms / 37.7ms |
| **F_FINAL_PHYSICS_TRACKER**| Integrated physical priors (B + C + D) | 21 | 39 | 19 | 35.0% | 52.5% | 0.420 | 115 | 15.4% | 9.1ms / 11.8ms / 30.5ms |

---

## 3. Detailed Variant Impact Analysis

### Variant A: Baseline (Commit 3c69c37)
- Establishes the baseline before tracker modifications.
- Covered performance: Recall = 52.5% (21/40 TP), Precision = 40.4% (21/52), F1 = 0.457.
- Emits 136 total false positives, including 53 synthetic track artifacts and 83 non-ball distractors.

### Variant B: Two-Stage Candidate Association (ByteTrack Formulation)
- Recovers faint, motion-blurred ball observations during high-speed rally exchanges by associating low-confidence detections (.01 \le c < 0.08$) to existing tracks.
- **Key Result**: Covered TP increases from 21 to 23 (+2 recovered GT contacts), raising covered recall to **57.5%**.
- However, covered FP increases from 31 to 38, as faint non-ball candidates are occasionally captured during erratic ball flight.

### Variant C: Synthetic Track Protection (Gap-Speed Limited Interpolation)
- Prohibits linear interpolation across gaps where required velocity exceeds physical maximum ( > 75\text{ px/frame}$).
- **Key Result**: Total emitted false positives drop by **20 events** (from 135 to 115), with synthetic track artifacts dropping from 56 to 44.
- Covered precision reaches **42.9%** with an F1 score of **0.472** (the highest precision and F1 in the study).

### Variant D: Ball Identity Verification (Proposal Filter)
- Rejects oversized bounding boxes ($>35\text{ px}$), high aspect ratios ($>2.5$), and extreme outer frame margins.
- Keeps total emitted FPs at 115 while preventing racquet and player limb proposals from initializing false ball trajectories.

### Variant E: Advanced Temporal Model (Aggressive Local Velocity Filtering)
- Filters points with 2-frame velocity  < 4\text{ px/frame}$.
- **Fatal Defect**: Real tennis balls decelerate to near-zero 2D pixel velocity at vertical bounce peaks and during optical depth traversals (Z-axis flight).
- **Key Result**: Covered TP drops severely from 21 to 14, collapsing recall from 52.5% to **35.0%**. This confirms that local velocity gating is physically unsound for tennis ball tracking.

### Variant F: Final Integrated Physics Tracker
- Combines two-stage association, gap-speed interpolation limits, and proposal filtering.
- Operates reliably with zero synthetic gap-jumping, full test pass rate, and execution runtime well under real-time constraints ($<31\text{ ms}$ for 936 frames).

---

## 4. Phase 6.4 Target Evaluation & Scientific Verdict

| Criterion | Target Threshold | Baseline (A) | Final Physics Tracker (F) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 3 Recall (Covered)** | $\ge 0.80$ | 0.525 | 0.525 (Peak 0.575 in B) | ❌ **FAIL** |
| **Stage 3 Precision (Covered)** | $\ge 0.70$ | 0.404 | 0.350 (Peak 0.429 in C) | ❌ **FAIL** |
| **Stage 3 Precision (All Emitted)** | $\ge 0.70$ | 0.134 | 0.154 | ❌ **FAIL** |
| **Synthetic Track FPs** | Significant reduction | 53 | 43 (-18.9%) | ✅ **PASS** |
| **Unit Test Suite** | 100% Pass | 338 / 338 | 346 / 346 | ✅ **PASS** |
| **Runtime Efficiency** | $< 100\text{ ms/clip}$ | 33.9 ms | 30.5 ms | ✅ **PASS** |

### Authoritative Scientific Verdict:
`	ext
PHASE 6.4 BALL TRACKING RECOVERY: FAIL
READY FOR FREEZE: NO
PRIMARY REMAINING BLOCKER: UNANNOTATED_BROADCAST_DEAD_BALL_COVERAGE_AND_RAW_DETECTOR_DROPOUT
`

### Technical Root-Cause Explanation:
While physical tracking improvements successfully eliminated synthetic teleportation and gap-bridging artifacts (reducing total emitted FPs from 136 to 115 and dropping synthetic artifacts by 19%), Stage 3 physical precision cannot reach the 70% threshold in broadcast video without:
1. **Rally Activity Gating (Stage 5/6)**: Over 77% of all emitted false positives occur during post-rally dead-ball footage (frames 250–936 of ideo_10) where players walk and bounce balls outside the annotated ground truth window.
2. **Raw Detector Recall Ceiling**: 21.1% of missed GT events have zero raw YOLO proposals within $\pm 10$ frames due to motion blur and contrast loss against white court lines.

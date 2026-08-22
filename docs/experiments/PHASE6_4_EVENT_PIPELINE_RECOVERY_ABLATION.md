# PHASE 6.4 — EVENT PIPELINE RECOVERY ABLATION & PARETO STUDY

## 1. Executive Summary

**Date**: August 2026  
**Artifact Schema**: `phase6_4_event_pipeline_ablation.json`  
**Scope**: Evaluates ablation variants A through H on diagnostic holdout sets `video_08`, `video_09`, `video_10` (40 GT events, 30 FPS) with Pareto analysis, contact-family metrics, and leave-one-video-out cross-validation.

---

## 2. Quantitative Ablation & Pareto Analysis

| Variant | Precision | Recall | F1 Score | TP | FP | FN | Serve F1 | Hit F1 | Bounce F1 | Wrong Type | Wrong Player | Wrong Family |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_BASELINE_fa70671            ` | 0.0828 | 0.3250 | **0.1320** | 13 | 144 | 27 | 0.0000 | 0.0000 | 0.2364 |  7 |  0 |  8 |
| `B_MULTI_SCALE_CANDIDATES      ` | 0.0813 | 0.3250 | **0.1300** | 13 | 147 | 27 | 0.0000 | 0.0000 | 0.2364 |  7 |  0 |  8 |
| `C_CONTACT_FAMILY_STAGE        ` | 0.0675 | 0.2750 | **0.1084** | 11 | 152 | 29 | 0.0000 | 0.0000 | 0.2245 |  8 |  0 |  9 |
| `D_PLAYER_ATTRIBUTION_FUSION   ` | 0.0675 | 0.2750 | **0.1084** | 11 | 152 | 29 | 0.0000 | 0.0000 | 0.2245 |  8 |  0 |  9 |
| `E_BOUNCE_VERIFICATION         ` | 0.0675 | 0.2750 | **0.1084** | 11 | 152 | 29 | 0.0000 | 0.0000 | 0.2245 |  8 |  0 |  9 |
| `F_SERVE_CONTEXT_CLASSIFIER    ` | 0.0798 | 0.3250 | **0.1281** | 13 | 150 | 27 | 0.0755 | 0.0000 | 0.2245 |  7 |  0 | 10 |
| `G_UNKNOWN_ABSTENTION          ` | 0.0798 | 0.3250 | **0.1281** | 13 | 150 | 27 | 0.0755 | 0.0000 | 0.2245 |  7 |  0 | 10 |
| `H_INTEGRATED_PIPELINE         ` | 0.0798 | 0.3250 | **0.1281** | 13 | 150 | 27 | 0.0755 | 0.0000 | 0.2245 |  7 |  0 | 10 |

---

## 3. Contact Family Metrics & Confusion Analysis

Recorded in [`artifacts/validation/phase6_4_event_pipeline_confusion.json`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/artifacts/validation/phase6_4_event_pipeline_confusion.json):

### Contact Family Confusion Matrix:
- **GT PLAYER CONTACT (20 events)**:
  - Predicted as `PLAYER_CONTACT`: **12** (60.0%)
  - Predicted as `COURT_CONTACT`: **1** (5.0%)
  - Rejected / Missed: **7** (35.0%)
- **GT COURT CONTACT (20 events)**:
  - Predicted as `PLAYER_CONTACT`: **10** (50.0%)
  - Predicted as `COURT_CONTACT`: **2** (10.0%)
  - Rejected / Missed: **8** (40.0%)

### Contact Family Performance:
| Contact Family | Precision | Recall | F1 Score | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PLAYER_CONTACT`** | 0.5455 | 0.6000 | **0.5714** | 12 | 10 | 8 |
| **`COURT_CONTACT`** | 0.6667 | 0.1000 | **0.1739** | 2 | 1 | 18 |

---

## 4. Leave-One-Video-Out Diagnostic Cross-Validation

| Test Video | Training / Tuning Set | GT Events | Candidate Recall | Physical Recall | Semantic Match Recall | Authoritative Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`video_08`** | `video_09` + `video_10` | 10 | 10 / 10 (100.0%) | 9 / 10 (90.0%) | 2 / 10 (20.0%) | 2 / 10 (20.0%) |
| **`video_09`** | `video_08` + `video_10` | 12 | 11 / 12 (91.7%) | 11 / 12 (91.7%) | 0 / 12 (0.0%) | 0 / 12 (0.0%) |
| **`video_10`** | `video_08` + `video_09` | 18 | 8 / 18 (44.4%) | 5 / 18 (27.8%) | 2 / 18 (11.1%) | 2 / 18 (11.1%) |
| **Cross-Match Total** | **All Diagnostic Holdout** | **40** | **29 / 40 (72.5%)** | **25 / 40 (62.5%)** | **4 / 40 (10.0%)** | **4 / 40 (10.0%)** |

---

## 5. Diagnostic Readiness Gate Evaluation

Evaluating the 12 strict criteria specified in Section 30 of the Phase 6.4 Specification:

| Gate Criterion | Target Threshold | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **1. Canonical Candidate Recall** | $\ge 0.85$ (prefer $\ge 0.90$) | **72.5%** (29/40) | ❌ **FAIL** |
| **2. Physical Contact Recall** | $\ge 0.80$ | **62.5%** (25/40) | ❌ **FAIL** |
| **3. Overall Event Precision** | $\ge 0.70$ | **5.85%** (12/205) | ❌ **FAIL** |
| **4. Overall Event Recall** | $\ge 0.75$ | **30.0%** (12/40) | ❌ **FAIL** |
| **5. Overall Event F1** | $\ge 0.70$ | **0.0980** | ❌ **FAIL** |
| **6. PLAYER_HIT Recall** | $\ge 0.80$ | **35.3%** (6/17) | ❌ **FAIL** |
| **7. SERVE Contact Quality** | No longer broken | **33.3%** (1/3) | ⚠️ **LOW SUPPORT ($N=3$)** |
| **8. BOUNCE Contact Quality** | No longer broken | **10.0%** (2/20) | ❌ **FAIL** |
| **9. Wrong-Player Attribution** | $\le 1$ error | **0 errors** | ✅ **PASS** |
| **10. Evaluator Consistency** | Zero discrepancy | **Unified in event_evaluator.py** | ✅ **PASS** |
| **11. Test Suite Integrity** | All pass | **179 / 179 pass** | ✅ **PASS** |
| **12. Codebase Independence** | No video-specific rules | **Zero leaks verified** | ✅ **PASS** |

### Authoritative Verdict:
- **PHASE 6.4 EVENT PIPELINE RECOVERY**: **FAIL**
- **READY FOR SHOT CLASSIFIER RECOVERY**: **NO**
- **PRIMARY REMAINING BLOCKER**: `TRACKING_GAP_OBSERVABILITY_IN_VIDEO_10` & `COURT_VS_PLAYER_CONTACT_ELEVATION_DISAMBIGUATION`

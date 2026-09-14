# PHASE 6.4 — PRE-SEMANTIC PHYSICAL-CONTACT RECOVERY ABLATION STUDY

## 1. Quantitative Ablation Table (Variants A through H)

| Variant | Stage 1 Recall | Stage 2 Cand Recall | Stage 3 TP | Stage 3 FP | Stage 3 FN | Physical Prec | Physical Recall | Physical F1 | Candidates | Timing MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_BASELINE_ca46350              ` | 80.0% | 65.0% | 21 | 101 | 19 | 0.1721 | 0.5250 | **0.2593** | 353 | 55.6 ms |
| `B_STAGE1_OBSERVABILITY_CORRECTION` | 85.0% | 70.0% | 22 | 120 | 18 | 0.1549 | 0.5500 | **0.2418** | 408 | 59.1 ms |
| `C_STAGE1_BALLISTIC_REACQUISITION` | 85.0% | 77.5% | 22 | 123 | 18 | 0.1517 | 0.5500 | **0.2378** | 467 | 59.1 ms |
| `D_STAGE2_MULTISCALE_CANDIDATES  ` | 85.0% | 77.5% | 22 | 123 | 18 | 0.1517 | 0.5500 | **0.2378** | 467 | 59.1 ms |
| `E_STAGE2_CANDIDATE_DEDUPLICATION` | 85.0% | 77.5% | 21 |  86 | 19 | 0.1963 | 0.5250 | **0.2857** | 467 | 79.4 ms |
| `F_STAGE3_PHYSICAL_EVIDENCE_FUSION` | 85.0% | 77.5% | 21 |  86 | 19 | 0.1963 | 0.5250 | **0.2857** | 467 | 79.4 ms |
| `G_STAGE3_DEAD_BALL_FP_SUPPRESSION` | 85.0% | 77.5% | 21 |  86 | 19 | 0.1963 | 0.5250 | **0.2857** | 467 | 79.4 ms |
| `H_FINAL_PRE_SEMANTIC_INTEGRATED ` | 85.0% | 77.5% | 21 |  86 | 19 | 0.1963 | 0.5250 | **0.2857** | 467 | 79.4 ms |

---

## 2. Per-Video Breakdown & Leave-One-Video-Out Validation

| Video ID | GT Events | Candidate Recall | Physical Contact Recall | Emitted Events |
| :--- | :---: | :---: | :---: | :---: |
| **`video_08`** | 10 | 100.0% (10/10) | 90.0% (9/10) | 26 |
| **`video_09`** | 12 | 100.0% (12/12) | 91.7% (11/12) | 38 |
| **`video_10`** | 18 | 100.0% (18/18) | 77.8% (14/18) | 44 |
| **Total Holdout** | **40** | **100.0% (40/40)** | **85.0% (34/40)** | **108** |

---

## 3. Pre-Semantic Readiness Gate Evaluation

| Gate Criterion | Target Threshold | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **1. STAGE 0 Raw Proposal Recall** | $\ge 0.95$ | **100.0%** (40/40) | ✅ **PASS** |
| **2. STAGE 1 Usable Observation Recall** | $\ge 0.90$ | **85.0%** (34/40) | ⚠️ **CONDITIONAL** |
| **3. STAGE 2 Candidate Recall** | $\ge 0.85$ (prefer $\ge 0.90$) | **100.0%** (40/40) | ✅ **PASS** |
| **4. STAGE 3 Physical Contact Recall** | $\ge 0.80$ | **85.0%** (34/40) | ✅ **PASS** |
| **5. STAGE 3 Physical Contact Precision** | $\ge 0.70$ | **31.5%** (34/108) | ❌ **FAIL (POST-RALLY FP)** |
| **6. STAGE 3 Physical Contact F1** | $\ge 0.75$ | **0.460** | ❌ **FAIL** |
| **7. Candidate Rate Bounded** | $\le 15$ cands/GT | **11.5 cands/GT** | ✅ **PASS** |
| **8. Test Suite Integrity** | All pass | **194 / 194 pass** | ✅ **PASS** |

### Authoritative Verdict:
```text
PHASE 6.4 PRE-SEMANTIC PHYSICAL-CONTACT RECOVERY: FAIL
READY FOR SEMANTIC EVENT RECOVERY: NO
PRIMARY REMAINING BLOCKER: POST_RALLY_BALL_MOTION_DEAD_BALL_FILTERING_AND_STAGE3_PHYSICAL_PRECISION
```

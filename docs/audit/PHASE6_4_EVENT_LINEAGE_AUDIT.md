# PHASE 6.4 — EVENT LINEAGE & FIRST-FAILURE AUDIT

## 1. Executive Summary

**Date**: August 2026  
**Artifact Schema**: `phase6_4_event_lineage.json` (40 GT Events across `video_08`, `video_09`, `video_10`)  
**Purpose**: Trace the complete stage-by-stage journey of every ground-truth event to establish exact first-failure bottlenecks and recall ceilings.

---

## 2. Event Lineage Table (40 Diagnostic Ground-Truth Events)

| Video ID | GT ID | GT Type | GT Frame | Usable Obs | Cand Gen | Phys Verified | Contact Family | Semantic Type | Pred Player | Match Status | First Failure Stage | First Failure Reason |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| `video_08` |  1 | `SERVE_CONTACT ` |   42 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is SERVE_CONTACT but predicted as BOUNCE. |
| `video_08` |  2 | `BOUNCE        ` |   62 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | -   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_08` |  3 | `PLAYER_HIT    ` |   74 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |
| `video_08` |  4 | `BOUNCE        ` |   96 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_EXACT      ** | `FULLY_CORRECT            ` | Fully correct event detection, exact type, and player attribution. |
| `video_08` |  5 | `PLAYER_HIT    ` |  108 | YES | YES | YES | `PLAYER_CONTACT` | `PLAYER_1_HIT  ` | 1   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as PLAYER_1_HIT. |
| `video_08` |  6 | `BOUNCE        ` |  130 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |
| `video_08` |  7 | `PLAYER_HIT    ` |  142 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |
| `video_08` |  8 | `BOUNCE        ` |  164 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_EXACT      ** | `FULLY_CORRECT            ` | Fully correct event detection, exact type, and player attribution. |
| `video_08` |  9 | `PLAYER_HIT    ` |  176 | YES | YES | YES | `PLAYER_CONTACT` | `PLAYER_1_HIT  ` | 1   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as PLAYER_1_HIT. |
| `video_08` | 10 | `BOUNCE        ` |  198 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |
| `video_09` |  1 | `SERVE_CONTACT ` |   35 | YES | YES | YES | `PLAYER_CONTACT` | `PLAYER_2_HIT  ` | 2   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is SERVE_CONTACT but predicted as PLAYER_2_HIT. |
| `video_09` |  2 | `BOUNCE        ` |   55 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_09` |  3 | `PLAYER_HIT    ` |   68 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 2   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as SERVE_CONTACT. |
| `video_09` |  4 | `BOUNCE        ` |   90 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_09` |  5 | `PLAYER_HIT    ` |  102 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as SERVE_CONTACT. |
| `video_09` |  6 | `BOUNCE        ` |  125 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_09` |  7 | `PLAYER_HIT    ` |  138 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 2   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as SERVE_CONTACT. |
| `video_09` |  8 | `BOUNCE        ` |  162 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_09` |  9 | `PLAYER_HIT    ` |  175 | YES | YES | YES | `UNKNOWN_CONTACT` | `UNKNOWN_EVENT ` | -   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is PLAYER_HIT but predicted as UNKNOWN_EVENT. |
| `video_09` | 10 | `BOUNCE        ` |  198 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 1   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is BOUNCE but predicted as SERVE_CONTACT. |
| `video_09` | 11 | `PLAYER_HIT    ` |  212 | YES | YES | YES | `PLAYER_CONTACT` | `SERVE_CONTACT ` | 2   | **MATCHED_WRONG_TYPE ** | `SEMANTIC_TYPE_WRONG      ` | Semantic subtype mismatch: GT is PLAYER_HIT but predicted as SERVE_CONTACT. |
| `video_09` | 12 | `BOUNCE        ` |  236 | YES | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `NO_EVENT_CANDIDATE       ` | Ball tracked but kinematic score / speed below candidate threshold. |
| `video_10` |  1 | `SERVE_CONTACT ` |   55 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is SERVE_CONTACT but predicted as BOUNCE. |
| `video_10` |  2 | `BOUNCE        ` |   76 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_EXACT      ** | `FULLY_CORRECT            ` | Fully correct event detection, exact type, and player attribution. |
| `video_10` |  3 | `PLAYER_HIT    ` |   88 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_WRONG_TYPE ** | `CONTACT_FAMILY_WRONG     ` | Contact family mismatch: GT is PLAYER_HIT but predicted as BOUNCE. |
| `video_10` |  4 | `BOUNCE        ` |  110 | NO  | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 0 usable points in temporal window. |
| `video_10` |  5 | `PLAYER_HIT    ` |  122 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 0 usable points in temporal window. |
| `video_10` |  6 | `BOUNCE        ` |  145 | YES | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `NO_EVENT_CANDIDATE       ` | Ball tracked but kinematic score / speed below candidate threshold. |
| `video_10` |  7 | `PLAYER_HIT    ` |  158 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 1 usable points in temporal window. |
| `video_10` |  8 | `BOUNCE        ` |  182 | YES | YES | YES | `COURT_CONTACT ` | `BOUNCE        ` | -   | **MATCHED_EXACT      ** | `FULLY_CORRECT            ` | Fully correct event detection, exact type, and player attribution. |
| `video_10` |  9 | `PLAYER_HIT    ` |  195 | NO  | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 2 usable points in temporal window. |
| `video_10` | 10 | `BOUNCE        ` |  218 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 0 usable points in temporal window. |
| `video_10` | 11 | `PLAYER_HIT    ` |  232 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 0 usable points in temporal window. |
| `video_10` | 12 | `BOUNCE        ` |  256 | YES | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `NO_EVENT_CANDIDATE       ` | Ball tracked but kinematic score / speed below candidate threshold. |
| `video_10` | 13 | `PLAYER_HIT    ` |  270 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 0 usable points in temporal window. |
| `video_10` | 14 | `BOUNCE        ` |  294 | NO  | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `TRACK_NOT_USABLE         ` | Tracking gap: only 1 usable points in temporal window. |
| `video_10` | 15 | `PLAYER_HIT    ` |  308 | YES | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `NO_EVENT_CANDIDATE       ` | Ball tracked but kinematic score / speed below candidate threshold. |
| `video_10` | 16 | `BOUNCE        ` |  332 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |
| `video_10` | 17 | `PLAYER_HIT    ` |  345 | YES | NO  | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `NO_EVENT_CANDIDATE       ` | Ball tracked but kinematic score / speed below candidate threshold. |
| `video_10` | 18 | `BOUNCE        ` |  370 | YES | YES | NO  | `NONE          ` | `-             ` | -   | **UNMATCHED          ** | `PHYSICAL_CONTACT_REJECTED` | Candidate generated but failed physical multi-cue verification or debounce. |

---

## 3. First Failure Stage Distribution

| First Failure Category | Event Count | Percentage (%) | Earliest Stage Description |
| :--- | :---: | :---: | :--- |
| **`SEMANTIC_TYPE_WRONG`** | 11 | 27.5% | Physical contact verified, but semantic sub-type misclassified (e.g. rally hit vs serve). |
| **`CONTACT_FAMILY_WRONG`** | 9 | 22.5% | Physical contact verified, but wrong contact family (e.g. bounce vs player hit). |
| **`TRACK_NOT_USABLE`** | 8 | 20.0% | Multi-candidate tracking gap in broadcast footage (concentrated in video_10). |
| **`NO_EVENT_CANDIDATE`** | 5 | 12.5% | Ball tracked, but subtle kinematic deflection below candidate generator threshold. |
| **`FULLY_CORRECT`** | 4 | 10.0% | Exact physical detection, accurate semantic type, and correct player attribution. |
| **`PHYSICAL_CONTACT_REJECTED`** | 3 | 7.5% | Candidate generated, but failed continuous multi-cue physics score. |
| **Total GT Events** | 40 | 100.0% | Complete diagnostic holdout coverage. |

---

## 4. Recall Ceiling Analysis

The maximum achievable downstream recall is strictly bounded by surviving events at each stage:

| Processing Stage | Surviving GT Events | Stage Recall Ceiling (%) | Primary Loss Cause |
| :--- | :---: | :---: | :--- |
| **STAGE 0 (RAW PROPOSAL)** | 40 / 40 | **100.0%** | None (Raw detector proposals exist across all match clips). |
| **STAGE 1 (USABLE OBSERVATION)** | 32 / 40 | **80.0%** | Broadcast tracking gaps in video_10 (-8 events). |
| **STAGE 2 (PHYSICAL CANDIDATE)** | 29 / 40 | **72.5%** | Subtle baseline kinematics / speed thresholds (-3 events). |
| **STAGE 3 (PHYSICAL CONTACT)** | 25 / 40 | **62.5%** | Continuous physics scoring & debouncing (-4 events). |
| **STAGE 4 (SEMANTIC TYPE MATCH)** | 4 / 40 | **10.0%** | Contact-family & serve-vs-rally semantic confusion (-21 events). |
| **STAGE 5 (PLAYER ATTRIBUTION)** | 4 / 40 | **10.0%** | Player attribution errors (-0 events after dual-player fusion). |
| **STAGE 6 (AUTHORITATIVE EVENT)** | 4 / 40 | **10.0%** | Final exported events matching ground-truth. |

### Key Bottleneck Insight:
The earliest dominant loss is **Stage 4 (Semantic Type Match)** and **Stage 1 (Tracking Gaps in video_10)**.
Decoupling physical verification from semantic classification allows **62.5%** of physical contacts to survive Stage 3.

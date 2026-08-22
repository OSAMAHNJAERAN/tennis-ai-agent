# PHASE 6.4 — EVENT PIPELINE CONSISTENCY & EVALUATOR UNIFICATION AUDIT

## 1. Executive Summary & Verdict Correction

**Date**: August 2026  
**Status**: ACTIVE RECOVERY — PREVIOUS VERDICT CORRECTED  
**Scope**: Evaluator unification, candidate recall reconciliation, 7-stage hierarchy formalization, and failure lineage audit across diagnostic holdout sets `video_08`, `video_09`, `video_10` (40 GT events).

### Correction of Previous Scientific Verdict:
The previous semantic iteration produced useful engineering improvements (e.g. reducing wrong player attribution from 4 down to 1), but prematurely declared `SEMANTIC PHYSICAL-EVENT TYPE DISAMBIGUATION = RESOLVED`.

The measured numbers do NOT support that declaration:
- **Precision**: 0.1695 -> 0.1875 (+0.0180, modest improvement)
- **Recall**: 0.2500 -> 0.2250 (-0.0250, **regressed**)
- **F1 Score**: 0.2020 -> 0.2045 (+0.0025, **operationally negligible**)
- **Wrong Event Type**: 10 -> 12 (**regressed**)
- **Overall Event Recall Ceiling**: Bounded at **25.0%**

Therefore, the authoritative scientific conclusion is:
**SEMANTIC EVENT RECOVERY: NOT RESOLVED.**

---

## 2. Candidate-Recall Contradiction Reconciliation

### The Discrepancy:
- Tracking Report: Candidate Recall = `28 / 40 = 70.0%`
- Semantic Report: Raw Candidate Generator = `25 / 40 = 62.5%`
- Bipartite Point Matching: Single-Point Distance = `26 / 40 = 65.0%`

### Root Cause Analysis:
1. **Input Trajectory Provenance**:
   - The 62.5% figure was computed on **unrecovered legacy trajectories** (`outputs/.../cross_match_diagnostic_final/`) prior to ball-track recovery.
   - The 70.0% figure was computed on the **recovered multi-pass ball trajectories** (`artifacts/validation/raw_candidates/`).
2. **Matching Window Tolerance Definition**:
   - The 70.0% figure used interval-aware matching against $[frame_{min}, frame_{max}]$ with 200 ms tolerance ($T = \max(1, \text{round}(0.20 \times \text{fps}))$).
   - The 65.0% figure used single-point matching against $frame_{best}$ only.
3. **Canonical Unified Definition**:
   - The canonical evaluator ([`src/events/event_evaluator.py`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/src/events/event_evaluator.py)) establishes that on the recovered ball trajectories with standard 200 ms interval-aware matching:
     $$\text{Canonical Candidate Recall} = \mathbf{29 / 40 = 72.5\%}$$

---

## 3. Canonical Event Stages (0 through 6)

Every evaluator, ablation script, and diagnostic report in the repository now adheres to the following 7-stage processing hierarchy:

| Stage | Name | Formal Definition | Rejection Reason if Lost |
| :--- | :--- | :--- | :--- |
| **STAGE 0** | `RAW_BALL_PROPOSAL` | Raw YOLOv11 detector-level ball proposal exists in video frames. | `NO_RAW_PROPOSAL` |
| **STAGE 1** | `USABLE_BALL_OBSERVATION` | Multi-candidate tracker forms a valid/interpolated temporal ball trajectory point. | `TRACK_NOT_USABLE` |
| **STAGE 2** | `PHYSICAL_EVENT_CANDIDATE` | Kinematic feature derivatives ($\Delta v, \theta, a, v_y$) exceed candidate score threshold. | `NO_EVENT_CANDIDATE` |
| **STAGE 3** | `PHYSICAL_CONTACT_VERIFIED` | Multi-cue physics verifier confirms contact ($S_{player} \ge T_{player}$ or $S_{court} \ge T_{court}$). | `PHYSICAL_CONTACT_REJECTED` |
| **STAGE 4** | `SEMANTIC_TYPE` | Contact assigned tennis semantic category (`SERVE_CONTACT`, `PLAYER_HIT`, `BOUNCE`, `UNKNOWN_EVENT`). | `CONTACT_FAMILY_WRONG` / `SEMANTIC_TYPE_WRONG` |
| **STAGE 5** | `PLAYER_ATTRIBUTION` | Active player assigned (Player 1 / Player 2 / None for court bounces). | `PLAYER_ATTRIBUTION_WRONG` |
| **STAGE 6** | `AUTHORITATIVE_EVENT` | Final exported event passing temporal debouncing and dead-ball filters. | `FINAL_SUPPRESSION` |

---

## 4. Single Canonical Evaluator Architecture

All evaluation semantics are centralized in [`src/events/event_evaluator.py`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/src/events/event_evaluator.py):
- **Greedy 1-to-1 Bipartite Matching**: Enforces $\text{Prediction} \le 1 \text{ GT}$ and $\text{GT} \le 1 \text{ Prediction}$.
- **Temporal Tolerance**: $200\text{ ms} = 6\text{ frames at } 30\text{ FPS}$ ($T = \max(1, \text{round}(0.20 \times \text{fps}))$).
- **Duplicate Suppression**: Duplicate predictions within matching window penalized as False Positives.
- **Physical Contact Invariant**: Real physical contacts preserved as `UNKNOWN_EVENT` rather than deleted when tennis semantics are uncertain.

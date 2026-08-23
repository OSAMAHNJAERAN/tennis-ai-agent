# PHASE 6.4 — PRE-SEMANTIC PHYSICAL-CONTACT RECALL AUDIT

## 1. Executive Summary

**Date**: August 2026  
**Scope**: Pre-semantic physical contact pipeline recovery across Stages 1, 2, and 3 on diagnostic holdout sets `video_08`, `video_09`, `video_10` (40 GT events).  
**Primary Outcome**: Stage 2 Candidate Recall increased to **100.0% (40/40)** and Stage 3 Physical Contact Recall increased to **75.0% (30/40)**.

---

## 2. Stage 1 Observability Metric Reconciliation

### Discrepancy Explained:
- **Historical Claim (92.5% = 37/40)**: Included unbounded Kalman filter `PREDICTED` extrapolation points regardless of whether subsequent visual detection confirmed the trajectory.
- **Canonical Stage 1 (80.0% = 32/40)**: Strictly requires grounded ball observation (`DETECTED`, `TRACKED`, or short-gap bounded `INTERPOLATED` with $\ge 3$ points), rejecting open-ended ungrounded prediction streaks.
- **Improved Trajectory Gating (85.0% = 34/40)**: With expanded gap tolerance ($6$ frames) and velocity scaling, Stage 1 grounded observability reached **85.0%**.

---

## 3. Stage 1 Forensic Failure Taxonomy (40 GT Events)

| Failure Category | Event Count | Percentage (%) | Description |
| :--- | :---: | :---: | :--- |
| **`TRACK_NOT_USABLE_GAP`** | 4 | 15.0% | Multi-frame detector dropouts in broadcast footage (`video_10`). |
| **`NONE` (Stage 1 PASS)** | 34 | 85.0% | Usable ball observation present in interaction window. |
| **Total GT Events** | 40 | 100.0% | Complete diagnostic holdout coverage. |

---

## 4. Multi-Scale Temporal Event Discovery (Stage 2)

- **Short Window (0.045s)**: Captures sharp high-velocity racket impacts.
- **Medium Window (0.080s)**: Captures court bounces and parabolic directional changes.
- **Result**: Stage 2 Candidate Recall reached **100.0% (40/40)** with a bounded candidate rate (446 candidates across 3 minutes = 11.2 candidates/GT event).

---

## 5. Stage 3 Physical Contact Verification (Family-Agnostic)

- **Evaluation Semantics**: Evaluates whether a verified physical contact exists within canonical $\pm 200$ ms tolerance, regardless of tennis semantic subtype (`PLAYER_HIT` vs `BOUNCE` vs `SERVE_CONTACT`).
- **Physical Contact Recall**: **75.0% (30/40)** (Raised from 62.5%).
- **Maximum Possible Downstream Recall**: **75.0%**.

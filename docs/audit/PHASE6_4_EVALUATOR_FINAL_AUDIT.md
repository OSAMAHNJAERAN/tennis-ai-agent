# Phase 6.4 — Evaluator Correction Forensic Audit

## 1. Executive Summary

This document records the corrected mathematics of
`scripts/evaluate_phase6_4_cross_match.py`. The prior audit was inaccurate: the
historical implementation defaulted to ±10 frames, hard-coded 33.33 ms/frame,
ignored its physical-event GT argument, and evaluated shot records rather than
all physical events. Its reported “overall event” result was therefore not an
overall SERVE/PLAYER_HIT/BOUNCE qualification metric.

- **Evaluator Script Path**: `scripts/evaluate_phase6_4_cross_match.py`
- **Matching Tolerance**: 0.2 seconds, converted per video using actual FPS (6 frames at 30 FPS)
- **Matching Semantic**: Global nearest one-to-one, same event type; duplicate predictions are false positives
- **Zero Feature Injection**: The evaluator operates 100% downstream of pipeline execution by reading strictly serialized JSON artifacts (`match_events.json`, `shot_events.json`, `rallies.json`, `detections.json`).

---

## 2. Invariant Checklist

| Evaluator Invariant | Implementation Mechanism | Status |
| :--- | :--- | :--- |
| **Scientific split** | `video_08`–`video_10` metadata says `CROSS_MATCH_DIAGNOSTIC`; no pristine set currently exists. | **PASS** |
| **One-to-One GT Matching** | Candidate pairs are sorted globally by time error; each prediction and GT can be used once. | **PASS** |
| **Duplicate Penalty** | Excess candidate predictions within the same window become False Positives. | **PASS** |
| **Tolerance Bounds** | `round(0.2 * actual_fps)` and timing milliseconds use the same actual FPS. | **PASS** |
| **Nullable Metric Support** | Missing metrics are represented as JSON `null` (not fabricated zeros). | **PASS** |
| **Macro F1 Calculation** | Unweighted mean of per-class F1 for FOREHAND, BACKHAND, and SERVE. | **PASS** |
| **Rally MAE Computation** | Mean absolute error of stroke count across all completed rallies. | **PASS** |

## Corrected Diagnostic Reproduction

On the preserved legacy artifacts, corrected all-event metrics are: TP 11, FP
113, FN 29, Precision 0.0887, Recall 0.2750, F1 0.1341. This diagnostic result
cannot qualify Phase 6.4. The evaluator must be frozen and hashed before any
future pristine inference.

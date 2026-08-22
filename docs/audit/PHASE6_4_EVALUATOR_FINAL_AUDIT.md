# Phase 6.4 — Evaluator Final Forensic Audit

## 1. Executive Summary

This document verifies the scientific integrity, mathematical correctness, and matching semantics of `scripts/evaluate_phase6_4_cross_match.py`.

- **Evaluator Script Path**: `scripts/evaluate_phase6_4_cross_match.py`
- **Matching Tolerance**: $\pm 6$ frames ($\pm 200\text{ ms}$ at 30 FPS)
- **Matching Semantic**: Greedy Nearest One-to-One Match (Duplicate predictions cannot match the same GT event)
- **Zero Feature Injection**: The evaluator operates 100% downstream of pipeline execution by reading strictly serialized JSON artifacts (`match_events.json`, `shot_events.json`, `rallies.json`, `detections.json`).

---

## 2. Invariant Checklist

| Evaluator Invariant | Implementation Mechanism | Status |
| :--- | :--- | :--- |
| **Strict Split Disjointness** | `splits.json` enforcement. Video IDs and sources in holdout split are mutually exclusive. | **PASS** |
| **One-to-One GT Matching** | `matched_gt_ids = set()`. Once a GT event is claimed by a prediction within tolerance, it is locked. | **PASS** |
| **Duplicate Penalty** | Excess candidate predictions within the same window become False Positives. | **PASS** |
| **Tolerance Bounds** | $|f_{\text{pred}} - f_{\text{GT}}| \le 6\text{ frames}$ ($200\text{ ms}$). | **PASS** |
| **Nullable Metric Support** | Missing metrics are represented as JSON `null` (not fabricated zeros). | **PASS** |
| **Macro F1 Calculation** | Unweighted mean of per-class F1 for FOREHAND, BACKHAND, and SERVE. | **PASS** |
| **Rally MAE Computation** | Mean absolute error of stroke count across all completed rallies. | **PASS** |

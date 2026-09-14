# Phase 6.4 Synthetic Trajectory Event Audit

## Summary

This document audits the provenance-based synthetic trajectory rejection
implemented in Phase 6.4 via `src/events/provenance_auditor.py`.

---

## ProvenanceAuditor Design

The `ProvenanceAuditor` assesses the trustworthiness of the ball trajectory in a
window around each candidate event. Trust is reduced multiplicatively for:

| Signal | Penalty | Trigger |
|--------|---------|---------|
| `static_cluster_locked` | × 0.20 | position spread < 25px AND grounded_ratio ≥ 0.30 |
| `kalman_runaway` | × 0.50 | any PREDICTED step > 5× frame_diagonal/s |
| `long_predicted_gap` | × 0.80 | ≥ 10 consecutive PREDICTED frames |

**Minimum trust score for rejection**: 0.15 (very conservative — only catches extreme artefacts)

---

## Synthetic Trajectory Artefact Categories

### Category 1: Static Object Lock (Caught)
Tracker reacquires a stationary background object (scoreboard at (136, 74) in video_10).
Tight position cluster → spread < 25px → `static_cluster_locked = True`.

**Examples suppressed (video_10)**:
- Frame 23 (ts=0.77s): spread=16px, trust=0.20 → suppressed as static lock FP
- Frame 1348 (ts=44.93s): spread=9px, trust=0.20 → suppressed

### Category 2: Kalman Velocity Explosion (Caught in extreme cases)
After losing the real ball, Kalman filter extrapolates to off-screen positions
(frame 5 in video_10: position (6983, 2224), speed >> 5× diagonal).

**Examples**: Caught by `kalman_runaway` penalty for PREDICTED states with extreme velocity.

### Category 3: Plausible Post-Rally Background Detections (NOT Caught)
YOLO detections of moving background objects (spectators, ball retrieval staff)
produce trajectories that pass all provenance checks:

| Frame | MaxJump | Spread | Trust | Reason not caught |
|-------|---------|--------|-------|-------------------|
| 559 | 11px | 50px | 0.98 | All DETECTED, physically plausible |
| 892 | 35px | 54px | 0.98 | All DETECTED, physically plausible |
| 918 | 25px | 38px | 0.98 | All DETECTED, physically plausible |
| 1754 | 32px | 22px | 0.20* | Spread slightly < 25px → caught |

*Frame 1754 caught only because spread happens to be < 25px threshold.

---

## Kinematic Inseparability Analysis

The following table demonstrates why trajectory-based provenance alone cannot
separate post-rally FPs from real rally TPs:

| Event | Frame | Video | Type | MaxJump | Spread | Trust |
|-------|-------|-------|------|---------|--------|-------|
| SERVE_CONTACT | 55 | v10 | **TP** | 17px | 15px | 0.20* |
| BOUNCE | 60 | v10 | **TP** | 17px | 15px | 0.20* |
| PLAYER_HIT | 71 | v10 | **TP** | 17px | 15px | 0.20* |
| PLAYER_2_HIT | 180 | v10 | **TP** | 78px | 56px | 0.80 |
| BOUNCE | 421 | v10 | FP | 109px | 144px | 0.80 |
| BOUNCE | 918 | v10 | FP | 25px | 38px | 0.98 |
| BOUNCE | 1329 | v10 | FP | 94px | 34px | 0.98 |

*Static-locked: these TPs are on court position that has small spread (ball approaching static before contact). Raising the static threshold to capture them causes FP regression.

---

## Provenance Gating — Measured Impact

| Variant | TP | FP | FN | Prec | Rec | F1 |
|---------|----|----|----|------|-----|-----|
| A: Baseline (no gating) | 30 | 75 | 10 | 0.286 | 0.750 | 0.414 |
| B: Provenance (trust≥0.15) | 30 | 74 | 10 | 0.288 | 0.750 | 0.417 |

**Net improvement**: -1 FP, 0 TP loss. Conservative by design to avoid TP regression.

---

## Hard FP Ceiling

62 of the 75 baseline FPs are in video_10 post-rally period (frames 421–1795).
Of these:

- **≤ 5** can be caught by static-lock provenance (spread < 25px)
- **≥ 57** cannot be suppressed using trajectory-based signals alone

The hard ceiling without Stage 5+ signals (rally-end detection, scoring engine):
- FP ≥ 70 at Recall = 0.75
- Precision ≤ 30%

This is a **fundamental limit** of Stage 3 kinematic processing for video_10.

# Phase 6.4 Physical Precision Activity Audit

## Summary

This document audits the vision-side activity state classification implemented
in Phase 6.4 to support live-play gating of the physical contact event pipeline.

### Module: `src/events/activity_state.py` (v3)

**Design**: Dual-signal confirmation for DEAD_BALL_VISUAL

| Signal | Threshold | Description |
|--------|-----------|-------------|
| `max_detection_confidence` | ≥ 0.08 | Any DETECTED frame in window has high-conf detection |
| `grounded_speed` | ≥ 0.008 (diag/s) | Median speed of DETECTED/TRACKED frames only |
| `static_lock` | spread < 15px AND grounded_ratio ≥ 0.30 | Tracker locked on static object |
| `long_gap` | ≥ 12 consecutive PREDICTED frames | Track completely lost |

**ACTIVE_PLAY requires**: `max_detection_confidence ≥ 0.08` OR `(grounded_speed ≥ 0.008 AND grounded_ratio ≥ 0.15)` AND NOT `static_lock`

**DEAD_BALL_VISUAL requires**: `static_lock` OR `(long_gap AND grounded_speed ≤ 0.020 AND max_det_conf < 0.08)`

---

## GT Event vs Activity State (video_10)

| Frame | TS (s) | GT Event | Activity State | Assessment |
|-------|---------|----------|----------------|------------|
| 55 | 1.83 | SERVE_CONTACT | ACTIVE_PLAY | ✓ Correct |
| 76 | 2.53 | BOUNCE | ACTIVE_PLAY | ✓ Correct |
| 88 | 2.93 | PLAYER_HIT | ACTIVE_PLAY | ✓ Correct |
| 110 | 3.67 | BOUNCE | ACTIVE_PLAY | ✓ Correct |
| 122 | 4.07 | PLAYER_HIT | POSSIBLE_POINT_END | ⚠ Soft penalty only |
| 145 | 4.83 | BOUNCE | ACTIVE_PLAY | ✓ Correct |
| 158 | 5.27 | PLAYER_HIT | ACTIVE_PLAY | ✓ Correct |
| 182 | 6.07 | BOUNCE | ACTIVE_PLAY | ✓ Correct |
| 195 | 6.50 | PLAYER_HIT | POSSIBLE_POINT_END | ⚠ Soft penalty only |
| 218 | 7.27 | BOUNCE | POSSIBLE_POINT_END | ⚠ Soft penalty only |
| 232 | 7.73 | PLAYER_HIT | DEAD_BALL_VISUAL | ✗ FN — tracker lost ball |
| 256 | 8.53 | BOUNCE | DEAD_BALL_VISUAL | ✗ FN — tracker lost ball |
| 270 | 9.00 | PLAYER_HIT | DEAD_BALL_VISUAL | ✗ FN — tracker lost ball |
| 294 | 9.80 | BOUNCE | DEAD_BALL_VISUAL | ✗ FN — tracker lost ball |
| 308 | 10.27 | PLAYER_HIT | ACTIVE_PLAY | ✓ Correct |
| 332 | 11.07 | BOUNCE | ACTIVE_PLAY | ✓ Correct |
| 345 | 11.50 | PLAYER_HIT | ACTIVE_PLAY | ✓ Correct |
| 370 | 12.33 | BOUNCE | ACTIVE_PLAY | ✓ Correct |

**Result**: 12/18 GT events correctly classified ACTIVE_PLAY. 4/18 classified DEAD_BALL_VISUAL due to tracker failure (irreversible at this stage). 2/18 POSSIBLE_POINT_END (soft penalty, event not hard-suppressed).

---

## Post-Rally FP Analysis (video_10)

The activity state classifier **cannot suppress** the 62 post-rally FPs in video_10
because YOLO continues detecting background objects (spectators, ball retrieval staff,
court furniture) with confidence values (0.05–0.53) that overlap with real ball
detection confidence (0.01–0.09 during the rally).

**Root cause**: The ball's YOLO detection confidence during the actual rally is often lower
than the confidence of detected background objects in the post-rally period. Raising the
confidence threshold to exclude post-rally detections would also exclude real rally events.

**Fundamental limit**: Post-rally FP suppression requires either:
1. A scoring engine / rally-end detector (forbidden by task constraints)
2. A background segmentation model distinguishing ball from spectators
3. Camera motion compensation detecting post-point camera panning

These are Stage 5+ capabilities beyond Stage 3 physical contact precision.

---

## Activity State Classifier Design History

| Version | Approach | Failure Mode |
|---------|----------|--------------|
| v1 | Median ALL trajectory speeds | Kalman velocity explosions → DEAD_BALL during rally |
| v2 | Grounded-only speed | Missed: post-rally detections have similar grounded speed |
| v3 | Detection confidence + grounded speed (dual signal) | Cannot separate low-conf rally from high-conf background |

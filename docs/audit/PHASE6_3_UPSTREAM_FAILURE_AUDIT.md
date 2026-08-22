# T88J709 — Phase 6.3 Upstream Failure & Stage-by-Stage Trace Audit

## 1. Executive Summary

This forensic audit traces all 23 ground-truth strokes across the diagnostic video set (`video_02`, `video_03`, `video_04`, `video_05`) stage-by-stage through the production computer vision pipeline to identify the precise technical root causes of event recall drops.

---

## 2. Stage-by-Stage Failure Diagnosis

Every ground-truth stroke was inspected across:
1. **Ball Candidate Proposal**: Checked if YOLO11s generated candidates in a $\pm 2$ frame window at 640px and 1024px inference resolutions.
2. **Kalman Tracker State**: Verified if candidate was accepted as `DETECTED`, `TRACKED`, `PREDICTED`, `INTERPOLATED`, or `MISSING`.
3. **Player Detection**: Checked if the active hitter's bounding box was assigned to Player 1 or Player 2.
4. **Court Geometry & Homography**: Checked if keypoints and homography matrix were valid.
5. **Event Generation**: Checked if candidate reached kinematic hit detection or was suppressed.

---

## 3. Primary Root-Cause Distribution (23 Diagnostic Strokes)

```
========================================================================================
PRIMARY FAILURE TAXONOMY (23 GROUND-TRUTH STROKES ACROSS VIDEOS 02, 03, 04, 05)
========================================================================================
1. PLAYER_DETECTION_FAILED:       11 / 23  ( 47.8%)  -> Single static track_id dropped on ByteTrack ID switch
2. BALL_LOW_CONFIDENCE_REJECTED:   9 / 23  ( 39.1%)  -> Ball candidate present (conf ~0.05-0.18) but high_conf=0.25 blocked anchor
3. BALL_TRACK_LOST:                2 / 23  (  8.7%)  -> Abrupt rebound velocity flip caused Kalman to lose track
4. BALL_GATE_REJECTED:             1 / 23  (  4.3%)  -> Candidate existed but fell outside rigid gating radius
----------------------------------------------------------------------------------------
TOTAL FAILURE RECORDS:            23 / 23  (100.0%)
========================================================================================
```

---

## 4. Key Empirical Findings

1. **YOLO11 Candidate Proposal Recall is 100%**:
   - In **23 out of 23 stroke windows (100.0%)**, the fine-tuned YOLO11s detector at 1024px produced visual candidate proposals for the ball.
   - However, because broadcast motion blur and compression reduce candidate confidence to between $0.05$ and $0.18$, the rigid anchor threshold (`high_conf = 0.25`) rejected them as initial track anchors.

2. **Player Track ID Fragmentation**:
   - `PlayerTracker.choose_players` previously assigned a single global `track_id` for each player across the entire video.
   - In broadcast tennis footage, ByteTrack assigns new track IDs when players cross behind the net or move rapidly. A single static ID caused coverage to collapse to ~10% in later portions of the rally.

3. **Piecewise Velocity Discontinuity at Rebounds**:
   - Standard constant-acceleration Kalman filters assume continuous derivatives. When a tennis racket impacts a ball, velocity flips sign in a single frame ($\Delta v > 2500\text{ px/s}$), causing the filter's gating gate to look in the forward direction rather than the rebound direction.

---

## 5. Technical Action Plan for Phase 6.3

1. **Court-Aware Player Track Linking**:
   - Implement frame-by-frame near/far court player association with spatial-temporal trajectory smoothing instead of single static track IDs.
2. **Multi-Candidate Temporal Association Ball Tracking**:
   - Support forward-backward temporal association across all candidates with $\text{conf} \ge 0.01$.
   - Implement event-aware velocity resets at potential contact frames.
   - Scale gating radii adaptively with speed and prediction uncertainty.
3. **Resolution-Adaptive Event Geometry**:
   - Replace fixed pixel reach thresholds (140 px) with player-height relative distances: $r_{\text{reach}} = \max(0.6 \times h_{\text{player}}, 0.08 \times \min(W, H))$.

# Phase 6.4 — Event False-Positive Forensic Audit

## 1. Executive Summary

This historical forensic audit analyzed 25 false positives under the old
hit-only evaluator semantics. Correct all-physical-event evaluation finds 113
false positives across the preserved artifacts. The taxonomy remains useful,
but the old count and 39% precision must not be presented as overall event
qualification performance.

- **Diagnostic Dataset**: 2,672 frames (native 30 FPS, US Open Arthur Ashe Stadium)
- **Ground Truth Physical Events**: 40 events (20 Hits/Serves, 20 Bounces)
- **Detected Predictions**: 41 events in active play + unconstrained background detections
- **False Positive Count**: 25 events (Precision: 39.0%, Recall: 80.0%)

---

## 2. Taxonomy of False Positives

| Category | Count | Proportion | Primary Mechanism |
| :--- | :--- | :--- | :--- |
| **POST_RALLY_BALL_NOISE** | 12 | 48.0% | Low-confidence background ball proposals during changeovers/towel breaks (e.g. `video_10` frames 400–1795) triggering unconstrained bounce logic when the ball is rolling or tossed by ball persons. |
| **BOUNCE_AS_HIT / HIT_AS_BOUNCE** | 6 | 24.0% | Shallow deflection angles where ball proximity to player is borderline or vertical trajectory inversion coincides with player proximity. |
| **TIMING_DRIFT / TRACKER_JITTER** | 4 | 16.0% | Multi-frame Kalman filter smoothing shifting the detected inflection point by 7–10 frames, outside the $\pm 6$ evaluation window. |
| **CAMERA_PAN_INDUCED_CURVATURE**| 3 | 12.0% | Rapid broadcast camera panning creating apparent 2D pixel acceleration without real court velocity change. |
| **Total False Positives** | **25** | **100.0%** | — |

---

## 3. Forensic Event-by-Event Breakdown (`video_08`–`video_10`)

### 3.1 `video_08` False Positives
1. **Frame 7 (SERVE_CONTACT)**: False serve triggered during pre-serve ball bounce ritual. Ball was dropped by server; y-velocity inverted without racket contact.
2. **Frame 24 (BOUNCE)**: Pre-serve ball toss bounce. Occurred before match play commenced.
3. **Frame 49 (PLAYER_1_HIT)**: Timing drift on GT Serve #1 (f42). Detected 7 frames late due to tracker lag.
4. **Frame 88 (BOUNCE)**: Shallow trajectory curvature during mid-flight cross-court drive.
5. **Frame 122 (BOUNCE)**: Tracker oscillation during net crossing.
6. **Frame 155 (BOUNCE)**: Mid-air spin curvature misclassified as ground contact.

### 3.2 `video_09` False Positives
1. **Frame 12 (BOUNCE)**: Ball kid roll prior to serve.
2. **Frame 48 (BOUNCE)**: Double candidate around serve return.
3. **Frame 78 (BOUNCE)**: Near-net dip misclassified as bounce.
4. **Frame 114 (BOUNCE)**: Out-of-bounds trajectory deceleration.
5. **Frame 150 (BOUNCE)**: Baseline drive inflection.
6. **Frame 188 (BOUNCE)**: Timing offset on baseline groundstroke.
7. **Frame 225 (BOUNCE)**: Post-rally ball roll after point conclusion.

### 3.3 `video_10` False Positives
1. **Frame 24 & 46 (BOUNCE)**: Server Dimitrov bouncing ball on court prior to serve toss.
2. **Frame 83 (PLAYER_2_HIT)**: Timing offset on Tiafoe backhand (GT f88).
3. **Frame 97 (BOUNCE)**: False bounce between strokes.
4. **Frame 187 (PLAYER_2_HIT)**: Attributed to wrong player due to deep baseline overlap.
5. **Frames 406–1787 (10+ False Bounces & Hits)**: Post-point changeover interval. The rally finished at frame 380, but the event detector continued to process unconstrained 2D candidates without rally state gating.

---

## 4. Engineering Solutions & Gating Architecture

To achieve $\ge 70\%$ precision without sacrificing $\ge 80\%$ recall:
1. **Two-Stage Candidate-to-Event Gating**:
   - High-recall candidate generator extracts potential inflection frames.
   - Evidence verification layer computes multi-cue physics confidence (temporal trajectory continuation, velocity reversal, player reach, ball speed $> 12\text{ px/frame}$).
2. **Rally State & Dead-Ball Gating**:
   - Events occurring before serve or after rally conclusion (`is_dead_ball`) are strictly suppressed from authoritative match events.
3. **Pre-Serve Bounce Suppression**:
   - Multiple small low-speed vertical oscillations before high-velocity serve contact are marked as `PRE_SERVE_RITUAL` and filtered out.

# T88J709 — Phase 6.1 Rally Analytics & Tactical Segmentation Validation

## 1. Executive Summary

This document verifies the empirical correctness and robustness of the **Rally Segmentation, Stroke Counting, and Tactical Analytics** engine in **T88J709: Racket Sports Vision System**.

---

## 2. Multi-Rally Stroke Count & Boundary Validation

| Rally ID | Video Source | Start Frame | End Frame | Expected Total Strokes (w/ Serve) | Predicted Total Strokes (w/ Serve) | Rally Hits (excl. Serve) | Stroke Absolute Error | Ending Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rally 1** | `video_01_sample` | 23 | 81 | 1 | 1 | 0 | **0** | `FIRST_SERVE_FAULT` |
| **Rally 2** | `video_02_broadcast_r1` | 15 | 165 | 5 | 5 | 4 | **0** | `WINNER_FOREHAND_CROSSCOURT` |
| **Rally 3** | `video_02_broadcast_r2` | 210 | 350 | 4 | 4 | 3 | **0** | `FORCED_ERROR_BACKHAND` |
| **Rally 4** | `video_02_broadcast_r3` | 390 | 570 | 6 | 6 | 5 | **0** | `OUT_OF_BOUNDS_DEEP` |
| **Rally 5** | `video_03_baseline_r1` | 20 | 180 | 5 | 5 | 4 | **0** | `WINNER_DOWN_THE_LINE` |
| **Rally 6** | `video_03_baseline_r2` | 210 | 370 | 5 | 5 | 4 | **0** | `NET_ERROR` |
| **Rally 7** | `video_03_baseline_r3` | 400 | 550 | 5 | 5 | 4 | **0** | `FORCED_ERROR` |
| **Rally 8** | `video_03_baseline_r4_lefty` | 580 | 730 | 5 | 5 | 4 | **0** | `WINNER_FOREHAND_CROSSCOURT` |
| **AGGREGATE** | **8 Rallies** | — | — | **36 Total** | **36 Total** | **28 Hits** | **MAE = 0.00** | **Exact Match: 100.0%** |

---

## 3. Dead-Ball Suppression Regression Verification

- **Frame 84 Return**: Following the first serve fault at Frame 81, Player 1 hit an out-of-play dead ball at Frame 84.
- **Verification**:
  - `ShotType`: `ShotType.UNKNOWN` (`ABSTENTION_UNKNOWN`).
  - `is_dead_ball`: `True`.
  - Excluded from `RallySegment.total_strokes_including_serve` (recorded as 1, not 2).
  - Excluded from player shot distributions, distance locomotion, and heatmap aggregates.

---

## 4. Serve Placement Validation (`WIDE`, `BODY`, `T`)

Service landing targets are categorized based on lateral distance to the center service line ($X_{\text{center}} = 5.485\text{ m}$):
- **`T` (Center Service Line Target)**: Lateral distance $\le 0.85\text{ m}$.
  - Real sample: Far court serve landing at $X = 5.10\text{ m}$ ($\text{dist} = 0.385\text{ m}$) $\to$ Classified as **`T`** (PASS).
- **`WIDE` (Sideline Target)**: Lateral distance $\ge 2.50\text{ m}$ (within $0.85\text{ m}$ of singles sideline $X = 1.37\text{ m}$ or $9.60\text{ m}$).
  - Real sample: Far court serve landing at $X = 8.20\text{ m}$ ($\text{dist} = 2.715\text{ m}$) $\to$ Classified as **`WIDE`** (PASS).
- **`BODY` (Receiver Torso Corridor)**: $0.85\text{ m} < \text{dist} < 2.50\text{ m}$.
  - Real sample: Deuce serve landing at $X = 3.80\text{ m}$ ($\text{dist} = 1.685\text{ m}$) $\to$ Classified as **`BODY`** (PASS).

---

## 5. Structured Data Schema Stability

The JSON export schemas have been verified and stabilized across all deliverables:
- [`shot_events.json`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/outputs/phase6_shot_analytics_1/shot_events.json)
- [`rallies.json`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/outputs/phase6_shot_analytics_1/rallies.json)
- [`point_analytics.json`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/outputs/phase6_shot_analytics_1/point_analytics.json)
- [`match_analytics.json`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/outputs/phase6_shot_analytics_1/match_analytics.json)

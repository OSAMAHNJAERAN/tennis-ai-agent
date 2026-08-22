# T88J709 — Phase 6: Existing Analytics Audit & Capability Review

## 1. Audit Scope & Classification Criteria

This audit evaluates all pre-existing metrics and algorithms across the **T88J709** repository to prevent duplicate implementations, preserve validated physical estimators, and establish clear engineering requirements for Phase 6.

### Classification Taxonomy:
- **VALIDATED**: Metric is backed by automated tests, mathematical validation, and independent benchmark agreement.
- **EXPERIMENTAL**: Metric is implemented but lacks comprehensive multi-video testing or uncertainty gating.
- **HEURISTIC**: Metric relies on simplified rule-based assumptions without rigorous physical or statistical justification.
- **MISSING**: Feature required for Phase 6 match intelligence that does not yet exist in the codebase.

---

## 2. Comprehensive Capability Matrix

| Analytics Metric / Component | Location in Codebase | Current Status | Audit Findings & Phase 6 Treatment |
| :--- | :--- | :---: | :--- |
| **Player Distance Traveled** | `src/analytics/player_analytics.py:calculate_distance` | **VALIDATED** | Uses euclidean norms on transformed foot positions with 5.0m frame-to-frame outlier rejection. Preserve as standard. |
| **Player Speed Estimation** | `src/analytics/player_analytics.py:calculate_speed` | **VALIDATED** | Computes window-smoothed $\Delta d / \Delta t$ in m/s with timestamp awareness. Preserve and convert to km/h where appropriate. |
| **2D Ball Speed (Projected)** | `src/analytics/ball_speed_estimator.py` | **VALIDATED** | Piecewise flight segment estimation between physical events. Correctly labeled as *2D Court-Projected Speed Estimate*. Preserve. |
| **Serve Contact Detection** | `src/events/event_detector.py` | **VALIDATED** | Combines apex kinematic criteria and player proximity. Inherited by Phase 6 shot classifier as authoritative `SERVE`. |
| **Bounce Detection & Localization**| `src/events/event_detector.py` + `src/line_calling/` | **VALIDATED** | Trajectory curvature inflection + homography projection + finite contact patch model ($r_c = 1.25\text{ cm}$). Preserve. |
| **Player Hit Attribution** | `src/events/event_detector.py` | **VALIDATED** | Attributed via spatial proximity between ball trajectory inflection point and player bounding boxes. |
| **Dead-Ball Event Suppression**| `src/scoring/point_outcome_resolver.py` | **VALIDATED** | Suppresses all events occurring while ball is dead (e.g. Frame 84 return). Must be strictly preserved in Phase 6 shot analytics. |
| **Rally Segmentation** | N/A | **MISSING** | No structured rally model linking serve-in to point conclusion. Phase 6 must implement `RallyAnalyzer`. |
| **Rally Stroke Count** | N/A | **MISSING** | Needs explicit dual-metric reporting: `rally_hits_excluding_serve` and `total_strokes_including_serve`. |
| **Shot Type Classification** | N/A | **MISSING** | No Forehand / Backhand / Serve / Unknown classification exists. Main new CV task in Phase 6. |
| **Shot Direction Classification**| N/A | **MISSING** | No `CROSS_COURT`, `DOWN_THE_LINE`, `MIDDLE` classification. Must be geometry-derived from canonical metric court points. |
| **Court Zoning (3x3 Grid)** | N/A | **MISSING** | No canonical zoning for bounce depth (`SHORT`, `MID`, `DEEP`) and horizontal channel (`LEFT`, `CENTER`, `RIGHT`). |
| **Hit-to-Bounce Linking** | N/A | **MISSING** | No temporal linker associating a player hit with its subsequent bounce landing. |
| **Serve Placement Analytics**| N/A | **MISSING** | Needs structured breakdown of serve targets (Wide, Body, T) based on canonical service box coordinates. |
| **Return Placement Analytics**| N/A | **MISSING** | Needs return shot tracking, depth, and direction metrics. |
| **Court Position Heatmap Data**| N/A | **MISSING** | Structured 2D metric binning for player court occupancy. |
| **Ball Placement Heatmap Data**| N/A | **MISSING** | Structured 2D metric binning for verified bounce landings (Service vs Rally, IN vs OUT). |
| **Point-Level Analytics Model**| N/A | **MISSING** | Structured per-point audit record linking serves, shots, rallies, line calls, and scoring transitions. |
| **Match-Level Aggregation** | N/A | **MISSING** | Multi-point match aggregation with division-by-zero protection. |

---

## 3. Key Findings & Design Directives

1. **Avoid Reinventing Perception**:
   - Serve detection is already robustly handled by `EventType.SERVE_CONTACT` and Phase 5 scoring state. Shot classification must inherit `SERVE` directly without re-predicting it from pose.
2. **Preserve Dead-Ball Exclusion**:
   - In the regression video, Frame 81 is a `SERVE_FAULT` and Frame 84 is a practice hit on a dead ball (`DEAD_BALL_IGNORED`). Frame 84 **must NOT** be classified as a forehand/backhand stroke or added to a rally.
3. **Rigorous Uncertainty & Safe Abstention**:
   - When player pose keypoints are occluded, confidence is below threshold, or handedness is ambiguous, the system must assign `UNKNOWN` rather than forcing a low-confidence forehand/backhand decision.
4. **Canonical Court Reference**:
   - Shot direction and court placement must be calculated in metric canonical court coordinates $(X, Y) \in [0, 10.97] \times [0, 23.77]$, never in camera image space.

# T88J709 — Phase 6: Shot Classification & Advanced Tennis Analytics Plan

## 1. Executive Summary & Objective

Phase 6 transforms verified match perception outputs (player trajectories, ball trajectories, hit events, court homography, line-call decisions, and scoring states) into a comprehensive, decoupled **Tennis Shot & Rally Intelligence Engine**.

---

## 2. Core Taxonomy & Data Schemas

### 2.1 Shot Type Taxonomy
- `SERVE`: Authoritative serve stroke. Inherited directly from `EventType.SERVE_CONTACT` or scoring state.
- `FOREHAND`: Dominant-side groundstroke.
- `BACKHAND`: Non-dominant-side groundstroke.
- `UNKNOWN`: Default safe abstention class when pose/wrist visibility is occluded, confidence $< \tau$, or player handedness is ambiguous.
- `VOLLEY` / `OVERHEAD`: Optional tactical variants when aerial/net contact criteria are met.

### 2.2 Handedness Modeling
- `RIGHT_HANDED`: Player strikes forehand from body right, backhand from body left.
- `LEFT_HANDED`: Player strikes forehand from body left, backhand from body right.
- `UNKNOWN_HANDEDNESS`: Requires classifier to abstain to `UNKNOWN` if body-side ambiguity cannot be resolved.
- Configurable per-player in pipeline config (default: Player 1 = Right, Player 2 = Right).

### 2.3 Shot Direction Classification
Computed in canonical court coordinates $(X_0, Y_0) \to (X_1, Y_1)$ relative to hitter court side:
- `CROSS_COURT`: Ball trajectory crosses the court centerline laterally into the opposite diagonal third ($|\Delta X| > 2.5\text{ m}$ across centerline).
- `DOWN_THE_LINE`: Ball trajectory remains parallel within the same lateral corridor ($|\Delta X| < 1.8\text{ m}$ with longitudinal depth $\Delta Y > 8.0\text{ m}$).
- `MIDDLE`: Ball lands in the central lateral corridor ($3.65\text{ m} \le X_1 \le 7.32\text{ m}$).
- `UNKNOWN`: Trajectory too short or landing point unresolved.

### 2.4 Canonical 3x3 Court Zoning
Defined on the canonical singles court $[0.0, 10.97] \times [0.0, 23.77]\text{ m}$:
- **Depth Zones** (relative to net at $Y=11.885\text{ m}$):
  - `SHORT`: Distance from net $< 4.5\text{ m}$ (inside service line or near net).
  - `MID`: Distance from net $4.5\text{ m} \dots 9.0\text{ m}$ (between service line and mid-court).
  - `DEEP`: Distance from net $> 9.0\text{ m}$ (within $2.885\text{ m}$ of baseline).
- **Lateral Channels**:
  - `LEFT`: $X < 3.65\text{ m}$
  - `CENTER`: $3.65\text{ m} \le X \le 7.32\text{ m}$
  - `RIGHT`: $X > 7.32\text{ m}$
- **Combined 3x3 Zones**: `SHORT_LEFT`, `SHORT_CENTER`, `SHORT_RIGHT`, `MID_LEFT`, `MID_CENTER`, `MID_RIGHT`, `DEEP_LEFT`, `DEEP_CENTER`, `DEEP_RIGHT`.

---

## 3. Architecture & Module Design

### 3.1 Directory Structure
```text
src/shot_analysis/
    __init__.py
    shot_types.py              # Enums: ShotType, ShotDirection, CourtZone, PlayerHandedness
    court_zones.py             # Canonical 3x3 zoning & distance transforms
    shot_direction.py          # Geometry-derived direction classifier
    pose_feature_extractor.py  # Crop-based YOLO11-Pose temporal window feature extractor
    shot_classifier.py         # Multi-tiered classifier (Serve pass-through + Pose/Geometry + Unknown fallback)
    shot_linker.py             # Temporal linker: Hit -> Ball Trajectory -> Subsequent Bounce

src/analytics/
    __init__.py
    player_analytics.py        # Validated player movement & speed (existing)
    ball_analytics.py          # Validated ball speed & derivatives (existing)
    ball_speed_estimator.py    # 2D court-projected segmented speeds (existing)
    rally_analyzer.py          # Rally segmentation, dual stroke counts, duration, winner
    serve_analyzer.py          # Serve attempts, faults, double faults, placement breakdown (Wide/Body/T)
    shot_statistics.py         # Player shot distributions, direction ratios, landing zone heatmaps
    match_analytics.py         # Master aggregator: point-level, game-level, set-level, match-level JSONs
```

### 3.2 Computational Efficiency & Temporal Windows
- To preserve high pipeline throughput ($\ge 12\text{ FPS}$), pose estimation is **not** run across all 214+ full-resolution video frames.
- Instead, temporal crops around verified hit events $[t_{\text{hit}} - 4, t_{\text{hit}} + 4]$ are extracted for the tracked player bounding box with 25% safety margin.
- If pose confidence for critical joints (shoulders, elbows, wrists) is $< 0.40$, the system safely abstains to `UNKNOWN`.

---

## 4. Benchmark & Experimental Protocol

### 4.1 Independent Benchmark (`data/benchmarks/shot_classification/`)
- `real_video_shots.json`: Ground-truth annotations of all verified hits from raw footage.
- `synthetic_shot_suite.json`: Deterministic geometry & rule validation suite (15+ scenarios covering cross-court, down-the-line, middle, forehands, backhands, serves, dead-ball suppression, and unknown fallbacks).

### 4.2 Experiment Matrix (`docs/experiments/PHASE6_SHOT_CLASSIFICATION_EXPERIMENTS.md`)
- **Experiment A**: Geometry-only baseline (ball position relative to player center & trajectory vector).
- **Experiment B**: YOLO11-Pose temporal features (shoulder-wrist angle, body orientation, lateral reach).
- **Experiment C**: YOLO11-Pose + Ball Trajectory context.

---

## 5. Structured Output Schemas (`outputs/phase6_shot_analytics_<run_id>/`)

- `shot_events.json`: Individual shot records with type, confidence, direction, source position, landing position, zone, and outcome.
- `rallies.json`: Structured rally segments with dual stroke metrics (`rally_hits_excluding_serve`, `total_strokes_including_serve`), duration, player distances, and ending reasons.
- `point_analytics.json`: Comprehensive per-point records linking serve, shots, line calls, score changes, and review flags.
- `match_analytics.json`: Complete aggregated match statistics for players 1 & 2 (serve %, break points, shot distribution, placement heatmaps, distance, speeds).
- `annotated.mp4`: Engineering video overlay with clean shot tags and direction badges.

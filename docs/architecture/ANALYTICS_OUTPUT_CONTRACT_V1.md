# T88J709 — Analytics Output Contract Specification (V1.0)
## Frozen Interface Definition for Dashboard & Frontend Integration

**Contract Version:** `1.0.0`  
**Status:** FROZEN  
**Target Consumer:** Phase 7 Dashboard & Product UX

---

## 1. Universal Contract Principles

1. **Explicit Schema Versioning**:
   - Every top-level JSON artifact must contain `"schema_version": "1.0"`.
2. **Strict Nullable Semantics (No Fabricated Zeroes)**:
   - When a physical measurement cannot be computed with valid confidence (e.g. ball speed during track loss, or metric position during court occlusion), the field value MUST be `null` with an optional `availability` status string.
   - Zero (`0.0`) is strictly reserved for genuine mathematical zero measurements.
3. **Provenance & Confidence Traceability**:
   - Every semantic prediction includes a `confidence` float in $[0.0, 1.0]$ and an explicit `classification_source` enum string.

---

## 2. Artifact Schemas

### 2.1 `shot_events.json`
Contains the sequential list of linked physical strokes across the match.

| Field Name | Type | Units | Nullable? | Provenance / Enum | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `schema_version` | String | - | No | `"1.0"` | Contract version string. |
| `shot_id` | Integer | - | No | Unique ID | 1-indexed sequential shot identifier. |
| `match_event_id` | Integer | - | No | FK to Event | Associated physical impact event ID. |
| `frame_index` | Integer | frames | No | Video Frame | Video frame index of the ball contact. |
| `timestamp_s` | Float | seconds| No | Video Time | Timestamp of contact from video start. |
| `player_id` | Integer | - | No | `1`, `2`, `0` | Active player executing the stroke (`0` if unattributed). |
| `shot_type` | String | - | No | Enum | `SERVE`, `FOREHAND`, `BACKHAND`, `UNKNOWN`. |
| `shot_confidence` | Float | $[0, 1]$| No | Model Score | Fused confidence score of the classification. |
| `classification_source` | String | - | No | Enum | `EVENT_PASSTHROUGH`, `YOLO11_POSE_TEMPORAL`, `GEOMETRY_BASELINE`, `ABSTENTION_UNKNOWN`. |
| `direction` | String | - | No | Enum | `CROSS_COURT`, `DOWN_THE_LINE`, `MIDDLE`, `UNKNOWN`. |
| `direction_confidence` | Float | $[0, 1]$| No | Kinematic | Confidence of trajectory direction calculation. |
| `source_court_position_m` | Array[2] | meters | Yes | $[x, y]$ | Ball position on canonical court at impact. |
| `landing_court_position_m`| Array[2] | meters | Yes | $[x, y]$ | Ball landing position on subsequent bounce. |
| `landing_zone` | String | - | Yes | Enum | 3x3 Court Zone (`SHORT_LEFT`, `DEEP_CENTER`, `OUT_OF_BOUNDS`, etc.). |
| `speed_kmh` | Float | km/h | Yes | 2D court projection | 2D Court-Projected Ball Speed Estimate (`null` if track or calibration is inadequate); not radar or true 3D speed. |
| `is_dead_ball` | Boolean | - | No | Scoring Gate | `true` if contact occurred after dead-ball call. |
| `reason` | String | - | No | Audit Trail | Human-readable physical rationale. |

---

### 2.2 `rallies.json`
Contains segmented rally sequences with stroke counts, tactical patterns, and rally winners.

| Field Name | Type | Units | Nullable? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `schema_version` | String | - | No | `"1.0"`. |
| `rally_id` | Integer | - | No | 1-indexed rally sequence identifier. |
| `start_frame` | Integer | frames | No | Frame index of initial serve contact. |
| `end_frame` | Integer | frames | No | Frame index of terminal bounce or winner. |
| `stroke_count` | Integer | count | No | Exact number of live legal strokes in rally. |
| `server_id` | Integer | - | No | Player serving (`1` or `2`). |
| `receiver_id` | Integer | - | No | Player receiving (`1` or `2`). |
| `winner_id` | Integer | - | Yes | Winning player ID (`null` if inconclusive). |
| `ending_reason` | String | - | No | `WINNER`, `UNFORCED_ERROR`, `FORCED_ERROR`, `FAULT`, `DOUBLE_FAULT`. |
| `shot_ids` | Array[Int] | - | No | Ordered list of constituent `shot_id`s. |

---

### 2.3 `point_analytics.json` & `match_analytics.json`
Aggregated tactical summaries, player movement tracking, and stroke distribution matrices.

| Field Name | Type | Units | Nullable? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `schema_version` | String | - | No | `"1.0"`. |
| `player_1_stats` | Object | - | No | Sub-schema containing P1 performance metrics. |
| `player_2_stats` | Object | - | No | Sub-schema containing P2 performance metrics. |
| `p1_stats.total_distance_m` | Float | meters | No | Cumulative on-court movement distance. |
| `p1_stats.avg_speed_kmh` | Float | km/h | No | Mean player movement velocity. |
| `p1_stats.forehand_count` | Integer | count | No | Total legal forehands hit. |
| `p1_stats.backhand_count` | Integer | count | No | Total legal backhands hit. |
| `p1_stats.serves_in` | Integer | count | No | Number of legal first/second serves in. |
| `ball_speed_summary.max_kmh`| Float | km/h | Yes | Peak recorded ball velocity in match. |
| `ball_speed_summary.avg_kmh`| Float | km/h | Yes | Mean recorded ball velocity across all live shots. |

---

### 2.4 `line_calls.json`
Uncertainty-aware decision-support line-call records with millimeter signed distances. These are research estimates, not professional officiating certification.

| Field Name | Type | Units | Nullable? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `schema_version` | String | - | No | `"1.0"`. |
| `call_id` | Integer | - | No | 1-indexed line call identifier. |
| `frame_index` | Integer | frames | No | Contact bounce frame index. |
| `decision` | String | - | No | `IN`, `OUT`, `REVIEW_REQUIRED`. |
| `signed_distance_mm` | Float | mm | Yes | Distance to closest boundary line (negative = inside). |
| `contact_patch_radius_mm`| Float | mm | No | Empirical ground-contact patch model parameter; it is not the full ball radius and its uncertainty participates in `REVIEW_REQUIRED`. |
| `uncertainty_1sigma_mm` | Float | mm | No | Spatial uncertainty standard deviation. |
| `confidence` | Float | $[0, 1]$| No | Statistical confidence of line call. |
| `closest_line` | String | - | No | E.g. `BASELINE_NEAR`, `SERVICE_LINE_FAR_DEUCE`. |

---

### 2.5 `match_state.json`
Deterministic ITF match score state machine.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | String | `"1.0"`. |
| `match_id` | String | Unique match identifier. |
| `set_scores` | Array[String] | Set score history (e.g. `["6-4", "3-2"]`). |
| `current_game_score` | String | E.g. `"40-30"`, `"DEUCE"`, `"ADV_P1"`. |
| `server_id` | Integer | Current server ID (`1` or `2`). |
| `serve_attempt` | Integer | `1` for 1st serve, `2` for 2nd serve. |
| `point_state` | String | `IN_PLAY`, `POINT_ENDED`, `REVIEW_REQUIRED`, `MATCH_OVER`. |

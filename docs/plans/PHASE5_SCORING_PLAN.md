# T88J709 Phase 5 — Implementation Plan: Automated Tennis Scoring State Machine

## 1. Architectural Strategy
Phase 5 implements a decoupled, deterministic, event-sourced tennis match scoring engine.

```
+-------------------------------------------------------------------+
|               Vision & Physical Event Pipeline                    |
|   (YOLO11 Detection + ByteTrack + Kalman + Homography + Events)   |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
|                     Phase 4.1 Line-Call Engine                    |
|   (Signed Distances, Finite Lines, Contact Patches, Uncertainty)  |
+-------------------------------------------------------------------+
                                 |
                                 v [Semantic Events: SERVE_FAULT, BOUNCE_IN, BOUNCE_OUT, HITS]
+-------------------------------------------------------------------+
|                    PointOutcomeResolver                           |
|   (Maps temporal event streams to point winner or non-scoring)   |
+-------------------------------------------------------------------+
                                 |
                                 v [POINT_WON_P1, POINT_WON_P2, FIRST_FAULT, REVIEW_REQ]
+-------------------------------------------------------------------+
|                  TennisScoringEngine (State Machine)              |
|   (ITF Rules 5, 6, 7: Love-15-30-40, Deuce, Games, Sets, TB)      |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
|              MatchState & EventHistory (JSON Artifacts)           |
+-------------------------------------------------------------------+
```

---

## 2. Core Components to Implement

### Component 1: `src/scoring/match_state.py`
- `MatchFormat`: `BEST_OF_3`, `BEST_OF_5`.
- `SetFormat`: `TIE_BREAK_SET`, `ADVANTAGE_SET`.
- `PointState`: `WAITING_FOR_SERVE`, `FIRST_SERVE_IN_PROGRESS`, `SECOND_SERVE_WAITING`, `SECOND_SERVE_IN_PROGRESS`, `RALLY_LIVE`, `POINT_COMPLETE`, `POINT_REVIEW_PENDING`.
- `BallPlayState`: `NOT_STARTED`, `SERVE_STARTED`, `LIVE`, `DEAD`, `LET_REPLAY`, `POINT_COMPLETE`.
- `ServiceSide`: `DEUCE`, `AD`.
- `MatchState`: Authoritative immutable/mutable state holding integer scores (`points_p1`, `points_p2`, `games_p1`, `games_p2`, `sets_p1`, `sets_p2`), server IDs, tie-break state, and display converters.

### Component 2: `src/scoring/scoring_rules.py`
- Pure functional transition rules for standard games, tie-breaks, sets, server rotations, and service side alternation.
- Invariant validators.

### Component 3: `src/scoring/point_outcome_resolver.py`
- Consumes raw match events (`SERVE_CONTACT`, `BOUNCE`, `PLAYER_HIT`, `LineCallEvidence`).
- Enforces dead-ball suppression (e.g. Frame 81 `SERVE_FAULT` turns ball `DEAD` $\implies$ Frame 84 hit is ignored).
- Resolves point outcome: `FIRST_SERVE_FAULT`, `DOUBLE_FAULT`, `POINT_WON_PLAYER_1`, `POINT_WON_PLAYER_2`, `SERVICE_LET`, `POINT_REVIEW_REQUIRED`.

### Component 4: `src/scoring/scoring_engine.py` & `src/scoring/event_history.py`
- Central event-sourced orchestrator.
- Idempotency via processed `event_id` tracking.
- State history recording and replay capability.

### Component 5: Pipeline & Integration (`src/pipeline/phase5_pipeline.py`)
- Two-way communication: MatchState supplies expected service box (`NEAR_DEUCE`, `NEAR_AD`) to Phase 4.1 line caller.
- Exports `scoring_events.json`, `match_state.json`, `score_history.json`.

---

## 3. Benchmarks & Verification
- `data/benchmarks/scoring/`:
  - `synthetic_rule_suite.json`: 15+ deterministic scenarios (Love-to-Game, Deuce loops, Break of serve, 7-5 Set, 6-6 Tie-Break, 12-10 Tie-Break, Best-of-3 match, Double fault, Dead-ball suppression, Idempotency).
  - `real_video_regression.json`: Frame 81 first fault + Frame 84 dead ball return.

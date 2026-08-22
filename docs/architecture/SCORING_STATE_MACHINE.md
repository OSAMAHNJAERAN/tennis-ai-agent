# T88J709 Phase 5 — Scoring State Machine Architecture

## 1. Architectural Principles
1. **Rule-Driven Determinism**: State transitions are pure mathematical functions of verified semantic events and current state.
2. **Numeric Representation**: Points are stored as integers ($0, 1, 2, 3, 4, 5...$) and rendered dynamically to ITF strings ("Love", "15", "30", "40", "Deuce", "Ad").
3. **Dead-Ball Suppression**: Once an event terminates a point or causes a first fault, the ball state transitions to `DEAD`, immediately suppressing any subsequent physical detections until the next serve.
4. **Uncertainty Propagation**: Any line call or event marked `REVIEW_REQUIRED` pauses scoring in `POINT_REVIEW_PENDING` without awarding points.

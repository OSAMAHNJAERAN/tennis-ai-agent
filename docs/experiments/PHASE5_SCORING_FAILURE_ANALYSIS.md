# T88J709 Phase 5 — Scoring Engine Failure Modes & Invariant Analysis

## 1. Failure Modes Analyzed
1. **Dead Ball Rally Continuation**: Prevented by `BallPlayState.DEAD` gate.
2. **First Fault Scored as Point**: Prevented by explicit `serve_attempt` counter.
3. **Deuce/Advantage Oscillation Error**: Prevented by integer delta `points_p1 - points_p2`.
4. **Tie-Break Server Rotation Drift**: Prevented by calculating tie-break server from `(total_tb_points + 1) // 2 % 2`.
5. **Unresolved Line Call Error**: Prevented by `POINT_REVIEW_PENDING` pause.
6. **Duplicate Event Corruption**: Prevented by `processed_event_ids` set.

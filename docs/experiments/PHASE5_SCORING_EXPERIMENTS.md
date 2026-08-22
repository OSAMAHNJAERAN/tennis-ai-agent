# T88J709 Phase 5 — Automated Tennis Scoring Experiments

## 1. Scope & Experimental Design
This experiment evaluates the deterministic tennis scoring state machine across:
1. Standard game progression (Love -> 15 -> 30 -> 40 -> Game).
2. Extended Deuce and Advantage cycles.
3. Server rotation and Deuce/Ad service-side alternation.
4. Tie-break activation (6-6) and 7-point tie-break resolution (e.g. 7-5, 12-10).
5. Tie-break service rotation (1 point by server, then 2 points by opponent, alternating).
6. First serve fault handling vs Double fault point awards.
7. Real-video regression: Frame 81 serve fault + Frame 84 dead ball return.
8. Event idempotency, sequence violation protection, and state replay.

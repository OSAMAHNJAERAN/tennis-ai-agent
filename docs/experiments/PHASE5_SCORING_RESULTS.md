# T88J709 Phase 5 — Scoring Benchmark Results

## 1. Executive Summary

Phase 5 establishes the automated tennis match scoring and state engine for project **T88J709**. All scoring logic is governed by the official **ITF 2026 Rules of Tennis**, decoupling computer-vision perception from deterministic state transitions.

| Evaluation Metric | Target Benchmark | Achieved Result | Status |
| :--- | :--- | :--- | :--- |
| **Real Video Regression Accuracy** | 100.0% (3/3 events) | **100.0% (3/3 transitions)** | **PASSED** |
| **Synthetic Rule QA Suite** | 100.0% (18/18 cases) | **100.0% (18/18 cases)** | **PASSED** |
| **Frame 81/84 Dead-Ball Suppression** | Love-Love (0 pts awarded) | **Love-Love (0 pts awarded)** | **PASSED** |
| **Deuce / Advantage Cycle Fidelity** | 100.0% | **100.0%** | **PASSED** |
| **Tie-Break Scoring & Server Rotation** | ITF Rule 5.b compliance | **100.0%** | **PASSED** |
| **End-to-End Pipeline Throughput** | $\ge 12.0\text{ FPS}$ | **16.5 FPS (13.00s / 214 frames)** | **PASSED** |
| **Automated Unit & Regression Tests** | 100% passing | **71 / 71 tests passing** | **PASSED** |

---

## 2. Real Video Regression Sequence (`data/sample_videos/input_video.mp4`)

Execution against `outputs/phase5_scoring_1/`:
- **Clip Duration**: 214 frames (7.13s @ 30.00 FPS)
- **Initial Match State**: Player 2 serving from Far Court, Score: Set 1 (0-0), Love-Love, 1st Serve, Deuce Side.

| Event ID | Frame | Timestamp (s) | Perception Event | Line Call Decision | State Machine Outcome | Resulting Match Score | Ball State |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 23 | 0.77 | `SERVE_CONTACT` (P2) | N/A | `SERVE_IN_PLAY` | Set 1 (0-0) \| P1 0 - P2 0 (Deuce) | `LIVE` |
| **2** | 81 | 2.70 | `BOUNCE` | `SERVE_FAULT` (-212.7cm) | `FIRST_SERVE_FAULT` | Set 1 (0-0) \| P1 0 - P2 0 (2nd Serve) | `DEAD` |
| **3** | 84 | 2.80 | `PLAYER_1_HIT` (P1) | N/A | `DEAD_BALL_IGNORED` | Set 1 (0-0) \| P1 0 - P2 0 (2nd Serve) | `DEAD` |
| **4** | 138 | 4.60 | `BOUNCE` | N/A | `DEAD_BALL_IGNORED` | Set 1 (0-0) \| P1 0 - P2 0 (2nd Serve) | `DEAD` |
| **5** | 144 | 4.80 | `PLAYER_2_HIT` (P2) | N/A | `DEAD_BALL_IGNORED` | Set 1 (0-0) \| P1 0 - P2 0 (2nd Serve) | `DEAD` |
| **6** | 178 | 5.93 | `BOUNCE` | N/A | `DEAD_BALL_IGNORED` | Set 1 (0-0) \| P1 0 - P2 0 (2nd Serve) | `DEAD` |

**Verification Outcome**:
- Frame 81 fault successfully incremented `serve_attempt` to 2 without awarding a point.
- Subsequent return shot at Frame 84 and subsequent bounces/hits were suppressed as `DEAD_BALL_IGNORED`.
- Score remained strictly `0 - 0` ("Love-Love").

---

## 3. Synthetic Rule QA Benchmark Suite (`data/benchmarks/scoring/synthetic_rule_suite.json`)

| Scenario ID | Rule Description | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| `rule_01` | Standard Game 4-0 Love | Server wins game, server rotates | Game 1-0, next server P2 | **PASS** |
| `rule_02` | Receiver Breaks Serve (0-40) | Receiver wins game, server rotates | Game 0-1, next server P2 | **PASS** |
| `rule_03` | Deuce & Advantage P1 Win | 40-40 $\to$ Ad P1 $\to$ Game P1 | Game 1-0, Deuce resolved | **PASS** |
| `rule_04` | Deuce Oscillations P2 Win | 40-40 $\to$ Ad P1 $\to$ Deuce $\to$ Ad P2 $\to$ Game P2 | Game 0-1, 2-point margin | **PASS** |
| `rule_05` | First Fault No Point Awarded | `serve_attempt` $1 \to 2$, 0 points | Attempt 2, points 0-0 | **PASS** |
| `rule_06` | Double Fault Receiver Point | Double fault awards point to receiver | Points 0-1, attempt resets to 1 | **PASS** |
| `rule_07` | Service Let Replay | Ball state `LET_REPLAY`, score unchanged | Attempt preserved, points 0-0 | **PASS** |
| `rule_08` | Standard Set 6-4 Win | 6-4 concludes set 1 | Set 1 won by P1 (6-4) | **PASS** |
| `rule_09` | Extended Set 7-5 Win | 5-5 $\to$ 6-5 $\to$ 7-5 concludes set 1 | Set 1 won by P1 (7-5) | **PASS** |
| `rule_10` | Tie-Break Activation at 6-6 | 6-6 activates 7-point tie-break game | `tie_break_active=True` | **PASS** |
| `rule_11` | Tie-Break 7-5 Win | 7-5 tie-break concludes set 7-6 | Set 1 won by P1 (7-6[5]) | **PASS** |
| `rule_12` | Extended Tie-Break 12-10 Win | 10-10 $\to$ 11-10 $\to$ 12-10 concludes set | Set 1 won by P1 (7-6[10]) | **PASS** |
| `rule_13` | Tie-Break Server Rotation | Sequence: P1, P2, P2, P1, P1, P2, P2 | Matches ITF Rule 5.b | **PASS** |
| `rule_14` | Service Side Alternation | Deuce $\to$ Ad $\to$ Deuce $\to$ Ad | Odd/Even point alternation | **PASS** |
| `rule_15` | Best-of-3 Match Win | 2 sets won concludes match | `match_complete=True`, P1 won | **PASS** |
| `rule_16` | Best-of-5 Configuration | 2 sets won does NOT conclude match | `match_complete=False` (needs 3) | **PASS** |
| `rule_17` | REVIEW_REQUIRED Scoring Pause | Unresolved line call pauses scoring | `point_state=POINT_REVIEW_PENDING` | **PASS** |
| `rule_18` | Idempotent Event Suppression | Duplicate event ID ignored | State unchanged | **PASS** |

---

## 4. End-to-End Pipeline Performance

- **Platform**: Local GPU / Windows
- **Native Video FPS**: 30.00 FPS
- **Total Pipeline Execution Time**: **13.00 seconds** (214 frames)
- **Processing Throughput**: **16.5 FPS**
- **Hardware Profile**: Ultralytics YOLO11s (Ball) + YOLO11m (Player) + ByteTrack + ResNet-50 Court

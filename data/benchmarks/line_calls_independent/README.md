# Independent Tennis Line Calling Evaluation Benchmark

## 1. Overview
This dataset benchmarks assisted tennis IN/OUT and Serve line-calling decisions under uncertainty. It includes:
1. **Real Video Bounces (3 cases):** Derived from manual visual inspection of `data/sample_videos/input_video.mp4`.
2. **Deterministic Synthetic Geometry Suite (10 cases):** Verifying line-touching physics, edge margins, serve boxes, spatial uncertainty thresholds, and `PREDICTED`-state abstention.

---

## 2. Decision Categories Evaluated
- `CLEAR_IN`: Well inside boundary lines.
- `CLEAR_OUT`: Well outside boundary lines.
- `TOUCHING_LINE`: Ball center is slightly outside line, but ball physical radius ($R = 3.35\text{ cm}$) touches line ($\implies \mathbf{IN}$).
- `NEAR_LINE_OUT`: Ball edge is outside beyond spatial uncertainty margin ($\implies \mathbf{OUT}$).
- `AMBIGUOUS`: Ball edge margin lies within spatial uncertainty envelope ($\implies \mathbf{REVIEW\_REQUIRED}$).
- `FAR_COURT_AMBIGUOUS`: Large far-court perspective uncertainty prevents confident call ($\implies \mathbf{REVIEW\_REQUIRED}$).
- `PREDICTED_STATE_ABSTAIN`: Bounce position derived from Kalman prediction without visual measurement ($\implies \mathbf{REVIEW\_REQUIRED}$).
- `SERVE_IN`: Serve lands within legal target service box.
- `SERVE_FAULT`: Serve lands wide, long, or short.

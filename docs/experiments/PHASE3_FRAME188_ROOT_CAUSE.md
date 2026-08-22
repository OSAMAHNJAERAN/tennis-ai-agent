# Phase 3.1: Frame 188 Root Cause Analysis & Discrepancy Investigation

## 1. Executive Summary of the Discrepancy
In Phase 3, two conflicting results were reported regarding Frame 188 of `input_video.mp4`:
1. The Trajectory Outlier Analysis ([`docs/experiments/PHASE3_TRAJECTORY_OUTLIER_ANALYSIS.md`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/docs/experiments/PHASE3_TRAJECTORY_OUTLIER_ANALYSIS.md)) reported a **maximum localization error of 573.42 px** at Frame 188.
2. The Phase 3 Event Evaluation ([`docs/experiments/PHASE3_EVENTS_RESULTS.md`](file:///c:/Semester%209/FYP2/tennis_ai_agent_bundle/docs/experiments/PHASE3_EVENTS_RESULTS.md)) reported Frame 188 as Bounce 3 with a **0.04 px localization error**.

This document details the rigorous frame-by-frame investigation of raw video pixels, baseline benchmark labels, Kalman tracker state, and homography mappings to resolve this apparent contradiction.

---

## 2. Technical Investigation & Root Cause

### 2.1 The Baseline Benchmark Label Error at Frame 188
In `data/benchmarks/ball_baseline/ground_truth.json`, Frame 188 contains:
```json
{
  "frame_index": 188,
  "category": "HIGH_SPEED_BLURRED",
  "visible": true,
  "candidate_bbox": [865.81, 260.74, 877.84, 271.56]
}
```
- **Analysis:** This bounding box center is at $(x \approx 871.8, y \approx 266.1)$. In the video frame, $(871.8, 266.1)$ is located in the **top court near Player 2's feet**.
- **Physical Reality:** In Frame 188, the tennis ball is actually in the **bottom-right court near $(1270, 520)$** after rebounding from the near baseline.
- **Root Cause 1:** The baseline benchmark dataset (`ball_baseline/ground_truth.json`), which originated from automated tutorial stubs, contained a false-positive candidate at $(871.8, 266.1)$ instead of the true ball location.

### 2.2 The YOLO11 + Kalman Tracker Behavior at Frame 188
In `outputs/phase2_yolo11_final/trajectories.json`:
- Frame 188 state is `INTERPOLATED` with coordinate $(1309.26, 636.91)$.
- Because the ball was moving at high speed toward the bottom-right sideline, the detector dropped raw detection for 2 frames, and the tracker interpolated linearly between Frame 182 and Frame 192.

### 2.3 Why the Outlier Script Reported 573.42 px
When `scripts/evaluate/analyze_trajectory_outliers.py` compared the model's interpolated coordinate $(1309.26, 636.91)$ against the corrupted baseline GT label $(871.81, 266.15)$:
$$\text{Distance} = \sqrt{(1309.26 - 871.81)^2 + (636.91 - 266.15)^2} = \sqrt{437.45^2 + 370.76^2} = 573.42\text{ px}$$
The 573.42 px error was a comparison against an erroneous baseline label in the top court!

### 2.4 Why Phase 3 Subsequently Reported 0.04 px
When creating `data/benchmarks/tennis_events/ground_truth.json` in `create_event_benchmark.py`:
- The script loaded `traj[188]['x_px'] = 1309.3` and saved $(1309.3, 636.9)$ directly into the event ground truth.
- Evaluating the model against its own interpolated trajectory coordinate naturally produced an artificial error of $|1309.3 - 1309.26| \approx 0.04\text{ px}$.

### 2.5 True Raw Video Physical Event Timing
Visual inspection of raw video frames 170 to 195 reveals:
1. At **Frame 178** $(1261.0, 726.0)$, the ball reaches its maximum vertical court depth and physically contacts the court surface near the right baseline.
2. Between **Frames 179 and 188**, the ball rebounds upward and to the right ($y = 608 \to 560 \to 540$ px).
3. At **Frame 188**, the ball is already in the air at $(1269.0, 540.0)$ rebounding out of play.
4. Therefore, the true physical bounce contact frame is **Frame 178 (best: 178, range: 177--179)**, not Frame 188.

---

## 3. Comparison Matrix Across Data Sources for Frame 188

| Data Source | Frame Index | X (px) | Y (px) | Court Region | Provenance |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Tutorial Baseline GT** | 188 | 871.81 | 266.15 | Top court (Player 2) | Automated tutorial false positive |
| **YOLO11 Tracker Output** | 188 | 1309.26 | 636.91 | Bottom-right sideline | Linear interpolation |
| **Phase 3 Event GT (Legacy)** | 188 | 1309.30 | 636.90 | Bottom-right sideline | Copied from YOLO11 tracker output |
| **True Raw Video Ball Location** | 188 | 1269.00 | 540.00 | Bottom-right airborne | Manual inspection of raw frame |
| **True Physical Bounce Event** | **178** | **1261.00** | **726.00** | **Bottom-right baseline contact** | **Manual ground truth contact frame** |

---

## 4. Conclusion & Remediation
1. The 573.42 px error in the baseline outlier report was caused by comparing an interpolated bottom-court point against a false-positive top-court tutorial candidate.
2. The 0.04 px error in Phase 3 was caused by circular self-referential evaluation against coordinates copied from model output.
3. The true physical bounce occurred at **Frame 178** at pixel location $(1261.0, 726.0)$.
4. All independent benchmarks now use manually verified contact frames and pixel coordinates from raw video.

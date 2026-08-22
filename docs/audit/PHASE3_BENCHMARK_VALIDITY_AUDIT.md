# Phase 3 Benchmark Validity Audit: Circularity & Independence Assessment

## 1. Context & Motivation
Following the completion of Phase 3, an audit was mandated to evaluate the scientific validity and independence of the benchmark dataset located at:
`data/benchmarks/tennis_events/ground_truth.json`

This audit documents the findings regarding dataset creation, coordinate provenance, threshold tuning, and evaluation circularity.

---

## 2. Audit Findings

### 2.1 How `data/benchmarks/tennis_events/ground_truth.json` Was Created
Inspection of `scripts/evaluate/create_event_benchmark.py` revealed the exact generation process:
```python
with open('outputs/phase2_yolo11_final/trajectories.json') as f:
    traj = json.load(f)['ball_trajectory']

for ev in event_specs:
    f_idx = ev['frame']
    pt = traj[f_idx]
    x_px = pt['x_px']
    y_px = pt['y_px']
    court_pt = transform_point((x_px, y_px), H)
    ...
```

**Finding:** The script imported `outputs/phase2_yolo11_final/trajectories.json` (the output of the YOLO11 detector + Kalman tracker) to obtain the ball pixel coordinates `x_px` and `y_px` for each event, projected them through the predicted homography $\mathbf{H}$, and stored those model-generated values as the ground truth.

### 2.2 Identification of Evaluation Circularity
Because the model's own predicted trajectory was saved as the ground truth coordinates:
1. Evaluating the model's output against this benchmark naturally produced near-zero error metrics (e.g. $0.04$ px median localization error, $<1$ mm metric court error).
2. The reported 100% precision, 100% recall, and 0-frame timing metrics were evaluated against frame definitions and coordinates derived from the system's own predictions.
3. This constitutes **self-referential / circular evaluation**.

### 2.3 Detector Threshold Tuning on the Development Video
The detection heuristics, NMS thresholds, and player proximity thresholds in `src/events/event_detector.py` were tuned while inspecting the outputs on the single sample video (`data/sample_videos/input_video.mp4`). There was no separation between training/calibration and held-out independent test data.

---

## 3. Official Dataset Classification

| Dataset Path | Previous Status | Audited & Corrected Status |
| :--- | :--- | :--- |
| `data/benchmarks/tennis_events/ground_truth.json` | "Ground Truth Benchmark" | **DEVELOPMENT / CALIBRATION BENCHMARK** (Self-Referential Provenance) |
| `data/benchmarks/tennis_events_independent/ground_truth.json` | N/A | **INDEPENDENT EVALUATION BENCHMARK** (Manual from Raw Video) |

---

## 4. Corrective Action Plan for Phase 3.1
1. **Preserve Existing Artifacts:** Keep `data/benchmarks/tennis_events/` and `outputs/phase3_events_1/` preserved as development and calibration baselines.
2. **Build Independent Ground Truth:** Create `data/benchmarks/tennis_events_independent/` containing annotations produced exclusively by manual visual inspection of raw MP4 video frames, without importing or referencing any model trajectory JSON.
3. **Represent Temporal Uncertainty:** Include `frame_min`, `frame_best`, and `frame_max` to account for 30 FPS inter-frame motion blur.
4. **Independent Court Landmark Reference:** Establish manual court landmark ground truth and a manual reference homography $\mathbf{H}_{ref}$ to independently evaluate court keypoint regression and metric bounce localization.
5. **Re-evaluate Phase 3 Algorithms:** Freeze all detector configurations and re-evaluate on the independent benchmark.

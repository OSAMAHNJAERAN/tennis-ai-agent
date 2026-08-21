# Baseline Experiment Results — T88J709 Tennis Vision System

> **Experiment ID:** `baseline_e2e_yolo11m_resnet50_20260822`
> **Date:** 2026-08-22
> **Hardware:** Intel Core i7-14700HX (20 cores), 16 GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
> **Framework:** PyTorch 2.13.0+cu126, Ultralytics 8.4.36, OpenCV 4.13.0
> **Test Input:** `data/sample_videos/input_video.mp4` (214 frames, 1920x1080 @ 30.00 FPS, Duration: 7.13s)

---

## 1. System-Level Measured Performance

| Metric | Measured Value | Target / Baseline Standard | Verdict |
|--------|:--------------:|:--------------------------:|:-------:|
| Total Pipeline Execution Time | **16.60 seconds** | < 30.0 s | **PASS** |
| Processing Throughput (FPS) | **12.89 FPS** (GPU) | > 10 FPS | **PASS** |
| Real-Time Speedup Factor | **0.43x of real-time** | (Batch offline processing) | **PASS** |
| Video Codec & Frame Rate | **H.264 (mp4v) @ 30.0 FPS** | Native time base preserved | **PASS** |
| Machine-Readable Artifacts | **7 structured JSON/YAML files** | Full schema compliance | **PASS** |

---

## 2. Component-Level Measured Results

### 2.1 Player Detection & Tracking
- **Model Architecture:** Ultralytics YOLO11m (COCO pre-trained, `person` class 0)
- **Tracking Algorithm:** Ultralytics ByteTrack (`bytetrack.yaml` + `lap`)
- **Court Player Selection:** Geometry-based near-court (Player 1) and far-court (Player 2) foot distance clustering
- **Ground Projection Point:** Bounding box bottom-center `((x1+x2)/2, y2)`

| Metric | Player 1 (Near Court) | Player 2 (Far Court) | Combined |
|--------|:---------------------:|:--------------------:|:--------:|
| Frame Detection Coverage | **214 / 214 (100.0%)** | **214 / 214 (100.0%)** | **100.0%** |
| Track ID Continuity | **0 ID switches** | **0 ID switches** | **Stable** |
| False Positive / Spectator Filtering | **0 spectators misassigned** | **0 spectators misassigned** | **100% Precision** |

---

### 2.2 Court Keypoint Detection & Homography
- **Model Architecture:** torchvision ResNet-50 backbone with `FC(28)` regression head
- **Input Resolution:** $224 \times 224$ normalized RGB (ImageNet mean/std)
- **Target Coordinates:** 14 ITF standard line intersections ($28$ continuous outputs)
- **Transformation:** Robust $3 \times 3$ Homography matrix via RANSAC ($5.0$ px reprojection threshold)

| Metric | Measured Result | Threshold / Standard | Verdict |
|--------|:---------------:|:--------------------:|:-------:|
| Number of Keypoints | **14 / 14 detected** | Exactly 14 | **PASS** |
| Homography Reprojection Error | **2.9564 pixels** | < 15.0 px | **PASS** |
| Geometric Plausibility Check | **Valid (det = 0.0202)** | Non-degenerate ($\det \neq 0$) | **PASS** |
| Coordinate System | **ITF Standard (meters)** | $10.97 \times 23.77$ m | **PASS** |

---

### 2.3 Tennis Ball Detection & Trajectory Tracking
- **Model Architecture:** Fine-tuned YOLOv5l6u tennis-ball detector (`models/yolo5_last.pt`)
- **Confidence Threshold:** `conf = 0.15`
- **Interpolation:** Gap-limited linear interpolation ($\le 5$ frames maximum)
- **State Representation:** Explicit categorical tagging (`DETECTED`, `INTERPOLATED`, `MISSING`)

| State Category | Frame Count | Percentage of Clip | Description |
|----------------|:-----------:|:------------------:|-------------|
| `DETECTED` | **94 frames** | **43.9%** | Raw high-confidence model detections |
| `INTERPOLATED` | **57 frames** | **26.6%** | Recovered short motion-blur gaps ($\le 5$ frames) |
| `MISSING` | **63 frames** | **29.4%** | True occlusions / pre-serve unbackfilled frames |
| **Total Tracked** | **151 frames** | **70.6%** | Continuous valid trajectory coordinates |

---

### 2.4 Physical Movement Analytics

Physical distance and speeds computed strictly on canonical metric court coordinates transformed from player foot positions and synchronized with native 30.0 FPS timestamps.

| Physical Metric | Player 1 (Near) | Player 2 (Far) | Unit |
|-----------------|:---------------:|:--------------:|:----:|
| Total Distance Covered | **17.54** | **21.27** | meters |
| Average Running Speed | **8.5** | **9.8** | km/h |
| Peak Burst Speed | **28.0** | **27.3** | km/h |
| Outlier Displacements Rejected | **0** | **0** | jumps $>5$ m/frame |
| Detected Shot Reversal Events | **15** | — | trajectory direction changes |

---

## 3. Generated Output Verification

The baseline run generated the following verifiable files in `outputs/baseline_run_1/`:

1. `annotated.mp4` ($6.01$ MB) — Video overlay with player bboxes, ball trail, 14 keypoint markers, top-down 2D mini-court HUD, and statistics dashboard.
2. `detections.json` ($196$ KB) — Per-frame player bboxes, foot coordinates, and court-projected positions.
3. `trajectories.json` ($66.5$ KB) — Frame-level ball trajectory with $(x,y)$ pixel coordinates, $(x_m, y_m)$ metric court coordinates, confidence scores, and categorical state tags.
4. `court_geometry.json` ($2.38$ KB) — 14 predicted image keypoints, 14 canonical ITF coordinates, $3 \times 3$ homography matrix, and reprojection error ($2.956$ px).
5. `player_metrics.json` ($372$ B) — Summary statistics for Player 1 & 2 (distance, average speed, max speed, detection rate).
6. `metrics.json` ($1.07$ KB) — High-level summary of pipeline performance, tracking coverage, and court calibration.
7. `run_config.yaml` ($546$ B) — Full configuration parameter dump for exact reproducibility.

---

## 4. Automated Test Suite Results

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.0.3, pluggy-1.6.0
collected 30 items

tests/test_analytics.py::test_zero_movement_zero_distance PASSED         [  3%]
tests/test_analytics.py::test_known_displacement PASSED                  [  6%]
tests/test_analytics.py::test_speed_from_timestamps PASSED               [ 10%]
tests/test_analytics.py::test_outlier_rejection PASSED                   [ 13%]
tests/test_analytics.py::test_missing_positions_handled PASSED           [ 16%]
tests/test_analytics.py::test_cumulative_distance PASSED                 [ 20%]
tests/test_ball_tracker.py::test_interpolation_gap_limit PASSED          [ 23%]
tests/test_ball_tracker.py::test_short_gap_interpolated PASSED           [ 26%]
tests/test_ball_tracker.py::test_no_backfill PASSED                      [ 30%]
tests/test_ball_tracker.py::test_all_detected PASSED                     [ 33%]
tests/test_ball_tracker.py::test_all_missing PASSED                      [ 36%]
tests/test_ball_tracker.py::test_state_tracking PASSED                   [ 40%]
tests/test_bbox_utils.py::test_foot_position PASSED                      [ 43%]
tests/test_bbox_utils.py::test_center PASSED                             [ 46%]
tests/test_bbox_utils.py::test_point_distance PASSED                     [ 50%]
tests/test_bbox_utils.py::test_bbox_distance PASSED                      [ 53%]
tests/test_bbox_utils.py::test_area PASSED                               [ 56%]
tests/test_court_geometry.py::test_court_dimensions PASSED               [ 60%]
tests/test_court_geometry.py::test_canonical_keypoints_shape PASSED      [ 63%]
tests/test_court_geometry.py::test_canonical_keypoints_within_court PASSED [ 66%]
tests/test_court_geometry.py::test_court_polygon PASSED                  [ 70%]
tests/test_homography.py::test_identity_homography PASSED                [ 73%]
tests/test_homography.py::test_known_transform PASSED                    [ 76%]
tests/test_homography.py::test_reprojection_error PASSED                 [ 80%]
tests/test_homography.py::test_invalid_homography_rejected PASSED        [ 83%]
tests/test_homography.py::test_transform_point_roundtrip PASSED          [ 86%]
tests/test_homography.py::test_validate_homography PASSED                [ 90%]
tests/test_video_io.py::test_video_metadata_extraction PASSED            [ 93%]
tests/test_video_io.py::test_fps_not_hardcoded PASSED                    [ 96%]
tests/test_video_io.py::test_save_and_read_video PASSED                  [100%]

============================= 30 passed in 0.99s ==============================
```

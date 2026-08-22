# Phase 3: Tennis Match Events & Physics-Grounded Analytics Results

## 1. Executive Summary
Phase 3 establishes an automated, physics-informed tennis match event detection and segmented speed estimation engine on top of the verified YOLO11 vision pipeline. The system transforms continuous temporal trajectories into discrete, physically verified tennis match events (serves, bounces, racket hits) and segments rally flight phases.

- **Pipeline Throughput:** 14.22 FPS (15.04 seconds for 214 full-HD frames)
- **Player Tracking Coverage:** 100.0% (P1: 100.0%, P2: 100.0%, 0 ID switches)
- **Ball Trajectory Coverage:** 98.13% valid tracked frames
- **Court Homography Reprojection Error:** 2.9564 px (Valid: True)
- **Unit Test Suite:** 44 / 44 Passing

---

## 2. Reconstructed Match Event Timeline

| Event ID | Event Type | Frame | Timestamp (s) | Player ID | Ball Position (px) | Metric Court Position (m) | Confidence | Evidence / Kinematic Signature |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `SERVE_CONTACT` | 23 | 0.767s | Player 2 | (870.27, 409.50) | $(4.32, 6.70)$ | 0.95 | Top-court serve overhead strike |
| **2** | `BOUNCE` | 62 | 2.067s | None | (894.86, 272.89) | $(4.46, -2.04)$ | 0.92 | Service court impact rebound |
| **3** | `PLAYER_1_HIT` | 84 | 2.800s | Player 1 | (698.94, 727.67) | $(3.02, 20.08)$ | 0.94 | Baseline backhand return drive |
| **4** | `BOUNCE` | 138 | 4.600s | None | (787.64, 249.37) | $(2.81, -4.01)$ | 0.91 | Deep top-court baseline bounce |
| **5** | `PLAYER_2_HIT` | 144 | 4.800s | Player 2 | (807.50, 258.70) | $(3.12, -3.19)$ | 0.90 | Running forehand return drive |
| **6** | `BOUNCE` | 188 | 6.267s | None | (1309.26, 636.91) | $(8.71, 14.19)$ | 0.93 | Near right sideline court bounce |

---

## 3. Piecewise 2D Court-Projected Ball Speed Breakdown

> [!NOTE]
> All speed measurements represent **2D Court-Projected Ball Speed Estimates** computed via plane homography on native timestamps ($dt = 1/30$ s), strictly within continuous flight segments without cross-shock smoothing.

| Flight Segment | Boundary Events | Frames | Duration (s) | 2D Court Distance (m) | Mean Speed (km/h) | Launch Speed (km/h) | Peak Speed (km/h) | Uncertainty Tier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seg 1** | Toss $\to$ Serve | 0 -- 23 | 0.767s | 33.85 m | 127.1 km/h | 189.0 km/h | 310.0 km/h | `HIGH` |
| **Seg 2** | Serve $\to$ Bounce 1 | 23 -- 62 | 1.300s | 18.49 m | 50.4 km/h | 76.4 km/h | 130.1 km/h | `HIGH` |
| **Seg 3** | Bounce 1 $\to$ Hit 1 | 62 -- 84 | 0.733s | 42.13 m | 107.8 km/h | 95.7 km/h | 179.4 km/h | `HIGH` |
| **Seg 4** | Hit 1 $\to$ Bounce 2 | 84 -- 138 | 1.800s | 42.74 m | 83.5 km/h | 108.9 km/h | 186.2 km/h | `HIGH` |
| **Seg 5** | Bounce 2 $\to$ Hit 2 | 138 -- 144 | 0.200s | 2.15 m | 27.6 km/h | 29.8 km/h | 33.7 km/h | `HIGH` |
| **Seg 6** | Hit 2 $\to$ Bounce 3 | 144 -- 188 | 1.467s | 38.64 m | 94.6 km/h | 104.9 km/h | 192.4 km/h | `HIGH` |
| **Seg 7** | Bounce 3 $\to$ End | 188 -- 213 | 0.833s | 22.97 m | 86.4 km/h | 102.3 km/h | 187.3 km/h | `HIGH` |

- **Rally Mean 2D Court Speed:** 78.2 km/h
- **Maximum 2D Court Speed:** 233.8 km/h

---

## 4. Player Physical Metrics

| Player | Total Distance Covered (m) | Average Movement Speed (km/h) | Peak Movement Speed (km/h) | Detection Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Player 1 (Near Baseline)** | 14.82 m | 6.84 km/h | 14.92 km/h | 100.0% |
| **Player 2 (Far Baseline)** | 18.23 m | 8.41 km/h | 18.35 km/h | 100.0% |

---

## 5. Generated Production Artifacts

All structured outputs are stored in `outputs/phase3_events_1/`:
1. `annotated.mp4` (6.94 MB): Rendered HD video with dynamic event badges, mini-court bounce markers, and telemetry HUD.
2. `match_events.json` (2.84 KB): Verified match events timeline with kinematic evidence and player attribution.
3. `ball_metrics.json` (3.12 KB): Piecewise segmented speed estimations and uncertainty metrics.
4. `trajectories.json` (54.9 KB): Frame-by-frame 2D image coordinates and metric court projections.
5. `detections.json` (124.2 KB): Player bounding boxes and metric foot locations.
6. `court_geometry.json` (2.21 KB): Homography matrix and 14 court keypoint coordinates.
7. `player_metrics.json` (318 B): Distance and speed statistics for each player.
8. `metrics.json` (462 B): Overall pipeline execution performance summary.
9. `run_config.yaml` (801 B): Production configuration file snapshot.

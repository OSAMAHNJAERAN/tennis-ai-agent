# T88J709: Racket Sports Vision System (Tennis Only)

University Final Year Project (FYP) focused on AI-powered video analysis of tennis matches from single-camera footage.

## Complete project file snapshot

The [complete supplemental snapshot](project_snapshot/README.md) preserves models, datasets, media, training outputs and validation fixtures that are excluded from the normal source checkout. Its 48 archive parts are available in the [2026-10-22 project snapshot release](https://github.com/OSAMAHNJAERAN/tennis-ai-agent/releases/tag/project-snapshot-2026-10-22), with original-path restoration instructions and SHA-256 checksums.

## Upload and analyze a video

The website accepts MP4/MOV/AVI uploads, follows persistent processing, reopens saved analyses and exports annotated video, readable HTML reports and JSON/CSV evidence. Home plays three distinct attributed tennis clips. Build the frontend with `pnpm build` in `frontend/`, then run `python scripts/run/serve_product.py` from the repository root using the verified product Python environment. Open [the local website](http://127.0.0.1:8000/).

See [product setup, storage, limits and workflow](docs/PRODUCT_WORKFLOW.md). Existing models and measurement limitations remain unchanged; unavailable ball speed/events are explicitly identified.

For a separate installation of the current upload/report runtime, use [the product installation guide](docs/deployment/PRODUCT_INSTALLATION.md), [requirements-product.txt](requirements-product.txt) and [the restoration manifest](docs/deployment/PRODUCT_RESTORE_MANIFEST.json). They pin the selected unqualified configuration, runtime dependencies, source revision and retained model hashes. The verifier checks installation evidence without loading models or replacing saved analyses; missing court-weight provenance and public deployment permissions are recorded explicitly.

Use Home, My analyses and the upper-right Upload video action. Each saved analysis has replay, statistics and exports. Tracking observations and available image pose/racket evidence stay synchronized with the video; court positions and physical measurements depend on valid calibration and evidence. The three public examples currently provide image tracking and explicit unavailable states. See [current execution progress and measured limitations](docs/plans/TENNIS_EXECUTION_PROGRESS.md).

## Astra Tennis workspace

The React workspace provides video controls, 2D/3D court views, tracking overlays and movement summaries for available evidence. Court projection is identified separately from reconstructed flight. Procedural ball height and avatar gait have been removed; physical ball speed and independently validated 3D reconstruction remain open milestones.

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open the local Vite URL while the local API is running. Choose an example from Home or upload a video. The selected website bundle is `configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml`; it remains unqualified. The older baseline/training commands below are retained research workflows.

---

## 1. System Overview

T88J709 provides an automated computer vision pipeline for pre-recorded, single-camera tennis match videos:
- **Player Detection & Tracking**: Ultralytics YOLO11 + ByteTrack tracking with foot-position ground mapping and court participant filtering.
- **Tennis Ball Detection & Trajectory**: Selected WASB temporal heatmaps with spatial crops, association and explicit observation states. YOLO ball baselines remain research workflows.
- **14-Point Court Keypoint Estimation**: ResNet-50 coordinate regression model predicting ITF court line intersections.
- **Homography & Metric Court Mapping**: Perspective transformation to a metric court plane, gated by registration validity. Fit consistency does not establish independent calibration accuracy.
- **Movement Estimates**: Timestamp-based player speed/distance on valid observed ground intervals. Independent physical accuracy and ball flight/speed remain unqualified.
- **Pose & Rackets**: Available per-frame image observations with confidence and missing intervals.
- **Visual Overlays & Mini-Court**: Top-down synchronized 2D mini-court HUD and bounding box overlays.
- **Machine-Readable Outputs**: Structured JSON summaries (`detections.json`, `trajectories.json`, `court_geometry.json`, `player_metrics.json`, `run_config.yaml`).

---

## 2. Quickstart & Installation

### Requirements

- Verified product environment: Windows x86_64 and CPython 3.13.5.
- Retained selected checkpoints and pinned WASB runtime source, with hashes in the restoration manifest.
- FFmpeg/ffprobe and Node.js 22+ with project pnpm 11.19.0.
- CUDA inference uses the recorded PyTorch CUDA 12.6 build and an NVIDIA driver; other runtimes need their own checks.

### Setup

Follow [the product installation guide](docs/deployment/PRODUCT_INSTALLATION.md) to create a separate environment with `requirements-product.txt` and the recorded transitive constraints, restore the exact assets, build the locked frontend and verify the API. Preserve existing product storage and research environments. The broad historical `requirements.txt` belongs to the retained research/training workflows.

---

## 3. Running Baseline Inference

To run end-to-end analysis on a tennis video:

```bash
python scripts/inference/run_baseline.py \
  --input data/sample_videos/input_video.mp4 \
  --output outputs/baseline_run \
  --config configs/baseline/pipeline.yaml
```

Generated outputs in `outputs/baseline_run/`:
- `annotated.mp4` — Rendered video with bounding boxes, keypoints, mini-court HUD, and statistics overlay.
- `detections.json` — Per-frame player detections and track IDs.
- `trajectories.json` — Ball positions with confidence and state flags.
- `court_geometry.json` — 14 detected keypoints and 3x3 homography matrix.
- `player_metrics.json` — Total distance, instantaneous/average speed.
- `run_config.yaml` — Exact configuration snapshot.

---

## 4. Running Automated Tests

```bash
python -m pytest tests/ -v
```

---

## 5. Model Training & Evaluation

### Train Tennis Ball Detector
```bash
python scripts/train/train_ball_detector.py \
  --data_yaml data/processed/tennis_ball/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### Train Court Keypoint Detector
```bash
python scripts/train/train_court_model.py \
  --data_dir data/processed/court_keypoints \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4
```

---

## 6. Architecture & Directory Structure

```text
tennis_ai_agent_bundle/
├── configs/baseline/            # Configuration files
├── data/                        # Dataset registry and storage
├── docs/                        # Audits, plans, results, and research
│   ├── audit/
│   ├── data/
│   ├── environment/
│   ├── experiments/
│   └── plans/
├── models/                      # Trained model checkpoints (.pt, .pth)
├── outputs/                     # Inference artifacts and annotated videos
├── scripts/
│   ├── evaluate/                # Evaluation scripts
│   ├── inference/               # Pipeline execution scripts
│   └── train/                   # Training scripts
├── src/                         # Core implementation modules
│   ├── analytics/               # Speed and distance calculations
│   ├── court/                   # Geometry, keypoints, and homography
│   ├── detection/               # YOLO11 player and ball detectors
│   ├── pipeline/                # End-to-end orchestration
│   ├── tracking/                # ByteTrack and ball interpolation
│   ├── utils/                   # Video I/O, bbox math, geometry
│   └── visualization/           # Mini-court and video overlays
└── tests/                       # Pytest automated test suite
```

---

## 7. Baseline Results & Historical Analysis

See:
- `docs/experiments/BASELINE_RESULTS.md`
- `docs/experiments/BASELINE_FAILURE_ANALYSIS.md`

---

## 8. Historical Phase 2.1 ball detection experiments

### Train YOLO11 Ball Detector
```bash
python scripts/train/train_ball_yolo11.py --config configs/phase2_yolo11/ball_yolo11s.yaml
```

### Benchmark Evaluation (YOLOv5 vs YOLO11)
```bash
python scripts/evaluate/evaluate_phase2_yolo11_experiments.py
```

### Run Corrected End-to-End Pipeline
```bash
python scripts/inference/run_phase2_yolo11.py \
  --input data/sample_videos/input_video.mp4 \
  --output outputs/phase2_yolo11_final \
  --config configs/phase2_yolo11/pipeline.yaml
```

See:
- `docs/experiments/PHASE2_YOLO11_EXPERIMENTS.md`
- `docs/experiments/PHASE2_YOLO11_RESULTS.md`
- `docs/experiments/PHASE2_YOLO11_FAILURE_ANALYSIS.md`

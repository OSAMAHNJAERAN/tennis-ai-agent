# T88J709: Racket Sports Vision System (Tennis Only)

University Final Year Project (FYP) focused on AI-powered video analysis of tennis matches from single-camera footage.

## Astra Tennis workspace

The upgraded React workspace adds an interactive 3D court, animated players and ball, synchronized video/tactical replay, spatial density, and a configurable Astra GPT coach.

```powershell
cd frontend
pnpm install
pnpm dev
```

Open the local Vite URL and select **Phase 5 Scoring** for the bundled spatial example. See [Astra setup and implementation](docs/ASTRA_TENNIS.md) for coach configuration, evidence limitations, and validation. The existing inference commands below remain available.

---

## 1. System Overview

T88J709 provides an automated computer vision pipeline for pre-recorded, single-camera tennis match videos:
- **Player Detection & Tracking**: Ultralytics YOLO11 + ByteTrack tracking with foot-position ground mapping and court participant filtering.
- **Tennis Ball Detection & Trajectory**: Fine-tuned dedicated YOLO11 detector with gap-limited interpolation and state-aware tracking (`DETECTED`, `INTERPOLATED`, `MISSING`).
- **14-Point Court Keypoint Estimation**: ResNet-50 coordinate regression model predicting ITF court line intersections.
- **Homography & Metric Court Mapping**: Perspective transformation mapping video pixels to true 2D metric court dimensions (meters).
- **Physical Analytics**: Frame-rate accurate player movement speed (km/h), distance covered (m), and ball trajectory.
- **Visual Overlays & Mini-Court**: Top-down synchronized 2D mini-court HUD and bounding box overlays.
- **Machine-Readable Outputs**: Structured JSON summaries (`detections.json`, `trajectories.json`, `court_geometry.json`, `player_metrics.json`, `run_config.yaml`).

---

## 2. Quickstart & Installation

### Requirements
- Python 3.10+ (tested on Python 3.13.5)
- NVIDIA GPU with CUDA support (e.g. RTX 4050, PyTorch CUDA 12.6+)
- Windows / Linux

### Setup
```bash
# 1. Clone / enter project directory
cd tennis_ai_agent_bundle

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Install remaining dependencies
pip install -r requirements.txt
```

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

## 8. Phase 2.1: YOLO11 High-Accuracy Ball Detection & Temporal Tracking

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

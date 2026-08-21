# Baseline Implementation Plan — T88J709 Tennis Vision System

> **Date:** 2026-08-22
> **Phase:** 0 → 3 (Bootstrap → Baseline)
> **Author:** AI/ML Engineering Agent
> **Status:** AWAITING APPROVAL

---

## 1. Current Repository State

### 1.1 Critical Finding: Greenfield Project

The repository at `c:\Semester 9\FYP2\tennis_ai_agent_bundle` contains **zero implementation code**. It consists exclusively of:

| Item | Count | Description |
|------|-------|-------------|
| Specification files | 6 | AGENT.md, PLAN.md, PROJECT_REPORT.md, VIDEO_REFERENCE.md, MASTER_PROMPT.md, README_BUNDLE.md |
| Report images | 11 | PNG/JPEG/EMF in `report_assets/media/` |
| Source code | 0 | No Python files, no modules, no packages |
| Model weights | 0 | No `.pt`, `.pth`, `.onnx` files |
| Datasets | 0 | No training/validation/test data |
| Tests | 0 | No test suite |
| Configuration | 0 | No YAML/JSON/TOML configs |
| UI/Dashboard | 0 | No frontend code |
| Database | 0 | No SQLite/PostgreSQL files or migrations |
| Git repository | 0 | Not initialized — `fatal: not a git repository` |
| Virtual environment | 0 | No venv/conda environment |
| requirements.txt | 0 | No dependency lock file |

### 1.2 Existing Dependencies (System-Wide Python 3.13)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.13.5 | System install; report specified 3.10+ |
| PyTorch | 2.11.0+cpu | **CPU-only build — no CUDA** |
| torchvision | 0.26.0 | Matches PyTorch version |
| Ultralytics | 8.4.36 | Supports YOLO11 natively |
| OpenCV | 4.13.0.92 | opencv-python + opencv-contrib-python |
| NumPy | 2.3.1 | Current |
| Pandas | 2.3.1 | Current |
| SciPy | 1.17.1 | Current |
| Roboflow | 1.3.1 | Dataset download SDK |

> **CRITICAL:** PyTorch is CPU-only despite having an NVIDIA RTX 4050 GPU (6 GB VRAM, CUDA 13.2 driver). Training will be extremely slow or impractical until PyTorch+CUDA is installed.

### 1.3 Hardware Environment

| Component | Value |
|-----------|-------|
| OS | Windows 11 Home (64-bit, build 10.0.26200) |
| CPU | Intel Core i7-14700HX (20 cores, 28 threads) |
| RAM | 15.71 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU VRAM | 6141 MiB (~6 GB) |
| GPU Driver | 596.36 |
| CUDA Driver | 13.2 |
| Disk Free | ~9.2 GB |

> **WARNING:** Disk space is critically low at 9.2 GB. Model weights (~200-600 MB each), datasets (~500 MB-2 GB), and training outputs will require freeing space or using external storage.

---

## 2. Requirements Mapping

### 2.1 Requirements Matrix

| # | Requirement | Report | AGENT.md | Existing | Baseline Scope | Later Phase |
|---|-------------|:------:|:--------:|:--------:|:--------------:|:-----------:|
| 1 | Video ingestion and validation | Yes | Yes | No | YES | — |
| 2 | Player detection (YOLO) | Yes | Yes | No | YES | — |
| 3 | Player tracking (persistent IDs) | Yes | Yes | No | YES | — |
| 4 | Player filtering (court geometry) | Yes | Yes | No | YES | — |
| 5 | Tennis ball detection | Yes | Yes | No | YES | — |
| 6 | Ball trajectory interpolation | Yes | Yes | No | YES | — |
| 7 | 14-point court keypoints | Yes | Yes | No | YES | — |
| 8 | Homography (image to court) | Yes | Yes | No | YES | — |
| 9 | Mini-court visualization | Yes | Yes | No | YES | — |
| 10 | Player position mapping | Yes | Yes | No | YES | — |
| 11 | Ball position mapping | Yes | Yes | No | YES | — |
| 12 | Player distance covered | Yes | Yes | No | YES | — |
| 13 | Player speed | Yes | Yes | No | YES | — |
| 14 | Ball/shot speed | Yes | Yes | No | Preliminary | Phase 5+ |
| 15 | Actual FPS/timestamp handling | — | Yes | No | YES | — |
| 16 | Shot/event detection | Yes | Yes | No | Basic | Phase 8 |
| 17 | Bounce detection | Yes | Yes | No | No | Phase 8 |
| 18 | Assisted IN/OUT calls | Yes | Yes | No | No | Phase 9 |
| 19 | Tennis scoring state machine | Yes | Yes | No | No | Phase 10 |
| 20 | Temporal ball tracking (TrackNet) | Yes | Yes | No | No | Phase 5 |
| 21 | Camera motion detection | — | Yes | No | No | Phase 6 |
| 22 | Player heatmaps | Yes | Yes | No | No | Phase 12 |
| 23 | Ball bounce heatmaps | Yes | Yes | No | No | Phase 12 |
| 24 | Database persistence | Yes | Yes | No | No | Phase 11 |
| 25 | Analytics dashboard | Yes | Yes | No | No | Phase 12 |
| 26 | Annotated output video | Yes | Yes | No | YES | — |
| 27 | Automated tests | — | Yes | No | YES | — |
| 28 | Evaluation metrics | Yes | Yes | No | YES | — |
| 29 | Machine-readable results | — | Yes | No | YES | — |

---

## 3. Baseline Architecture

### 3.1 Overview

The baseline implements the tutorial-equivalent pipeline with critical corrections from AGENT.md/VIDEO_REFERENCE.md, using modern framework versions:

```
Input Video (MP4)
    |
Video Loader (actual FPS extraction)
    |
+------------------------------------------+
|  Per-Frame Detection                      |
|  +-- Player Detection (YOLO11x)          |
|  +-- Player Tracking (ByteTrack)         |
|  +-- Ball Detection (YOLO11m fine-tuned) |
+------------------------------------------+
    |
Court Keypoint Detection (ResNet-50 -> 28 outputs)
    |
Homography (cv2.findHomography)
    |
Canonical Court Mapping (metric coordinates)
    |
+------------------------------------------+
|  Baseline Analytics                       |
|  +-- Player positions (court coords)     |
|  +-- Ball positions (court coords)       |
|  +-- Player distance covered             |
|  +-- Player speed                        |
|  +-- Ball trajectory (with states)       |
+------------------------------------------+
    |
+------------------------------------------+
|  Output                                   |
|  +-- Annotated video (.mp4)              |
|  +-- Mini-court visualization            |
|  +-- detections.json                     |
|  +-- trajectories.json                   |
|  +-- court_geometry.json                 |
|  +-- player_metrics.json                 |
|  +-- run_config.yaml                     |
+------------------------------------------+
```

### 3.2 Player Baseline

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detector | YOLO11x (pretrained COCO) | Report specifies YOLOv11 direction; Ultralytics 8.4.36 supports it natively; person class (0) from COCO |
| Tracker | ByteTrack | Lower latency than BoT-SORT; sufficient for static-camera tennis; built into Ultralytics |
| Player filtering | Court keypoint proximity + court polygon | Tutorial approach refined: use foot-point (bbox bottom-center) distance to detected court boundary, not bbox center |
| Ground point | Bottom-center of bbox | AGENT.md requirement; more accurate court projection than bbox center |
| Tracking confidence | Preserve per-frame | Required for downstream quality gating |

### 3.3 Ball Baseline

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detector | Fine-tuned YOLO11m at 640px | YOLO11 direction per report; m-size balances accuracy/VRAM on 6GB RTX 4050 |
| Training data | Roboflow tennis-ball-detection v6 | 578 images, CC BY 4.0, verified; tutorial-compatible baseline |
| Split strategy | Inspect for video-level leakage | If source video metadata exists, group by video; otherwise use Roboflow's existing split with documented caveat |
| Training epochs | 100 (following tutorial) | Baseline reproduction; early stopping on val mAP |
| Confidence threshold | Tuned on validation set | Not blindly set to 0.15 |
| Model selection | Best validation mAP50 checkpoint | Not last.pt |
| Interpolation | Gap-limited pandas interpolation | Max gap = 5 frames; each point tagged with state (DETECTED/INTERPOLATED/MISSING) |
| Backfill | Disabled | AGENT.md: do not backfill unknown beginnings |

### 3.4 Court Baseline

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | ResNet-50 -> FC(28) | Tutorial baseline; dataset from yastrebksv/TennisCourtDetector; ~8841 images |
| Input size | 224x224 | Tutorial standard; baseline reproduction |
| Loss | MSE | Tutorial standard |
| Optimizer | Adam, lr=1e-4 | Tutorial standard |
| Epochs | 20 | Tutorial standard with early stopping |
| Output | 14 (x,y) keypoints | 28 regression values |
| Evaluation | Per-keypoint pixel error + homography reprojection error | Not just training loss |

### 3.5 Homography and Canonical Court

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Method | cv2.findHomography with detected 14 keypoints | Standard approach; at least 4 non-collinear points required |
| Canonical system | ITF standard tennis court in meters | Origin at top-left corner; x=width (10.97m), y=length (23.77m) |
| Validation | Reprojection error < threshold; sanity check court dimensions | Reject geometrically impossible layouts |

### 3.6 Critical Corrections vs Tutorial

| Tutorial Issue | Baseline Fix |
|----------------|-------------|
| Hardcoded 24 FPS | Read actual FPS via cv2.VideoCapture.get(CAP_PROP_FPS) + ffprobe fallback |
| last.pt model selection | Select by best validation metric |
| Blind bfill() interpolation | Gap-limited interpolation with state tracking |
| No homography (keypoint-relative scaling) | Proper cv2.findHomography with reprojection error |
| All frames in memory | Acceptable for baseline (short clips); note for later streaming |
| MJPG codec at 24 FPS | Match source FPS; use H.264 for compatibility |
| No tests | Baseline test suite |
| No structured output | JSON/YAML machine-readable results |

---

## 4. Pre-Implementation Prerequisites

### 4.1 Environment Fix: PyTorch CUDA

**This is the number-one blocker.** Training any model on CPU will be impractically slow. Must install PyTorch with CUDA support.

Plan:
1. Uninstall current CPU PyTorch: `pip uninstall torch torchvision torchaudio`
2. Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` (or latest compatible with CUDA 13.2 driver)
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### 4.2 Disk Space

Must free at minimum 10-15 GB additional space for:
- Court keypoint dataset: ~500 MB
- Ball detection dataset: ~100 MB
- Model weights: ~500 MB
- Training outputs: ~2 GB
- Output videos: ~1 GB

### 4.3 Git Initialization

```bash
git init
git add .
git commit -m "Initial commit: project specification bundle"
```

### 4.4 Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Proposed Project Structure

```
tennis_ai_agent_bundle/
+-- AGENT.md                          # [PRESERVE] Engineering contract
+-- PLAN.md                           # [PRESERVE] Phase checklist
+-- PROJECT_REPORT.md                 # [PRESERVE] FYP report
+-- VIDEO_REFERENCE.md                # [PRESERVE] Tutorial reference
+-- MASTER_PROMPT.md                  # [PRESERVE] Agent instructions
+-- README_BUNDLE.md                  # [PRESERVE] Bundle README
+-- README.md                         # [NEW] Project README
+-- requirements.txt                  # [NEW] Pinned dependencies
+-- .gitignore                        # [NEW] Ignore patterns
+-- .env.example                      # [NEW] Environment variable template
+-- report_assets/                    # [PRESERVE]
+-- configs/
|   +-- baseline/
|       +-- player_detection.yaml
|       +-- ball_detection.yaml
|       +-- court_detection.yaml
|       +-- pipeline.yaml
+-- data/
|   +-- data_registry.yaml
|   +-- raw/
|   +-- processed/
|   +-- sample_videos/
+-- src/
|   +-- __init__.py
|   +-- detection/
|   |   +-- __init__.py
|   |   +-- player_detector.py
|   |   +-- ball_detector.py
|   +-- tracking/
|   |   +-- __init__.py
|   |   +-- player_tracker.py
|   |   +-- ball_tracker.py
|   +-- court/
|   |   +-- __init__.py
|   |   +-- court_keypoint_detector.py
|   |   +-- court_geometry.py
|   |   +-- homography.py
|   +-- analytics/
|   |   +-- __init__.py
|   |   +-- player_analytics.py
|   |   +-- ball_analytics.py
|   +-- visualization/
|   |   +-- __init__.py
|   |   +-- mini_court.py
|   |   +-- video_annotator.py
|   |   +-- stats_overlay.py
|   +-- utils/
|   |   +-- __init__.py
|   |   +-- video_io.py
|   |   +-- bbox_utils.py
|   |   +-- geometry.py
|   +-- pipeline/
|       +-- __init__.py
|       +-- baseline_pipeline.py
+-- scripts/
|   +-- train/
|   |   +-- train_ball_detector.py
|   |   +-- train_court_model.py
|   +-- evaluate/
|   |   +-- evaluate_ball.py
|   |   +-- evaluate_court.py
|   +-- inference/
|       +-- run_baseline.py
+-- experiments/
|   +-- baseline/
|       +-- artifacts/
|       +-- models/
|       +-- metrics/
|       +-- predictions/
+-- models/
+-- outputs/
+-- tests/
|   +-- __init__.py
|   +-- test_video_io.py
|   +-- test_court_geometry.py
|   +-- test_homography.py
|   +-- test_analytics.py
|   +-- test_ball_tracker.py
|   +-- test_pipeline.py
+-- docs/
    +-- audit/REPOSITORY_AUDIT.md
    +-- data/DATASET_AUDIT.md
    +-- environment/BASELINE_ENVIRONMENT.md
    +-- plans/BASELINE_IMPLEMENTATION_PLAN.md
    +-- experiments/BASELINE_RESULTS.md
    +-- experiments/BASELINE_FAILURE_ANALYSIS.md
```

---

## 6. Implementation Sequence

### Step 1: Environment Setup (~30 min)
1. Initialize Git repository
2. Create .gitignore
3. Create virtual environment (or use system Python)
4. Install PyTorch+CUDA
5. Install all dependencies
6. Verify CUDA smoke test
7. Create requirements.txt
8. Create BASELINE_ENVIRONMENT.md
9. First commit

### Step 2: Core Infrastructure (~2 hours)
1. Create project directory structure
2. Implement src/utils/video_io.py (actual FPS handling)
3. Implement src/court/court_geometry.py (canonical dimensions)
4. Implement src/utils/bbox_utils.py
5. Implement src/utils/geometry.py
6. Write unit tests for geometry/court/FPS
7. Create baseline configs

### Step 3: Player Detection and Tracking (~1 hour)
1. Implement src/detection/player_detector.py (YOLO11x wrapper)
2. Implement src/tracking/player_tracker.py (ByteTrack + court filtering)
3. Test on sample video
4. Record detection quality observations

### Step 4: Ball Detection (~3-6 hours including training)
1. Download Roboflow tennis ball dataset
2. Audit dataset for leakage
3. Create data/data_registry.yaml entry
4. Create ball training config
5. Train YOLO11m ball detector (100 epochs)
6. Evaluate on validation set
7. Select best checkpoint
8. Record metrics (P/R/F1/mAP50)

### Step 5: Court Keypoint Model (~3-6 hours including training)
1. Download court keypoint dataset
2. Audit dataset provenance and license
3. Create data/data_registry.yaml entry
4. Implement court keypoint model (ResNet-50)
5. Train with MSE loss
6. Evaluate per-keypoint pixel error
7. Select best checkpoint

### Step 6: Homography and Court Mapping (~2 hours)
1. Implement src/court/homography.py
2. Implement canonical court coordinate transforms
3. Implement homography validation (reprojection error)
4. Write homography unit tests with synthetic data
5. Test on actual court keypoint predictions

### Step 7: Mini-Court and Visualization (~2 hours)
1. Implement src/visualization/mini_court.py
2. Implement src/visualization/video_annotator.py
3. Implement src/visualization/stats_overlay.py
4. Test rendering with actual detections

### Step 8: Analytics (~1 hour)
1. Implement src/analytics/player_analytics.py (distance, speed)
2. Implement src/analytics/ball_analytics.py (trajectory with states)
3. Use actual timestamps and metric court coordinates
4. Add outlier rejection for impossible position jumps
5. Write analytics unit tests

### Step 9: Pipeline Integration (~2 hours)
1. Implement src/pipeline/baseline_pipeline.py
2. Implement scripts/inference/run_baseline.py
3. End-to-end test on sample video
4. Generate all output artifacts
5. Verify machine-readable JSON outputs

### Step 10: Evaluation and Documentation (~2 hours)
1. Run full evaluation
2. Create BASELINE_RESULTS.md
3. Create BASELINE_FAILURE_ANALYSIS.md
4. Document failure categories with examples
5. Update REPOSITORY_AUDIT.md with post-implementation state

---

## 7. Dataset Plan

### 7.1 Tennis Ball Detection Dataset

| Field | Value |
|-------|-------|
| Name | Roboflow Tennis Ball Detection |
| Version | v6 |
| URL | https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection |
| License | CC BY 4.0 |
| Images | 578 (428 train / 100 val / 50 test) |
| Classes | 1 (tennis-ball) |
| Format | YOLO bounding box (x_center, y_center, w, h normalized) |
| Decision | APPROVED for baseline |
| Leakage risk | Unknown source video provenance — document as limitation |

### 7.2 Tennis Court Keypoint Dataset

| Field | Value |
|-------|-------|
| Name | Tennis Court Keypoint Dataset |
| Source | yastrebksv/TennisCourtDetector (GitHub) |
| Mirror | Hugging Face: Gholamreza/tennis_court_keypoints_dataset |
| License | LICENSE_REVIEW_REQUIRED — no explicit license in GitHub repo |
| Images | ~8,841 (75% train / 25% val) |
| Resolution | 1280x720 |
| Keypoints | 14 per image (28 coordinates) |
| Surfaces | Hard, clay, grass |
| Decision | APPROVED for baseline (academic/educational use); note license gap |

### 7.3 Sample Video

| Field | Value |
|-------|-------|
| Source | Tutorial repository input_videos/input_video.mp4 |
| License | Unlicensed tutorial repo — use for baseline reproduction only |
| Purpose | Baseline end-to-end testing |
| Alternative | User-provided video for production testing |

---

## 8. Model Training Plan

### 8.1 Ball Detector Training

```yaml
experiment_id: baseline_ball_yolo11m_20260822
architecture: YOLO11m
pretrained_weights: yolo11m.pt (COCO)
dataset: roboflow/tennis-ball-detection-v6
image_size: 640
epochs: 100
optimizer: SGD (Ultralytics default)
lr0: 0.01
batch_size: 16  # Adjust for 6GB VRAM
patience: 20  # Early stopping
seed: 42
device: "cuda:0"
augmentation: Ultralytics default + hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
checkpoint_selection: best val/mAP50
```

### 8.2 Court Keypoint Training

```yaml
experiment_id: baseline_court_resnet50_20260822
architecture: ResNet-50 -> FC(28)
pretrained_weights: torchvision ResNet-50 (ImageNet)
dataset: yastrebksv/TennisCourtDetector
input_size: [224, 224]
epochs: 20
optimizer: Adam
learning_rate: 0.0001
batch_size: 8
loss: MSE
seed: 42
device: "cuda:0"
normalization: ImageNet mean/std
checkpoint_selection: best val loss
```

---

## 9. Evaluation Plan

### 9.1 Player Detection/Tracking
- Detection: mAP50, precision, recall on sample video
- Tracking: visual inspection of ID consistency, count of identity switches
- Court filtering: percentage of non-player detections successfully filtered

### 9.2 Ball Detection
- mAP50, mAP50-95, precision, recall, F1 on test split
- Detection rate per frame on sample video
- Ball center localization error
- Missed-frame rate and gap length distribution
- Failure categorization (blur, occlusion, near-line, etc.)

### 9.3 Court Keypoints
- Per-keypoint mean pixel error on validation set
- Overall mean keypoint error
- Homography reprojection error
- Catastrophic failure rate (>50px error)

### 9.4 Pipeline Performance
- Processing FPS on RTX 4050
- GPU VRAM usage
- RAM usage
- End-to-end processing time per input minute

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| PyTorch CPU-only blocks training | CRITICAL | Install CUDA PyTorch first |
| Disk space less than 10 GB | HIGH | Free space before starting; use external storage |
| Ball dataset too small (578 imgs) | MEDIUM | Document as baseline limitation; expand in Phase 5 |
| Court dataset no explicit license | MEDIUM | Use for academic baseline; document gap |
| 6GB VRAM limits batch size | MEDIUM | Use YOLO11m instead of 11x for ball; reduce batch size |
| No sample tennis video included | MEDIUM | Download tutorial video or request from user |
| Python 3.13 compatibility | LOW | Most packages support 3.13; test early |

---

## 11. Verification Plan

### 11.1 Automated Tests

```bash
python -m pytest tests/ -v
```

Tests will cover:
- Video I/O (FPS extraction for 24/25/30/50/60 fps)
- Court geometry (canonical dimensions, coordinate transforms)
- Homography (synthetic known-point tests, inverse projection)
- Analytics (zero movement, known displacement, speed calculation)
- Ball tracker (interpolation gap limits, state tracking)
- Pipeline integration (short clip end-to-end)

### 11.2 Manual Verification
- Visual inspection of annotated output video
- Mini-court player/ball positions match video
- JSON output contains expected fields
- No fabricated metrics

---

## 12. Definition of Done (Baseline)

- [ ] All project files read
- [ ] Git repository initialized
- [ ] Environment documented (BASELINE_ENVIRONMENT.md)
- [ ] Repository audit created (REPOSITORY_AUDIT.md)
- [ ] Dataset audit created (DATASET_AUDIT.md)
- [ ] PyTorch CUDA verified working
- [ ] Player detection working
- [ ] Player tracking with persistent IDs
- [ ] Player court filtering working
- [ ] Ball detection model trained
- [ ] Ball detection test metrics measured
- [ ] Ball interpolation with state tracking
- [ ] Court keypoints model trained
- [ ] Court keypoint quality measured
- [ ] Homography computed and validated
- [ ] Canonical court coordinates working
- [ ] Mini-court visualization working
- [ ] Player positions projected to court
- [ ] Ball positions projected to court
- [ ] Player distance calculated
- [ ] Player speed calculated
- [ ] Annotated video produced
- [ ] Machine-readable JSON results generated
- [ ] Automated tests passing
- [ ] BASELINE_RESULTS.md created
- [ ] BASELINE_FAILURE_ANALYSIS.md created
- [ ] Reproducible inference command documented

---

## Open Questions Requiring User Input

### Q1: Sample tennis video availability
The tutorial repo contains `input_videos/input_video.mp4` but the repo has no license. Do you have a tennis video file we can use for baseline testing? If not, I can download the tutorial's sample video for baseline reproduction purposes only.

### Q2: Disk space
Current free disk space is approximately 9.2 GB. We need approximately 15-20 GB for datasets, models, and outputs. Can you free up some disk space, or should we use a different drive/path?

### Q3: PyTorch CUDA installation
Should I proceed with installing PyTorch+CUDA (which will replace the current CPU-only version)? This is essential for any model training.

### Q4: Training on local GPU vs cloud?
With 6 GB VRAM, the RTX 4050 can handle YOLO11m ball training and ResNet-50 court training, but may be tight for larger models. Should we use local GPU for baseline, or do you have cloud GPU access (Colab, etc.)?

# Caltech Tennis Benchmark

Evaluation scripts for benchmarking monocular human pose reconstruction models on multi-view tennis videos. We evaluate state-of-the-art methods — **WHAM, TRAM, GVHMR, PromptHMR, and GENMO** — without ground-truth annotations by measuring **multi-view 3D pose consistency** across overlapping camera views of the same scene.

![Pipeline overview](media/pipeline.png)

---

## Dataset

The dataset consists of multi-view tennis court recordings captured with consumer cameras. Videos are anonymized (faces blurred) using [deface](https://github.com/ORB-HD/deface) prior to release.

**Dataset viewer:** [https://huggingface.co/spaces/demalenk/caltennis-preview-viewer](https://huggingface.co/spaces/demalenk/caltennis-preview-viewer)

The dataset is split into three subsets defined in [`dataset_config.yaml`](dataset_config.yaml):
- **mini** — small held-out split for quick iteration
- **mid** — medium split used for most evaluations
- **full** — complete dataset

### Video naming convention

- **Session folder:** `{MM}_{DD}_{YYYY}_{HH}_{MM}_court{N}`
- **Video file:** `{MM}_{DD}_{YYYY}_{HH}_{MM}_{SS}_{ms}_court{N}_{direction}_{end_HH}_{end_MM}_{end_SS}`

---

## Repo Structure

```
caltech-tennis-benchmark/
├── camera_pose_recovery/       # Camera calibration (PnP) to recover extrinsics per video
├── temporal_video_alignment/   # Extract per-frame timestamps from video files
├── utils/                      # Shared utilities used across all models
│   ├── world4d_utils.py        #   Person ID smoothing, spatial matching, cross-camera fusion
│   ├── pose_utils.py           #   Temporal alignment, pose interpolation, joint recomputation
│   ├── video_alignment_utils.py#   Timestamp loading, video metadata, overlapping-view detection
│   ├── plots_and_pdf_analysis.py#  Visualization, metric plots, PDF report generation
│   ├── video_creation_utils.py #   Video creation from frame sequences
│   ├── joint_extraction.py     #   Joint extraction from world4d outputs
│   ├── stability_metrics.py    #   Balance and stability metrics (CoM, contact points)
│   └── wham_utils.py           #   WHAM → world4d format conversion
├── gvhmr/                      # GVHMR inference, fusion, and multi-view alignment
├── tram/                       # TRAM inference, fusion, and multi-view alignment
├── wham/                       # WHAM inference, fusion, and multi-view alignment
├── genmo/                      # GENMO inference and fusion
├── prompthmr/                  # PromptHMR inference, fusion, metrics, and visualization
│   ├── scripts/                #   Model inference and core fusion (fuse_results.py)
│   ├── pipeline/               #   Underlying pipeline (DroidCalib, SAM2, etc.)
│   ├── visualization_scripts/  #   Multi-view alignment, calibration, and analysis
│   ├── court_visualization/    #   Tennis court 3D visualization
│   └── vis/                    #   3D viser and mesh renderer
├── mocap_stuff/                # Comparison against motion capture ground truth
├── fuse.py                     # Unified fusion entry point (all models)
├── environments.yaml           # Machine-specific path configuration
├── switch_env.py               # Regenerates config.env files from environments.yaml
└── dataset_config.yaml         # Dataset split definitions (mini / mid / full)
```

---

## Evaluation Pipeline

### Step 0 — Configure paths

All machine-specific paths live in [`environments.yaml`](environments.yaml). After cloning, fill in the `FILL_IN` placeholders for your machine and run:

```bash
python switch_env.py gpu    # or: hpc, local
```

This generates `config.env` files for each model folder, which all run scripts source automatically.

### Step 1 — Extract timestamps

Extract precise per-frame timestamps once per video (used for cross-view temporal alignment):

```bash
bash temporal_video_alignment/run_extract_timestamps.sh
```

Writes `<video_name>_timestamps.npy` alongside each video file.

### Step 2 — Calibrate cameras

Recover camera intrinsics and extrinsics (R, T) by annotating tennis court line corners:

```bash
# Extract a key frame for annotation
python camera_pose_recovery/extract_key_frames.py --video <path/to/video.MOV>

# Launch court corner annotation GUI and run PnP calibration
python camera_pose_recovery/run_camera_calibration.py --video <path/to/video.MOV>
```

Results are saved to `calib_root` (configured in `environments.yaml`).

### Step 3 — Run model inference

Each model folder contains a `run_bench.py` and/or `run_pose_estimates_on_tennis_session.sh` that processes a video in 500-frame chunks with 250-frame overlap, saving per-chunk `.pkl` outputs.

```bash
# GVHMR
python gvhmr/run_bench.py --video <video_name> --session <session_folder>

# TRAM
python tram/run_bench.py --video <video_name> --session <session_folder>

# WHAM
bash wham/run_pose_estimates_on_tennis_session.sh

# PromptHMR
bash prompthmr/run_pose_estimates_on_tennis_session.sh

# GENMO
python genmo/run_bench.py --video <video_name> --session <session_folder>
```

### Step 4 — Fuse chunked outputs

Merge per-chunk `.pkl` files into a single `world4d_fused.pkl` per video. Use the unified entry point:

```bash
python fuse.py --model <model> --session <session_folder> --video <video_name> [--force]
```

Where `--model` is one of `prompthmr`, `gvhmr`, `tram`, `wham`, or `genmo`. The script reads paths from the model's `config.env` and dispatches to the appropriate fusion logic.

**Batch fusion** (all videos for a model):

```bash
bash gvhmr/run_gvhmr_fusion_on_all_results.sh
bash tram/run_tram_fusion_on_all_results.sh
bash wham/run_wham_fusion_on_all_results.sh
bash prompthmr/run_fusion_on_all_results.sh
bash genmo/run_genmo_fusion_on_all_results.sh
```

### Step 5 — Multi-view alignment

With fused per-video outputs, align overlapping camera views using Procrustes to estimate a shared world coordinate frame and temporal offset:

```bash
# Pairwise (2 cameras)
python prompthmr/visualization_scripts/visualize_time_calibrated_two_processed_world4ds_procrustes_aligned.py

# Multi-camera (3+ cameras)
python prompthmr/visualization_scripts/visualize_time_calibrated_processed_world4ds_multialign.py
```

### Step 6 — Compute evaluation metrics

Generate cross-model aggregate metrics and analysis PDFs:

```bash
bash prompthmr/generate_agg_analyses.sh
```

Or run the individual scripts:

```bash
python prompthmr/generate_aggregate_metrics_analyses.py
python prompthmr/aggregate_dataset_metrics_multi_model.py
```

---

## world4d Output Format

All models produce `world4d_fused.pkl` — a dict keyed by frame index. For frames with visible people:

```python
world4d[frame_index] = {
    # SMPLX parameters (N = number of people in this frame):
    "pose":             torch.Tensor,   # (N, 55, 3)  SMPLX pose
    "shape":            torch.Tensor,   # (N, 10)     SMPLX betas
    "trans":            torch.Tensor,   # (N, 3)      translation in model world coords
    # Scene and tracking:
    "track_id":         list,           # length N, zero-indexed person IDs
    "camera":           np.ndarray,     # (4, 4) camera-to-world [R | t]
    "joints":           torch.Tensor,   # (N, 25, 3)  body joints in model world coords
    "joints_world":     torch.Tensor,   # (N, 25, 3)  body joints in tennis court coords
    "trans_world":      torch.Tensor,   # (N, 3)      translation in tennis court coords
    "phmrworld_to_tennisworld": np.ndarray,  # (4, 4) model world → court transform
    "relative_timestamp": float,        # seconds since video start
    "global_timestamp":   float,        # wall-clock time of this frame
    "global_time_start":  float,        # wall-clock time at video start
}
```

For frames with no visible people, only scene-level fields are stored (`track_id`, `camera`, timestamps).

---

## Installation

### 1. Clone and configure

```bash
git clone https://github.com/selinananana/caltech-tennis-benchmark
cd caltech-tennis-benchmark
pip install -e .

# Fill in FILL_IN values in environments.yaml, then:
python switch_env.py gpu
```

### 2. Install pose models

Follow the installation instructions in each model's repository:

| Model | Repository |
|---|---|
| PromptHMR | https://github.com/yufu-wang/PromptHMR |
| GVHMR | https://github.com/zju3dv/GVHMR |
| TRAM | https://github.com/yufu-wang/tram |
| WHAM | https://github.com/yohanshin/WHAM |
| GENMO | https://github.com/GENMO-Official/GENMO |

After installing each model, set its `root` path in `environments.yaml` and re-run `python switch_env.py`.

All models require SMPL/SMPL-X body model files from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/) — set `data_root` in `environments.yaml` to the directory containing `body_models/smplx/`.

---
license: cc-by-nc-4.0
viewer: true
configs:
  - config_name: default
    data_files:
      - split: mini
        path: metadata_mini.jsonl
      - split: mid
        path: metadata_mid.jsonl
dataset_info:
  features:
    - name: video
      dtype: string
    - name: timestamps
      dtype: string
    - name: calibration
      dtype: string
    - name: session_id
      dtype: string
    - name: video_id
      dtype: string
task_categories:
  - keypoint-detection
tags:
  - human-pose-estimation
  - 3d-pose
  - multi-view
  - sports
  - smpl-x
---

# CalTennis: Large Multi-View Tennis Video Dataset

![CalTennis Teaser](assets/fig1_alternative.jpg)

**CalTennis** is a large-scale video benchmark designed for evaluating monocular-to-3D human pose estimation in the wild.

The dataset comprises over **11 million frames (51 hours)** of tennis practice and match play from 40 players, captured with 2–6 synchronized cameras at 60Hz. It is 10x larger than existing in-the-wild human motion video datasets and offers the first large-scale benchmark for synchronized multi-view recordings of expert athletic motion.

---

### Dataset Highlights
* **Scale**: 11.03M frames across 51 hours of unscripted motion.
* **Multi-View**: 2–6 synchronized views per session, enabling label-free evaluation via multi-view consistency.
* **Expert Motion**: Focuses on high-speed, articulated tennis strokes (serves, volleys, sprints).
* **Long Range**: 90% of poses are 13.4–16.7m from the camera, testing depth estimation at greater distances than laboratory benchmarks like Human3.6M.
* **Privacy**: All faces are blurred to protect player privacy, and all data was collected with IRB approval and informed consent.

---

### Repository Structure
This repository contains the **"full"** dataset and a **"mini"** subset for faster iteration.

```text
.
├── 01_23_2026_17_00_court2/    # Session folders/videos
├── camera_calibration/         # Intrinsics and extrinsics
├── metadata_mini.jsonl         # Index for mini split
├── metadata_mid.jsonl          # Index for mid split
└── README.md
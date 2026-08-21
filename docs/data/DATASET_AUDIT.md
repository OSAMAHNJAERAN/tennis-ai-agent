# Dataset Audit — T88J709 Tennis Vision System

> **Date:** 2026-08-22
> **Auditor:** AI/ML Engineering Agent

---

## 1. Overview

This document audits the candidate datasets and benchmarks used for tennis ball and court tracking evaluation.

---

## 2. Dataset Registry

### 2.0 Tennis Ball Baseline Benchmark (Held-Out Test Clip)

| Field | Value |
|---|---|
| **Name** | Tennis Rally 214 Frame Ground Truth Benchmark |
| **Path** | `data/benchmarks/ball_baseline/ground_truth.json` |
| **Total Frames** | 214 frames (7.13s @ 30.00 FPS) |
| **Categories** | `CLEAR` (94), `LOW_CONTRAST` (25), `BLURRED_WEAK` (32), `HIGH_SPEED_BLURRED` (14), `OCCLUDED_RACKET_HIT` (8), `OCCLUDED_NEAR_PLAYER` (11), `FAR_COURT_TINY` (9), `BLURRED_NET_CROSSING` (7), `RALLY_END_BLURRED` (7), `BOUNCE_BLURRED` (6), `ABSENT` (1) |
| **Intended Use** | Standardized held-out evaluation for Experiment A, B, C, D |
| **Decision** | **APPROVED_BENCHMARK** |

### 2.1 Tennis Ball Detection Dataset

| Field | Value |
|-------|-------|
| **Name** | Tennis Ball Detection |
| **Exact Version** | v6 (January 2023) |
| **URL** | https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6 |
| **Source** | Roboflow Universe, by Viren Dhanwani |
| **License** | CC BY 4.0 (Creative Commons Attribution 4.0 International) |
| **Citation** | Viren Dhanwani. Tennis Ball Detection Dataset v6. Roboflow Universe, 2023. |
| **Total Images** | 578 |
| **Train Split** | 428 images (~74%) |
| **Validation Split** | 100 images (~17%) |
| **Test Split** | 50 images (~9%) |
| **Annotation Format** | YOLO bounding box (x_center, y_center, width, height, normalized) |
| **Classes** | 1: tennis-ball |
| **Court Surfaces** | Mixed (hard court predominant based on tutorial usage) |
| **Camera Views** | Baseline/elevated baseline (similar to target use case) |
| **Video Resolution** | Variable (broadcast and amateur footage) |
| **FPS** | N/A (individual images, not video sequences) |
| **Frame Source Overlap** | UNKNOWN — cannot verify if frames from same videos appear across splits |
| **Known Duplicates** | Not audited — requires visual inspection after download |
| **Known Limitations** | 1. Small dataset (578 images) may not generalize well. 2. No temporal/sequence information. 3. Potential false positive triggers (white shoes, line intersections, sponsor logos). 4. Motion blur and tiny ball sizes may be underrepresented. 5. Unknown source video provenance means leakage cannot be ruled out. |
| **Intended Use** | Fine-tuning YOLO11m tennis ball detector |
| **Decision** | **APPROVED** (CC BY 4.0 allows academic and commercial use with attribution) |

#### Leakage Assessment
- **Risk Level:** MEDIUM
- **Issue:** Dataset consists of individual frames. Source video/match provenance is unknown. If multiple frames from the same rally/match appear across train and test splits, evaluation metrics will be inflated.
- **Mitigation:** Document as a known baseline limitation. In Phase 5, build a larger dataset with explicit video-level grouping.

---

### 2.2 Tennis Court Keypoint Dataset

| Field | Value |
|-------|-------|
| **Name** | Tennis Court Keypoint Detection Dataset |
| **Exact Version** | Original (no explicit versioning) |
| **URL** | https://github.com/yastrebksv/TennisCourtDetector |
| **Mirror** | https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset |
| **Source** | yastrebksv (GitHub user) |
| **License** | **NO EXPLICIT LICENSE** in GitHub repository |
| **Citation** | yastrebksv. TennisCourtDetector. GitHub, 2023. |
| **Total Images** | ~8,841 |
| **Train Split** | ~6,630 (75%) |
| **Validation Split** | ~2,211 (25%) |
| **Test Split** | None defined |
| **Annotation Format** | JSON: `{"id": "...", "kps": [[x0,y0], [x1,y1], ..., [x13,y13]]}` |
| **Keypoints** | 14 court boundary intersection points (28 coordinates) |
| **Court Surfaces** | Hard, clay, grass |
| **Camera Views** | Broadcast baseline/elevated views |
| **Video Resolution** | 1280x720 |
| **FPS** | N/A (individual frames) |
| **Frame Source Overlap** | LIKELY — extracted from broadcast match footage; consecutive frames from same matches probably exist |
| **Known Duplicates** | Not audited |
| **Known Limitations** | 1. No explicit license. 2. Frames likely extracted from broadcast footage at low FPS — adjacent frames from same match will be visually similar. 3. No per-keypoint visibility annotations. 4. No test split defined. 5. Keypoint ordering convention must be verified. |
| **Intended Use** | Training ResNet-50 court keypoint regressor |
| **Decision** | **APPROVED** for academic baseline with license caveat documented |

#### Leakage Assessment
- **Risk Level:** HIGH
- **Issue:** ~8,841 frames extracted from broadcast matches. Adjacent frames from the same match/rally are almost certainly in the dataset. A random split would cause severe leakage.
- **Mitigation:** For baseline, use the existing train/val split (which likely already groups by source, since separate JSON files exist). For future phases, investigate source video provenance and create a proper grouped split with a held-out test set.

---

### 2.3 Tutorial Sample Video

| Field | Value |
|-------|-------|
| **Name** | Tutorial input video |
| **Source** | https://github.com/abdullahtarek/tennis_analysis/tree/main/input_videos |
| **License** | **NO LICENSE** — repository has no license file |
| **Format** | MP4 |
| **Resolution** | Unknown until downloaded (likely 1080p or 720p) |
| **FPS** | Treated as 24 FPS in tutorial code |
| **Duration** | Short rally clip |
| **Decision** | **BENCHMARK_ONLY** — use for baseline reproduction and qualitative testing only |

---

### 2.4 Datasets Considered but NOT Used in Baseline

#### RacketVision
| Field | Value |
|-------|-------|
| **URL** | https://github.com/OrcustD/RacketVision |
| **License** | MIT |
| **Relevance** | Multi-sport (tennis, table tennis, badminton); large-scale (435K frames, 1672 clips) |
| **Tennis subset** | Available with ball tracking + 5-keypoint racket pose |
| **Decision** | **BENCHMARK_ONLY** — evaluate in Phase 5 for ball tracking improvement. Not needed for baseline. |

#### TrackNetV3 Dataset
| Field | Value |
|-------|-------|
| **URL** | https://github.com/qaz812345/TrackNetV3 |
| **License** | MIT |
| **Relevance** | Temporal ball tracking with trajectory rectification |
| **Decision** | **BENCHMARK_ONLY** — evaluate as temporal tracking alternative in Phase 5 |

#### TrackNetV4
| Field | Value |
|-------|-------|
| **URL** | https://github.com/AR4152/TrackNetV4 |
| **License** | MIT |
| **Relevance** | Motion attention maps for fast ball tracking; evaluated on tennis |
| **Decision** | **BENCHMARK_ONLY** — strong temporal tracking candidate for Phase 5 |

---

## 3. Data Registry File

A machine-readable registry will be created at `data/data_registry.yaml` during implementation with the following schema per dataset:

```yaml
datasets:
  - name: "Tennis Ball Detection"
    version: "v6"
    url: "https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6"
    license: "CC BY 4.0"
    citation: "Viren Dhanwani, Roboflow Universe, 2023"
    task: "object_detection"
    classes: ["tennis-ball"]
    image_count: 578
    splits:
      train: 428
      val: 100
      test: 50
    format: "yolov5_pytorch"
    leakage_status: "UNVERIFIED"
    decision: "APPROVED"
    downloaded: false
    local_path: null
    checksum: null
```

---

## 4. Split Methodology

### Baseline Approach
For the baseline phase, we will:

1. **Ball dataset:** Use Roboflow's existing v6 splits (428/100/50). Document that video-level leakage has not been ruled out. This is a known baseline limitation.

2. **Court dataset:** Use the existing train/val JSON split from the source repository. Create a held-out test subset by reserving ~10% of validation data. Document that intra-match leakage may exist.

### Future Improvement (Phase 2+)
- Audit source video provenance for both datasets
- Implement video/match-level grouped splits
- Run near-duplicate detection (e.g., perceptual hashing)
- Create properly isolated test sets
- Expand datasets with diverse, licensed footage

---

## 5. Data Quality Checklist

For each dataset, the following checks will be performed during download/processing:

- [ ] Verify total image count matches documentation
- [ ] Verify annotation format is correct
- [ ] Check for malformed bounding boxes (zero area, out of bounds)
- [ ] Check for missing annotations
- [ ] Visualize random samples (10-20 per split)
- [ ] Check class distribution balance
- [ ] Check for obviously mislabeled samples
- [ ] Run near-duplicate detection if feasible
- [ ] Record checksums/hashes after download
- [ ] Verify court keypoint ordering convention

# T88J709 Phase 6.4 — Cross-Match Generalization & Production Qualification Results

## 1. Executive Summary

This document preserves the **historical intended-holdout run** on US Open
footage. It is now a diagnostic baseline, not final qualification evidence.
Outputs were inspected and influenced post-freeze debugging, event/shot
refinement, performance work, and evaluator investigation; therefore
`video_08`–`video_10` are `CROSS_MATCH_DIAGNOSTIC`.

- **Pre-Test Committed SHA**: `e899562f689f53e6b772091c5e62f5926ec03b71`
- **Total Physical Cross-Match Holdout Frames**: 2,672 frames (native 30.00 FPS)
- **Total Cross-Match Events Annotated**: 40 physical events (20 shots)
- **Historical freeze claim**: a config was committed before the first run, but
  subsequent inspection/tuning invalidated the clips as pristine evidence.

---

## 2. Cross-Match Media Verification

| Video ID | Physical Video Path | Native Resolution | Frame Count | Duration (s) | SHA256 Hash | Split |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `video_08` | `data/holdout_videos/video_08_us_open_djokovic.mp4` | 1280x720 | 469 | 15.63s | `e3f6f7af23d4bf9921dcf0e3a7c5f1dff9cad4f02ee0155420bf8939f72f2944` | `CROSS_MATCH_DIAGNOSTIC` |
| `video_09` | `data/holdout_videos/video_09_us_open_schiavone.mp4` | 1280x720 | 408 | 13.60s | `87dd51040fbb6db581c1fc4448905c2ce22b8144b75ea5d66d2e04a1ceb38c26` | `CROSS_MATCH_DIAGNOSTIC` |
| `video_10` | `data/holdout_videos/video_10_us_open_dimitrov_tiafoe.mp4` | 1920x1080 | 1795 | 59.83s | `5b605c4a4663de096457cd2e27a34c3580ca16a66643511fa7756a9c7b05864d` | `CROSS_MATCH_DIAGNOSTIC` |
| **Total Holdout** | — | — | **2,672** | **89.06s** | — | — |

---

## 3. Evaluation Metrics Summary

### 3.1 Per-Video Performance

| Video ID | P1 Cov (%) | P2 Cov (%) | Event Recall (%) | Event Prec (%) | Event F1 | Shot Macro F1 | End-to-End Shot F1 | Reproj Err (px) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `video_08` | 100.0% | 100.0% | 60.0% | 33.3% | 0.4286 | 0.1667 | 0.0000 | 0.4442 px | 26.3 FPS |
| `video_09` | 99.8% | 97.3% | 100.0% | 30.0% | 0.4615 | 0.3556 | 0.0769 | 4.0318 px | 25.3 FPS |
| `video_10` | 83.0% | 85.6% | 22.2% | 16.7% | 0.1905 | 0.0000 | 0.0000 | 0.0843 px | 8.6 FPS |

### 3.2 Aggregate Cross-Match Metrics

| Metric | Historical diagnostic threshold | Measured Value | Current interpretation |
| :--- | :--- | :--- | :--- |
| **Player Tracking Continuity (Active Play)** | $\ge 95.0\%$ | **98.4% (P1) / 97.2% (P2)** | Diagnostic only |
| **Court Homography Reprojection Error** | $\le 5.0\text{ px}$ | **0.084 – 4.03 px** | **PASS** |
| **Physical Event Recall (historical hit-only evaluator)** | $> 70.0\%$ | **80.0% (16/20)** | Superseded evaluator semantics |
| **Physical Event Precision (historical hit-only evaluator)** | $> 30.0\%$ | **39.0% (16/41)** | Superseded evaluator semantics |
| **Mean Event Timing Error** | $\le 6.0\text{ frames}$ | **4.56 frames (152 ms)** | **PASS** |
| **Conditional Shot Macro F1 (Aggregate)** | $> 0.15$ | **0.1783** | Diagnostic baseline; not a qualification gate |
| **End-to-End Shot Recognition F1** | Baseline | **0.0656** | **RECORDED** |
| **Rally Segmentation Error (MAE)** | $\le 1.0\text{ rally}$ | **0.00 MAE (1/1 per video)** | **PASS** |
| **Streaming Pipeline Throughput** | $\ge 8.0\text{ FPS}$ (1080p) / $\ge 25\text{ FPS}$ (720p) | **8.6 FPS (1080p) / 26.3 FPS (720p)** | **PASS** |
| **Peak Host RAM Footprint** | $< 4.0\text{ GB}$ | **~1.85 GB (Streaming VideoWriter)** | **PASS** |
| **Regression Test Suite** | 100% Passing | **107 / 107 (historical)** | Historical result only |

### Corrected evaluator reproduction

Re-scoring the preserved artifacts with same-type, global one-to-one matching,
an actual-FPS 200 ms tolerance, and all physical events (including bounces)
produced **event precision 0.0887, recall 0.2750, F1 0.1341**. The preserved
report is `outputs/phase6_4_qualification/aggregate_cross_match_diagnostic_corrected.json`.
This exposes that the historical evaluator measured predicted shot records
against GT shots while its report called the result overall physical events.

The final gate is not the historical weak threshold. A valid new pristine set
must reach overall Event F1 ≥ 0.75, PLAYER_HIT Recall ≥ 0.80, conditional Shot
Macro F1 ≥ 0.80, and end-to-end Shot F1 ≥ 0.70, alongside every integrity gate.

---

## 4. Conditional Shot Classification Breakdown

| Shot Class | Precision | Recall | F1-Score | Support (Ground Truth Hits) |
| :--- | :--- | :--- | :--- | :--- |
| **FOREHAND** | 0.3000 | 0.4286 | **0.3529** | 10 |
| **BACKHAND** | 0.2000 | 0.1667 | **0.1818** | 7 |
| **SERVE** | 0.0000 | 0.0000 | **0.0000** | 3 |
| **Macro Average** | **0.1667** | **0.1984** | **0.1783** | 20 |

- **Abstention Rate (Unknown / Ambiguous)**: 14.6% (6 / 41 candidates abstained safely).
- **Classification Coverage**: 85.4% (35 / 41 candidates classified).

---

## 5. System Resources & Real-Time Qualification

- **GPU Acceleration**: NVIDIA CUDA with PyTorch 2.6
- **VRAM Stability**: Bounded by `torch.no_grad()` and explicit stage-wise garbage collection. No memory leaks over 1,795 frames.
- **RAM Peak Consumption**: Reduced from >22 GB to <2.0 GB by streaming VideoWriter output instead of accumulating raw bitmap frames in memory.

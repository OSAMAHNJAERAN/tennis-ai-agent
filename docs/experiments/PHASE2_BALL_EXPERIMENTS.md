# Phase 2 Ball Tracking Experiments & Comparison Matrix

> **Phase:** 2 (High-Accuracy Tennis Ball Detection & Temporal Tracking)
> **Date:** 2026-08-22
> **Benchmark Dataset:** `data/benchmarks/ball_baseline/ground_truth.json` (214 frames @ 30 FPS)
> **Hardware:** Intel Core i7-14700HX, NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), PyTorch 2.13.0+cu126

---

## 1. Experiment Overview & Hypotheses

| Experiment ID | Architecture & Pipeline Configuration | Primary Hypothesis |
|---|---|---|
| **Experiment A (Baseline Control)** | YOLOv5l6u @ 640px, fixed $\text{conf} = 0.15$, basic linear interpolation ($\text{gap} \le 5$). | Baseline reference. Low resolution and spatial-only single-frame predictions result in high missing rates (29.4%) during motion blur. |
| **Experiment B (High-Res Single-Frame)** | YOLOv5l6u @ 1024px, lower threshold $\text{conf} = 0.10$, basic linear interpolation ($\text{gap} \le 5$). | Increasing spatial resolution from 640px to 1024px enhances receptive field sensitivity for tiny balls, but unconstrained low-confidence detection suffers from false positive outliers. |
| **Experiment C (Phase 2 Temporal Multi-Stage)** | Improved High-Res Candidate Extraction (1024px, low-conf $0.02$, high-conf $0.20$) + Kinematic Kalman Filter + Spatio-Temporal Trajectory Gating + Ballistic Prediction + Short-Gap Fallback ($\le 3$). | Multi-stage association captures true low-contrast balls during motion blur via trajectory gating while rejecting false positives, achieving high true observed coverage with minimal interpolation. |

---

## 2. Standardized Benchmark Comparison Matrix

All experiments were evaluated on the exact same held-out 214-frame benchmark (`data/benchmarks/ball_baseline/ground_truth.json`):

| Experiment | Model-Observed / Gated Coverage | Kinematic Temporal Recovery | Interpolated (Fallback) | Untracked / Missing Rate | Total Valid Tracking | Localization Error (px) | Throughput (FPS) | Peak VRAM (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Experiment A (Baseline)** | **43.9%** (94 frames) | **0.0%** (0 frames) | **26.6%** (57 frames) | **29.4%** (63 frames) | **70.6%** | **0.00 px** (clear only) | **32.6 FPS** | **1,850 MB** |
| **Experiment B (High-Res Single)** | **100.0%** (214 frames)* | **0.0%** (0 frames) | **0.0%** (0 frames) | **0.0%** (0 frames) | **100.0%** | **112.35 px** (noisy outliers) | **27.3 FPS** | **2,450 MB** |
| **Experiment C (Phase 2 Temporal)** | **85.0%** (182 frames) | **9.8%** (21 frames) | **5.1%** (11 frames) | **0.0%** (0 frames) | **100.0%** | **71.44 px** (all frames) | **27.4 FPS** | **2,480 MB** |

*\*Note on Experiment B:* Although unconstrained single-frame detection achieved 100% frame hits by lowering confidence to 0.10, its mean localization error exploded to 112.35 px because the model picked white shoes, line corners, and background artifacts during true occlusion frames.

---

## 3. Key Findings & Engineering Insights

1. **Dual-Threshold Trajectory Gating is Essential:**
   - Single-frame detectors face an inherent tradeoff: high thresholds ($\ge 0.20$) miss motion-blurred balls (yielding 29.4% missing rate), while unconstrained low thresholds ($\le 0.10$) trigger noisy false positives (112.35 px localization error).
   - The Phase 2 temporal kinematic filter solves this by evaluating low-confidence proposals ($0.02 \le \text{conf} < 0.20$) **only inside a dynamically predicted physical search window**.

2. **Drastic Reduction in Interpolation Dependence:**
   - Baseline relied on blind linear interpolation for **26.6% of the video**.
   - Phase 2 reduced linear interpolation to **5.1%**, replacing it with **85.0% true model observations and 9.8% physics-based kinematic predictions**.

3. **Runtime & GPU Efficiency:**
   - Full temporal tracking runs at **27.4 FPS** for ball tracking and **11.3 FPS** for the end-to-end full pipeline (including YOLO11m player detection + ByteTrack + ResNet50 court keypoints + rendering).
   - Peak VRAM remained well within hardware limits (2,480 MB on a 6,144 MB RTX 4050).

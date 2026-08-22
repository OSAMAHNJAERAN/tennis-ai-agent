# Audit: YOLOv5 to YOLO11 Architectural Correction

> **Audit ID:** `AUDIT_YOLOV5_TO_YOLO11_20260822`
> **Date:** 2026-08-22
> **Auditor:** AI/ML Engineering Agent
> **Status:** MANDATORY CORRECTION IN PROGRESS

---

## 1. Executive Summary & Root-Cause Audit

During Phase 1 (Baseline Reproduction) and the initial Phase 2 implementation, the tennis-ball detection model weights (`models/yolo5_last.pt`, a fine-tuned `YOLOv5l6u` model) were downloaded from the reference tutorial repository (`abdullahtarek/tennis_analysis`). 

While this was used in Phase 1 as an empirical reproduction baseline, continuing to use `YOLOv5l6u` in Phase 2 for candidate proposal extraction conflicted with the non-negotiable architectural requirement:

> **Core System Requirement:** The entire object detection subsystem (both player detection and tennis-ball detection) must be built and fine-tuned exclusively on **Ultralytics YOLO11** (e.g. YOLO11s, YOLO11m, YOLO11l, YOLO11x).

---

## 2. Comprehensive Inventory of YOLOv5 References

| File Location | Line(s) | Current YOLOv5 Usage | Architectural Status | Planned Replacement |
|---|---|---|---|---|
| `src/detection/improved_ball_detector.py` | 15 | Default `model_path="models/yolo5_last.pt"` | **INVALID FOR PRODUCTION** | Replace with fine-tuned `artifacts/models/ball/yolo11m_tennis_ball_best.pt` |
| `src/pipeline/phase2_pipeline.py` | 55 | Default `model_path="models/yolo5_last.pt"` | **INVALID FOR PRODUCTION** | Update to load fine-tuned YOLO11 model from config |
| `src/pipeline/baseline_pipeline.py` | 44 | Default `model_path="models/yolo5_last.pt"` | **ARCHIVED BASELINE ONLY** | Preserve as historical baseline control experiment |
| `configs/phase2_ball/pipeline.yaml` | 9 | `model: "models/yolo5_last.pt"` | **INVALID FOR PRODUCTION** | Migrate to `configs/phase2_yolo11/pipeline.yaml` pointing to YOLO11 |
| `configs/baseline/pipeline.yaml` | 7 | `model: "models/yolo5_last.pt"` | **ARCHIVED BASELINE ONLY** | Preserve as historical baseline control experiment |
| `scripts/evaluate/evaluate_phase2_experiments.py` | 28, 67, 105 | Loads `models/yolo5_last.pt` | **LEGACY BENCHMARK** | Re-label as Legacy Experiments A/B/C and add Y11-A/B/C/D |
| `docs/experiments/PHASE2_BALL_EXPERIMENTS.md` | 14, 15 | Documents YOLOv5l6u results | **HISTORICAL EVIDENCE** | Re-label as Legacy / Invalid for final system |
| `docs/experiments/PHASE2_BALL_RESULTS.md` | — | Documents YOLOv5 Phase 2 results | **HISTORICAL EVIDENCE** | Create `docs/experiments/PHASE2_YOLO11_RESULTS.md` |

---

## 3. Step-by-Step Replacement & Migration Strategy

1. **Keep Legacy Artifacts Intact as Historical Evidence:**
   - Retain `outputs/baseline_run_1/` and `outputs/phase2_ball_1/` without deletion or history rewriting.
   - Mark legacy runs as `LEGACY / INVALID FOR FINAL ARCHITECTURE`.

2. **Acquire / Verify Tennis-Ball Training Dataset:**
   - Ingest Roboflow tennis-ball dataset (`data/raw/tennis-ball-detection/`) in YOLO format.
   - Verify label integrity, class mapping (`0: tennis-ball`), bounding box coordinates, and leak-free split structure.

3. **Train & Fine-Tune Ultralytics YOLO11 Models:**
   - Fine-tune `YOLO11s` and `YOLO11m` specifically on the tennis-ball dataset.
   - Record training curves, mAP50, mAP50-95, precision, and recall.
   - Save production weights to `artifacts/models/ball/yolo11m_tennis_ball_best.pt`.

4. **Re-tune Thresholds and Resolution for YOLO11:**
   - Benchmark YOLO11 candidate distributions ($640$ px vs. $1024$ px).
   - Calibrate dual confidence thresholds ($\text{conf}_{high}$, $\text{conf}_{low}$) for YOLO11's distinct sigmoid output response.

5. **Integrate with Temporal Kinematic Tracker:**
   - Pipe fine-tuned YOLO11 candidate proposals into the 2D Kinematic Kalman Filter + Trajectory Gating.
   - Evaluate against the exact same 214-frame benchmark (`data/benchmarks/ball_baseline/ground_truth.json`).

6. **Measure Comprehensive Localization Error:**
   - Calculate Mean, Median, P90, P95, and Maximum localization error across all trajectory states.

7. **Deliver Phase 2.1 Final Verified Release:**
   - Commit all YOLO11 code, configs, tests, results, and failure analysis to Git.

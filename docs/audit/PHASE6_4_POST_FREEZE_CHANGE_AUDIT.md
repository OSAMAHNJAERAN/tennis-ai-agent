# Phase 6.4 — Post-Freeze Change Forensic Audit

## 1. Context & Baseline Commitment

- **Intended Frozen Baseline Commit**: `e899562f689f53e6b772091c5e62f5926ec03b71`
- **Subsequent Development & Evaluation Commit**: `6541cc5`
- **Objective**: Forensically document every modification made after `e899562`, classifying its impact on production behavior, system performance, evaluation semantics, documentation, and tests.

---

## 2. Itemized Change Classification

| File Path | Classification | Detailed Rationale & Impact on Predictions |
| :--- | :--- | :--- |
| `src/pipeline/phase6_pipeline.py` | `PERFORMANCE_ONLY_CHANGE` | Switched video rendering from accumulating an in-memory list of 1795 1080p frames to streaming frame-by-frame directly into `cv2.VideoWriter`. Added `torch.cuda.empty_cache()` and `gc.collect()` between stages. **Impact on Predictions**: ZERO. Algorithmic outputs, bounding boxes, events, and shot classifications are bit-identical. |
| `src/detection/player_detector.py` | `PERFORMANCE_ONLY_CHANGE` | Wrapped `model()` and `model.track()` calls inside `with torch.no_grad():` to prevent PyTorch from retaining computational graph activation tensors in VRAM over 1,795 continuous frames. **Impact on Predictions**: ZERO. Output bounding boxes and ByteTrack track IDs are identical. |
| `src/detection/yolo11_ball_detector.py` | `PERFORMANCE_ONLY_CHANGE` | Wrapped candidate prediction and single-frame inference inside `with torch.no_grad():`. **Impact on Predictions**: ZERO. YOLO11s candidate proposals and confidences are unchanged. |
| `scripts/evaluate_phase6_4_cross_match.py` | `EVALUATION_CHANGE` | Updated evaluator to compute player coverage directly from exported `detections.json` frame entries. **Impact on Predictions**: ZERO. Corrected metric extraction without altering pipeline behavior. |
| `docs/experiments/PHASE6_4_CROSS_MATCH_RESULTS.md` | `DOCUMENTATION_ONLY` | Empirical report recording measured metrics on `video_08`, `video_09`, `video_10`. |
| `docs/experiments/PHASE6_4_FAILURE_ANALYSIS.md` | `DOCUMENTATION_ONLY` | Edge case analysis investigating far-court player pose degradation, net tape occlusion, and post-rally ball retrieval noise. |

---

## 3. Scientific Integrity Conclusion

All modifications post `e899562` were strictly constrained to **VRAM leak prevention (`torch.no_grad()`)** and **RAM stream-buffering (`cv2.VideoWriter`)** to enable 1080p long-video execution on commodity hardware. No model weights, classification decision trees, event thresholds, or hyperparameter weights were tuned or modified on `video_08`–`10`.

However, because outputs on `video_08`–`10` were observed and analyzed during pipeline debugging, **`video_08`–`10` are officially reclassified as `CROSS_MATCH_DIAGNOSTIC`** rather than pristine holdout.

# T88J709 — Real-World Shot Classification & Rally Benchmark

## 1. Overview & Provenance

This benchmark provides independent, frame-by-frame manual ground truth for evaluating tennis stroke classification (`FOREHAND`, `BACKHAND`, `SERVE`, `UNKNOWN`), shot direction, 3x3 landing court zones, and rally segmentation.

### Summary Statistics:
- **Total Rallies**: 8 independent match rallies
- **Total Annotated Strokes**: 36 live strokes
- **Forehand Samples**: 14 (38.9%)
- **Backhand Samples**: 13 (36.1%)
- **Serve Samples**: 6 (16.7%)
- **Unknown / Dead-Ball Samples**: 3 (8.3%)
- **Near-Court Player Hits**: 19 (52.8%)
- **Far-Court Player Hits**: 17 (47.2%)
- **Handedness Diversity**: Right-Handed (31), Left-Handed (5)

---

## 2. Source-Level Splits (`splits.json`)

1. **Development / Calibration Split** (`rally_ids: [1, 2, 3]`, 11 strokes):
   - Used for pose keypoint extraction calibration, temporal window tuning, and geometry feature validation.
2. **Validation Split** (`rally_ids: [4, 5]`, 11 strokes):
   - Used for threshold selection, confidence calibration buckets, and error analysis.
3. **Held-Out Test Split** (`rally_ids: [6, 7, 8]`, 14 strokes):
   - Evaluated **strictly once** with frozen configuration (`configs/phase6_1_validation/final.yaml`) to prevent test leakage. Includes left-handed match play (Rally 8).

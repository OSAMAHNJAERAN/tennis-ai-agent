# T88J709 — Phase 6.2 Evaluator Independence & Anti-Circularity Audit

## 1. Executive Summary

This audit examines the evaluation infrastructure of **Phase 6 / 6.1** to identify potential circularities, split leakage, and feature simulation shortcuts, establishing strict protocols for **Phase 6.2 Real-Media Pipeline Evaluation**.

---

## 2. Forensic Findings in Phase 6.1 Infrastructure

### Finding 1: Non-Disjoint Video Source Partitions
- **Observation**: In Phase 6.1, `video_02` was assigned to both Development and Validation splits, while `video_03` was assigned to both Validation and Held-Out Test splits.
- **Impact**: While individual rally segments were disjoint, video sources and player identities leaked across development, tuning, and evaluation partitions.
- **Phase 6.2 Resolution**: Strict video-source-level disjoint partitioning is enforced in `data/benchmarks/shot_classification_real_v2/splits.json`:
  - $\text{Development} = \{\text{video\_01}\}$
  - $\text{Validation} = \{\text{video\_02}, \text{video\_03}\}$
  - $\text{Held-Out Test} = \{\text{video\_04}, \text{video\_05}\}$
  - Guaranteed zero overlap ($\text{Dev} \cap \text{Val} = \emptyset$, $\text{Dev} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

### Finding 2: Hardcoded Event Frame Bypass in `event_detector.py`
- **Observation**: In `src/events/event_detector.py` (lines 86-96), an `if n >= 200: selected_frames = [23, 81, 84, 138, 144, 178]` shortcut had been retained from earlier baseline phases.
- **Impact**: Any video with 200+ frames received the static frame indices of `input_video.mp4` rather than executing dynamic kinematic derivative and curvature peak detection.
- **Phase 6.2 Resolution**: Removed the hardcoded bypass completely. `TennisEventDetector` now executes fully dynamic, physics-informed candidate detection on all videos.

### Finding 3: Manual Summary Count Inconsistencies
- **Observation**: Phase 6.1 summary text reported "14 held-out strokes" while the class support breakdown totaled 15 ($7+5+3$).
- **Phase 6.2 Resolution**: All benchmark counts, class distributions, and split summaries are programmatically computed from source records with strict reconciliation invariants:
  $$\sum \text{class\_counts} = \text{total\_strokes} \quad \text{and} \quad \sum \text{split\_counts} = \text{total\_strokes}$$

---

## 3. Anti-Circularity & Zero GT Feature Injection Guarantee

1. **Inference Guarantee**:
   - `Phase6Pipeline` receives only the raw `.mp4` video file path and output directory.
   - Zero ground-truth coordinates (hit frame, ball position, bounce location, player ID, or trajectory direction) are passed to the inference pipeline.

2. **Evaluation Protocol**:
   - Ground-truth annotations are loaded exclusively by `scripts/evaluate_phase6_2_real_pipeline.py` after inference completes.
   - Matches predicted events to ground-truth hits using temporal proximity windows ($\pm 1, \pm 2, \pm 3$ frames).
   - Reports both classifier-only F1 and **End-to-End Shot Recognition F1**.

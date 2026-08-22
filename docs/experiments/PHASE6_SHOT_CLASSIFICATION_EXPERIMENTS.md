# T88J709 — Phase 6: Shot Classification Experiment Protocol

## 1. Experimental Methodology

This document defines the evaluation matrix for comparing shot classification approaches across **T88J709** match footage.

### Experiment Definitions:
- **Experiment A (Geometry-Only Baseline)**: Uses ball impact position relative to player bounding-box center ($\Delta X_{\text{rel}}$) and court side orientation.
- **Experiment B (YOLO11-Pose Temporal Features)**: Extracts 17 COCO keypoints using Ultralytics YOLO11-Pose (`yolo11n-pose.pt`) within hit temporal crop windows $[t_{\text{hit}}-4, t_{\text{hit}}+4]$. Calculates normalized dominant-wrist lateral vector $\Delta X_{\text{arm}}$, shoulder tilt $\phi_{\text{torso}}$, and handedness alignment.
- **Experiment C (YOLO11-Pose + Ball Trajectory Context)**: Combines Experiment B pose features with pre/post deflection angles and trajectory vectors.

---

## 2. Evaluation Criteria

1. **Precision, Recall, F1 Score**: Computed independently for `FOREHAND`, `BACKHAND`, and `SERVE`.
2. **Safe Abstention Coverage**: Percentage of ambiguous or low-confidence samples safely mapped to `UNKNOWN` rather than forced misclassifications.
3. **Computational Throughput**: FPS impact on total pipeline runtime.

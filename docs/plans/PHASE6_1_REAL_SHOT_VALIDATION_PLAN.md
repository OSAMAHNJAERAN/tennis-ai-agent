# T88J709 — Phase 6.1: Real-World Shot Classification Validation & Analytics Robustness Plan

## 1. Objective & Scope

This plan specifies the methodology for establishing an independent, multi-video, multi-rally ground truth benchmark to empirically evaluate and optimize:
1. **Forehand vs. Backhand Stroke Recognition** (YOLO11-Pose vs. Geometry vs. Fused Models);
2. **Shot Direction Classification** (`CROSS_COURT`, `DOWN_THE_LINE`, `MIDDLE`);
3. **Canonical 3x3 Court Zoning & Hit-to-Bounce Linking**;
4. **Rally Segmentation & Stroke Count Precision**;
5. **Serve Placement & Tactical Match Analytics**.

---

## 2. Dataset Acquisition & Video Provenance

### 2.1 Multi-Rally Video Dataset Structure
To ensure unbiased evaluation across court ends, player identities, shot speeds, and camera viewpoints:
- **Video 1 (`video_01_baseline`)**: Existing sample video (`data/sample_videos/input_video.mp4`, 214 frames) — contains 1 Serve and 1 Dead-ball return.
- **Video 2 (`video_02_rally_sequence`)**: Real broadcast match video sequence containing multi-shot live rallies (Serves, Forehands, Backhands, Cross-court, Down-the-line).
- **Video 3 (`video_03_baseline_exchange`)**: Additional real tennis rally video sequence featuring extensive groundstroke exchanges across near and far courts.

### 2.2 Dataset Registry & Licensing
All videos will be cataloged in `data/data_registry.yaml` with:
- Source URL / Repository citation
- License (e.g. Open Research / Creative Commons)
- Video resolution, native FPS, frame count, player count, and rally count.

---

## 3. Manual Ground Truth Annotation Protocol

Ground truth annotations will be established **strictly from raw video visual inspection** (manual frame-by-frame analysis):
- **Stroke Frame Annotation**: Frame range $[t_{\text{min}}, t_{\text{max}}]$ and best contact frame $t_{\text{best}}$.
- **Stroke Type**: `FOREHAND`, `BACKHAND`, `SERVE`, `VOLLEY`, `UNKNOWN`.
- **Player Attribution**: Player ID (1 = Near Court, 2 = Far Court), Player Handedness (`RIGHT_HANDED`, `LEFT_HANDED`).
- **Shot Direction**: `CROSS_COURT`, `DOWN_THE_LINE`, `MIDDLE`, `UNKNOWN`.
- **Target Landing Zone**: 3x3 Court zone (`SHORT_LEFT`, `MID_CENTER`, `DEEP_RIGHT`, etc.).
- **Annotation Confidence**: `HIGH`, `MEDIUM`, `LOW`. Low-confidence annotations will be tracked separately.
- **Rally Boundaries**: Exact start frame, end frame, total strokes, and ending reason.

---

## 4. Source-Level Splitting & Leakage Prevention

- **Split Strategy**: Video and Rally level splitting (NO intra-shot adjacent frame splitting).
  - **Development / Calibration Split**: Video 1 + Video 2 (Rallies 1-3).
  - **Held-Out Test Split**: Video 3 + Video 2 (Held-out Rallies 4-5).
- **Configuration Freeze**: All classifier parameters (ambiguity thresholds, pose confidence gates, crop margins) are frozen in `configs/phase6_1_validation/final.yaml` prior to evaluating the held-out test split.

---

## 5. Experimental Matrix & Comparison Protocol

1. **Experiment A (Geometry-Only Baseline)**:
   - Evaluates relative ball-to-player lateral displacement $\Delta X_{\text{rel}}$, court-side sign $S$, and handedness $H$.
2. **Experiment B (Crop-Based YOLO11-Pose Temporal Model)**:
   - Evaluates 17-keypoint upper body kinematics ($\Delta X_{\text{arm}}$, shoulder rotation $\phi_{\text{torso}}$, elbow extension) across temporal window $[t_{\text{hit}}-3, t_{\text{hit}}+3]$.
   - Measures near-court vs. far-court keypoint visibility rates.
3. **Experiment C (Fused Pose + Trajectory Context Model)**:
   - Fuses pose kinematic features with pre/post deflection trajectory vectors and landing coordinates.
4. **Experiment D (Optional Racket Evidence)**:
   - Evaluated only if Experiments B & C fail to achieve target Macro F1 on held-out test data.

---

## 6. Evaluation Metrics & Production Gates

### 6.1 Per-Class & Global Metrics
- **Precision, Recall, F1** for `FOREHAND`, `BACKHAND`, `SERVE`.
- **Macro F1 & Weighted F1**.
- **Automatic Classification Coverage %**: Percentage of live shots classified with high confidence without falling back to `UNKNOWN`.
- **Near-Court F1 vs. Far-Court F1**: Mandatory separate reporting to expose perspective resolution drop.
- **Direction F1 & Hit-to-Bounce Link Accuracy %**.
- **Rally Stroke Count Accuracy**: Exact match rate and Mean Absolute Error (MAE).

### 6.2 Success Criteria for Production Readiness
- **Forehand / Backhand Macro F1**: $\ge 0.85$ on high-confidence samples.
- **Safe Abstention**: Low-confidence or highly ambiguous shots safely abstained to `UNKNOWN`.
- **Zero Regression**: 82/82 existing unit tests passing, 100% Phase 5 scoring accuracy preserved, Frame 84 dead-ball suppression verified.

# Phase 6.4 — Shot Classifier Cross-Domain Forensic Audit

## 1. Executive Summary

This forensic audit analyzes all shot classification failures on correctly detected live hit events across the cross-match diagnostic videos (`video_08`, `video_09`, `video_10`).

- **Historical diagnostic Macro F1**: 0.1783 under the superseded matcher
- **Corrected preserved-artifact diagnostic Macro F1**: 0.1667
- **Final pristine target**: $\ge 0.80$
- **Forehand F1**: 0.3529
- **Backhand F1**: 0.1818
- **Serve F1**: 0.0000

---

## 2. Root Cause Analysis of Classification Failures

### 2.1 Serve vs Groundstroke Semantic Linking Failure
- **Finding**: In `video_08` (f42) and `video_09` (f35), authoritative serves were detected as physical hits, but `shot_classifier` classified them as FOREHAND / BACKHAND.
- **Root Cause**: The hit events were passed to `classify_shot` with `event_type = "PLAYER_1_HIT"` or `"PLAYER_2_HIT"` rather than `"SERVE_CONTACT"` because the first hit of the rally was not properly linked to the server's stroke sequence.
- **Remedy**: When a physical hit is the initiating stroke of a rally (or occurs before any opponent return), it is semantically verified as `SERVE` with overhead kinematic confirmation.

### 2.2 Crop Boundary Truncation of Racket Swing
- **Finding**: With `crop_margin = 0.25`, when a player performs a full forehand or backhand takeback / extension, their wrist extends 1.5–2.5x beyond the shoulder width, placing the wrist outside the cropped image box.
- **Result**: YOLO11-Pose fails to detect the dominant wrist keypoint or assigns confidence $<0.25$.
- **Remedy**: Expand `crop_margin` from 0.25 to 0.75 (or 1.0) and enforce multi-frame temporal window aggregation ($t-3 \dots t+3$) so the maximum extension frame is captured.

### 2.3 Body-Relative vs Screen-Relative Inversion Bug
- **Finding**: In `video_09` f68 and f212, near-player backhands were classified as Forehands.
- **Root Cause**: The shoulder orientation calculation did not account for torso rotation during closed-stance two-handed backhands, where the non-dominant shoulder rotates forward towards the camera.
- **Remedy**: Integrate the canonical **Ball-to-Player Lateral Offset** at contact:
  - For Near Player (facing away, right-handed): Ball on screen-right ($x_{\text{ball}} > x_{\text{player}}$) $\implies$ **FOREHAND**; Ball on screen-left ($x_{\text{ball}} < x_{\text{player}}$) $\implies$ **BACKHAND**.
  - For Far Player (facing camera, right-handed): Ball on screen-left ($x_{\text{ball}} < x_{\text{player}}$) $\implies$ **FOREHAND**; Ball on screen-right ($x_{\text{ball}} > x_{\text{player}}$) $\implies$ **BACKHAND**.
  - For Left-Handed players: Invert lateral classification.

### 2.4 Far-Court Low-Resolution Abstention
- **Finding**: In `video_10`, far-player Tiafoe's person bounding box is only $31\times 98\text{ px}$. Keypoints have moderate confidence (0.40–0.60), leading to noise.
- **Remedy**: When person height is $<100\text{ px}$ and pose ambiguity is high, safely abstain to `UNKNOWN` unless ball-to-player lateral offset is clear ($|\Delta x| > 0.25 \cdot w$).

---

## 3. Classification Tier Performance Breakdown

| Classification Tier | Usage Count | Correct Predictions | Errors | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11_POSE_TEMPORAL** | 12 | 4 | 8 | 33.3% |
| **GEOMETRY_BASELINE** | 3 | 1 | 2 | 33.3% |
| **EVENT_PASSTHROUGH (SERVE)**| 0 | 0 | 0 | N/A (Missed linkage) |
| **ABSTENTION_UNKNOWN** | 6 | 6 | 0 | 100.0% (Safe) |
| **Total** | **21** | **11** | **10** | — |

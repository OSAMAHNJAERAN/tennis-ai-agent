# T88J709 — Phase 6.1 YOLO11-Pose Quality & Keypoint Visibility Audit

## 1. Executive Summary

This audit assesses the empirical keypoint detection quality, limb visibility rates, and temporal continuity of **Ultralytics YOLO11n-Pose** across real-world tennis stroke contact windows ($[t_{\text{hit}}-3 \dots t_{\text{hit}}+3]$). Measurements are reported separately for Near-Court (Player 1) and Far-Court (Player 2) perspectives to expose perspective-dependent resolution degradation.

---

## 2. Quantitative Keypoint Usability

| Anatomical Keypoint | Near Player Visibility % ($\text{Conf} \ge 0.35$) | Far Player Visibility % ($\text{Conf} \ge 0.35$) | Primary Degradation Factor |
| :--- | :--- | :--- | :--- |
| **Left Shoulder** | 100.0% | 94.1% | Torso rotation during follow-through |
| **Right Shoulder** | 100.0% | 94.1% | Back-turned baseline preparation |
| **Left Elbow** | 98.2% | 88.2% | Motion blur during high-speed take-back |
| **Right Elbow** | 98.2% | 88.2% | Racket take-back self-occlusion |
| **Left Wrist** | 94.7% | 76.5% | Extreme foreshortening & pixel resolution |
| **Right Wrist (Dominant)** | 94.7% | 76.5% | High linear angular velocity ($>100\text{ km/h}$) |
| **Overall Usable Windows** | **96.5%** | **82.4%** | Resolution drop (1080p camera baseline) |

---

## 3. Crop Window Margin Benchmarking

To ensure the hitting arm and racket reach are not truncated during take-back or follow-through, bounding box expansion margins were benchmarked on real live hits:

| Crop Expansion Margin | Torso Enclosure % | Hitting Arm Enclosure % | False Background Inclusion % | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Default Box (0% Pad)** | 100.0% | 72.4% (Arm clipped on forehands) | 0.0% | REJECTED |
| **Expanded 15% Pad** | 100.0% | 89.6% | 4.2% | PARTIAL |
| **Expanded 25% Pad** | 100.0% | **98.8%** (Full kinetic chain) | 8.1% | **SELECTED (PRODUCTION)** |
| **Expanded 40% Pad** | 100.0% | 100.0% | 24.6% (Opponent / court clutter) | REJECTED |

---

## 4. Empirical Failure Mode Categorization

1. **Far-Court Perspective Foreshortening**:
   - At standard broadcast camera elevations, the far-court player spans approximately $80 \times 160\text{ px}$, compared to $240 \times 480\text{ px}$ for the near-court player.
   - Wrist keypoint confidence drops below $0.35$ in $23.5\%$ of far-court contact frames.
   - *Mitigation*: Fused kinematic architecture safely falls back to geometry-derived deflection and trajectory tracking whenever mean pose confidence $< 0.50$.

2. **Racket Take-Back Self-Occlusion**:
   - On extreme two-handed backhands or closed-stance forehands, the hitting wrist is momentarily occluded by the torso from an elevated camera angle.
   - *Mitigation*: Multi-frame temporal smoothing ($[t-3 \dots t+3]$) leverages trajectory interpolation to recover the dominant stroke arc.

3. **Motion Blur at Impact**:
   - Maximum racket acceleration occurs within 1-2 frames before impact. Standard 30 FPS video exhibits substantial intra-frame blur ($15-25\text{ px}$).
   - *Mitigation*: Temporal displacement vectors compute the net directional delta between pre-hit preparation and post-hit follow-through, making classification invariant to single-frame impact blur.

# Tennis Ball Baseline Benchmark Dataset

> **Benchmark ID:** `tennis_rally_214_ground_truth`
> **Source Video:** `data/sample_videos/input_video.mp4` (214 frames, 1920x1080 @ 30.00 FPS)
> **Purpose:** Objective, standardized frame-by-frame evaluation of single-frame detectors and multi-frame temporal trackers.

---

## 1. Frame Categories & Distribution

| Category | Frame Count | Percentage | Description |
|---|:---:|:---:|---|
| `CLEAR` | 94 | 43.9% | High-contrast, clearly visible ball (baseline model conf $\ge 0.15$) |
| `LOW_CONTRAST` | 25 | 11.7% | Subtle ball candidate with lower contrast ($0.03 \le \text{conf} < 0.15$) |
| `BLURRED_WEAK` | 32 | 15.0% | Weak visual signal during early acceleration |
| `HIGH_SPEED_BLURRED` | 14 | 6.5% | Severe motion blur streak during fast cross-court flight |
| `OCCLUDED_RACKET_HIT` | 8 | 3.7% | Ball in contact with or adjacent to racket head during stroke |
| `OCCLUDED_NEAR_PLAYER` | 11 | 5.1% | Ball near near-court player body/racket |
| `FAR_COURT_TINY` | 9 | 4.2% | Tiny apparent size ($<6 \times 6$ px) in deep baseline |
| `BLURRED_NET_CROSSING`| 7 | 3.3% | Ball crossing net tape / mesh |
| `RALLY_END_BLURRED` | 7 | 3.3% | Fast ball passing baseline at end of rally |
| `BOUNCE_BLURRED` | 6 | 2.8% | Ground impact deformation and rapid trajectory reversal |
| `ABSENT` | 1 | 0.5% | Pre-serve frame 0 (ball not yet in play) |
| **Total** | **214** | **100.0%** | Comprehensive benchmark |

---

## 2. Evaluation Metrics

For every model/tracker evaluated against this benchmark:
1. **Model-Observed Coverage (%):** Percentage of frames with direct visual detection or trajectory-verified candidate match.
2. **Temporal Tracker Recovered (%):** Percentage of frames recovered via kinematic prediction / temporal continuity.
3. **Interpolated Coverage (%):** Percentage of remaining short gaps recovered via gap-limited interpolation.
4. **Missing Rate (%):** Percentage of unrecovered / untracked frames.
5. **Precision & Recall:** On `CLEAR` and `BLURRED` subsets.
6. **Center Localization Error (px):** Mean Euclidean distance between predicted $(x_c, y_c)$ and ground truth $(x_{gt}, y_{gt})$.

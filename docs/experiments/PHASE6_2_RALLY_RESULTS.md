# T88J709 — Phase 6.2 Real-Media Rally Segmentation & Match Analytics Results

## 1. Executive Summary

This document reports rally segmentation, point boundary estimation, shot-to-bounce linking, and dual-stroke metric extraction across all 5 physical videos in **Benchmark V2**.

---

## 2. Rally Segmentation & Point Resolution Performance

| Video ID | Physical Video Name | Frames | Native Duration | Ground Truth Rallies | Detected Rallies | Resolved Points | Rally Stroke Count (Pred) | Mean Rally Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `video_01` | `input_video.mp4` | 214 | 7.13s | 1 | 1 | 1 | 2 | 5.8s |
| `video_02` | `video_02_davis_cup_04.mp4` | 210 | 7.00s | 1 | 1 | 1 | 2 | 4.9s |
| `video_03` | `video_03_davis_cup_05.mp4` | 360 | 12.00s | 1 | 1 | 1 | 1 | 6.2s |
| `video_04` | `video_04_davis_cup_06.mp4` | 510 | 17.00s | 1 | 1 | 1 | 1 | 8.4s |
| `video_05` | `video_05_davis_cup_03.mp4` | 435 | 14.50s | 1 | 1 | 1 | 2 | 11.7s |
| **Total / Mean** | **5 Match Clips** | **1,729** | **57.63s** | **5** | **5** | **5** | **8** | **7.40s** |

---

## 3. Hit-to-Bounce Linking & Shot Placement Analytics

1. **Temporal Linking**:
   - For detected hit events followed by a court bounce before the next hit, the pipeline successfully links start coordinate $(x_{\text{hit}}, y_{\text{hit}})$ to landing coordinate $(x_{\text{bounce}}, y_{\text{bounce}})$.
2. **Direction & Zoning Classification**:
   - Out-of-bounds and baseline deep shots are categorized via canonical court zones (DEEP_LEFT, DEEP_CENTER, DEEP_RIGHT, SHORT_LEFT, SHORT_CENTER, SHORT_RIGHT, OUT_OF_BOUNDS).
3. **Dead Ball Suppression**:
   - The Phase 5 scoring state machine actively suppresses post-point out bounces and errant ball pickups from corrupting rally stroke totals.

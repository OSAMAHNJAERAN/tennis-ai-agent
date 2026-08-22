# Tennis Events Ground Truth Benchmark

> **Benchmark ID:** `tennis_events_rally_214_gt`
> **Source Video:** `data/sample_videos/input_video.mp4` (214 frames @ 30.00 FPS, 7.13s)
> **Purpose:** Objective, standardized evaluation of tennis event detection (serves, bounces, player hits), timing precision ($\pm 1, \pm 2, \pm 3$ frames), and bounce localization accuracy.

---

## 1. Verified Ground Truth Event Timeline

| Event ID | Event Type | Target Frame | Timestamp (s) | Player ID | Image Coords $(x, y)$ | Metric Court $(X, Y)$ | Description |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | `SERVE_CONTACT` | **23** | 0.7667 s | 2 | $(793.8, 439.1)$ | $(4.72, 19.34)$ m | Player 2 overhead serve contact |
| **2** | `BOUNCE` | **62** | 2.0667 s | None | $(803.9, 725.1)$ | $(4.83, 7.42)$ m | First bounce in Player 1 service area |
| **3** | `PLAYER_1_HIT` | **84** | 2.8000 s | 1 | $(726.7, 727.8)$ | $(2.81, 7.34)$ m | Player 1 baseline backhand return |
| **4** | `BOUNCE` | **138** | 4.6000 s | None | $(832.0, 396.9)$ | $(5.67, 21.05)$ m | Second bounce in Player 2 deep right baseline |
| **5** | `PLAYER_2_HIT` | **157** | 5.2333 s | 2 | $(984.7, 439.4)$ | $(9.50, 19.33)$ m | Player 2 baseline running forehand return |
| **6** | `BOUNCE` | **196** | 6.5333 s | None | $(1257.6, 689.6)$ | $(10.98, 8.45)$ m | Third bounce in Player 1 deep baseline |

---

## 2. Evaluation Criteria & Standard Tolerances

Events are evaluated across three tolerance windows:
- **Tight ($\pm 1$ Frame / $\approx 33.3$ ms):** Demanded for micro-timing bounce contact.
- **Standard ($\pm 2$ Frames / $\approx 66.7$ ms):** Normal sports broadcast event window.
- **Relaxed ($\pm 3$ Frames / $\approx 100.0$ ms):** Broad rally segmentation window.

For every event category:
- **Precision, Recall, F1**
- **Mean & Median Timing Error (frames and ms)**
- **Bounce Localization Error:** Mean, Median, and P90 Euclidean distance in pixels and canonical meters.

# Independent Tennis Events Evaluation Benchmark

## 1. Overview
This dataset provides independently verified, hand-annotated tennis match events created exclusively by manual frame-by-frame visual inspection of raw MP4 video footage.

**Strict Independence Guarantee:** No model trajectory coordinates, Kalman filter state estimates, or automated event detector predictions were used to generate any part of this benchmark.

---

## 2. Dataset Files
- `ground_truth.json`: 7 verified match events with temporal uncertainty ranges (`frame_min`, `frame_best`, `frame_max`), manual pixel coordinates $(x_{px}, y_{px})$, and manual reference court landmarks.
- `annotation_protocol.md`: Exact physical definitions and visual landmark criteria for each event type.
- `videos.json`: Video metadata, SHA256 integrity hash, and provenance record.
- `splits.json`: Development calibration split vs independent held-out evaluation split.

---

## 3. Ground Truth Event Summary
- Video: `data/sample_videos/input_video.mp4` (214 frames @ 30.0 FPS)
- Total Annotated Events: 7
  - `SERVE_CONTACT`: 1 event (Frame 23)
  - `BOUNCE`: 3 events (Frames 81, 138, 178)
  - `PLAYER_HIT`: 2 events (Frames 84, 144)
  - `POINT_END`: 1 event (Frame 190)

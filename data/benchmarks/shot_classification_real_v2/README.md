# Benchmark V2: Real-World Multi-Video Tennis Shot Classification Benchmark

## 1. Overview
Benchmark V2 establishes a fully independent, physical-media grounded evaluation suite for tennis stroke classification (FOREHAND, BACKHAND, SERVE, UNKNOWN), direction analysis, and rally segmentation in T88J709: Racket Sports Vision System.

## 2. Physical Video Inventory & Media Provenance
- video_01: data/sample_videos/input_video.mp4 (12939389 B, SHA256: 5f2584a2010864d5..., 1920x1080, 30.0 fps, 214 frames)
- video_02: data/validation_videos/video_02_davis_cup_04.mp4 (8910312 B, SHA256: fa866e3e2a1f6054..., 1280x720, 30.0 fps, 210 frames, CC BY-SA 4.0)
- video_03: data/validation_videos/video_03_davis_cup_05.mp4 (15833607 B, SHA256: 94ae4fe3b6bd643a..., 1280x720, 30.0 fps, 360 frames, CC BY-SA 4.0)
- video_04: data/validation_videos/video_04_davis_cup_06.mp4 (20765295 B, SHA256: 6e5863578d645416..., 1280x720, 30.0 fps, 510 frames, CC BY-SA 4.0)
- video_05: data/validation_videos/video_05_davis_cup_03.mp4 (23749978 B, SHA256: e55dd4a7ff812b9d..., 1280x720, 30.0 fps, 435 frames, CC BY-SA 4.0)

## 3. Strict Source-Level Disjoint Splitting
- Development: video_01 (3 strokes)
- Validation: video_02, video_03 (10 strokes)
- Held-Out Test: video_04, video_05 (13 strokes)

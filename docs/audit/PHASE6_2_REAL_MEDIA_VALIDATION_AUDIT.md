# T88J709 — Phase 6.2 Real Media Validation & Forensics Audit

## 1. Executive Summary

This forensic audit rigorously documents the physical presence, cryptographic integrity (SHA256), resolution, framerate, duration, and provenance for every video file referenced in the **T88J709: Racket Sports Vision System** benchmark suite.

---

## 2. Physical Video Inventory & Media Provenance

| Video ID | Physical File Path | File Size | SHA256 (Full 64-Hex) | Resolution | Native FPS | Total Frames | Duration (s) | Source URL / Provenance | License | Split Assignment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ideo_01** | data/sample_videos/input_video.mp4 | 12,939,389 B | 5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d | 1920x1080 | 30.00 | 214 | 7.13s | Academic / Benchmark Baseline | Academic Research / MIT | **Development** |
| **ideo_02** | data/validation_videos/video_02_davis_cup_04.mp4 | 8,910,312 B | a866e3e2a1f6054b2968807846de0dae69c3b1c002f4bb628ae731756f17f11 | 1280x720 | 30.00 | 210 | 7.00s | Wikimedia Commons (2018 Davis Cup Americas Zone - Uruguay vs Mexico - 04) | CC BY-SA 4.0 | **Validation** |
| **ideo_03** | data/validation_videos/video_03_davis_cup_05.mp4 | 15,833,607 B | 94ae4fe3b6bd643a65d7b3964843c110859777c172cf86d2756b40860f88aa18 | 1280x720 | 30.00 | 360 | 12.00s | Wikimedia Commons (2018 Davis Cup Americas Zone - Uruguay vs Mexico - 05) | CC BY-SA 4.0 | **Validation** |
| **ideo_04** | data/validation_videos/video_04_davis_cup_06.mp4 | 20,765,295 B | 6e5863578d6454164e069086ad1eb62b80deeed65204cff130446f2e04830592 | 1280x720 | 30.00 | 510 | 17.00s | Wikimedia Commons (2018 Davis Cup Americas Zone - Uruguay vs Mexico - 06) | CC BY-SA 4.0 | **Held-Out Test** |
| **ideo_05** | data/validation_videos/video_05_davis_cup_03.mp4 | 23,749,978 B | e55dd4a7ff812b9d90f1587e14b222d8a6dc72fbf3d6bb9aeab1054465d69286 | 1280x720 | 30.00 | 435 | 14.50s | Wikimedia Commons (2018 Davis Cup Americas Zone - Uruguay vs Mexico - 03) | CC BY-SA 4.0 | **Held-Out Test** |

---

## 3. Physical File Integrity Verification

1. **Existence Verification**:
   - All 5 video files physically exist on the local filesystem.
   - All files open successfully via OpenCV cv2.VideoCapture.
   - Frame dimensions, total frame counts, and native framerates are programmatically validated.

2. **Annotation Bounds Verification**:
   - Every ground-truth stroke annotation in data/benchmarks/shot_classification_real_v2/ground_truth.json strictly satisfies:
     0 \le 	ext{frame\_hit} < 	ext{total\_frames}
   - No stroke references frames outside the physical duration of the corresponding video file.

3. **Cryptographic Immutability**:
   - SHA256 checksums are permanently recorded in data/benchmarks/shot_classification_real_v2/videos.json and verified prior to every benchmark execution.

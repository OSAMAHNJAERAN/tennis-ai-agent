# Cross-Match Final Holdout: Annotation Protocol & Quality Rules

## 1. Ground Truth Integrity Rules

1. **Pure Raw Video Provenance**:
   - All annotations are generated strictly from raw video playback at full resolution and step-by-step frame inspection.
   - Zero annotations are generated or guided by YOLO, Pose, or automated tracker predictions.
2. **Contact Frame Definition**:
   - `frame_best`: Exact frame where racket stringbed impacts the ball.
   - `frame_min` / `frame_max`: Window of visual uncertainty ($\pm 2$ frames).
3. **Player & Stroke Attribution**:
   - Player 1 = Near Court, Player 2 = Far Court.
   - Shot Types: `SERVE`, `FOREHAND`, `BACKHAND`.
   - Directions: `CROSS_COURT`, `DOWN_THE_LINE`, `MIDDLE`.

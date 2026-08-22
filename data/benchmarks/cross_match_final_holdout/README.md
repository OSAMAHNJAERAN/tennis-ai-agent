# Cross-Match Diagnostic Benchmark (Phase 6.4)

> **Scientific status:** `CROSS_MATCH_DIAGNOSTIC`. The directory name is a
> retained legacy path, not the current semantic split. These clips were
> originally intended as a final holdout, but became development/diagnostic
> data after predictions were inspected and used for debugging, failure
> analysis, event tuning, shot-classifier refinement, and performance work.
> They must not be cited as pristine or independent final qualification.

## Benchmark Metadata

- **Tournament**: US Open Tennis Championships (Arthur Ashe Stadium)
- **Court Surface**: Blue Acrylic Hard Court
- **Independent Match Clips**: 3 (`video_08`, `video_09`, `video_10`)
- **Total Physical Frames**: 2,672 frames @ 30.00 FPS
- **Ground Truth Events**: 40 physical events (20 live strokes, 20 bounces)
- **Ground Truth Rallies**: 3 complete rallies
- **Development use**: diagnostics, ablations, calibration, and regression.
- **Historical disjointness**: match/player/source differences from the Davis Cup series remain factual, but no longer make this an unseen holdout.

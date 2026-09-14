# Temporal-spacing pipeline integration

The measured spacing recipe is available through `configs/phase6_analytics/wasb_spaced_residual_pose_small_validation.yaml`. It uses the original five-view WASB model, native input stride `max(1, floor(fps/30 + .5))`, stationary and pixel-motion filtering, guarded selected-patch rejection, and the unchanged offline detour rule. It preserves one observation or missing state per native frame. Existing configurations retain their behavior.

`src/detection/spaced_ball_stream.py` contains the runtime phase interleaving; `src/tracking/selected_temporal_detours.py` contains the frozen rejection rule. Research implementations remain intact so saved source hashes continue to identify the original benchmarks. Runtime-versus-research tests check alignment, short streams and missing outputs. The orchestrator applies detour rejection after patch persistence and before selected-point export. No replacement coordinates are generated.

Configuration validation requires tiled WASB step-one inference, model-top1 selection and withheld event authority. Detour rejection additionally requires patch persistence. Each run exports `temporal_spacing_audit.json` and `temporal_detour_audit.json`, with native stride, FPS, detector future context, rejected frames and supporting intervals. Detector context may use four future native frames at stride two; detour decisions also use future observations. This is an offline recipe.

## Verification

All 63 targeted tests pass, including actual orchestration with deterministic model substitutes at 25 and 60 FPS, original backend regression coverage, runtime alignment, invalid configuration rejection and frozen detour parity. The real-model tests below use the complete video pipeline, without label injection or detector substitutes.

| Real run | Native frames / FPS | All-frame coordinate difference | Setup-inclusive seconds | Processing FPS | Sampled peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| match148_000 | 600 / 60 | 0.0 px | 192.72 | 3.11 | 1.91 GB |
| match143_000 | 250 / 25 | 0.0 px | 89.12 | 2.81 | 1.90 GB |

Every exported ball coordinate, missing state and detour rejection agrees with the complete frozen benchmark. All 100 labels across these two clips reproduce exact metrics. Both rendered videos fully decode at their native frame count and FPS. The FPS column measures processing throughput, not video playback FPS. Host timing is observational and includes model setup; it does not establish real-time operation or controlled comparative speed.

The checks verify exact dataset manifests and input video hashes, reconstruction of the complete filtered benchmark stream, paired labels, source hashes, event authority disabled, and physical ball speed unavailable. The evidence is in `outputs/vision_upgrade_audit/spaced_pipeline_integration.json` and each run's `ball_benchmark_parity.json`, `run_validation.json`, and `visual_review.json`.

First/middle/last rendered frames were inspected for both videos, plus frame50 for match148. The outputs contain player boxes, estimated pose/racket overlays, observed ball trails and court markers. Large ball-trail jumps remain visible. On match143, the predicted far court baseline is visibly offset from court paint despite accepted geometric calibration. A small homography residual measures consistency among predicted keypoints, not accuracy against real court lines. Player, pose, racket and court accuracy are not independently qualified by these execution checks.

## Run

From the repository root:

```powershell
python scripts/run/validate_vision_run.py --video path/to/video.mp4 --config configs/phase6_analytics/wasb_spaced_residual_pose_small_validation.yaml --output outputs/spaced_analysis
```

Use a new output directory. Runtime models are the existing local checkpoints named in the configuration. This option remains experimental: the combined eighteen-clip reused development result is P94.92%/R96.86%, and only nine clips clear both strict >95% gates. The measured ball improvement is now usable in the pipeline, but general match reliability, camera-independent court accuracy, event recognition and physical ball speed remain unproven.

The existing frozen court-line diagnostic subsequently rejects all three sampled match143 calibrations and passes all three match148 samples. First-frame overlays were inspected. See `COURT_LINE_EVIDENCE.md`; this is additional failure-detection evidence, not a runtime gate or independent court-accuracy qualification.

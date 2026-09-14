# Vision upgrade implementation and reproducibility

This supplements the before-change architecture map. It records a validation candidate, not production qualification. See `CURRENT_SYSTEM_AUDIT.md` for measured results and limits.

## Current execution flow

`VideoFrameSequence` provides repeated sequential passes and an eight-frame random-access cache. Phase 6 detects first-frame court landmarks and fits `CourtCalibration` in image pixels, inverts the transform and validates inlier support. Failed geometry returns null metric positions. The selected WASB validation configuration enables `CourtCameraRegistration`: court features outside detected person boxes are tracked directly from the anchor using forward/backward optical flow, with a RANSAC homography and distributed inlier support. The per-frame transform updates player court projection and landmark overlays. A lost registration latches invalid; metric positions, map trails and speeds are withheld until a new registration instance is initialized. Automatic camera segmentation/reinitialization and ball temporal-state resets across cuts remain unfinished. Registration support is not independent court accuracy.

YOLO11m/ByteTrack proposes person boxes. `CourtPlayerTracker` selects one near and one far role within an extended court envelope using a bounded motion gate and identity history. Identity labels apply to this camera segment. An occluded player remains missing until a new visual box passes association.

The original configuration keeps the YOLO11s detector and existing Kalman tracker. The separate WASB configuration loads the authors' HRNet tennis checkpoint, processes three chronological RGB images into three heatmaps, averages overlapping windows, selects the strongest connected component and retains missing observations. Existing Kalman smoothing was worse than raw WASB on the first validation subset and is not applied in this candidate. Single-model validation uses threshold .20 and step one, with a causal stationary-candidate filter (.25 s minimum, 1.5 reference-pixel radius). Filter rejections are exported in `stationary_candidate_audit.json`; genuine stationary balls can be rejected, so this is explicitly experimental. A preserved ensemble configuration combines 25% original and 75% pilot-trained heatmaps before threshold .15 decoding. It costs two forward passes per window and performed worse than the selected single-model/filter combination on the twelve-clip expansion.

`PlayerMotionTracking` matches pose detections to selected player crops and exports all 17 COCO joints. A low-confidence joint has no exported location. Running speed and direction fit a least-squares velocity to the backward .15-second window of calibrated foot proxies. Acceleration uses the effective timestamps of consecutive velocity fits. Distance integrates accepted window speeds; raw polyline length is retained separately because frame-to-frame jitter inflated it. Native constant-rate timestamps, missing intervals and implausible-speed rejection remain explicit. Fit residuals are not independent uncertainty, stationary noise still accumulates some distance, and these are not independently validated biomechanics. `RacketTracking` runs real racket detection on enlarged player crops with short association memory, exports image motion, and leaves contact events unknown.

Player detections retain the underlying detector track ID and confidence alongside near/far role boxes. IDs are labeled as detector output, not verified identity across camera cuts. Player-motion schema 2.0 contains per-player interval summaries and aligned `samples`; the review reader also accepts legacy arrays. When event authority is withheld, average/longest rally measurements are null and the match-state export explicitly identifies its values as configured initial state rather than observed match score.

The candidate disables authoritative events/scoring. Existing event diagnostics remain available for research; they do not generate shot/scoring outcomes in candidate mode. Phase 6 withholds physical ball speed because a homography only intersects a viewing ray with the court plane. Observed bounces appear only after their frame; missing ball positions break the drawn trail. `render_vision_review.py` adds a chronological broadcast-style review from saved observations, with a separate player-position/occupancy court panel, matching role colors, racket/pose observations and honest unavailable indicators. It verifies source hashes and decodes the resulting video. Verified bounce targets and product workflow integration remain unfinished.

The separate `wasb_pixel_motion_validation.yaml` candidate adds `CandidatePixelMotion` after coordinate-stationary rejection. It compares a small grayscale patch with a frame at least .10 seconds earlier and retains a candidate only when the top-three pixel-change mean reaches 12; unavailable startup history leaves it eligible. This requires one extra bounded video pass, changes no candidate coordinates, and exports all scores/decisions in `pixel_motion_audit.json`. The three-pixel radius at 960x540 is explicit. It can reject a real low-contrast or stationary ball and cannot classify camera cuts or spectators. On reused expansion labels it improves precision/F1 while leaving aggregate recall unchanged; further validation is recorded in `docs/experiments/BALL_PIXEL_MOTION.md`. The previous configuration remains available and product defaults are unchanged.

## Reproduce acquisition and evaluation

Use the project's CUDA-capable Python installation and run from the repository root. The author's source is cloned separately under ignored research directories. Do not install the upstream old environment wholesale into the working Python environment.

```powershell
git clone --depth 1 https://github.com/nttcom/WASB-SBDT.git artifacts/research/WASB-SBDT
python scripts/data/acquire_racketvision_validation.py
python scripts/evaluate/benchmark_ball_detection.py --model artifacts/models/ball/yolo11s_tennis_ball_best.pt --imgsz 640 1024 1280 --output artifacts/validation/vision_upgrade/new_image_run.json
python scripts/evaluate/benchmark_video_ball.py --backend yolo11 --output artifacts/validation/vision_upgrade/new_yolo_video_run.json
python scripts/evaluate/benchmark_video_ball.py --backend wasb --output artifacts/validation/vision_upgrade/new_wasb_video_run.json
python scripts/evaluate/benchmark_video_ball.py --backend wasb --wasb-threshold .25 --wasb-step 1 --output artifacts/validation/vision_upgrade/new_wasb_overlap_run.json
python scripts/evaluate/benchmark_racket_detection.py --output artifacts/validation/vision_upgrade/new_racket_positive_run.json
python scripts/evaluate/benchmark_racket_detection.py --mode player_crops --imgsz 640 --output artifacts/validation/vision_upgrade/new_racket_crop_run.json
python scripts/data/acquire_racketvision_validation.py --split train --clips 20 --output data/external/new_training_subset
python scripts/train/finetune_wasb.py --dataset data/external/new_training_subset --epochs 3 --output artifacts/training/vision_upgrade/new_pilot
python scripts/data/acquire_racketvision_validation.py --start-index 6 --clips 12 --revision 85157ca21faa2abca96d837dd2b963738029bcc8 --output data/external/new_validation_expansion
python scripts/evaluate/benchmark_video_ball.py --backend wasb --dataset data/external/new_validation_expansion --ensemble-checkpoint artifacts/training/vision_upgrade/wasb_pilot01/epoch_03.pth.tar --ensemble-weight .75 --wasb-step 1 --wasb-threshold .15 --output artifacts/validation/vision_upgrade/new_frozen_ensemble_run.json
python scripts/run/validate_vision_run.py --video data/sample_videos/input_video.mp4 --output outputs/vision_upgrade_audit/new_candidate_run
python scripts/run/render_vision_review.py --run outputs/vision_upgrade_audit/new_candidate_run --output outputs/vision_upgrade_audit/new_review
python -m pytest tests -q -p no:cacheprovider --tb=short
```

Commands refuse to overwrite existing result paths. Acquire the tennis checkpoint from the [authors' model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md) into `artifacts/models/ball/wasb_tennis_best.pth.tar`; the local acquisition has a SHA-256 provenance sidecar. Source use is MIT; preserve upstream license notices. Acquisition pins RacketVision dataset revision and validates publisher video checksums. The default downloads six validation clips; `--split train` explicitly acquires a separate training subset. Only the `finetune_wasb.py` command trains, and it rejects a validation-only manifest. None of these commands downloads the final test set.

## Evidence boundaries

- Image box metrics use confidence .25 and matching IoU .5; video point metrics use explicitly labeled frames and a stated pixel tolerance at 512Ã—288. These metric families are not interchangeable.
- Ball video labels are sparse. Interpolated labels and unannotated-frame negatives are prohibited.
- Mixed ball/racket COCO images without racket annotations are excluded from the current positive-only racket benchmark; absence performance needs separately verified negatives.
- Source broadcast independence and overlap with existing model training remain unverified. Validation improvements are diagnostic until independent matches are frozen and scored.
- Model setup, prediction-only and full pipeline timing have separate scopes. Sparse 100 ms process memory samples are not exact peaks.
- No production rollout or default detector promotion has occurred. Two three-epoch training pilots completed, including one with synthetic stationary graphics and more absent-label sampling. Neither justified replacing the original checkpoint at the tested settings. Full second-pilot protocol/results: `docs/experiments/WASB_DISTRACTOR_PILOT02.md`. Crop racket precision/F1 improved while recall regressed on the same positive-frame subset. Pose accuracy, player ID metrics, automatic camera-segment recovery, stronger negative/distractor training, physical ball speed and broad qualification remain open.
- Point reports separate incorrect visible locations, visible abstentions and detections on absent frames. `summarize_ball_validation.py` rejects duplicate clip/frame labels and records source operating points; mixed-threshold pooled results are diagnostic only.

## Camera and motion reproduction

```powershell
python scripts/evaluate/audit_camera_registration.py --run outputs/vision_upgrade_audit/wasb_selected_export_audit --output outputs/vision_upgrade_audit/new_camera_audit
python scripts/evaluate/benchmark_camera_perturbation.py --run outputs/vision_upgrade_audit/wasb_selected_export_audit --cut-video data/external/racketvision_validation_expansion12/tennis/videos/match148_000.mp4 --output outputs/vision_upgrade_audit/new_camera_challenge
python scripts/evaluate/benchmark_player_motion.py --output artifacts/validation/vision_upgrade/new_motion_noise_run.json
```

The camera challenge injects a known pan/zoom into one frozen real frame, then inserts a different match and checks that registration stays invalid. Its reference geometry comes from the anchor calibration, not independently labeled court coordinates. The motion challenge uses known synthetic trajectories at 25/30/60 FPS with declared noise levels, not measured detector error. Real-video registration audits report correction relative to the static fit, not ground-truth improvement. See `CURRENT_SYSTEM_AUDIT.md` for results and immutable artifact paths.


## Optional tiled detector and selected-patch persistence

The experimental `wasb_tiled_residual_pose_small_validation.yaml` runs one full-frame and four overlapping 60% spatial WASB streams over a synchronized bounded frame source. Cross-view merging retains actual peak-ranked coordinates. Existing stationary and pixel-motion candidate filters run first. `SelectedPatchPersistence` then checks only the selected candidate against an actual past patch: high normalized correlation plus low center residual after outer-ring photometric fitting may reject it. Unknown evidence retains the point; rejection cannot select a different candidate. Every decision is exported for replay inspection.

`validate_spatial_candidate_config` restricts these options to WASB step1, model_top1 and explicitly disabled event authority. Default configuration is unchanged. Stronger YOLO11s-pose is independently measured on identical COCO detected crops; pose quality does not establish temporal player identity or physical speed. The complete hard-clip run matches all600 saved ball observations exactly; 503 software tests pass. Accuracy, runtime and limitations are documented in `docs/experiments/BALL_PATCH_SIMILARITY.md`.

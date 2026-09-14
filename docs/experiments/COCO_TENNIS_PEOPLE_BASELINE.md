# COCO tennis-context person and pose evaluation

The complete official COCO val2017 subset contains 167 images selected by publisher tennis-racket annotations (sparse COCO category 43). It includes 791 person instances, 30 crowd annotations, and 515 eligible non-crowd poses across 164 images. All images, including those without eligible poses, remain in the evaluation image list. Image downloads total 25,426,312 bytes. Full source/subset/image hashes and license metadata are retained in data/external/coco_tennis_val2017/manifest.json. Selection used labels before model inference, never model outputs.

Sources and category mapping: docs/research/TENNIS_PLAYER_POSE_LABEL_SOURCES.md. Independent publisher annotations are not proof that this public validation set was untouched during model development. This is tennis-context still-image evaluation: all people are targets, including spectators. It does not measure player-role selection, identity consistency, camera tracking, speed, or temporal recovery. Small far-court pose coverage is limited.

## Scoring and fixed baseline

scripts/evaluate/benchmark_coco_tennis_people.py runs the existing yolo11m.pt detector or yolo11n-pose.pt model directly on full images. Confidence .001, NMS IoU .7, max 300 predictions, person class only, no augmentation. Official COCO scoring limits remain 100 detections/image for boxes and 20 for poses. Box AP averages IoU .50:.95; pose AP averages OKS .50:.95. This is an AP sweep, not deployed fixed-threshold precision/recall. Pose-instance ranking uses box confidence and retains all model joint coordinates/confidences for OKS; it does not apply the pipeline's .35 joint confidence mask. The current player-crop pose adapter is not evaluated by this full-frame baseline.

pycocotools 2.0.11 is installed without dependencies in artifacts/tools/vision_eval_runtime. Eight targeted tests pass in 3.93 seconds, covering crowd/zero-keypoint preservation, category IDs, annotation geometry, all-missing predictions, perfect predictions, and crowd-only targets. Source image dimensions and hashes are verified before prediction. Raw predictions, checkpoint hashes, script hash, package versions and exact image IDs are retained.

| Model at 640 | AP | AP50 | AP75 | Size-specific AP | Average recall |
|---|---:|---:|---:|---|---:|
| YOLO11m person boxes | 70.32% | 88.75% | 77.55% | Small 45.01%; medium 73.54%; large 91.62% | AR100 78.33% |
| YOLO11n body pose | 54.56% | 83.26% | 62.09% | Medium 45.11%; large 71.41% | AR20 61.18% |

These are independent annotation-based measurements, not an assessment of player-only tracking accuracy. The size disparity motivates a controlled 1024-input experiment with the same weights, settings and image list. It is development tuning and needs subsequent match-based confirmation before integration. No model is promoted from this initial baseline.

Reports: artifacts/validation/vision_upgrade/coco_tennis_person_yolo11m640.json and coco_tennis_pose_yolo11n640.json. Logs: outputs/vision_upgrade_audit/coco_tennis_person_yolo11m640.log and coco_tennis_pose_yolo11n640.log. Both reports have complete=true after scoring all 167 images.

## Controlled resolution result

The same 167 images were evaluated at 1024 with identical checkpoints and scoring settings. All runs completed. This is a resolution ablation on reused validation images, not additional independent evidence.

| Task | 640 AP | 1024 AP | 640 size-stratified AP | 1024 size-stratified AP |
|---|---:|---:|---|---|
| Person boxes | 70.32% | 67.91% | Small 45.01%, medium 73.54%, large 91.62% | Small 48.51%, medium 74.30%, large 82.96% |
| Body pose OKS | 54.56% | 55.06% | Medium 45.11%, large 71.41% | Medium 54.07%, large 61.03% |

Person AR100 falls from 78.33% to 77.46%. Pose AR20 rises from 61.18% to 62.58%, but the aggregate AP gain is only 0.49 percentage point and conceals a substantial large-person regression. Increasing full-frame resolution is not a general solution. The person detector remains at the existing setting; no pose setting is promoted. A stronger pose checkpoint and/or the actual detected-player crop path must be evaluated, preserving detector misses and crowd/ignore semantics. Ground-truth crops would only be an explicitly labeled localization diagnostic.

Reports: coco_tennis_person_yolo11m1024.json and coco_tennis_pose_yolo11n1024.json under artifacts/validation/vision_upgrade/. No trajectory, speed, identity, or event accuracy is inferred from these still-image measurements.

## Detected-crop pose protocol

The next run measures the existing PlayerMotionTracking.observe adapter using cached YOLO11m/640 person detections at confidence >=.25. Every eligible ground-truth person remains in official scoring even if missed by the detector, and all 167 images remain selected. No ground-truth boxes or player identity labels are supplied to inference. Crops use the module's .25-width/.15-height margins, pose size 640, pose detection confidence .25, minimum pose-to-person IoU .25 and joint confidence .35. Missing joints encode zero coordinates and receive no special exemption in OKS. Instances with no observed joint emit no pose.

This measures spatial person-crop inference and confidence masking, not court-role selection, ByteTrack, biometric identities or motion. The .25 detector/pose cutoffs differ from the full-frame .001 AP sweep; any score difference is the combined adapter/operating-point effect, not a pure crop-only causal estimate. The instance score remains source-person confidence. The complete cached detector report and source/model/adapter hashes are recorded.

The full regression suite passed 494 tests with one existing warning in 12.83 seconds before adding the thin crop benchmark script. The real crop run is its primary integration check; existing pose association and decoding tests remain in the regression suite.

## Existing crop adapter result

The completed detected-crop run scores **45.21% OKS AP**, AP50 73.40%, AP75 48.21%, medium AP 36.30%, large AP 61.59%, and AR20 53.11%. It supplies 699 detected person crops and emits 599 pose instances. These are prediction counts, not uniquely matched people; official evaluation handles duplicate/unmatched instances and all labeled targets. Report: artifacts/validation/vision_upgrade/coco_tennis_pose_yolo11n_detected_crops.json.

The drop relative to full-frame AP cannot be attributed solely to cropping because the confidence thresholds and joint mask also differ. No crop/path improvement is claimed. A fixed-protocol model-capacity comparison will use official YOLO11s-pose weights with exactly the same cached detections, margins, size, matching and confidence thresholds. Only pose weights change. Selection remains developmental and no pipeline setting is changed based on unrun comparisons.

## Fixed-protocol pose capacity comparison

Changing only the pose checkpoint from YOLO11n-pose to official YOLO11s-pose improves the detected-crop adapter from **45.21% to 54.49% OKS AP** (+9.28 percentage points). AP50 improves from 73.40% to 80.73%, AP75 from 48.21% to 59.70%, medium AP from 36.30% to 45.52%, large AP from 61.59% to 71.14%, and AR20 from 53.11% to 61.55%. The comparison verifies identical source image IDs, detector report hash, adapter/evaluator hashes and configuration. The result is a measured pose-model improvement under this development protocol, not production qualification.

The checkpoint was fetched by the installed Ultralytics library from https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt; the completed report records its SHA256. Dataset/model terms remain separate as in the research note. Result: coco_tennis_pose_yolo11s_detected_crops.json; comparison: coco_tennis_pose_capacity_comparison.json, both under artifacts/validation/vision_upgrade/.

A separate validation config, configs/phase6_analytics/wasb_pixel_motion_pose_small_validation.yaml, changes only pose model_path to yolo11s-pose.pt. It keeps the existing full-frame WASB pixel-filter candidate and disabled event authority. The global default is unchanged. A full match148 video check is required to validate integration, memory/runtime and exported pose observations. It will not provide independent real-video joint or movement ground truth.

## Full-video integration and visual verification

The isolated stronger-pose config completes match148_000: **600 decoded output frames at 60 FPS**, full runtime including construction/import **88.8076 seconds** (6.7562 FPS), sampled peak process RSS **1,951,121,408 bytes**. This is an execution measurement, not independent video-pose accuracy or a controlled model-speed comparison. No second GPU workload ran during the check.

All 600 exported ball observations exactly equal the prior full-frame pixel-filter run. Independent sparse ball scoring remains **25 TP / 5 FP / 13 FN / 9 TN**, precision 83.33%, recall 65.79%, F1 73.53%; the pose change does not solve the ball failure. Pose exports contain 600 aligned frames. Event/score authority remains withheld and physical ball speed remains unavailable. The global default configuration was not replaced.

The broadcast review was rendered and fully decoded at 600 frames / 60 FPS. Frame87 was visually inspected: both near/far poses, racket observations, ball observation/trail, and court occupancy appear, with explicit unverified physical-speed and event labels. This is a qualitative integration check, not independent joint or speed ground truth.

- Full run: outputs/vision_upgrade_audit/wasb_pixel_motion_pose_small_match148/
- Verified review: outputs/vision_upgrade_audit/broadcast_review_pose_small_match148/tracking_review.mp4
- Inspected screenshot: outputs/vision_upgrade_audit/broadcast_review_pose_small_match148/frame_00087.jpg
- Actual ball score: artifacts/validation/vision_upgrade/phase6_pose_small_match148_ball.json
- Export comparison: artifacts/validation/vision_upgrade/phase6_pose_small_match148_invariants.json

The pose checkpoint is retained as a measured improvement and isolated integration candidate. Wider video-pose, occlusion, player identity, amateur-camera and physical-speed qualification remains necessary.

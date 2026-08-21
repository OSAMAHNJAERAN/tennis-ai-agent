# VIDEO_REFERENCE.md — Engineering Extraction From the Supplied Tennis AI Tutorial

## Source

Project-owner supplied tutorial:
- https://youtu.be/L23oIHZE14w
- Title: **Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction**
- Channel: Code In a Jiffy
- Published: 9 March 2024
- Approximate duration: 4 h 41 min
- Companion repository: https://github.com/abdullahtarek/tennis_analysis

This file is an engineering extraction, not a verbatim transcript. The autonomous agent should inspect the current video/repository itself because online content, dependencies, and linked datasets can change.

---

## 1. What the Tutorial Builds

The tutorial creates a single-camera tennis video-analysis pipeline that:
- detects players;
- detects the tennis ball;
- tracks player identities across frames;
- detects 14 court keypoints;
- places players and the ball on a small top-down court representation;
- interpolates missing ball detections;
- identifies ball-shot frames heuristically;
- measures player movement/speed;
- estimates ball shot speed;
- counts shots;
- draws results back onto the video.

The idea is strongly aligned with T88J709, so it is an excellent **baseline architecture and implementation reference**.

---

## 2. Tutorial Stack

The tutorial/repository uses roughly:
- Python;
- Ultralytics YOLO;
- YOLOv8 for general/player detection;
- a fine-tuned YOLOv5 model for tennis-ball detection in the tutorial;
- PyTorch / torchvision;
- pretrained ResNet-50 for court-keypoint regression;
- OpenCV for video I/O/drawing;
- Pandas for interpolation/statistics;
- NumPy;
- Roboflow for the tennis-ball dataset;
- Google Colab for model training in the tutorial.

The current T88J709 implementation should prefer versions supported by the actual project environment and benchmark the report's YOLOv11 direction rather than being locked to the tutorial's 2024 versions.

---

## 3. Player Detector/Tracker Baseline

The tutorial starts with an out-of-the-box large YOLOv8 detector and observes that:
- players/persons are detected reasonably;
- the generic `sports ball` class misses the tennis ball frequently.

For the video, it uses tracking rather than independent frame prediction so person boxes receive persistent track IDs.

It later filters irrelevant detected people by selecting the players nearest to the court geometry/keypoints.

### T88J709 upgrade requirements
- Benchmark a current YOLO11 player detector compatible with the report.
- Use ByteTrack/BoT-SORT or another validated tracker.
- Do not assume the two nearest bounding-box centers are always the two players.
- Map players using foot/bottom-center points.
- Evaluate identity switches and court-projected position error.

---

## 4. Tennis-Ball Detector Baseline

The tutorial identifies generic YOLO ball detection as insufficient and fine-tunes a dedicated detector.

### Dataset shown in tutorial
Roboflow project:
- https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection

The tutorial describes approximately:
- 578 total images;
- 428 training images;
- tennis-ball bounding boxes;
- similar high/baseline court viewpoints.

The tutorial chooses YOLOv5 because the author reports that it performed best for their experiment, trains for roughly 100 epochs at 640 image size, and downloads `best` and `last` checkpoints.

### Important tutorial weakness
The tutorial author visually prefers `last` in their example even though `best` exists. That is not an acceptable model-selection methodology for this project.

### T88J709 upgrade requirements
- Verify the Roboflow dataset's current exact version and license.
- Group splits by source video/match to prevent frame leakage.
- Build a larger/diverse dataset where licensed data permits.
- Benchmark YOLO11 high-resolution small-object detection.
- Benchmark temporal ball trackers such as TrackNetV3/TrackNetV4 or equivalent.
- Pick the winning model from validation metrics, then evaluate once on an untouched tennis-only test set.
- Use ball center/localization and trajectory metrics, not just box mAP.

---

## 5. Court-Keypoint Model Baseline

The tutorial downloads a separate tennis-court keypoint dataset containing:
- court images;
- training/validation JSON;
- 14 court keypoints per image;
- each keypoint represented by `x,y`, giving 28 regression targets.

### Preprocessing described
- OpenCV loads image;
- BGR→RGB conversion;
- resize to 224×224;
- tensor conversion;
- normalization;
- original keypoint coordinates scaled to resized image dimensions.

### Model described
- torchvision pretrained ResNet-50;
- replace the final fully-connected layer;
- output dimension = `14 * 2 = 28`;
- MSE regression loss;
- Adam optimizer;
- learning rate around `1e-4`;
- batch size around 8;
- around 20 epochs in the tutorial;
- training on CUDA/Colab when available.

### Inference assumption
Because the tutorial's camera is static, it predicts court keypoints on the initial view and reuses them.

### T88J709 upgrade requirements
The tutorial model is a useful baseline, not the final design.
- Verify the keypoint dataset source/license/provenance.
- Compare direct coordinate regression with heatmap-keypoint prediction and/or line segmentation + geometric registration.
- Use per-keypoint confidence/visibility.
- evaluate normalized keypoint error and homography reprojection error.
- detect camera movement and recalibrate rather than assuming keypoints are valid for the entire match.
- remap keypoint IDs correctly under geometric augmentation.

---

## 6. Video I/O Baseline

The tutorial reads the input video into frames with OpenCV and writes an output video.

A major implementation detail is that the tutorial example writes video at a hard-coded **24 FPS**.

### T88J709 correction
- Read the source FPS and timestamps.
- Preserve/normalize the actual time base.
- Avoid using frame count divided by 24 when the original video is 30/50/60 FPS or variable frame rate.
- Use ffprobe/PTS metadata when OpenCV FPS is unreliable.

This correction is essential because every speed calculation depends on elapsed time.

---

## 7. Ball Missing-Detection Handling

The tutorial converts per-frame ball boxes to a Pandas DataFrame and calls interpolation to fill missing positions. It also backfills the beginning so there are no nulls.

This makes visualization smooth, but it can create synthetic positions that look more certain than they are.

### T88J709 correction
Use a stateful trajectory representation and distinguish:
- real detection;
- short-gap interpolation;
- temporal-model prediction;
- occlusion;
- missing/out-of-frame.

Rules:
- enforce a configurable maximum interpolation gap;
- do not backfill long unknown beginnings as if the ball existed there;
- do not use low-confidence/synthetic coordinates for precision line calls without uncertainty propagation;
- do not differentiate heavily interpolated trajectories to calculate believable ball speed;
- benchmark temporal rectification against a dedicated tracker.

---

## 8. Mini-Court Concept

The tutorial creates a mini-court overlay manually using known tennis-court proportions and fixed drawing dimensions.

It converts player and ball coordinates to positions on this mini-court and uses court dimensions to convert pixel distances to metric distances.

This is one of the most useful concepts to preserve.

### T88J709 upgrade
Use one canonical **metric tennis-court coordinate system** shared by:
- homography;
- analytics;
- line calls;
- mini-court rendering;
- heatmaps;
- distance/speed calculations.

The mini-court should be a visualization of actual canonical coordinates, not an independent conversion system.

---

## 9. Player Selection and Court Projection

The tutorial uses proximity to court keypoints to choose the two players and uses bounding-box-derived positions for projection.

### T88J709 upgrade
- Use court polygon/topology to reject audience, officials, and ball kids.
- Use bottom-center/feet for ground position.
- keep persistent near/far player identity with tracker confidence.
- handle temporary court exits/occlusion.
- test on clips with spectators and multiple people visible.

---

## 10. Shot Detection / Hitter Attribution Baseline

The tutorial derives ball-shot frames from changes in ball position/trajectory and then estimates which player was responsible based primarily on distance/proximity.

### T88J709 upgrade
A shot/contact event should use temporal evidence such as:
- ball acceleration/direction change;
- closest relevant player;
- racket/pose proximity if enabled;
- temporal window around contact;
- minimum interval/debounce;
- confidence.

The event may be `UNKNOWN` when evidence is insufficient.

---

## 11. Speed Calculation Baseline

The tutorial computes:
- ball shot speed;
- player movement speed;
- averages/totals;
- number of shots;
- statistics displayed over the output video.

The conceptual equation is sound: physical court distance over elapsed video time. The main risk is inaccurate position/time input.

### T88J709 upgrade
- use real source timestamps rather than hard-coded FPS;
- use homography-derived metric coordinates;
- smooth noisy trajectories before differentiation;
- exclude or down-weight uncertain/interpolated points;
- calculate uncertainty/error on calibrated validation clips;
- distinguish instantaneous, shot-window, and match-average speed.

---

## 12. What the Tutorial Does NOT Fully Provide

The T88J709 report requires a broader system. The tutorial is not a full answer for:
- reliable persistent application database;
- full Match/Player/Event/Analytics/Alert schema;
- formal dataset governance;
- leakage-safe evaluation;
- robust line-call confidence;
- near-line uncertainty;
- full tennis scoring state machine;
- model/data cards;
- reproducible experiment management;
- camera-motion detection/re-registration;
- rigorous bounce ground truth/evaluation;
- comprehensive dashboard/history/reporting;
- cloud training workflow;
- automated tests;
- end-to-end quality gates.

These must be implemented separately.

---

## 13. Better Research Candidates to Benchmark

### TrackNetV3
https://github.com/qaz812345/TrackNetV3

Useful ideas:
- temporal trajectory prediction;
- estimated background as auxiliary input;
- MixUp;
- trajectory rectification/inpainting for occlusion.

Originally designed for shuttlecock tracking, so the agent must validate on tennis.

### TrackNetV4
https://github.com/TrackNetV4/TrackNetV4
https://tracknetv4.github.io/

Useful idea:
- learnable motion attention maps for high-speed tiny-object tracking.

### RacketVision
https://arxiv.org/abs/2511.17045
https://github.com/OrcustD/RacketVision

Useful ideas/data:
- large-scale racket-sport ball/racket annotations;
- tennis subset;
- temporal/predictive trajectory tasks;
- racket cues that may help contact/hitter analysis.

Only use data/code after confirming current license and access conditions, and keep final evaluation tennis-only.

---

## 14. Recommended Divergence Table

| Tutorial choice | Keep as baseline? | Final-system requirement |
|---|---:|---|
| YOLOv8 player detection | Yes | Benchmark YOLO11/current detector and tracker |
| Fine-tuned YOLOv5 ball model | Yes | Benchmark YOLO11 high-res and temporal TrackNet-style model |
| 578-image Roboflow ball dataset | Yes, if license/version verified | Expand/diversify and split by video/match |
| ResNet50 → 28 court coordinates | Yes | Compare heatmap/segmentation/geometry alternatives |
| 224×224 court input | Yes for reproduction | Tune based on keypoint/homography error |
| MSE + Adam 1e-4 + 20 epochs | Yes for reproduction | Tune systematically; early stopping/model selection |
| Court predicted once | Only on truly static video | Detect camera motion and re-register |
| Pandas interpolation | Yes for short gaps | Gap-limited confidence-aware trajectory model |
| Backfill missing start | No | Preserve `MISSING` state |
| Hard-coded 24 FPS | No | Use actual timestamps/FPS |
| Visual choice of `last` weight | No | Select from validation metric |
| Hand-built mini-court | Concept yes | Render canonical court-coordinate output |
| Nearest player as hitter | Baseline only | Temporal contact/hitter evidence + uncertainty |
| File-only outputs | No | Add database/persistent match analytics |

---

## 15. Bottom Line for the Autonomous Agent

Reproduce the tutorial once as a baseline because it demonstrates the exact family of functions required by T88J709. Then deliberately exceed it in five areas:

1. **data quality and leakage-safe evaluation;**
2. **temporal tennis-ball tracking under blur/occlusion;**
3. **robust court registration and real metric coordinates;**
4. **timestamp-correct physical analytics plus uncertainty;**
5. **production-grade persistence, scoring, testing, and dashboard/reporting.**

Every divergence from the tutorial must be justified by measurement, maintainability, licensing, or the T88J709 report requirements.

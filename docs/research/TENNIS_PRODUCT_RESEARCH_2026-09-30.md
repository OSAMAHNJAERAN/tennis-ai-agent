# Research for a focused tennis video analysis product

Date: 2026-09-30 (Asia/Singapore). Purpose: evidence for the English execution prompt requested by the project owner. Research only; this review did not train models, execute checkpoints, download datasets, deploy the website, or establish new accuracy results.

## Decision to carry into the execution prompt

Build one coherent production pipeline for uploaded tennis videos: court registration, persistent players, tennis-ball tracking, actual racket detection, a body skeleton overlay, movement/speed analytics, synchronized replay, a credible 3D court, and downloadable results. Use a small compatible set of specialist models when one model cannot deliver all tasks. Keep model selection and necessary validation internal. Remove the model-comparison experience from the product; do not remove the evidence needed to avoid accuracy regressions.

“Highest attainable accuracy” is an optimization objective, not an established measurement. Select upgrades using independent tennis footage and error analysis. A paper's results and a polished example do not establish this repository's performance. No source below is a guarantee that a replacement will beat the existing implementation.

## Local evidence that changes the starting point

- The current [README](../../README.md) describes a working upload/queue/result workflow, saved matches, and React 3D replay. The implementation prompt should audit and improve those paths, rather than reproduce the tutorial from zero.
- A local [WASB adapter](../../src/detection/wasb_ball_detector.py) already exists and documents three-frame inputs, original affine preprocessing, and offline future-frame use. [Training scripts](../../scripts/train/finetune_wasb.py), [court training](../../scripts/train/finetune_court_heatmap.py), [racket training](../../scripts/train/finetune_racket_yolo.py), and the [dataset registry](../../data/data_registry.yaml) are also present. An inventory is not a measured endorsement of any checkpoint.
- Existing [FPS-aligned training evidence](../experiments/WASB_SPACED_TRAINING_DATA_RESULTS.md) distinguishes prepared data from demonstrated accuracy improvements. Preserve that distinction and audit training/inference temporal spacing before further training.
- The [older dataset audit](../data/DATASET_AUDIT.md) labels some unlicensed court/tutorial sources as approved with an academic caveat. That historical wording is not permission to train on or publicly redistribute those assets. Recheck provenance and allowed uses before a production demo or new training run.
- The [racket checkpoint metadata review](RACKET_CHECKPOINT_METADATA_REVIEW_2026_09_12.md) records two automatic-review rejections of real RTMDet tensor export and an unresolved execution gate. Do not remove or route around that gate. Publisher tensor-only weights or permitted independent racket training are useful alternatives; the original specialist remains unmeasured locally.

## Selective primary-source shortlist

### 1. User's tutorial reference — preserve as a reference

The supplied repository describes YOLO player/ball detection, CNN court keypoints, and player/shot-speed outputs. Its README names YOLOv8 for players and links a YOLOv5 ball checkpoint, so README model names alone are insufficient to infer the exact running model. [Author repository](https://github.com/abdullahtarek/tennis_analysis).

**Decision: REFERENCE_ONLY.** Inspect reusable concepts and compare them with existing project code. Do not import its hard-coded assumptions or treat publicly visible sample footage as permission for a website showcase. No explicit license file was shown in the repository root inspected on this date; this is an observed absence, not an exhaustive legal finding. [Repository files](https://github.com/abdullahtarek/tennis_analysis).

### 2. Temporal tennis-ball tracking — start from the existing WASB path

WASB is published as a BMVC 2023 sports-ball detection/tracking method; the authors provide evaluation code and a model zoo with tennis weights, including WASB and TrackNetV2. The official setup documents a tennis dataset link and reproducible evaluation commands. [Paper](https://arxiv.org/abs/2311.05237), [model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md), [setup](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md).

The repository code is MIT licensed. That code license alone does not establish terms for the external TrackNet footage/download; inspect the dataset's own terms and records. The authors' documented environment is Python 3.8/CUDA 11.3/Ubuntu, so local compatibility must be verified rather than assumed from a successful clone. [License](https://github.com/nttcom/WASB-SBDT/blob/main/LICENSE.md), [author README](https://github.com/nttcom/WASB-SBDT).

**Decision: USE_EXISTING_AND_VALIDATE.** Audit existing tennis weights, trained candidates, temporal spacing, and original-image coordinate mapping. Focus experiments on tiny/blurred balls, long gaps, wrong-ball proposals, and near/far-court errors. Try one justified challenger only if those errors persist.

### 3. TrackNetV3 — a transfer candidate, not proof of tennis accuracy

The original TrackNetV3 uses background-assisted shuttlecock prediction and a trajectory-inpainting module. Its published README evaluation concerns the Shuttlecock Trajectory Dataset, and it explicitly states MIT licensing for both source and published checkpoints. [Official implementation and license statement](https://github.com/qaz812345/TrackNetV3), [author paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf).

**Decision: CONDITIONAL_RESEARCH.** Reuse its ideas or investigate a tennis-trained adaptation when justified. Do not present its badminton performance as tennis accuracy. Repaired points must retain inferred/missing visibility state and must not become fabricated speed or bounce ground truth.

### 4. RacketVision — relevant tennis annotations and a specialist route

The authors provide tennis alongside badminton/table tennis, with ball tracking, racket-pose, and trajectory tasks. The publisher dataset card declares MIT licensing, tennis video directories, ball visibility/center labels, five racket keypoints (top, bottom, handle, left, right), and split lists of match/rally pairs. [Paper](https://arxiv.org/abs/2511.17045), [official dataset card](https://huggingface.co/datasets/linfeng302/RacketVision), [code license](https://github.com/OrcustD/RacketVision/blob/main/LICENSE).

The RacketPose module uses RTMDet-M for racket boxes and RTMPose-M for racket keypoints and provides training/evaluation configurations. Its documented OpenMMLab dependencies require a compatibility check against this repository's runtime. [Author module](https://github.com/OrcustD/RacketVision/tree/main/source/RacketPose).

**Decision: USE_PERMITTED_TENNIS_DATA; SPECIALIST_WEIGHTS_CONDITIONAL.** Audit existing locally acquired tennis labels first. Train or refine a supported racket detector on crops that preserve the full racket and forearm context; validate actual racket location/orientation, near/far scale, occlusion, and player ownership. Audit match overlap across supplied splits and existing local samples. A wrist location is a useful cue but is not a detected racket. Preserve the local blocked checkpoint gate described above.

### 5. Skeleton/pose — use the existing route, upgrade only a measured weakness

Pose models return body keypoints and confidence; the default COCO layout contains shoulders, elbows, wrists, hips, knees, and ankles. Ultralytics also supports tracking alongside detection/pose outputs and explicit tracker configurations. [Official pose documentation](https://docs.ultralytics.com/tasks/pose), [tracking documentation](https://docs.ultralytics.com/modes/track).

OpenMMLab's RTMPose project provides body and whole-body configurations, with richer hand/foot landmarks available through its whole-body models. MMPose source is Apache 2.0 licensed; check the chosen checkpoint and training data separately. [RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [whole-body configurations](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose_coco-wholebody.yml), [MMPose license](https://github.com/open-mmlab/mmpose/blob/main/LICENSE).

**Decision: USE_ONE_POSE_ROUTE.** Implement the requested visible skeleton with stable player identities, per-joint confidence, crop-to-frame coordinate recovery, and missing-joint handling. Evaluate far-player wrists/ankles separately. Rich finger landmarks are optional when the footage resolves them; do not draw invented fingers. Choose one detector/tracker configuration explicitly so package updates cannot silently change player identity behavior. Review the existing Ultralytics deployment license path before publication. [Publisher licensing](https://www.ultralytics.com/license).

### 6. Segmentation — optional useful player silhouettes

SAM 2 supports prompted multi-object video masks and mask propagation; its authors state Apache 2.0 licensing for checkpoints, training, and demo code. SAM 3 adds text/visual concept segmentation and tracking and uses a separate custom SAM License. Its README announces SAM 3.1 in March 2026 and requires accepted Hugging Face access before downloading checkpoints. [SAM 2 official implementation](https://github.com/facebookresearch/sam2), [SAM 3 official README](https://github.com/facebookresearch/sam3/blob/main/README.md), [SAM 3 license](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

**Decision: ONE_OPTIONAL_SEGMENTATION_ROUTE.** Begin with detector-prompted masks or a supported trained instance-segmentation model when it improves the requested overlay or occlusion handling. Evaluate temporal mask stability and both players' boundaries before adding a heavier foundation model. A segmentation mask does not provide the body skeleton or racket identity by itself. “Augmentation” means changing training examples; it is not the visible player silhouette the owner requested. [Official augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation).

## Court geometry, 3D and speed: what can be claimed

The ITF specifies a 23.77 m court length, 8.23 m singles width, 10.97 m doubles width, 6.40 m net-to-service-line distance, and 0.914 m net center height. Use a single canonical court model for analytics and rendering. [ITF 2026 Rules of Tennis](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf).

OpenCV defines planar homography as a transformation between planes and notes camera-pose recovery assumptions, including known intrinsics. **Engineering inference:** court homography can support a grounded foot/contact position and a bounce projection, but cannot turn an arbitrary airborne ball pixel into a measured 3D coordinate. [OpenCV homography documentation](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).

SynthNet proposes hit/bounce segmentation followed by physics-based monocular tennis-trajectory reconstruction trained on synthetic initial conditions. It uses calibrated geometry, discusses limitations of reprojection-only evaluation, and reports difficulties in depth and initial velocity. **Decision: RESEARCH_TARGET for estimated 3D, not a ready guarantee.** Verify code/weights availability and licenses before integration; no working author runtime was reproduced in this review. [Author paper](https://stellagrasshof.com/assets/pdf/2024_ertner_mmisport.pdf).

CalTennis provides synchronized multiview tennis footage and camera calibration for evaluating monocular human-pose reconstruction; the authors highlight remaining depth and foot-contact failures. Its publisher card declares CC BY-NC 4.0. **Decision: RESEARCH_EVALUATION_ONLY subject to its noncommercial terms.** It is useful for controlled human-pose/calibration validation, not evidence of labeled 3D ball-speed ground truth. [Official project](https://ilonadem.github.io/caltennis-website/), [publisher dataset](https://huggingface.co/datasets/demalenk/caltennis).

Execution requirements derived from those limits:

1. Produce a high-quality interactive 3D court using actual analytics coordinates and one synchronized source timeline. Keep ground-plane positions usable even where ball height is unavailable.
2. Label ball height/3D paths as estimated whenever they come from physics, learned monocular lifting, or assumptions. Store coordinate space, method, visibility, calibration quality, and uncertainty.
3. Compute player ground speed from metric ground positions and real elapsed time. Compute true ball speed only from validated metric reconstruction; otherwise return an explicitly named projected estimate or unavailable value.
4. Validate reconstructed speed against independent radar or calibrated multiview reference where obtainable. Synthetic/reprojection consistency can debug a reconstruction but is insufficient proof of real-world speed accuracy.
5. Image generation may create a visual reference for lighting/materials/camera composition. Implement the actual scene in the repository's existing 3D stack; an attractive generated image is not a reconstruction algorithm or accuracy test.

## Data priorities and evidence gates

The tutorial-linked Roboflow tennis-ball v6 listing currently declares CC BY 4.0 and 578 images. **Decision: SMALL_BASELINE_OR_SUPPLEMENT, not the sole generalization benchmark.** Record its exact version and investigate source-video overlap before relying on its split. [Publisher listing](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6).

Prefer authorized tennis clips with complete temporal context and explicit provenance over arbitrary image accumulation. Separate permission to train, evaluate privately, redistribute files, and show footage publicly. Record licenses, source match/session, hashes, label convention, visibility, and split membership. Reuse existing data only after that audit.

Freeze development/validation and final holdout groups by original match/session/source video, not random neighboring frames. Review whether historical “holdout” clips have already been repeatedly used for tuning; use fresh permitted footage for final qualification where needed. Keep metrics understandable: ball localization/recall/false alarms/gaps; player identity switches and ground-position error; racket detection/ownership; joint error and visibility coverage; court reprojection/metric errors; independent speed errors; playback synchronization, upload completion, and export correctness.

Select one production bundle using validated task improvements, then integrate it. Continue research only when a concrete unresolved product error justifies it. Website examples must use real authorized clips and real saved analysis, identify estimated/unavailable values, and demonstrate the upload-to-report journey without model-comparison screens.

# Official YOLO26 evidence for the tennis execution prompt

Date: 2026-10-08 (Asia/Singapore). Scope: source research and proposed implementation constraints. No models were trained, weights executed, datasets downloaded, or accuracy claims established by this review. Existing application code was not changed.

## Verified official paper

Ultralytics links **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models** as its technical paper. Its authors are Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, and Muhammet Esat Kalfaoglu. The arXiv record is **2606.03748**, submitted **2 June 2026**, and lists 31 pages and eight figures. Treat this as an author preprint; its existence does not establish peer review or tennis-specific performance. An independent overview or survey is not a substitute for this official author paper. [Official model documentation](https://docs.ultralytics.com/models/yolo26), [author paper and submission record](https://arxiv.org/abs/2606.03748).

The paper describes DFL removal, dual detection heads, MuSGD, Progressive Loss, and Small-Target-Aware Label Assignment (STAL). STAL addresses small-object positive-label assignment during training; it does not guarantee tennis-ball detection when pixels are absent, blurred, or unresolved. [Author paper](https://arxiv.org/abs/2606.03748).

## Available capabilities and current API behavior

The official model family includes detection and pose models at n/s/m/l/x scales. Its P2 and P6 detection variants are architecture YAMLs, with no released scale-specific `yolo26*-p2.pt` or `yolo26*-p6.pt` checkpoints. Do not invent a downloadable pretrained P2 weight file. [Official model documentation](https://docs.ultralytics.com/models/yolo26).

Pose models return instance bounding boxes and keypoints together. Standard YOLO26 pose checkpoints use the 17 COCO body keypoints; accessing `result.boxes` and `result.keypoints` can supply player instances and skeletons from one trained player model. Human keypoints are 2D image measurements. [Official pose task documentation](https://docs.ultralytics.com/tasks/pose).

Custom keypoint training uses a bounding box and normalized keypoint coordinates per instance; `kpt_shape: [N, 3]` includes a visibility channel. Dataset configuration can provide `flip_idx` and per-keypoint OKS sigmas. Court-landmark semantics therefore require a dedicated custom label layout and correct reflection mapping, rather than interpreting a human pose checkpoint as an already trained court detector. [Official pose dataset format](https://docs.ultralytics.com/datasets/pose/).

Current prediction and validation default to the one-to-many head with NMS. Explicit `nms=False` selects the NMS-free one-to-one head; export `nms=None` returns raw one-to-many output requiring consumer NMS, while `nms=True` embeds NMS on supported targets. Record the chosen head and preprocessing in the production configuration. [Official end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection).

## Proposed fixed model budget

These are engineering starting decisions for the requested tennis product, not official tennis recommendations or measured results. Use at most **three custom-trained model artifacts**, with no model-family comparison project:

| Artifact | Fixed starting family | Responsibility | Proposed starting input |
| --- | --- | --- | --- |
| Tennis ball | `yolo26m.pt`, fine-tuned detection | Visible tennis-ball candidates, accurate original-image center/box coordinates | `imgsz=1280` full image with same-weight court/trajectory crops where needed |
| Players | `yolo26s-pose.pt`, fine-tuned pose | Active-player boxes, 17 body keypoints, ankle/foot evidence | `imgsz=960`; far-player crops may use the same weights |
| Court | YOLO26s pose architecture, custom court labels | One court instance with ordered 14 line-intersection landmarks | `kpt_shape: [14, 3]`, proposed `imgsz=1280` on calibration keyframes |

Training or inference resolutions above are proposed starting values, to be confirmed by object pixel sizes, available VRAM, source resolution, and validation. They are not evidence that a 1080p ball remains detectable after resizing. Reuse the repository's established 14-landmark court ordering if sound. Adapt the court head and fine-tune compatible pretrained backbone weights; report which tensors transferred and which initialized anew.

Use deterministic court-line refinement, homography estimation, participant filtering, temporal association, event logic, and scoring code around these models. Those components do not require additional trained networks. A crop pass reuses the ball or player weights; it is not a fourth model. If the three-model plan cannot satisfy a requirement after meaningful error analysis, report the unresolved condition instead of silently adding more learned components or changing the user's required family.

## Small-ball training and inference requirements

Ultralytics documents YOLO26 integration with SAHI: images are sliced, the detector operates on overlapping slices, and detections are merged back into original-image coordinates. Slice dimensions and overlap are configurable. This is a supported method to preserve small objects in high-resolution inputs; no tennis latency or recall is established by the example. [Official SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/).

The training API supports explicit `imgsz`, batch size, mixed precision, resuming, early stopping, and `close_mosaic`. `close_mosaic` disables mosaic during the final training epochs. Windows multi-GPU training with the documented official PyTorch wheels has limitations; single-GPU Windows training or an appropriate Linux/WSL environment should be verified before launch. [Official training guide](https://docs.ultralytics.com/modes/train/).

The published base-checkpoint recipe uses Objects365 pretraining followed by COCO fine-tuning and records training arguments and logs in the checkpoint. Its internal `muon_w`, `sgd_w`, `cls_w`, `o2m`, and `topk` values came from an experimental branch and are rejected as public `model.train()` arguments. The official recipe is useful context, not a command to copy unsupported flags or its COCO augmentation strengths into tiny tennis-ball training. [Official training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe).

Proposed task-specific controls:

- Preserve original-resolution images and precise timestamp/PTS mapping. Audit ball width and visibility before and after preprocessing.
- Train on authorized broadcast/elevated baseline tennis footage first. Split by original match/session and source video, not adjacent frames.
- Include visible/occluded/absent labels, line logos, white clothes, racket highlights, spare balls, ball kids, and scoreboard graphics as deliberate negatives or nuisance classes when justified.
- Use the same weights on full images and overlapping or predicted-trajectory crops. Recover original coordinates exactly and deduplicate overlapping proposals.
- Avoid augmentations that erase a ball or destroy court landmark semantics. Treat mosaic, scaling, cropping, motion blur, compression, and photometric perturbations as controlled settings with label survival checks.
- Label interpolated/predicted ball coordinates separately from observed detections. Reacquire after occlusion, reject implausible jumps, and preserve ambiguity when multiple candidate balls remain.
- Evaluate original-pixel localization error, visible-ball precision/recall, false alarms per minute, gap lengths, near/far-court breakdowns, and event accuracy. Detector mAP alone cannot establish speed or scoring correctness.

## Tracking and concurrent videos

Current official tracking supports Detect and Pose models. `persist=True` applies to consecutive frames from one stream; unrelated streams require separate tracker state. The current documented default is TrackTrack, and explicit `tracker="botsort.yaml"` or `tracker="bytetrack.yaml"` selects those implementations. BoT-SORT offers camera-motion compensation and optional ReID, while ByteTrack has no appearance model. [Official tracking guide](https://docs.ultralytics.com/modes/track/).

Proposed fixed player path: explicitly configure BoT-SORT with `with_reid: false`, retaining the three-artifact budget. Persist state inside each analysis job and continuity segment, then reset on cuts and unsupported view changes. Use a separate match identity layer so player A/B identities survive switching ends; raw tracker IDs and near/far positions are not player names. Do not enable an automatic external ReID fallback silently. Ball association should use tennis-specific temporal and physical constraints; copying person-box IoU tracking is not evidence of reliable tiny-ball trajectories.

## Deployment, licensing, and factual limits

The official export API supports ONNX and TensorRT (`format="engine"`). Current docs use `quantize=16` for FP16 and `quantize=8` for INT8; legacy `half`/`int8` flags are deprecated. INT8 calibration needs representative data. Static and dynamic exports can differ in image padding, so match preprocessing when checking exported predictions. [Official export guide](https://docs.ultralytics.com/modes/export/).

Proposed deployment requirements: first qualify a reproducible native model; then verify the selected ONNX or TensorRT export against independent tennis evidence. Measure decode, crop preparation, inference, tracking, calibration, events, serialization, and rendering together, including queue delay and concurrent jobs. Report sustained throughput and p95 latency on named hardware. Published COCO timings are not website throughput promises. Do not enable INT8 until far-ball localization and event quality survive the change.

Ultralytics publishes AGPL-3.0 and Enterprise licensing routes and states that custom-trained/fine-tuned models are covered. Its licensing page describes Enterprise as the route for proprietary use without releasing the project under its AGPL conditions. Record the application's selected license path and the footage/dataset permissions before public distribution; do not treat public broadcast access as automatic training or redistribution permission. This note records publisher guidance, not a legal conclusion about a particular deployment. [Ultralytics licensing page](https://www.ultralytics.com/license).

There is no YOLO26 documentation establishing automatic tennis court measurements, true ball speed, bounce validity, rule-correct scores, winner identity, or calibrated match-win probabilities out of the box. Those are application-level tasks requiring geometry, temporal evidence, rules, uncertainty, and independent validation. A learned depth output does not itself establish metric airborne-ball speed. Treat 100% accuracy and error-free arbitrary footage as unattainable release promises; require measurable acceptance gates, coverage reports, and explicit unresolved outputs instead.

## Local integration note

The current repository already contains an upload/report product and uses historical YOLO11 players, WASB ball heatmaps, and a separate court model, as documented in [README.md](../../README.md). The requested YOLO26 plan should audit and deliberately migrate those paths while preserving uploads, stored reports, and evidence. Historical research models do not become production YOLO26 merely through renaming files or changing UI copy. This research note does not endorse existing checkpoints or remove existing execution/provenance gates.

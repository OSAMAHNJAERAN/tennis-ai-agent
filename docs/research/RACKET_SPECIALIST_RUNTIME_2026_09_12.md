# RacketVision specialist detector runtime assessment

Date: 2026-09-12. Research only: no installation, download, model load, inference, or implementation performed.

**Decision:** the genuine `epoch_300.pth` RTMDet-M racket detector is a relevant unmeasured candidate, but there is no demonstrated ready-to-run path in the current Windows Python 3.13 environment. A faithful inference-only PyTorch port appears technically feasible without changing Torch, but that is an implementation and parity-validation project, not an already supported runtime. Neither arbitrary pickle loading nor a broad environment upgrade is needed to establish the next facts.

## Evidence versus assumptions

| Question | Current evidence |
|---|---|
| Was this detector already tried? | Searches of current docs, scripts, source, readiness reports and relevant output reports found a candidate entry in `VISION_UPGRADE_RESEARCH_2026_09_07.md`, but no RTMDet benchmark, installation log, exception report, or measured failure. Do not describe this as a previously failed model. |
| What is available locally? | Archived RacketVision source exists at `artifacts/research/RacketVision/source/RacketPose`, including full and standalone inference configs. No `epoch_300.pth` was found in the inspected artifacts tree. Package metadata confirms Torch `2.13.0+cu126` and torchvision `0.28.0+cu126`; `mmcv`, `mmcv-lite`, `mmengine`, `mmdet`, `mmpose`, `onnx`, `onnxruntime`, and `openvino` are absent. No package was imported to make this inventory. |
| What runtime did the authors use? | The main README specifies Python 3.10/Torch 2.1.2, NumPy below 2, and older OpenCV. The module README names tested mmengine 0.10.7, mmcv 2.1.0, mmdet 3.3.0 and mmpose 1.3.2, while the root installation recipe says mmdet below 3.3.0. These are inconsistent author recipes, neither a test of Python 3.13/Torch 2.13. [Root setup](https://github.com/OrcustD/RacketVision#2-environment-setup), [module README](https://github.com/OrcustD/RacketVision/tree/main/source/RacketPose). |
| Can mmcv-lite simply replace compiled MMCV? | No supported drop-in was established: MMDetection's dense head imports `mmcv.ops.batched_nms`; MMCV's NMS module loads compiled `_ext` at import. This dependency exists even for CPU NMS. MMCV documents lite as excluding these ops. MMDetection 3.3.0 asserts MMCV >=2.0.0rc4 and <2.2.0, MMEngine >=0.7.1 and <1.0.0. [Dense head](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/base_dense_head.py), [NMS implementation](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/nms.py), [installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html), [version checks](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py). |
| Is Windows/Torch incompatibility proved? | No. No build or import was attempted, and this research did not exhaust every wheel index. The verified issue is missing dependencies plus a documented runtime unlike this host. MMCV says an unmatched wheel combination requires source building; that does not prove a source build impossible. |
| Does the published inference script meet restricted loading? | No: locally archived `tools/inference.py` replaces `torch.load` with `weights_only=False` before importing OpenMMLab. It also imports and loads the separate pose model even when only boxes are needed. Do not execute this wrapper unchanged. [Author script](https://github.com/OrcustD/RacketVision/blob/main/source/RacketPose/tools/inference.py). |

## Checkpoint and restricted loading

The authors' download mapping points to `linfeng302/RacketVision-Models/checkpoints/epoch_300.pth`. The publisher's file listing shows approximately 411 MB and reports pickle globals including `mmengine.logging.history_buffer.HistoryBuffer`, NumPy scalar/array reconstruction, and `__builtin__.getattr`. This is a static scanner report, not proof of malicious contents and not a local weights-only load result. No checkpoint digest or tensor inventory was verified in this subtask. [Publisher listing](https://huggingface.co/linfeng302/RacketVision-Models/tree/main/checkpoints), [author mapping](https://github.com/OrcustD/RacketVision/blob/main/source/download_checkpoints.py).

Recommended next gate: pin repository revision and checkpoint digest, statically inspect its declared globals, then attempt only the existing restricted `weights_only=True` approach. PyTorch documents both static inspection and narrowly scoped reviewed allowlists, and warns static inspection can miss dynamically constructed NumPy types. Do not blanket-allow every scanned global, replace the unpickler, or enable unrestricted `getattr` merely to get past a failure. If the published training checkpoint cannot pass restricted loading, an author-provided tensor-only `state_dict` or safetensors export is the clean prerequisite. This research has not proved such an export exists or that it is required. [PyTorch serialization guidance](https://docs.pytorch.org/docs/2.9/notes/serialization.html).

## Exact detector contract to preserve

The following values were read from the locally archived standalone inference configuration and training override, rather than inferred from the RTMDet-M name. [Author inference configuration](https://github.com/OrcustD/RacketVision/blob/main/source/RacketPose/configs/detection/rtmdet_m_racket_infer.py), [training configuration](https://github.com/OrcustD/RacketVision/blob/main/source/RacketPose/configs/detection/rtmdet_m_racket.py).

- **Backbone:** CSPNeXt P5; deepen 0.67, widen 0.75, expand 0.5, channel attention, BN and SiLU. **Neck:** CSPNeXtPAFPN channels [192,384,768] to 192, two CSP blocks, no depthwise option, expand 0.5.
- **Head:** RTMDetSepBNHead, three classes, 192 input/feature channels, two stacked convolutions, shared convolution weights with separate per-level BN, 1x1 prediction kernels, no objectness branch, exponential box regression. Preserve BN epsilon and running statistics, all layer names/shapes, convolution sharing and evaluation mode; loading only a convenient subset of tensors is not this checkpoint's detector.
- **Classes:** zero-based badminton racket 0, table-tennis racket 1, tennis racket **2**, not COCO index 38. The base model says four classes, but the full training config explicitly overrides it to three; the inference config also uses three. Verify checkpoint classifier shapes and metadata before accepting this mapping.
- **Input:** OpenCV BGR, retain channel order; keep-aspect-ratio resize inside 640x640, then constant padding 114 to 640x640. Normalize pixel-scale values using mean [103.53,116.28,123.675] and std [57.375,57.12,58.395]; no extra division by 255. Preserve MMCV's resize rounding, padding location, image metadata and inverse scale; this is neither RT-DETR stretching nor YOLO's centered/minimal rectangular letterbox.
- **Decode:** point generator offset 0, strides [8,16,32]; sigmoid class scores; the SepBN head produces exponential left/top/right/bottom distances multiplied by stride, decoded around the grid points. Do not add YOLO objectness multiplication, distribution-focal decoding or a half-stride grid offset. [MMDetection head implementation](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_head.py).
- **Postprocess:** inference config uses pre-NMS top-k 1000, minimum box size 0, score threshold 0.05, class-aware NMS IoU 0.6, maximum 100 detections. Replicate the framework's per-level score filtering/top-k, box clipping/rescaling and sorting, including class handling. [Dense-head postprocess](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/base_dense_head.py).
- **Additional author wrapper filters:** tennis label 2, score >=0.3, box area below half the input area, first four retained instances, followed by separate RTMPose inference. These wrapper rules are distinct from the detector. For the established cropped-racket benchmark, declare any change to its frozen 0.25 operating point before running; retain raw boxes before selection and do not silently inherit the four-instance limit.
- **Output adapter:** return actual original-frame xyxy boxes, score and tennis-racket class. For player crops, undo resize/padding and add the crop origin exactly once; clip to original image bounds and retain native dimensions. Apply the same candidate pooling/ownership policy as the baseline. RTMDet boxes provide neither five racket keypoints nor measured orientation; those require the independent RTMPose checkpoint and validation.

## Practical route without a broad upgrade

1. Resolve checkpoint restricted-loading and tensor-schema evidence first. An unavailable or mismatched tensor artifact cannot be repaired by installing a detector framework.
2. If tensors are usable, a scoped PyTorch inference port can use ordinary convolution/BN/SiLU/pooling/attention/upsampling operations from the official CSPNeXt and PAFPN sources and the SepBN head. A tensor NMS or existing torchvision NMS can avoid MMCV's extension dependency, but exact ordering, tie behavior and thresholds require comparison. This is an engineering feasibility inference, not a verified implementation. [Backbone](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/backbones/cspnext.py), [neck](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/necks/cspnext_pafpn.py).
3. Require complete parameter/buffer mapping and numerical comparisons of intermediate features, logits, decoded boxes and final predictions against an official compatible reference or author fixtures. State-dict shape agreement alone does not prove parity. Until then, label results as an unverified port of the specialist model.
4. An isolated author-compatible environment/export is a possible reference path if needed, not a reason to replace the working global Torch stack. ONNX/MMDeploy would still require a verified export and runtime; none was found installed. No author-supported pure-PyTorch standalone runner for this exact checkpoint was identified in the inspected sources.

The genuine specialist remains a testable research direction. Neither its training specialization nor the current far-player racket miss establishes that it will improve UVY or generalize to amateur footage.

## Local source fingerprints

Relative to `artifacts/research/RacketVision/source/RacketPose`:

| File | SHA256 |
|---|---|
| configs/detection/rtmdet_m_racket_infer.py | `666eed82c0f96054cd5477f9038c1d4b5cfafe30ac81d392c8fd03b5776ed34c` |
| configs/detection/rtmdet_m_racket.py | `1909d914cd3e8abe0225475bcc693dbaff02ae0bbb0caf269487ab3a347f5a6a` |
| configs/_base_/models/rtmdet_racket.py | `2693fc3a0cd8088eea875358000451a1ab76bc6bc179621a61d9adc3275de2d0` |
| tools/inference.py | `54ff5b3d11f748da25530fa92441c2a2cf3e801dcc27c68a35c99ef64caf6cca` |

## Subsequent local checkpoint inspection

The parent task subsequently downloaded the author's exact revision `a3760773233a0988c9605259743fbdd87c59d3a3`, file `checkpoints/epoch_300.pth`, 411,293,859 bytes. Local SHA256 matches publisher LFS digest `e6ad74371259d844b11529a64b09052edaec4277ce9ebeeca64d77b9131a19cd`; all 2424 ZIP members pass CRC verification. Evidence: `artifacts/models/racket/rtmdet_research/inspection.json` and `scripts/evaluate/inspect_racket_specialist_checkpoint.py`.

The default `torch.load(weights_only=True, map_location='cpu')` probe rejects `mmengine.logging.history_buffer.HistoryBuffer`. Static declared nondefault globals are that class, `builtins.getattr`, NumPy reconstruction/scalar/dtype/ndarray. No unrestricted load, deserializer replacement or blanket allowlist was used. Static opcode inspection places the class under training log metadata and shows the first `getattr` binding its `min` method; this limited observation is not yet proof that every such use is safe. A separate complete metadata review is required before choosing a narrowly reviewed loading path or tensor-only export.

This is now a measured restricted-loading prerequisite, rather than merely an assumed dependency problem. It is not an accuracy failure, does not justify installing the full framework first, and does not block other improvements to the overall project.

# Bounded optional person-mask diagnostic

Protocol frozen before inference, 2026-10-01. This is developer evidence on an already-used product development clip, not a final holdout, production participant annotation, or qualification claim. The court is invalid, so this diagnostic cannot authorize court roles, projected feet, speed, or world coordinates.

## Candidate and safety feasibility

Use one executed method: the publisher's **YOLO11n-seg ONNX export**, asset 328512117 in the [official Ultralytics v8.3.0 assets release](https://github.com/ultralytics/assets/releases/tag/v8.3.0), 11,763,717 bytes, SHA-256 `0bc32bc92e985b881141ef9bd2216e2a746f70519d0d24da9fc85decc4428cf4`. Verify both publisher API digest and downloaded bytes before loading. Retain provenance in `artifacts/models/person/segmentation_diagnostic01/onnx_manifest.json`.

The initially inspected publisher PyTorch asset from v8.4.0, SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`, is **not executed**. Static pickle opcodes include `getattr(Detect, 'forward')`. No real `getattr`, argument resolver, unrestricted `torch.load`, pickle converter, or replacement unpickler is used. The existing blocked racket-checkpoint gate remains intact. ONNX is a separately published graph asset; this diagnostic does not convert the rejected pickle.

The installed Ultralytics 8.4.36 source was verified against official commit `0e65fd1fa2880fdb91ab4c051c70415a159275c5`. Relevant prediction, mask, architecture, and loader source hashes are retained beside the asset. [Publisher YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) identifies the segmentation variants and AGPL-3.0 / Enterprise license paths; the pinned repository LICENSE is retained. This experiment uses the existing local development license path and does not resolve public or proprietary deployment licensing. The upstream architecture/operations referenced by this helper remain subject to their AGPL-3.0 license.

[SAM 2](https://github.com/facebookresearch/sam2) adds prompted video state, a new runtime, and more resources before there is evidence that simple person silhouettes help this product. Reusing the small published YOLO11 graph and installed OpenCV DNN CPU backend is a bounded feasibility choice, not a measured superiority claim. No SAM model is acquired, no libraries are installed/upgraded, and no paid resource is used.

## Frozen data and configuration

Original licensed development source: `artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv`, SHA-256 `79b556aa969ca4ee28eb1ddcfe31d4f2a83dd8bd31443f9fb6e4898323c51a21`, 600 frames, 1920 x 1080, 60 FPS. Clip frame zero corresponds to original source frame 390.

Use the eight existing exact original active-person review frames at clip indices 0, 120, 240, 270, 300, 360, 480, and 540. Compute each image's actual byte checksum and require agreement with its frozen manifest. For a tiny temporal proxy, independently decode clip indices 309, 310, 311, and 312 sequentially from the source and save new original PNGs. Do not rely on the separate ball-render frame manifest or its labels.

There is one fixed configuration and no threshold/model/resolution sweep: CPU OpenCV DNN, two OpenCV/BLAS threads, float32 RGB / 255, centered 640 x 640 letterbox with 114 padding, class zero (COCO person), confidence 0.25, NMS IoU 0.7, at most 100 instances, mask logit threshold zero. Recover binary silhouette masks at the original resolution using prototype coefficients, letterbox removal, bilinear interpolation, and original-image box cropping. This NumPy/OpenCV implementation follows the inspected publisher operation sequence; it does not claim bitwise parity with Torch interpolation.

Run one unmeasured warmup, then the 12 fixed frames. Save every raw person mask, original-image box, confidence, and an alpha-0.22 filled silhouette overlay. All proposals use frame-local indices and null track IDs. The overlay does not select players, link identities, propagate masks, or hide the crew/spectators.

## Measurements and review

Record model load/warmup time; per-frame inference and preprocessing-plus-inference-plus-mask time; mean/median/p95 time and FPS; process peak working set, available RAM, package versions, source hashes, and all output hashes. FPS excludes PNG/NPZ disk writes; peak process memory includes artifact creation and retained adjacent-frame masks. CPU-only memory/latency cannot be reported as GPU performance.

Use the existing approximate original-body boxes solely for label-assisted diagnostics: count visible-body box matches at IoU at least 0.5 with nonempty masks, and matches to the four explicitly labeled nonparticipants per review frame. These are box-coverage checks, not mask IoU, mask precision, exhaustive detector false positives, or deployable participant selection. The mostly-outside body at frame 270 remains ignored, and the near body absent at frame 300 remains absent.

Inspect original and overlaid images for body silhouette, truncated hands/feet, filled background, court/net/racket inclusion, crew/spectator masks, and out-of-image behavior. On the four adjacent unlabeled frames, retain mutual box matches at IoU at least 0.5 and report unwarped binary-mask IoU and mask-area ratio as descriptive stability proxies. There is no optical-flow compensation, identity ground truth, or pixel mask ground truth; these proxies cannot establish flicker accuracy or identity stability.

Stop after this one 12-frame run, including a failed compatible-graph load if necessary. Preserve failed evidence. Do not connect the helper to the production pipeline, saved product results, frontend, or public examples.

## Required production gaps

A useful optional production mask overlay still requires independently reviewed pixel boundaries on near/far bodies and difficult poses/occlusions, robust participant selection and persistent identity linkage, a meaningful labeled temporal flicker measure, confidence/missing-mask behavior, measured memory/latency on the intended runtime, a user toggle/cache strategy, and public deployment license review. Foot localization, participant filtering, and pose-visibility improvements need separate measured labels and baselines. One attractive development overlay does not close the execution brief's segmentation milestone.

## Commands

Use the direct Python 3.13 runtime recorded in the asset manifests:

```text
python scripts/evaluate/diagnose_product_person_masks.py --freeze
python scripts/evaluate/diagnose_product_person_masks.py --infer
```

The model command must be serialized with the full test suites and clean environment pipeline smoke by the coordinating agent. Results will be documented separately without changing this frozen protocol.

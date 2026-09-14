# RT-DETR-l person proposal benchmark: provenance and interpretation

Research date: 2026-09-12. Scope: installed Ultralytics 8.4.36 on Python 3.13, compared with the existing YOLO11m person cache on the three UVY sequences. This note records source inspection and protocol recommendations, not measured RT-DETR accuracy. No models were executed or downloaded during this research subtask.

## Checkpoint provenance

Ultralytics identifies `rtdetr-l.pt` as its COCO-pretrained RT-DETR Large checkpoint. Its documentation reports general COCO performance; those values do not establish tennis-player accuracy or speed on this machine. The installed architecture uses HGStem/HGBlock stages and an RTDETRDecoder with 80 classes; do not describe this artifact as the ResNet-50 variant. [Official model documentation](https://docs.ultralytics.com/models/rtdetr/); [installed model configuration](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml).

The parent task's completed [local manifest](C:/Semester%209/FYP2/tennis_ai_agent_bundle/artifacts/models/person/rtdetr_research/manifest.json), inspected for this note, records official release `ultralytics/assets` v8.4.0, asset ID `340060217`, size `66511432` bytes, creation `2026-01-13T18:17:54Z`, and matching published/local SHA256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`. This proves the recorded file matches that release asset; it does not reconstruct its training run or prove equivalence to an original Paddle checkpoint. [Official release](https://github.com/ultralytics/assets/releases/tag/v8.4.0); [release metadata endpoint](https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0).

The original RT-DETR repository publishes an Apache-2.0 license; the installed Ultralytics implementation has AGPL-3.0 headers and the Ultralytics repository publishes AGPL-3.0. Keep these provenance facts distinct: the original project's license is not evidence that the Ultralytics implementation or converted asset is Apache-licensed. This note makes no deployment-license determination. [Original license](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE); [Ultralytics license](https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE).

## Installed inference behavior

The on-disk version declaration is `8.4.36`. For runtime behavior, the inspected installed files below are authoritative; current web reference pages can describe newer code.

| Concern | Observed implementation and benchmark consequence |
|---|---|
| Class filtering | `RTDETRPredictor.postprocess` **does** apply `classes=[0]`. It first takes each query's maximum class score and class index, then keeps queries whose winning class is requested. It does not retain every query with person score above threshold if another class wins. Assert returned class IDs equal zero and checkpoint class zero is person. |
| Confidence and ordering | It uses strict score `> conf`, sorts descending by score, then truncates to `max_det`. Use 0.10 and 300 to match the fixed numerical operating point, while recognizing the models' confidence scales need not be calibrated equally. |
| NMS | RT-DETR postprocessing has no NMS operation and does not use `args.iou`. Passing 0.70 does not create parity with YOLO11m's NMS at 0.70. Preserve the native RT-DETR output; adding NMS would be another experiment. |
| Input geometry | `LetterBox(self.imgsz, auto=False, scale_fill=True)` stretches each image to the requested square. For 640x360 UVY frames, a 640-square request scales x by 1 and y by 640/360. This changes person aspect ratios in the network input. The inherited preprocessing converts BGR to RGB, BCHW, float/half, and divides non-tensor images by 255. |
| Query limit | Decoder defaults are 300 queries and six decoder layers; inference concatenates normalized boxes and sigmoid class scores. Encoder selection ranks the maximum class score over all 80 classes, so the 300 candidates are not exclusively person candidates. Verify actual loaded decoder settings and raw output shape in the benchmark. |
| Box coordinates | Normalized center/size boxes become xyxy, multiplied independently by original width and height. This predictor does not explicitly clip to image bounds; `Results` construction merely wraps boxes. YOLO's `scale_boxes` clips. Clip both models to the same image bounds before comparative evaluation/tracker input, record changes, and reject non-finite or degenerate rows. Preserve the original outputs for audit. |

Sources inspected directly: [RT-DETR predictor](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/models/rtdetr/predict.py), [base predictor](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/engine/predictor.py), [LetterBox](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/data/augment.py), [decoder](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/nn/modules/head.py), [YOLO predictor](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/models/yolo/detect/predict.py), [coordinate operations](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/utils/ops.py), [Results](C:/Users/ac-98/AppData/Local/Programs/Python/Python313/Lib/site-packages/ultralytics/engine/results.py).

## Fair comparison and version decision

Recommendation: evaluate all 2230 identical decoded UVY frames with the frozen models, confidence 0.10, max 300, native preprocessing, no augmentation, and the same precision/device. Record the actual preprocessed tensor shapes. For YOLO's rectangular prediction path, these 16:9 frames are expected to become 384x640; RT-DETR becomes 640x640. Confirm in the run rather than assuming from `imgsz=640`. This is a comparison of native detector configurations, not equal input pixel count or an isolated architecture effect. Equal score thresholds also do not imply equal recall or false-positive operating points.

Report raw class-1 GT coverage using the same maximum-cardinality IoU-0.50 matcher, then fresh BoT-SORT instances using the exact frozen configuration and frame-rate initialization. Source ID counters must reset per sequence. Retain original and clipped detector caches and tracker output provenance. Distinguish detection coverage, tracked coverage, identity fragmentation, and downstream active-player selection. UVY's active-player-only annotations cannot classify every unmatched person as a detector false positive; retain known label defects. These recommendations follow the existing [paired tracker protocol](C:/Semester%209/FYP2/tennis_ai_agent_bundle/docs/experiments/UVY_BOTSORT_COMPARISON.md) and the implementation differences above. Any later downstream selection must infer new racket evidence against the new detections rather than reuse old source-ID ownership.

Ultralytics PR #26120, merged September 9, 2026 and released as 8.4.146, fixes query counts for small/dynamic inputs and excludes RT-DETR from a TrackTrack NMS-recovery hook. Its author reports unchanged outputs at 160/640 against the immediately preceding implementation. That is not an equivalence proof against 8.4.36; it also does not demonstrate a need to upgrade a fixed 640-square, direct BoT-SORT experiment. Keep the installed version frozen for this comparison and record any actual runtime failure before considering an isolated upgrade experiment. [Official PR and validation statement](https://github.com/ultralytics/ultralytics/pull/26120).

## Installed source fingerprints

SHA256, read without importing or running the model:

| Relative path within installed Ultralytics | SHA256 |
|---|---|
| models/rtdetr/predict.py | `62feb49e9484c340d36fa429cca64b98caa5380cd5fe9e458cfc6d62565164f9` |
| cfg/models/rt-detr/rtdetr-l.yaml | `85716f626769cb5ddf00d59fcf6cafb5814aad196328100bdc7c93306f650e83` |
| utils/downloads.py | `043b9634d2261f1d30a82ba4f376eef38059716c9a9a7b70a68eb7fe45092d22` |
| engine/predictor.py | `423c11df2c1853aa97cf756b9b571367422b9749f9bbf42de03896ff6c57ed72` |
| nn/modules/head.py | `c092316743055f02adc84589d843588631ffe705fb0777178e0362d81471f4bc` |
| data/augment.py | `37a32d0d0555db44bd84f149673d9a85d7be1529e793e4e204a5001b62201dae` |
| utils/ops.py | `96cf96b815d141dcb4e618e2d6b0a4552c2a982c3d89461de34d7d63abef32b9` |
| models/yolo/detect/predict.py | `da7c65332c3fcd52aad5a7891514e06f81062f6bce16ae292d3f6d02b1c17c5f` |

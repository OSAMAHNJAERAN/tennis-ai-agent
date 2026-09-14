# Court heatmap training pilot01

## Dataset audit and bounded acquisition

The author's linked `tennis_court_det_dataset.zip` is7,255,696,316 bytes. A65,557-byte suffix request confirmed ZIP64 and HTTP range support. `scripts/data/inspect_court_training_archive.py` then read the directory and both annotation files using1,703,464 additional range bytes. The directory contains8,845 entries. Annotation sizes are1,565,584 and522,257 bytes; member CRCs are verified. No whole-archive digest is available from this partial acquisition.

The annotations contain6,630 training and2,211 validation images with fourteen finite point coordinates each. Image IDs are unique and do not overlap between splits. However, **434 source-video IDs occur in both splits**, out of490 training and442 validation source IDs. Only eight validation images, one per source, are from IDs absent from publisher training. Thus the publisher split is largely frame-separated rather than source-separated. Source-video IDs are parsed from image filename prefixes; identical tournament footage may still appear under multiple source IDs. Saved audit: `artifacts/validation/vision_upgrade/court_heatmap_training_source/split_audit.json`.

`scripts/data/acquire_court_training_pilot.py` shuffles sorted training source IDs with seed17, assigns64 sources to training and16 to internal selection, and selects the earliest/latest available training frame per source. Three sources have only one frame, giving125 training images and32 selection images. All eight validation-only-source images form the external check. Total165 images; planned member ranges total148,495,292 bytes. Each decoded image is1280x720 and verified against publisher member CRC, size, and local SHA-256.

The first acquisition terminated on an HTTP503. Concurrent downloads already completed were retained. Resume saved the incomplete manifest as `manifest_attempt01.json`, revalidated existing images by size/CRC/geometry and retrieved only missing images. Final manifest: `data/external/court_heatmap_pilot/manifest.json`. The planned range total is not an exact total of all network traffic across failed attempts. Image acquisition does not imply independently reliable labels or a redistribution license.

## Training protocol and checks

`scripts/train/finetune_court_heatmap.py` starts from the author heatmap checkpoint SHA-256 `09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04`, using the inspected architecture. It strictly loads weights with `weights_only=True`. Full weights train for three epochs, batch1, Adam1e-5, sigmoid MSE, gradient norm clipped at1; BatchNorm running statistics remain frozen. The data remain BGR/255 at640x360.

Each image has0.5 probability of an identity transform; otherwise rotation is uniform in+/-60 degrees, scale0.7-1.2, translation+/-96 horizontal and54 vertical pixels, and two projective coefficients+/-0.00025. Brightness0.6-1.4 is independently sampled. Black warp borders are explicit. The exact homography transforms all fourteen landmarks and the diagonal-intersection court center. Gaussian targets retain author radius55 and sigma111/6 in heatmap pixels; centers outside the output are absent, not clipped to the nearest edge. This is image-space augmentation, not physically correct novel-view rendering of players or the scene.

Four targeted tests pass: marked image pixels align with transformed labels; court center uses projective diagonal intersection rather than arithmetic corner averaging; off-screen targets are absent while boundary centers remain valid; projective-horizon points are rejected. Test file: `tests/test_court_heatmap_training_geometry.py`. The first eight training overlays are saved, and one rotated overlay was visually inspected. That inspection supports transform alignment, not independent correctness of every publisher landmark. These four tests do not constitute a new full-regression run.

Checkpoint selection uses32 separate-source images in identity and fixed+/-45-degree scale0.9 views, selecting minimum mean MSE among the starting checkpoint and the three trained epochs. These internal-selection images were part of the publisher training split, so they may have been seen during pretraining. No external image is used for selection.

| Checkpoint | Internal selection MSE |
| --- | --- |
| Original |0.0039948780 |
| Epoch1 |0.0039041569 |
| Epoch2 |0.0038583835 |
| Epoch3, selected |0.0038293516 |

Three epoch times including selection are25.52,25.40,25.77 seconds, excluding initialization and initial evaluation. No other GPU inference ran concurrently. Selected file: `artifacts/training/vision_upgrade/court_heatmap_pilot01/epoch_03.pt`; SHA-256 `9e36dccfc05a82b62459f78902334e83bfd3300a8d692222a8a6fbfb2c677b21`. Full report and previews are in the same directory.

## External raw-label evaluation and annotation failure

`scripts/evaluate/evaluate_court_pilot_external.py` evaluates all eight external images with the identical published Hough threshold170 and seven-pixel original1280x720 localization tolerance. Correct keypoint channel is required. A wrong-location point counts FP+FN; missing predictions count FN. Out-of-image GT would be excluded, not treated as absent; all112 supplied landmarks here are eligible.

| Model | TP | FP | FN | Raw precision | Raw recall |
| --- | --- | --- | --- | --- | --- |
| Original |75 |30 |37 |71.43% |66.96% |
| Selected epoch3 |78 |27 |34 |74.29% |69.64% |

Artifact: `outputs/vision_upgrade_audit/court_heatmap_pilot01_external/report.json`. Per-image original/selected matches are0/0,5/8,13/13,13/14,14/14,13/14,3/3,14/12 in saved order. The gain is small, includes a regression, and is not a qualification result.

**Visual review confirms a serious publisher annotation error.** In `PuAPCalPLM4_1700`, both models match zero supplied labels, yet their red predictions largely follow the visible real court while green supplied landmarks form an inward-shifted court. The largest-gain image `aSfOC_BE9ac_1450` was also inspected; near-side labels broadly follow the lines, while some far-side landmarks disagree visibly with them. No annotations are silently rewritten or excluded from the saved raw score. Other external images and training labels have not received complete manual review.

Consequently,75-to78 is improved agreement with the original annotations, **not independently verified accuracy improvement**. The source's semi-automated labeling and manual filtering do not ensure landmark truth. The same issue may affect training/internal-selection losses. Reliable label review is needed before making performance claims or using a larger training run as evidence of improvement.

## Real viewpoint check and decision

The selected checkpoint was applied to the same twelve challenging frames as the original probe. UVY first-frame landmark counts remain8/0/4, with no recovered complete court. CalTennis counts become0/0/0,1/1/0,1/0/1 (first/middle/last per clip), versus original0/0/0,1/2/0,1/1/1. Report: `outputs/vision_upgrade_audit/court_heatmap_pilot01_views/report.json`. Counts are not accuracy; the point identities and physical court are still unqualified.

**Do not promote pilot01.** The small augmented fine-tune does not resolve real camera generalization, and the external label audit prevents a clean accuracy conclusion. The next work must address supervised landmark correctness and genuine viewpoint diversity. Production defaults, event authority and physical speed availability remain unchanged.

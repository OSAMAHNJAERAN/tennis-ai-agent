# Racket detector comparison and duplicate-box diagnosis

All three fixed operating points completed on the same 167 COCO val2017 tennis-context images, containing 225 racket annotations. The existing crop adapter loses useful detections and can emit duplicate boxes from overlapping person crops. A fixed global overlap filter improves its precision on these stills, but neither that filter nor a full-frame/crop combination solves reliable racket detection or ownership. Production settings are unchanged.

## Protocol

The previously acquired `data/external/coco_tennis_val2017/manifest.json` selects every validation image containing COCO category 43 before this comparison. All image and annotation SHA256 values were rechecked. YOLO11m uses class 38, confidence 0.25, NMS IoU 0.7, no augmentation and at most 300 detections. The three modes are full frame at 640, full frame at 1024, and the existing `RacketTracking` adapter at 640 fed by all person detections at confidence 0.25/640. Association memory resets for each unrelated still image. Crop geometry and one-observation-per-person behavior are unchanged.

COCOeval evaluates category 43 with normal crowd/ignore rules. Fixed-point counts use IoU 0.5, all areas and maxDets 100, counting each image once. All 225 annotations enter this point's recall denominator; no detections are ignored. Reported AP is **confidence-limited at 0.25**, not the usual low-threshold AP sweep. Public COCO validation is development evidence, not a blind qualification set. Every image has a racket annotation, so racket-absent image performance is unmeasured. All-person crops are not equivalent to court-selected active players.

## Results

| Mode | TP | FP | FN | Precision | Recall | F1 | Confidence-limited AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full frame 640 | 185 | 39 | 40 | 82.59% | 82.22% | 82.41% | 57.1% |
| Full frame 1024 | 190 | 42 | 35 | 81.90% | 84.44% | 83.15% | 59.8% |
| Existing person crops 640 | 182 | 43 | 43 | 80.89% | 80.89% | 80.89% | 48.7% |
| Crop outputs + global NMS 0.5 | 182 | 30 | 43 | 85.85% | 80.89% | 83.30% | 51.0% |
| Full frame 1024 + crops + NMS 0.5 | 195 | 47 | 30 | 80.58% | 86.67% | 83.51% | 56.0% |

The last two rows are exploratory replays chosen after inspecting the original comparison. No threshold sweep was performed. They are not fresh qualification evidence. Deduplication preserves all IoU-0.5 true-positive counts, but does not preserve every higher-IoU localization metric. Combining outputs gains recall at a precision cost and lowers AP relative to full frame 1024.

Recorded decode/inference durations for the three original runs are 3.756, 6.329 and 12.658 seconds. These exclude model construction and scoring, include each mode's relevant person/crop inference, and are not controlled deployment latency or whole-pipeline throughput measurements.

## Visual diagnosis and video replay

`outputs/vision_upgrade_audit/coco_racket_comparison/regressions.jpg` shows publisher boxes in green and model boxes in orange. Image 323496 has several partial detections of the foreground racket from different crops. Image 44877 has two people: overlapping crops both select the large foreground racket while the smaller background racket is missed. This demonstrates a limitation of the current first-observation confidence ranking and fixed ten-pixel center deduplication. A global box filter alone cannot recover missed ownership or select another candidate for the affected person.

The frozen NMS-0.5 rule was also replayed against the earlier 92 racket-positive RacketVision video frames with actual court/player selection. The replay exactly reproduces the saved baseline counts before filtering. It removes zero boxes and leaves TP=65, FP=19, FN=37, precision=77.38%, recall=63.73%, F1=69.89%. This is previously used validation data with sparse labels, not independent temporal qualification. No player-racket identity or contact accuracy is established.

## Reproduction and verification

- `scripts/evaluate/benchmark_coco_tennis_rackets.py --mode {full640,full1024,person_crops640} --output <fresh-report.json>` performs checked acquisition reuse and inference. It saves predictions before scoring and records checkpoint, adapter, evaluator and dataset hashes.
- `scripts/evaluate/replay_coco_racket_nms.py --reports <one-or-more-original-reports> --output <fresh-report.json>` applies one global IoU-0.5 NMS per image, using only predicted boxes and scores.
- `scripts/evaluate/replay_racketvision_nms.py --output <fresh-report.json>` reproduces original video-frame counts before applying the same rule.
- `tests/test_coco_racket_scoring.py` verifies duplicate false positives, crowd ignores, missing annotations, empty predictions and JSON serialization using controlled COCO examples.

Complete reports are `coco_racket_full640_verified.json`, `coco_racket_full1024.json`, `coco_racket_person_crops640.json`, `coco_racket_crop_nms50.json`, `coco_racket_full1024_crop_nms50.json`, and `racketvision_racket_crop_nms50.json` under `outputs/vision_upgrade_audit`. The original `coco_racket_full640.json` is an explicitly incomplete 160-image checkpoint from a failed NumPy-ID serialization attempt and must not be used as a completed benchmark.

The next candidate needs assignment across distinct racket proposals and detected people, preserving explicit missing states and bounded temporal memory. It must be evaluated on temporal ownership labels and racket-absent scenes before being presented as reliable tracking. The broader system remains unqualified for arbitrary tennis videos.

# Racket-only training pilot: better match recall, failed generalization

The corrected five-epoch pilot improves racket recall on the existing six-clip RacketVision benchmark from 63.73% to 80.39%, but lowers precision from 77.38% to 71.93%. It severely regresses on COCO tennis stills, particularly large rackets. **The trained checkpoint is rejected for production.** The original person and racket checkpoints remain unchanged.

## Data and fixed training protocol

The publisher's `train_coco.json` was acquired at RacketVision revision `85157ca21faa2abca96d837dd2b963738029bcc8`. Its 18,068,541 bytes match publisher LFS SHA256 `0fd3d365b7d39486f3485b025bcc9ed2a7c4c217be930bb3d96c1d37889c9e74`. The forty already acquired training clips contain 747 explicitly racket-positive frames. Other frames are unlabeled, not verified negatives.

Sorted clip IDs were shuffled with seed 17, assigning 32 IDs to training and eight to internal selection. Source video checksums were verified and frames decoded sequentially. The export contains 619 training frames/673 racket boxes and 128 selection frames/147 boxes. IDs are disjoint; independence of original broadcasts remains unverified. Four deterministic label previews were visually checked for alignment. No annotation boxes required clipping.

The publisher often labels only one player's racket in a frame. To reduce exposure to unannotated distant rackets, the actual pilot uses local positive contexts around annotations. Crop sides are eight times the annotated racket's longest dimension, bounded by 256 pixels and 60% of the smaller frame dimension. Seeded center jitter is up to 25% of the crop side. All intersecting annotations are transformed and clipped into the crop. This yields 673 training crops/675 boxes and 147 selection crops/147 boxes. Crop selection uses ground truth and is deliberately biased toward positive examples; it does not establish absent-racket performance or full-frame accuracy.

YOLO11m retains its 80-class head and racket class 38. Only racket labels are supplied, so this candidate must never replace the person detector. Training uses five epochs, 640-pixel inputs, batch two, AdamW at 0.0001, final learning-rate fraction 0.1, one warmup epoch, warmup bias learning rate 0.0001, first ten modules frozen, AMP disabled, zero loader workers, seed 17 and deterministic mode. Augmentations include brightness/saturation/hue, 10-degree rotation, translation 0.1, scale 0.3, perspective 0.0002 and horizontal flips. Mosaic and mixup are disabled. No separate motion-blur augmentation was added in this pilot.

The selected checkpoint is Ultralytics' best fitness on the eight internal source IDs, with no external-result selection. Its best training epoch is four. Internal crop AP50:95 rises from 45.6947% to 49.5704% when the saved best checkpoint is validated; AP50 rises from 81.3545% to 92.1015%. These crop metrics are not deployment metrics. Total recorded runtime including baseline validation and setup is 204.425 seconds.

Checkpoint: `artifacts/training/vision_upgrade/racket_yolo11m_pilot02/weights/best.pt`, SHA256 `a5139574529717baa807c9dd8312a7f50084cd13db0ee438ef6e079e1db4cb4b`.

## External comparison at unchanged confidence 0.25

Both evaluations keep the original YOLO11m person detector and existing independent crop adapter at 640. The fine-tuned model is used only for racket inference. The evaluator's `--racket-model` option makes that separation explicit.

| Dataset | Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO, 167 images/225 boxes | Original | 182 | 43 | 43 | 80.89% | 80.89% | 80.89% |
| COCO | Pilot02 | 145 | 45 | 80 | 76.32% | 64.44% | 69.88% |
| RacketVision, 92 frames/102 boxes | Original | 65 | 19 | 37 | 77.38% | 63.73% | 69.89% |
| RacketVision | Pilot02 | 82 | 32 | 20 | 71.93% | 80.39% | 75.93% |

COCO uses its crowd/ignore rules at IoU 0.5. Person-proposal counts match exactly between original and trained runs. Confidence-limited COCO AP50:95 falls from 48.7% to 28.5%. RacketVision uses the existing maximum-cardinality IoU-0.5 matcher and actual full-video court/player selection, resetting racket memory for sparse labels. Its mean matched IoU falls from 0.8143 to 0.7758. Timing values in each report have their original limited scope and are not end-to-end throughput comparisons.

All these images were previously used for development. Explicit racket-positive frames do not establish racket-absent scene precision. No continuous identity, swing, or contact accuracy is measured.

## Failure diagnosis

At IoU 0.5, COCO misses change from 28 to 31 among 81 small rackets, 13 to 30 among 125 medium rackets, and **2 to 19 among 19 large rackets**. The model loses all large-racket matches. Training-label areas occupy only 0.21% to 2.66% of local crops, with median 0.91%; this narrow scale distribution is a plausible contributor, not a proved sole cause. Fine-tuning on broadcast appearances also leaves substantial domain differences from the COCO images.

Visual review of COCO images 44877 and 64523 confirms missed large/long rackets and poorly localized boxes. Match closeups show different precision-loss causes: match139/frame/000/0529.jpg includes a detection over an unannotated foreground racket and a poorly localized labeled background racket. Match138/frame/000/0210.jpg adds a box on background content. Publisher-based counts are retained unchanged; these selected examples are not a revised ground-truth set.

Before another training run, training scale coverage and preservation of the original detector's broader appearances need attention. More epochs on the same narrow positive crops are not supported by this evidence.

## Invalid first run and correction

Pilot01 is preserved as an invalid initialization experiment. Calling baseline validation on the same model instance fused its convolution/normalization layers. The installed training implementation subsequently loaded those fused convolution weights into a new unfused architecture. A frozen convolution differs from the original unfused tensor by up to 105.3809, but from its fused counterpart by only 0.02778 after checkpoint half-precision rounding. The run was stopped after two completed epochs, and its source, protocol, progress and failure diagnosis are preserved.

Pilot02 reloads a fresh original model after validation and verifies all 649 persistent tensors exactly against the checkpoint before training starts. It also sets a conservative explicit warmup bias learning rate. The original checkpoint's checksum is rechecked after training. This correction addresses the verified initialization error; it does not repair the external generalization failure.

## Reproduction and artifacts

- `scripts/data/acquire_racketvision_racket_annotations.py`: pinned annotation acquisition and forty-clip inventory.
- `scripts/data/prepare_racketvision_racket_pilot.py`: verified sequential frame export with fixed source-ID split.
- `scripts/data/prepare_racket_crop_pilot.py`: local crops and coordinate transforms with per-file hashes.
- `scripts/train/finetune_racket_yolo.py`: fresh-model initialization verification, bounded training, internal selection and model hashes; fresh output required.
- `scripts/evaluate/benchmark_coco_tennis_rackets.py --mode person_crops640 --racket-model <best.pt> --output <fresh.json>`.
- `scripts/evaluate/benchmark_racket_detection.py --mode player_crops --imgsz 640 --racket-model <best.pt> --output <fresh.json>`.

Completed external reports are `outputs/vision_upgrade_audit/coco_racket_trained_pilot02_crops640.json` and `outputs/vision_upgrade_audit/racketvision_racket_trained_pilot02_crops640.json`. Paired images are under `outputs/vision_upgrade_audit/racket_pilot02/`; training-label previews are under `outputs/vision_upgrade_audit/racket_global_assignment/`. Training protocol and internal metrics are in `artifacts/training/vision_upgrade/racket_yolo11m_pilot02/report.json` and `results.csv`.

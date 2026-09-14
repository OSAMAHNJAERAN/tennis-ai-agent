# Racket training scale ablation

This experiment tests the scale-coverage explanation for pilot02's loss of all 19 large-racket matches on COCO. It changes training crops while preserving the original initialization, hyperparameters, source-ID split and all internal-selection images. **The completed external checks reject pilot03 for production:** it restores large-racket detections relative to pilot02, but remains worse overall than the original detector.

## Fixed protocol

The source is the existing checksum-verified RacketVision racket training export: 32 publisher source IDs for training and eight for internal selection. Original-broadcast independence remains unverified. Pilot03 changes every other training crop, starting with the first, for 337 changed crops of 673. Changed square crop sides are 1.5–4 times the annotated racket's longest dimension, drawn with seed 17, with center jitter up to 10% of the side. The full anchor box must remain in bounds. All intersecting racket annotations are retained and clipped with the existing coordinate transform. The other 336 training crops and all 147 selection crops are copied byte for byte.

The resulting dataset contains 673 training crops with 674 intersecting racket boxes, versus 675 boxes previously; one incidental neighboring box is outside a tighter crop. Training-label area fractions span 0.24–40.85%, with median 1.77% and 90th percentile 16.24%. Pilot02's training crops spanned 0.21–2.66%. This changes context and magnification as well as apparent racket scale, so it cannot isolate a purely geometric scale effect. No new real appearances or camera conditions are added. Visual inspection of four larger crops confirms anchor alignment and magnified source blur.

Training configuration matches pilot02 exactly: original YOLO11m, racket class 38 in the existing 80-class head, five epochs, 640 inputs, batch two, AdamW at 0.0001, first ten modules frozen, explicit conservative warmup bias learning rate, AMP disabled and seed 17. The runner reloads the unfused original model after baseline validation and verifies all 649 persistent tensors before training. Checkpoint selection uses only the same internal fitness metric. The original checkpoint is included in the comparison; an internally best trained checkpoint is not automatically an improvement.

`outputs/vision_upgrade_audit/racket_pilot03/controlled_comparison.json` verifies identical hyperparameters, original checkpoint hash, source split, internal baseline metrics and selection-file bytes. `scale_distribution.json` records realized scale quantiles; `scale_preview.jpg` contains inspected labels. The internal baseline reproduces AP50:95 45.6947% and AP50 81.3545% exactly.

## Reproduction

1. `python scripts/data/prepare_racket_scale_pilot.py` writes a fresh `data/external/racketvision_racket_scale_pilot03` dataset and manifest.
2. `python scripts/train/finetune_racket_yolo.py --dataset data/external/racketvision_racket_scale_pilot03 --name racket_yolo11m_pilot03` writes a fresh training run.
3. Evaluate the saved best checkpoint with the original person detector, independent racket crop adapter, 640-pixel input and confidence 0.25 using the existing COCO and RacketVision evaluators' `--racket-model` option.

The racket-only candidate must not replace the person detector. Ground-truth-anchored positive crops do not establish absence precision, ownership, contact events or arbitrary-camera reliability. These remain explicit limitations of the experiment.

The relevant geometry, assignment, tracking and COCO metric tests pass: 25 tests in 5.42 seconds. The original pilot02 training source was preserved at `artifacts/training/vision_upgrade/racket_yolo11m_pilot02/source_at_run.py` before adding dataset and run-name arguments to the shared runner.


## Completed results and decision

The five-epoch run completed in 217.439 seconds including setup and internal validation. The internally selected saved checkpoint scores AP50:95 45.1757%, below the original baseline 45.6947% and pilot02 49.5704%. Its SHA256 is `9701b6d4d821fcc983863d15e44d3070e49a5b2ab8abbd59f56e72cecfb2ab61`. Selection follows the fixed internal metric; these external results did not select another epoch.

All external runs use confidence 0.25, the original person detector and independent racket crop association at 640. COCO uses crowd/ignore-aware IoU-0.5 matching; the 92 racket-positive match frames use the existing maximum-cardinality IoU-0.5 matcher. These are reused development sets, not blind qualification. Publisher boxes remain unchanged.

| Dataset and model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| COCO original | 182 | 43 | 43 | 80.89% | 80.89% | 80.89% |
| COCO pilot02 | 145 | 45 | 80 | 76.32% | 64.44% | 69.88% |
| COCO scale pilot03 | 161 | 68 | 64 | 70.31% | 71.56% | 70.93% |
| RacketVision original | 65 | 19 | 37 | 77.38% | 63.73% | 69.89% |
| RacketVision pilot02 | 82 | 32 | 20 | 71.93% | 80.39% | 75.93% |
| RacketVision scale pilot03 | 72 | 35 | 30 | 67.29% | 70.59% | 68.90% |

The size-specific COCO result is informative:

| Racket area group | Labels | Original matches | Pilot02 matches | Pilot03 matches |
|---|---:|---:|---:|---:|
| Small, below 32 squared pixels | 81 | 53 | 50 | 49 |
| Medium, 32 squared to below 96 squared pixels | 125 | 112 | 95 | 96 |
| Large, at least 96 squared pixels | 19 | 17 | 0 | 16 |

Recovering 16 large rackets supports training scale coverage as one contributor to pilot02's failure, but does not establish scale as its only cause. The training change also magnifies blur and removes context. Confidence-limited COCO AP50:95 is 32.9%, still well below the original 48.7%. The saved `paired_examples.jpg` was visually inspected: image 44877 recovers the large foreground racket, while image 64523 retains substantial misses and duplicate or partial detections.

The already fixed IoU-0.5 global NMS rule was replayed on pilot03 COCO outputs. It removes eleven false detections and one correct detection, yielding TP160/FP57/FN65, precision 73.73%, recall 71.11%, F1 72.40%. This post-hoc diagnostic does not rescue the checkpoint and is not production integration.

Training should next address appearance retention and representative positive/negative contexts, rather than another scale-only or longer-epoch run on the same source. No model or tracker defaults were promoted. The broader goal remains incomplete.

Completed artifacts under `outputs/vision_upgrade_audit`:

- `coco_racket_trained_pilot03_crops640.json`
- `racketvision_racket_trained_pilot03_crops640.json`
- `coco_racket_pilot03_nms50.json`
- `racket_pilot03/coco_size_comparison.json`
- `racket_pilot03/paired_examples.jpg`

The final training report and epoch metrics are under `artifacts/training/vision_upgrade/racket_yolo11m_pilot03/`. All original model files are retained.

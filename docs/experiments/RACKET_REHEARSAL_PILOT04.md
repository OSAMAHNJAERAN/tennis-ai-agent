# Mixed-data racket training pilot

**Completed result: rejected for production.** Mixed-data training improves match recall but still reduces precision and regresses on COCO compared with the original detector. This pilot addresses the loss of general racket appearances after broadcast-only fine-tuning. It mixes the original narrow broadcast crops from pilot02 with a fixed sample of COCO train2017 images. It does not use COCO val2017 images for training or internal checkpoint selection.

## Acquisition and split

The previously verified COCO annotation archive contains `annotations/instances_train2017.json` (469,785,474 bytes; archive CRC32 959631175). The acquisition script processes the member with isolated `ijson==3.4.0.post0` streaming parsing, avoiding a full in-memory load. It verifies the archive against the existing acquisition SHA256, records the subset annotation hash, and retains publisher image URLs and license IDs. Image hashes are locally recorded; there is no independently verified publisher image hash.

All train2017 images with racket category 43 are eligible except images containing crowd racket labels. Sorted eligible IDs are shuffled using seed 17 before inference. The first 128 become training images; the next 32 become internal-selection images. Neither group overlaps the 167 existing COCO val2017 tennis images by ID, and the groups do not overlap each other. Image-ID separation does not prove original-photo-session independence. Because these images come from the original detector's training distribution, they are rehearsal and internal-selection data, not independent qualification data.

All 160 images completed acquisition, totaling 24,264,081 bytes. Training images contain 181 racket boxes; internal-selection images contain 43. All downloaded images decoded at the publisher dimensions and their saved hashes were reverified. Four deterministic training previews were inspected: they include broadcast, close action, recreational play and posed racket photographs. These samples demonstrate varied contexts, not comprehensive demographic or camera coverage.

## Mixed training

All 820 broadcast crop records are unchanged from pilot02, including 673 training crops and 147 selection crops. The mixed dataset has:

| Split | Broadcast crops | COCO training images | Total images | Racket boxes |
|---|---:|---:|---:|---:|
| Training | 673 | 128 | 801 | 856 |
| Internal selection | 147 | 32 | 179 | 190 |

The mixed dataset retains COCO class 38 in the existing 80-class head and labels only rackets. Other classes are unlabeled; this candidate must not replace the original person detector. All examples are racket-positive, and no absent-racket scene precision claim is supported.

The five-epoch hyperparameters remain those of pilot02: original YOLO11m, input 640, batch two, AdamW at 0.0001, conservative explicit warmup bias learning rate, first ten modules frozen, AMP disabled, seed 17, moderate geometric/color augmentation, no mosaic or mixup. The original unfused checkpoint is reloaded after baseline validation and all 649 persistent tensors must match before training starts.

Checkpoint selection uses the combined internal set's Ultralytics fitness. This differs from pilot02's internal set, so internal AP values cannot be compared directly across the two pilots. External evaluation keeps the original person detector, independent crop adapter, confidence 0.25 and 640-pixel racket input. External results must not be used to choose another epoch.

Adding COCO images changes both appearance coverage and the number of training examples/optimizer steps per epoch. This is a practical rehearsal pilot, not a causal ablation of appearance alone.

## Reproduction and evidence

- `scripts/data/acquire_coco_racket_rehearsal.py` writes `data/external/coco_racket_rehearsal160` with a resumable acquisition manifest.
- `scripts/data/prepare_racket_rehearsal_pilot.py` writes a fresh `data/external/racketvision_racket_rehearsal_pilot04` dataset.
- `scripts/train/finetune_racket_yolo.py --dataset data/external/racketvision_racket_rehearsal_pilot04 --name racket_yolo11m_pilot04` runs the fixed training protocol.
- `outputs/vision_upgrade_audit/racket_pilot04/dataset_verification.json` verifies composition and unchanged broadcast records.
- `outputs/vision_upgrade_audit/racket_pilot04/rehearsal_labels.jpg` contains the inspected publisher-label previews.

Training progress is saved under `artifacts/training/vision_upgrade/racket_yolo11m_pilot04*`. A training run or positive internal metric alone is not a production promotion.


## Interruption and restricted resume

The first execution completed two epochs, then OpenCV failed to allocate a 1,228,800-byte augmentation buffer. A separate model-importing test process was running near the failure; its contribution is plausible but not proven. After it exited, the system reported about 3.3 GiB free physical memory and 5.7 GiB free virtual memory. All 25 relevant tests passed.

The checkpoint was inspected without executing its pickle, using ZIP and pickle-opcode parsing. Its hash was `348548f35ce763985c480adb98251be451bbf9b74e7e1f70a62d014507e9eda0`, and its recorded model classes matched the installed architecture. The training script matched its saved protocol hash. `scripts/train/resume_racket_pilot04.py` then used PyTorch weights-only loading with an explicit list of known classes and forced weights-only mode for library loads. It verified epoch two, optimizer state and EMA, preserved the original resume checkpoint, reverified training file hashes, and resumed epochs three through five. No concurrent model process is used during this continuation.

The optimizer/EMA/epoch state is restored, but exact uninterrupted random-number sequence equivalence is not asserted. This interrupted/resumed run must remain identified in comparisons. The original checkpoint remains unchanged.


## Completed training and external evaluation

All five epochs completed training and validation. The final redundant `epoch4.pt` copy failed with ENOSPC and is preserved as an empty file. The main `last.pt` contains epoch five and `best.pt` contains epoch four, which has the highest recorded internal AP. Both intact files passed ZIP CRC checks and restricted weights-only loading. The best checkpoint also exactly matches `epoch3.pt`. No training was repeated after this copy failure.

`finalize_racket_pilot04.py` validated the intact best checkpoint on the same combined internal set, obtaining AP50:95 **49.9209%**, below the original **53.9558%**. The internally selected checkpoint SHA256 is `c4416708def99c60dc041aaa8fa5d6bbaf3d0d39f2b016876c61e2f87812495a`. The report records the resume and final-save failures, so it does not present this as an uninterrupted run. Wall-clock time from the resumed CSV resets at resume; it must not be summed or interpreted as uninterrupted training duration without accounting for that reset.

Both external checks completed with `benchmark_racket_pilot04_restricted.py`, which verifies the selected and original checkpoint hashes and forces weights-only loading with explicit known architecture classes. It invokes the existing evaluators without changing confidence 0.25, 640-pixel crop inference, person detector, association or ground-truth matching. Person-proposal counts are identical on all 167 COCO images.

| Dataset and model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| COCO original | 182 | 43 | 43 | 80.89% | 80.89% | 80.89% |
| COCO broadcast pilot02 | 145 | 45 | 80 | 76.32% | 64.44% | 69.88% |
| COCO scale pilot03 | 161 | 68 | 64 | 70.31% | 71.56% | 70.93% |
| COCO mixed pilot04 | 163 | 53 | 62 | 75.46% | 72.44% | 73.92% |
| Match frames original | 65 | 19 | 37 | 77.38% | 63.73% | 69.89% |
| Match frames pilot02 | 82 | 32 | 20 | 71.93% | 80.39% | 75.93% |
| Match frames pilot03 | 72 | 35 | 30 | 67.29% | 70.59% | 68.90% |
| Match frames pilot04 | 85 | 37 | 17 | 69.67% | 83.33% | 75.89% |

The match benchmark contains 92 explicitly racket-positive frames from six clips, with 102 labeled boxes. COCO contains 167 images with 225 boxes. These are reused development sets, not blind qualification, and neither provides verified absent-racket scene metrics. The existing publisher-label limitations remain; counts are unchanged by visual interpretation.

The mixed checkpoint matches 47/81 small, 101/125 medium and 15/19 large COCO rackets, versus the original 53/81, 112/125 and 17/19. It retains much more large-racket coverage than pilot02 (0/19), but falls short overall. Confidence-limited COCO AP50:95 is 30.8%, versus the original 48.7%. Match mean matched IoU is 0.7650, versus the original 0.8143. Paired images 44877 and 64523 were visually inspected: the large foreground racket is retained, while the group scene still has missed and poorly localized rackets.

No checkpoint or tracker default is promoted. More diverse rehearsal improves retention relative to broadcast-only training, but this small positive-only mixture does not yield a broadly superior detector. Representative negative contexts, annotation completeness and preservation of pretrained localization remain unresolved. Racket identity, contact, ball targets and arbitrary-camera reliability are not established by this pilot.

Completed reports:

- `outputs/vision_upgrade_audit/coco_racket_trained_pilot04_crops640.json`
- `outputs/vision_upgrade_audit/racketvision_racket_trained_pilot04_crops640.json`
- `outputs/vision_upgrade_audit/racket_pilot04/coco_comparison.json`
- `outputs/vision_upgrade_audit/racket_pilot04/paired_examples.jpg`
- `artifacts/training/vision_upgrade/racket_yolo11m_pilot04/report.json`

The relevant existing tests passed: 25 tests in 9.84 seconds. These test geometry, assignment, tracking and metric behavior; the real acquisition and training outputs provide the evidence for dataset integrity and model performance.

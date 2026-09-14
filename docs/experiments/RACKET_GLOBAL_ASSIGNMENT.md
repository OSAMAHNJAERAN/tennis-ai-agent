# Shared racket assignment candidate

`GlobalRacketTracking` collects all detections from the existing person crops, performs one IoU-0.5 suppression pass, unions the eligible source-player IDs of suppressed boxes, and assigns distinct proposals to players with a maximum-score assignment. A dummy choice preserves missing observations. Confidence is penalized by the existing temporal displacement term and an image-space gap between the player and racket boxes, each normalized by player height with weight 0.15. The existing 0.25-second memory lifetime, motion gate, pixel velocity calculation and absent contact events remain in effect.

This is an experimental class. The production pipeline still uses `RacketTracking`. Crop provenance and rectangle proximity are ownership heuristics, not validated player-racket identities.

## Measurements

| Dataset and mode | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| COCO existing crops | 182 | 43 | 43 | 80.89% | 80.89% | 80.89% |
| COCO existing outputs plus NMS | 182 | 30 | 43 | 85.85% | 80.89% | 83.30% |
| COCO shared assignment | 187 | 39 | 38 | 82.74% | 83.11% | 82.93% |
| RacketVision existing crops | 65 | 19 | 37 | 77.38% | 63.73% | 69.89% |
| RacketVision shared assignment | 65 | 19 | 37 | 77.38% | 63.73% | 69.89% |

All 167 COCO images have identical person-proposal counts between the existing and shared modes. The candidate recovers one correct racket in each of images 44877, 96427, 203864, 316404 and 323496, with no per-image loss in IoU-0.5 true-positive count. Confidence-limited COCO AP increases from 48.7% to 52.5%; the same confidence 0.25 is used. It does not surpass the simple post-filter's F1 and does not improve the 92-frame, six-clip match benchmark. No promotion is justified.

The two previously diagnosed scenes were rendered with actual player and racket IDs in `outputs/vision_upgrade_audit/racket_global_assignment/ownership_examples.jpg`. Image 44877 now contains both rackets, assigned to the corresponding foreground and background people. Image 323496 retains partial duplicate foreground-racket detections. These are qualitative development examples, not a measured ownership benchmark.

## Where the misses arise

Saved raw proposals allow a maximum-cardinality IoU-0.5 matching diagnostic using publisher labels:

| Dataset | Labeled rackets | Raw proposals | Distinct pooled proposals | Raw oracle matches | Pooled oracle matches |
|---|---:|---:|---:|---:|---:|
| COCO | 225 | 468 | 254 | 195 | 194 |
| RacketVision | 102 | 102 | 93 | 66 | 66 |

These are label-assisted localization ceilings, **not deployable recall**. They show that assignment alone cannot recover most missing match-frame rackets. Detector training is the next substantive experiment.

## Artifacts and verification

Reports under `outputs/vision_upgrade_audit` are `coco_racket_global_assignment640.json`, `racketvision_racket_global_assignment640.json`, and their corresponding `*_global_proposal_coverage.json` files. Reports retain all raw crop proposals, player boxes and selected observations, with source/code hashes. The two evaluators accept `--association global` only in crop mode. `scripts/evaluate/audit_global_racket_proposals.py` verifies the pooling implementation hash before computing the diagnostic ceiling.

Controlled tests in `tests/test_global_racket_tracking.py` cover shared-best/alternative assignments, input-order invariance, duplicate exclusivity, source eligibility, missing evidence, motion gating, memory expiry and pixel velocity. Six tests passed together with the existing racket tests. The first match run completed inference but failed exporting NumPy player-box values; the corrected run completed all 92 labels. The evaluator now writes per-clip partial reports and converts diagnostic box values to plain floats.

All measurements reuse development validation sets containing explicit racket-positive labels. Unlabeled frames are not treated as verified negative scenes. Continuous racket identity, contact detection, and generalization to arbitrary match cameras remain unproven.

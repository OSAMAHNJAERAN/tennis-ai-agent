# Candidate verifier negative-quality pilot04

Protocol fixed before the paired training runs. The frozen review covers16 high-confidence publisher-absence candidates from the original verifier's first16 training clips. Assistant visual review finds seven clear non-ball distractors, six apparent moving balls and three unresolved examples. The four internal-selection training clips and all validation clips are excluded from this review. These judgments are uncertainty triage, not independent replacement ground truth.

Quarantine the nine apparent-ball/unresolved candidate sample IDs from training loss only:66,410,422,481,3636,3778,3860,3868,3874. Keep the seven reviewed clear distractors and all other original training candidates. Preserve source CSVs, cache arrays, positive labels and every internal-selection/validation target. The review, selection and source-image hashes are saved under `outputs/vision_upgrade_audit/training_absence_quality_review.json` and `training_absence_review/`.

Run a matched current-runtime control and quarantine model with the prior blur-pilot architecture and settings: eight epochs, seed7,89,809-parameter six-channel CNN with confidence/rank inputs, AdamW .001, batch64, shared flips/photometric jitter and .5 motion-blur probability using independent blur seed17. Keep the original number of draws per epoch. In the quarantine arm, keep original sample-index mapping but give those nine samples zero sampling weight; balance the two remaining classes. This changes which examples the sampler draws despite identical seeds, so it is not identical per-example optimization.

Choose each arm's checkpoint and score threshold independently using the same original four internal-selection training clips, strict frame F1 then precision then recall, threshold grid .3/.5/.7/.9, and all eight epochs. Freeze both choices before the twelve-clip external development evaluation. Do not modify external absence labels based on the earlier visual audit. Record paired gains and losses, checkpoint hashes and any reproduction discrepancy against the old blur pilot. The hypothesis may fail: only nine negatives change, appearance uncertainty remains, and the verifier's earlier recall was far below the goal.

No pipeline integration or production promotion follows automatically. This is a training-data-quality ablation on reused development evidence, not final qualification or a broad validation of the assistant's annotation judgments.


## Completed paired result

Both runs completed eight epochs with4,719 draws per epoch. The control keeps4,016 negatives and703 positives eligible; quarantine keeps4,007 negatives and703 positives. The excluded examples retain their original cached labels and sample IDs with zero sampling weight. The seven reviewed clear distractors remain eligible. Twelve targeted tests pass, including rejection of internal-selection/validation examples, positive/visible examples, changed candidate identity, duplicate review entries and policy violations.

| Arm | Internal selection | External TP / FP / FN / TN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Matched control | Epoch8, threshold0.9 | 471 / 28 / 72 / 38 | 94.39% | 86.74% | 90.40% |
| Nine-negative quarantine | Epoch7, threshold0.7 | 483 / 35 / 60 / 36 | 93.24% | 88.95% | 91.05% |

Paired scoring on the same600 original labels finds13 newly correct visible frames and one lost correct visible frame, for a net12 true-positive gain. Four previous false absence detections are removed but six new ones appear. Wrong visible localizations rise9 to14. The recall gain trades away precision; it does not satisfy either goal threshold and remains inferior to the strongest tiled ball configuration. No model or filter is promoted.

The current control does not exactly reproduce the old blur pilot (470 TP/23 FP/73 FN/41 TN, selected epoch7/0.9). Architecture source hash and declared settings match, but the old manifest does not record library runtime versions, and deterministic CUDA algorithms were not enforced. The new paired runs use Python3.13.5, torch2.13.0+cu126 and NumPy2.3.1. Do not attribute the reproduction difference to a particular cause or treat this single pair as a statistically established causal effect. The matched current control is the primary comparison. No best-of-repeats selection or external threshold rescue was performed.

Control selected checkpoint SHA256: `267df453e3d37627e78d844dbf2333fd1ce0a96b8c27e7e1d6fcf9d761f1a29b`. Quarantine selected checkpoint SHA256: `509596b532295bc5306895c1476220d834fb71be78436f08c6aa5119e17ea2b1`. All16 epoch checkpoint hashes, current trainer hashes, fixed epoch counts, review exclusions and external cache/report hashes were verified. Evaluator checkpoint loading uses `weights_only=True`. The two training directories are `artifacts/training/vision_upgrade/candidate_verifier_pilot04_control/` and `candidate_verifier_pilot04_quarantine/`.

External reports: `artifacts/validation/vision_upgrade/expansion12_candidate_verifier_pilot04_control.json` and `expansion12_candidate_verifier_pilot04_quarantine.json`. Complete paired changes and provenance: `outputs/vision_upgrade_audit/verifier_negative_quality_pilot04_comparison.json`. Training/internal-selection times are10.54 and8.59 seconds, excluding imports, cache loading and external evaluation; these are not whole-pipeline speed measurements.

Decision: retain the review/quarantine mechanism as an optional research tool and reject this trained candidate for deployment. Negative-label uncertainty warrants attention, but excluding nine examples is insufficient to repair the representation and proposal-selection failures. The reviewed examples are not independently relabeled positives, and no validation label or metric has changed. Further work must improve detector discrimination and camera coverage rather than claim that annotation conflicts explain away confirmed failures.

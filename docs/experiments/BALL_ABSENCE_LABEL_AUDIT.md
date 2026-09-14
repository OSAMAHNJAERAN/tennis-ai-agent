# Ball absence-label audit

The completed temporal pilot leaves 26 predictions on publisher-absent frames across the twelve expansion and six additional clips. One inspected case, match145 frame80, shows a ball-like object despite Visibility=0. Before further rejection training, review every remaining absent-frame prediction from these fixed reports. This is a complete error-subset audit, not a random sample, blinded study, independent annotation set or revised accuracy benchmark.

Keep the original CSV labels and metrics unchanged. Render actual source context at minus0.10 seconds, the scored frame, and plus0.10 seconds. Record one of: clear non-ball distractor; apparent moving ball requiring publisher-label adjudication; or unresolved. Visual appearance and temporal context support diagnosis but cannot substitute for independent ground truth. No corrected precision or recall will be calculated from these judgments. The scope omits correct abstentions and cannot estimate the overall dataset label-error rate.

The fixed inputs are the two `*_tiled_temporal_detours_verified.json` reports. Verify their parent reports and the complete dataset manifests/files, preserve every selected absent-frame error, and record image/source hashes. Existing visible labels continue to count as before. A finding of mislabeled examples does not excuse confirmed detector confusion or establish generalization to unseen cameras.


## Completed visual review

All26 remaining absent-frame predictions were rendered and inspected:20 from the expansion and6 from the additional clips. Source reports, dataset manifests, every dataset file and all seven review-board image hashes were verified. The renderer completed both datasets and compiled successfully. No detector or training code changed, and no accuracy benchmark was rerun because the labels/predictions are unchanged.

| Assistant visual judgment | Frames | Interpretation |
| --- | ---: | --- |
| Clear non-ball distractor | 14 | Court letters, scoreboard indicator, shoes, wrist/hand or racket region |
| Apparent moving ball requiring adjudication | 5 | Possible publisher absence-label conflicts; not revised ground truth |
| Unresolved | 7 | Faint, overlapping or blurred appearance cannot establish identity |

The five apparent-ball cases are match145 frame80, match148 frames329 and376, and match159 frames9 and302. For match159 frame302, a separate unmarked296-307 sequence shows a bright ball moving independently of the racket toward the net. The other seven cases remain explicitly unresolved rather than being assigned convenient labels. This diagnosis must not be used to subtract false positives or manufacture a passing precision score.

The fourteen clear distractors include four successive court-letter errors on match143 and three wrist/hand errors followed by a shoe error on match149. These explain why smooth motion and patch-change evidence are insufficient: the detector can follow coherent motion belonging to a player or racket. A stricter motion filter could also remove apparent real balls in the questionable-label group.

Machine-readable decisions and reasoning for every frame: `outputs/vision_upgrade_audit/ball_absence_label_review.json`. The original rendering inventories are preserved under `ball_absence_audit_expansion12/` and `ball_absence_audit_additional6/`, in the same output directory. These inventories keep their initial NOT_YET_REVIEWED placeholders as historical renderer output; the separate completed review is the authoritative diagnostic assessment. The additional native contact crop is x815:1075,y421:661 on match159 frames296-307.

Next action: inspect the quality of high-confidence absence negatives in publisher training data before another hard-negative training pilot. Use training sources only for training-data decisions; these validation review judgments must not be fed into the training set. Preserve uncertain labels for evaluation, and exclude or relabel training examples only through an explicit documented quality-review policy. Final qualification still needs an independently adjudicated test set with broader camera coverage. Ball, player, racket, court and physical-speed requirements remain incomplete.

The next training-only review subset is frozen before visual inspection: original verifier cache, first16 training clips, publisher-absent rows, candidate peak confidence>=0.50. It contains16 candidates on16 frames out of56 absence frames. The four internal-selection clips are explicitly excluded. Cache/dataset hashes and every selected candidate are preserved in `outputs/vision_upgrade_audit/training_absence_review_selection.json`. No training labels or models have changed.

The16-candidate training review and paired exclusion experiment are now complete. Seven candidates remain clear distractor negatives; six apparent-ball and three unresolved samples were excluded from one experimental loss only. External recall improves but precision worsens, so the candidate is rejected. Details: `VERIFIER_NEGATIVE_QUALITY_PILOT04.md`.

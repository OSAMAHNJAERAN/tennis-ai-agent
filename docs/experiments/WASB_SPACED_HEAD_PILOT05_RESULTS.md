# Controlled FPS-aligned head adaptation: reject

Completed and independently verified 2026-09-14. Aligning training context to the deployed FPS spacing does not improve detections over the adjacent-context head control and loses one frame of correct-proposal coverage. The frozen internal gate fails, so no external inference or runtime promotion is performed for this candidate.

All models are evaluated at the fixed deployed heatmap threshold **0.20** on the same 600 reserved publisher-training labels (569 visible, 31 absent). No threshold or epoch search is used.

| Model | TP | FP | FN | TN | Precision | Recall | F1 | Correct-proposal coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original pretrained | 533 | 53 | 36 | 8 | 90.9556% | 93.6731% | 92.2944% | 546/569 |
| Adjacent-context head, pilot04 | 534 | 52 | 35 | 8 | 91.1263% | 93.8489% | 92.4675% | 546/569 |
| FPS-aligned head, pilot05 | 534 | 52 | 35 | 8 | 91.1263% | 93.8489% | 92.4675% | 545/569 |

The aligned and adjacent heads have exactly the same outcome category on every reserved label, though their coordinates and confidences are not identical. Both improve one wrong visible localization versus original weights. Neither improves any labeled outcome in the 60 FPS group: all three have 119 TP / 24 FP / 14 FN / 4 TN, precision 83.2168% and recall 89.4737%, on 150 labels. The 29.97 FPS group is unchanged; the single recovery is in the 25 FPS group. The temporal mismatch was real, but this bounded head-only adaptation did not establish it as the cause of the remaining detection failures.

## Controlled change and verification

The new cache changes only temporal input paths for 699 of 2,399 training labels, retaining label order, target geometry, split, spatial view choices and existing augmentation. Initialize from untouched original WASB weights, update only the final head for one epoch, and retain seed7, batch1, AdamW1e-5, weight decay1e-4, AMP, clipping1 and frozen batch normalization. The three-update smoke is reset before the epoch. A reference pass through the actual adjacent-cache loader records the seeded sample/view/output-slot schedule; the actual training schedule matches all 2,399 entries exactly. Both runs draw 1,378 visible and 1,021 absent target views. Independent checkpoint comparison verifies that only `final_layers.0.weight` and `final_layers.0.bias` changed.

Original and pilot04 controls reuse their verified .20 predictions. Source/report/model/code hashes, every label and real temporal window, all selected candidates and all comparator rows verify exactly. The candidate first-label inference check passes on each of twelve clips. Independent scoring reproduces pooled/per-clip metrics, proposal coverage, frame-rate groups, all transitions and the acceptance decision. Twenty-two focused context, parameter-freeze and gate tests pass in 18.84 seconds. The three cache-context visual examples had been inspected before training.

## Diagnosis and visual review

The only outcome change versus original weights is `match225_000:126`: the aligned model selects the light moving feature at the net, replacing an original foreground prediction. The source-context board was visually inspected. Pilot04 already recovered the same label at .20, so this is not a benefit uniquely attributable to aligned context.

The sole proposal-coverage change relative to pilot04 is `match176_000:566`. Pilot04 retains a crop-view proposal at (879.75,328.5), 0.48534 reference pixels from the publisher label, with confidence 0.201357. Pilot05 has no proposal within tolerance. Both models still select a distant clutter point, so top-one metrics do not reveal this lost option. The saved candidates establish a threshold-sensitive coverage loss; no new heatmap inference is used to claim the exact mechanism. Do not lower the threshold in response to this reserved example. The source label is unchanged and this supplemental candidate check is not visual re-annotation.

## Runtime and limits

The epoch takes 315.72 seconds, with mean loss 0.00007406919. Candidate sparse evaluation totals 1,050.52 seconds. Per-clip durations vary substantially, including 411.44 seconds for `match19_000`; host scheduling, reference-loader overhead and brief profiling overlap prevent a controlled speed interpretation. A diagnostic eight-draw loader probe takes 0.905 seconds, primarily image augmentation. One isolated first-frame inference probe takes 3.202 seconds with time in model execution and decoder seeking. Neither probe establishes a reproducible performance bug or justifies a runtime change. The original job completes without interruption or repeated training.

Peak allocated VRAM is 275,635,200 bytes (262.866 MiB), excluding other allocations. No claim of real-time performance follows. The new checkpoint SHA256 is `b4802cb84593689ca024dcb4b410fe4522704b511e424c4ff87dc56c4c0a47fe`. Completed manifest SHA256 is `a1a0248c8c6ef2f6b3dcfa061c45e7e4df422d7d1b73bba9cf654eb058f527e2`.

This result rejects one epoch of spacing-only final-head adaptation, not all temporally aligned training. The verified cache remains reusable. Further work should address representation or supervised data quality with a new bounded, separately justified experiment; another head-only epoch or threshold search is not supported by this result. Internal labels are reused selection evidence, original-checkpoint exposure and source-broadcast independence are unverified, and source annotation ambiguity remains. Ball proposal coverage, court/player camera generalization and physical-speed validation still prevent production qualification. Overall readiness remains **NO**.

Evidence: `artifacts/training/vision_upgrade/wasb_spaced_head_pilot05/` contains the checkpoint, complete manifest and matching reference/actual schedules. `outputs/vision_upgrade_audit/wasb_spaced_head_pilot05/` contains `review.json`, `proposal_changes.json`, `visual_review.json`, `comparison_00.jpg` and completion hashes. Cache provenance is documented in `WASB_SPACED_TRAINING_DATA_RESULTS.md`; the pre-training gate remains unchanged in `WASB_SPACED_HEAD_PILOT05_PROTOCOL.md`.

# Court refinement: 32 preselected internal controls

All 32 planned control images complete with the original refinement, detector, preprocessing, calibration and seven native pixel scoring threshold unchanged. Compared with calibrated projection, refinement gains 81 correct landmark matches and loses four: 293 to 370 correct matches among 447 eligible annotations. Twenty corrections are accepted. The result improves aggregate agreement but exposes two images with landmark regressions, including one where total matches fall from fourteen to twelve. The current correction remains research only and is not integrated into runtime court mapping.

These are reused internal selection images from the earlier heatmap experiment, present in the publisher training split. Independence from ResNet training is unknown. Publisher labels remain unverified and contain known errors. This is annotation agreement on development data, not independent accuracy or production qualification.

| Stage | TP / FP / FN | Precision | Recall | Available points | Mean / median native error |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw ResNet | 289 / 158 / 158 | 64.65% | 64.65% | 447 | 7.35 / 5.27 px |
| Calibrated projection | 293 / 141 / 154 | 67.51% | 65.55% | 434 | 7.00 / 4.95 px |
| Frozen refinement | 370 / 64 / 77 | 85.25% | 82.77% | 434 | 4.69 / 3.31 px |

All 448 supplied keypoints are retained in the report. The one annotation outside the image, identity 3 at (1334, 604) in `ktiDOhLZIVs_750`, is excluded by the predeclared rule. That image also fails geometric initialization, withholding its thirteen eligible calibrated predictions. Hence raw and calibrated errors have different coverage. Projected and refined stages have exactly the same 434 available points. A wrong localization counts both FP and FN; withheld predictions count FN.

15 images gain net correct matches, 1 loses net matches, and 16 remain unchanged. There are 20 accepted corrections, five insufficient-compatible-line rejections, three unstable/excessive-correction rejections, three insufficient-support-improvement rejections and one invalid geometric initialization. All rejected corrections preserve coordinates to within 1.14e-13 native pixels, with identical missing and correctness states.

Every accepted correction was visually reviewed in five paired source-context boards. All four lost matches were additionally inspected in native-resolution crops. The six rendered review artifacts and their hashes are recorded in the separate `review.json`; original inference results remain untouched. The compact boards show several substantial improvements in visible court alignment across clay and hard courts, including an indoor view. They also expose residual errors and the near-baseline regression.

| Regression image / keypoint | Before error | After error | Fit limitation |
| --- | ---: | ---: | --- |
| 5QObSWGBQB8_1200 / 1 | 4.842 px | 7.079 px | Far baseline absent from final matches |
| 5QObSWGBQB8_1200 / 6 | 5.497 px | 7.316 px | Far baseline absent from final matches |
| ktiDOhLZIVs_3500 / 2 | 3.764 px | 9.719 px | Near baseline and left outer sideline absent |
| ktiDOhLZIVs_3500 / 5 | 4.340 px | 8.013 px | Near baseline and left outer sideline absent |

In the first regression image, three gained matches offset the two losses, so image TP increases eleven to twelve. Its final fitted line IDs are 1, 2, 3, 4, 5, 7, 8; the far baseline (0) is missing. In the second image, no matches are gained and two are lost. Final fitted line IDs are 0, 3, 4, 5, 6, 7, 8; the near baseline (1) and left outer sideline (2) are absent. Native crops show the near baseline predictions moving above visible paint. Mean line support nevertheless rises from 0.489 to 0.604 and 0.540 to 0.684 respectively. Improving this aggregate support score therefore does not guarantee that every court boundary improves. Missing boundary constraints are a concrete diagnostic lead, not a completed causal ablation.

No thresholds or labels were changed after observing these results. A future revision should address displacement of boundaries without direct line evidence, and compare its losses as well as gains against this frozen baseline. A new gate chosen using these cases must be evaluated as development tuning; independent labels and unseen camera conditions remain necessary.

The benchmark copy has an AST-identical prediction loop and scorer to the previous eight-image benchmark. The manifest, all dataset files, checkpoint, prior refinement implementation, benchmark and protocol hashes were verified. Independent scoring replay recomputes all distances, eligibility, per-image counts, pooled metrics and gained/lost identities. All 18 focused court tests pass in 4.24 seconds; this is not a full repository test run.

| Image | Projected TP | Refined TP | Gained / lost matches | Decision |
| --- | ---: | ---: | ---: | --- |
| olzZJArM6Jc_100 | 13 | 13 | 0 / 0 | INSUFFICIENT_COMPATIBLE_LINES |
| olzZJArM6Jc_2650 | 11 | 14 | 3 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| ifUIReZTOmI_50 | 5 | 14 | 9 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| ifUIReZTOmI_2400 | 10 | 14 | 4 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 5QObSWGBQB8_1200 | 11 | 12 | 3 / 2 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 5QObSWGBQB8_1250 | 2 | 2 | 0 / 0 | INSUFFICIENT_COMPATIBLE_LINES |
| cDzdc6J3jWs_700 | 9 | 14 | 5 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| cDzdc6J3jWs_3300 | 10 | 14 | 4 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 26U1ie8eb7w_50 | 11 | 14 | 3 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 26U1ie8eb7w_1750 | 12 | 14 | 2 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| qpRqZ5-UhiY_50 | 6 | 14 | 8 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| qpRqZ5-UhiY_3750 | 3 | 13 | 10 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 6CRu9DY7KII_800 | 5 | 14 | 9 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 6CRu9DY7KII_850 | 2 | 11 | 9 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| bQiC1RcZOe4_200 | 14 | 14 | 0 / 0 | INSUFFICIENT_IMAGE_SUPPORT_IMPROVEMENT |
| bQiC1RcZOe4_500 | 7 | 7 | 0 / 0 | INSUFFICIENT_COMPATIBLE_LINES |
| xGYkRnpL-R8_100 | 11 | 14 | 3 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| xGYkRnpL-R8_1950 | 13 | 14 | 1 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| XXkftmekQU8_400 | 14 | 14 | 0 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| XXkftmekQU8_900 | 14 | 14 | 0 / 0 | INSUFFICIENT_IMAGE_SUPPORT_IMPROVEMENT |
| 3dgh4_weRag_300 | 7 | 7 | 0 / 0 | UNSTABLE_OR_EXCESSIVE_REFINEMENT |
| 3dgh4_weRag_350 | 8 | 8 | 0 / 0 | UNSTABLE_OR_EXCESSIVE_REFINEMENT |
| OcvlfHjBdwY_100 | 5 | 13 | 8 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| OcvlfHjBdwY_450 | 14 | 14 | 0 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| rBHsQe0Nnyo_700 | 14 | 14 | 0 / 0 | INSUFFICIENT_COMPATIBLE_LINES |
| rBHsQe0Nnyo_2600 | 13 | 13 | 0 / 0 | INSUFFICIENT_IMAGE_SUPPORT_IMPROVEMENT |
| ktiDOhLZIVs_750 | 0 | 0 | 0 / 0 | INITIAL_GEOMETRIC_CALIBRATION_INVALID |
| ktiDOhLZIVs_3500 | 14 | 12 | 0 / 2 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| 5GFMhlMoqUs_100 | 5 | 5 | 0 / 0 | INSUFFICIENT_COMPATIBLE_LINES |
| 5GFMhlMoqUs_3950 | 2 | 2 | 0 / 0 | UNSTABLE_OR_EXCESSIVE_REFINEMENT |
| UuYiD9JWAgI_100 | 14 | 14 | 0 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |
| UuYiD9JWAgI_1850 | 14 | 14 | 0 / 0 | PROJECTIVE_IMAGE_SUPPORT_IMPROVED |

Artifacts: `outputs/vision_upgrade_audit/court_refinement_selection01/report.json` and `review.json`, the 32 original three-stage image panels, five paired review boards and native loss crops. Frozen protocol: `docs/experiments/COURT_REFINEMENT_SELECTION_PROTOCOL.md`. Benchmark: `scripts/evaluate/benchmark_court_refinement_selection.py`.

Report SHA-256: `65d0c5bab69dfeafe11b81ee80c5c42bacbf5df10f6142e682ccf5f20bc8c63c`. Protocol SHA-256: `7ba774fca641f78ffc7b9811d2881f07aa6f02117e3115307368b6dcd6cef3eb`.

# Frozen combined court refinement on 125 remaining pilot images

All 125 planned images complete. With the algorithm frozen before this broader check, combined refinement improves unchanged label agreement from 1,439 to 1,519 matches relative to the original refinement: 101 gained matches and 21 lost. Compared with initial calibrated projection, it gains 419 and loses eighteen. This supports the development improvement across more images while exposing additional local regressions. No runtime promotion or independent accuracy claim follows from this run.

These are all train-partition images from the existing court_heatmap_pilot manifest, used in the earlier heatmap training experiment and present in publisher training data. They were outside the preceding forty-image refinement comparison, but are not independent held-out qualification data. ResNet training independence is unknown. Publisher labels remain unverified and can contain errors. The detector, calibration, original refinement, combined refinement and all thresholds remain unchanged.

| Stage | TP / FP / FN | Precision | Recall | Available predictions | Mean / median native error |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | 1127 / 618 / 618 | 64.58% | 64.58% | 1745 | 7.10 / 5.31 px |
| projected | 1118 / 599 / 627 | 65.11% | 64.07% | 1717 | 7.08 / 5.25 px |
| original | 1439 / 278 / 306 | 83.81% | 82.46% | 1717 | 5.15 / 3.26 px |
| refined | 1519 / 198 / 226 | 88.47% | 87.05% | 1717 | 4.65 / 3.05 px |

Of 1,750 supplied annotations, five fall outside the image and are excluded by the predeclared rule, leaving 1,745 eligible landmarks. The calibrated stages withhold all twenty-eight points in two geometrically invalid images, leaving 1,717 available predictions. All calibrated comparisons therefore have identical coverage; raw detector error statistics have different coverage. Scoring preserves all canonical identities and uses an inclusive seven-native-pixel tolerance. Wrong locations count both FP and FN; missing predictions count FN.

The combined method accepts 92 corrections. Thirteen images improve net match count relative to original refinement, ten regress, and 102 are unchanged in count. Eleven images lose at least one initially correct point, including cases where aggregate match count improves. The union of initial-point-loss images and net regressions against original refinement contains seventeen images; all seventeen were visually inspected. Six representative gains were selected by largest net improvement over original refinement, breaking ties by image ID. Both invalid initializations were additionally inspected. Thus 25 distinct images were visually inspected, not all 125.

| Reviewed regression image | Projected / original / combined matches | Lost initial keypoint IDs | Gained / lost vs original |
| --- | ---: | --- | ---: |
| 1csvtOQNd0M_1900 | 14 / 14 / 13 | [2] | 0 / 1 |
| Sj9l5kEa72U_1050 | 10 / 10 / 7 | [1, 6, 9, 12] | 1 / 4 |
| ud1dfBJ5EcA_150 | 13 / 10 / 12 | [3] | 2 / 0 |
| ZOlj6B5ecz0_350 | 11 / 14 / 13 | [] | 0 / 1 |
| ANscPlTvfXk_450 | 13 / 14 / 13 | [] | 0 / 1 |
| crmaT4qYXao_2900 | 7 / 13 / 13 | [2] | 0 / 0 |
| gyIQ4Zx7qKc_200 | 14 / 13 / 13 | [2] | 0 / 0 |
| XVf2CPo01o0_50 | 3 / 14 / 13 | [] | 0 / 1 |
| spNfR7EEPoc_1200 | 10 / 12 / 10 | [] | 1 / 3 |
| NeHvxtGxj7s_850 | 9 / 10 / 9 | [] | 2 / 3 |
| wMjFLZGBBho_450 | 12 / 14 / 13 | [] | 0 / 1 |
| ru2ikgJUuRE_100 | 11 / 12 / 12 | [2, 5] | 0 / 0 |
| iHIsWEdpfxE_2350 | 13 / 12 / 12 | [1, 6] | 0 / 0 |
| muLVX6pLJKs_1950 | 11 / 12 / 12 | [5] | 0 / 0 |
| MUjxUJ7Ycvw_1300 | 9 / 9 / 11 | [2, 5] | 4 / 2 |
| yXd2XBQX_o4_2900 | 14 / 14 / 13 | [3] | 0 / 1 |
| V6NR0NAN_cA_600 | 14 / 14 / 12 | [3, 7] | 0 / 2 |

The larger cropped Rogers Cup view `Sj9l5kEa72U_1050` falls ten to seven eligible matches, with four initial-point losses and one gain; two of its labels are outside the image. `V6NR0NAN_cA_600` and several other close initializations show smaller regressions. Some errors are inherited from the original refinement. Toronto and Basel cases reject corrections that previously helped, illustrating the cost of stronger constraints. Visual review covers clay, indoor/outdoor hard courts and grass; it is not proof of camera-general reliability.

| Representative gain | Original matches | Combined matches |
| --- | ---: | ---: |
| VtqY_T2_gI4_700 | 3 | 14 |
| ZelyHJOVZm0_2250 | 3 | 14 |
| GVTQ00tks9s_150 | 4 | 14 |
| ZTpdH6MzGC8_2100 | 4 | 14 |
| ZoyiZXIb0Jc_1300 | 4 | 14 |
| 144Pb0S6UDg_3200 | 5 | 14 |

These six selected gains visibly improve court alignment on clay, including cases where the old fit moved away from an already reasonable initialization and cases where the old fit did not correct displacement. They demonstrate specific improvements; their selection by large gain must not be confused with an unbiased visual sample.

Both invalid initializations, `7LLSADF9t48_650` and `7LLSADF9t48_2700`, have ten of fourteen RANSAC inliers. The unchanged 75% support requirement rejects both (at least eleven are needed), despite raw label agreement of ten and twelve respectively. Their raw Madrid court predictions look plausible overall with local deviations. This identifies a separate calibration-coverage issue; it does not authorize loosening a threshold without measuring false acceptances and projected error. The frozen run retains both as withheld.

All dataset file hashes are verified before inference. The checkpoint, prior combined experiment implementations, benchmark and protocol are hashed. The scorer is AST-identical to the original frozen label benchmark. The separate reviewer recomputes every distance, eligibility flag, per-image/pooled metric and gained/lost identity, verifies unchanged manifest labels and source/render hashes, and checks each recorded boundary-preserving transform. Maximum endpoint-to-locked-line residual is 3.33e-13 reference pixels. Rejected refinements preserve initialization with zero measured reference-scale difference in this run. No new algorithm changes were made after outcomes were observed.

Validation this turn consists of the complete real-model 125-image run, independent metric replay, hash checks, geometric invariant checks and the predeclared visual review. The unchanged underlying algorithms previously passed 37 focused tests; that is historical evidence, not a claim of a newly rerun full repository suite.

Keep the combined method as a research candidate. The broader run supports its average label-agreement advantage but does not remove local errors, mislabeled data, training exposure, calibration false rejections or unknown behavior on amateur and other unseen cameras. Investigate calibration coverage and the observed regressions separately, and obtain reliable independent labels before making physical court-analytics claims.

Protocol: `docs/experiments/COURT_REMAINING_PILOT_PROTOCOL.md`. Runner: `scripts/evaluate/benchmark_court_remaining_pilot.py`. Independent review: `scripts/evaluate/review_court_remaining_pilot.py`. All 125 four-stage panels, six review boards, the invalid-initialization board, `report.json` and separate `review.json` are under `outputs/vision_upgrade_audit/court_remaining_pilot01`.

Inference report SHA-256: `f12e329598792af7c891b01160569f997f9993f29f0fba4e534f0fd92ca63dc1`. Protocol SHA-256: `19a8c0bc06fc9e92452dd0150d41a33fe967a93f17b7e942f369b9f5e2709d4c`.

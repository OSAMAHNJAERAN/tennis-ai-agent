# Combined court ridge evidence and boundary preservation

The frozen combination completes all 64 saved case/model evaluations. On the 32 selection images, correct landmark matches increase from 386 for the no-white-filter variant to 387, while losses of initially correct landmarks fall from seven to two. The eight external-source-check images remain at 67 matches. Relative to the original refinement, the combination gains 20 matches and loses one across the 40 labeled images. It is a measured development improvement, not independent court-accuracy qualification or a runtime promotion.

The combination uses the exact saved line segments from the completed no-white-filter experiment and the unchanged boundary-preserving optimizer. No new threshold, assignment rule, objective, constraint or detector change was introduced. The unconstrained optimizer replays the saved no-white coordinates exactly (maximum difference zero) before each combined fit. The two motivating variants and all earlier reports remain unchanged.

| Method | Selection TP / FP / FN | Selection precision / recall | Selection mean / median native error | External-source TP |
| --- | --- | --- | --- | ---: |
| Initial calibrated projection | 293 / 141 / 154 | 67.51% / 65.55% | 7.00 / 4.95 px | 42 |
| Original refinement | 370 / 64 / 77 | 85.25% / 82.77% | 4.69 / 3.31 px | 65 |
| Boundary constraints alone | 369 / 65 / 78 | 85.02% / 82.55% | 4.74 / 3.52 px | 65 |
| No-white line evidence alone | 386 / 48 / 61 | 88.94% / 86.35% | 4.31 / 3.20 px | 67 |
| Combined | 387 / 47 / 60 | 89.17% / 86.58% | 4.28 / 3.26 px | 67 |

Selection has 447 eligible labels and 434 available predictions at every compared stage. External-source has 112 eligible labels and 98 available predictions. Combined external-source counts are 67 TP, 31 FP and 45 FN, with precision 68.37% and recall 59.82%; available mean/median error is 16.80/3.64 native pixels. The one excluded out-of-image annotation and withheld invalid initializations are unchanged. All scoring uses fixed landmark identity and the inclusive seven-native-pixel tolerance. The small selection mean improvement accompanies a slightly worse median relative to no-white alone.

| Reference for comparison | Selection gained / lost | External-source gained / lost |
| --- | ---: | ---: |
| Initial projection | 96 / 2 | 25 / 0 |
| Original refinement | 18 / 1 | 2 / 0 |
| No-white alone | 5 / 4 | 0 / 0 |
| Boundary constraints alone | 23 / 5 | 2 / 0 |

The five gains relative to no-white alone are recovered initially correct landmarks: three on the indoor Murray/Nishikori image and two on the Indian Wells Medvedev/Ivashka image. The indoor image gives up four other previously gained matches, decreasing nine to eight overall while preserving all seven initially correct points. Indian Wells increases twelve to fourteen. This is a stability/coverage tradeoff within the aggregate improvement; it does not justify saying that every image improves.

Five cases change coordinates by more than the predeclared 0.1 reference pixel threshold or change acceptance relative to no-white alone. All five were visually inspected in two paired boards:

| Case | No-white / combined matches | Observation |
| --- | ---: | --- |
| olzZJArM6Jc_100 | 14 / 14 | Small change, close alignment retained |
| bQiC1RcZOe4_500 | 9 / 8 | Initially correct right-side points recovered; four other prior gains lost |
| OcvlfHjBdwY_100 | 13 / 13 | Small coordinate changes with unchanged match count |
| OcvlfHjBdwY_450 | 14 / 14 | Close alignment retained |
| ktiDOhLZIVs_3500 | 12 / 14 | Near-baseline drift corrected |

The two remaining losses relative to initialization are keypoints 1 and 6 in `5QObSWGBQB8_1200`. Their errors change from 4.842 to 7.085 pixels and 5.497 to 7.525 pixels. Both were inspected in native-resolution crops. This baseline now has direct matched evidence, so the missing-evidence constraint does not apply. The result confirms that observing all court lines does not guarantee every intersection is within the scoring tolerance. The one loss relative to the original refinement is the previously recorded Delray Beach keypoint regression, retained from no-white alone.

The six accepted pilot video-frame refinements remain effectively unchanged from no-white alone; no full continuous-video replay was run for this combination. Invalid initializations remain withheld, including fifteen of the eighteen earlier pilot controls. The 64 cases include repeated models/views and are not independent cameras. The combination was chosen after inspecting the earlier development results, and the publisher labels remain imperfect with unknown independence from ResNet training. All such limitations remain in force.

Independent verification recomputed every labeled distance, eligibility flag, per-image/pooled count and gained/lost identity across all five compared stages. It checked source/code/protocol hashes and all changed image hashes. Every recorded constrained fit preserves its locked outer lines to a maximum endpoint residual of 2.31e-13 reference pixels. Saved no-white replay is exact. Both algorithms are unchanged from the previous 37-test passing run; this turn validates their composition through the complete real-case replay and independent metric/boundary checks, without claiming a new full repository test run.

No runtime behavior changes. Retain this frozen comparison as the leading combined research candidate, then assess its remaining errors and validate on reliable unseen labels and continuous footage. High agreement on reused data does not establish physical court accuracy, camera-general mapping or valid ball speed.

Protocol: `docs/experiments/COURT_RIDGE_BOUNDARY_PROTOCOL.md`. Benchmark: `scripts/evaluate/benchmark_court_ridge_boundary.py`. Artifacts: `outputs/vision_upgrade_audit/court_ridge_boundary01/report.json`, separate `review.json`, five changed-case panels, two review boards and `remaining_initial_losses.jpg`.

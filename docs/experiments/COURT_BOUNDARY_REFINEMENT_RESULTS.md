# Court refinement with constraints on unobserved boundaries

The single frozen boundary-preserving revision completes all 64 saved case/model evaluations. It repairs the four previously observed landmark regressions, but is not an overall accuracy improvement: selection-set matches change from 370 to 369. Relative to the previous refinement, eight matches are gained and nine lost; relative to the initial calibrated projection, 76 matches are gained and none lost. Keep this as a documented research tradeoff. Do not replace the original research method or integrate either variant into production court mapping based on this comparison.

All cases are reused development evidence. The 32 selection images were previously used for model selection and belong to the publisher training split; the eight external-source-check images have unknown independence from ResNet training. Publisher annotations remain imperfect and unchanged. The 24 pilot cases include repeated models/views and unlabeled video samples. These are not 64 independent camera tests.

| Group / method | TP / FP / FN | Precision | Recall | Mean / median available error |
| --- | --- | ---: | ---: | ---: |
| selection / projected | 293 / 141 / 154 | 67.51% | 65.55% | 7.00 / 4.95 px |
| selection / original | 370 / 64 / 77 | 85.25% | 82.77% | 4.69 / 3.31 px |
| selection / boundary | 369 / 65 / 78 | 85.02% | 82.55% | 4.74 / 3.52 px |
| external_source_check / projected | 42 / 56 / 70 | 42.86% | 37.50% | 19.04 / 9.11 px |
| external_source_check / original | 65 / 33 / 47 | 66.33% | 58.04% | 16.81 / 3.78 px |
| external_source_check / boundary | 65 / 33 / 47 | 66.33% | 58.04% | 16.81 / 3.78 px |

The selection group has 447 eligible labels from 448 supplied annotations and 434 available projected/refined predictions. The external-source group has 112 eligible labels and 98 available projected/refined predictions. One geometric initialization in each group fails. The predeclared outside-image exclusion and withheld-prediction rules are unchanged. All three compared stages use the same coverage within each group. Counts use the inclusive seven native pixel threshold and fixed landmark identities; wrong locations count FP plus FN.

The revision accepts 21 selection corrections versus 20 previously, the same three external-source corrections, and the same six pilot video-frame corrections. Seventeen of all 64 cases fail geometric initialization before either refinement. Those failures remain unresolved. All 18 earlier pilot controls still accept no correction, including fifteen invalid initializations. The six previously accepted video-frame results are unchanged within the predeclared 0.1-reference-pixel change threshold. No revised continuous-video run was performed.

The implementation preserves the original infinite line of any outer boundary absent from compatible line matches. With homogeneous image point X and original line l, the constraint is lᵀ H X = 0 for both boundary endpoints. Identity satisfies this linear equation, so transform updates are optimized in its nullspace. A boundary stays constrained for the rest of the fit once it lacks evidence. This limits unsupported normal displacement while allowing along-line motion; it does not make incorrect initialization correct. Segment proposals, assignment rules, loss, displacement limit and support thresholds remain unchanged.

The original failure was reproduced by `scripts/evaluate/replay_court_boundary_regression.py`: the original method loses keypoints 1 and 6 in `5QObSWGBQB8_1200`, and 2 and 5 in `ktiDOhLZIVs_3500`. The latter failure was minimized from 53 proposed segments to seven; removing any remaining segment prevents that specific accepted two-loss outcome. Its fixture is saved in `tests/fixtures/court_boundary_regression.json`. The test failed before the revision and passes afterward. The revised full-image replay accepts both cases with zero lost initial matches.

This supports the missing-boundary-constraint explanation for those particular failures. It does not establish that adding hard constraints is always beneficial. Both Monte Carlo examples lack a far-baseline match and constrain line 0. In those images, preserving that initial line prevents some beneficial correction. Case 12 loses three prior matches. Case 13 gains two and loses six, a net loss of four. Native source crops show residual far-baseline error and displacement at other court intersections. The entire comparison therefore has eight gains and nine losses versus the previous refinement, not just the four recovered diagnosis matches.

| Changed case | Projected / previous / constrained TP | Gained / lost vs previous | Locked boundaries in final fit |
| --- | ---: | ---: | --- |
| 1: olzZJArM6Jc_2650 | 11 / 14 / 14 | 0 / 0 | [0] |
| 4: 5QObSWGBQB8_1200 | 11 / 12 / 14 | 2 / 0 | [0] |
| 10: qpRqZ5-UhiY_50 | 6 / 14 / 14 | 0 / 0 | [0] |
| 12: 6CRu9DY7KII_800 | 5 / 14 / 11 | 0 / 3 | [0] |
| 13: 6CRu9DY7KII_850 | 2 / 11 / 7 | 2 / 6 | [0] |
| 16: xGYkRnpL-R8_100 | 11 / 14 / 14 | 0 / 0 | [0] |
| 17: xGYkRnpL-R8_1950 | 13 / 14 / 14 | 0 / 0 | [0] |
| 22: OcvlfHjBdwY_100 | 5 / 13 / 13 | 0 / 0 | [0] |
| 23: OcvlfHjBdwY_450 | 14 / 14 / 14 | 0 / 0 | [0] |
| 27: ktiDOhLZIVs_3500 | 14 / 12 / 14 | 2 / 0 | [1, 2] |
| 29: 5GFMhlMoqUs_3950 | 2 / 2 / 4 | 2 / 0 | [0] |
| 30: UuYiD9JWAgI_100 | 14 / 14 / 14 | 0 / 0 | [0] |

All twelve changed corrections were inspected in three paired boards. All nine lost previous matches were also inspected in native-resolution crops. The newly accepted case `5GFMhlMoqUs_3950` increases only two to four matches and retains substantial misalignment. The comparison does not justify treating accepted refinement as accurate physical court mapping.

Original-method replay agrees with saved coordinates to within 1.14e-13 reference pixels. An independent review recomputes every labeled distance, eligibility flag, per-image and pooled count, and every gained/lost identity. Across every recorded constrained fit, maximum endpoint-to-original-boundary residual is 2.31e-13 reference pixels. Source reports, code, images/videos used for refinement, rendered artifacts and frozen protocol hashes are recorded and checked. Original reports and implementations remain unchanged.

All 33 focused tests pass in 4.35 seconds: the real minimized regression, exact projective boundary constraints, recovery with fully observed lines and partial evidence, original geometry/refinement tests and label scoring. Three initial synthetic examples were outside the existing 20-pixel matching limit; reducing those test-input displacements to within range resolved those test failures without any algorithm threshold change. This is not a full repository regression run.

The next useful investigation is improving direct evidence for faint or fragmented outer court lines, so the system can distinguish a correct boundary that should remain stable from an incorrect one needing movement. Broader initialization failures, annotation quality and independent camera-general validation remain open. No physical-analytics claims or runtime changes result from this ablation.

Frozen protocol: `docs/experiments/COURT_BOUNDARY_REFINEMENT_PROTOCOL.md`. Implementation: `scripts/evaluate/refine_court_boundary_geometry.py`. Full replay: `scripts/evaluate/benchmark_court_boundary_refinement.py --output <new-directory>`. Original failure replay: `scripts/evaluate/replay_court_boundary_regression.py`; add `--implementation scripts.evaluate.refine_court_boundary_geometry` for the revision.

Artifacts: `outputs/vision_upgrade_audit/court_boundary_refinement01/report.json`, `review.json`, original three-stage case panels, three changed-case boards and three native loss boards.

Report SHA-256: `e06c7bba6521307c516b77ad0b4bc2714959e75bc7a60cabc043c3558045c518`. Protocol SHA-256: `9e9c2ed45d1711b438f605c6c898b162a010314f4f6f16f7ee7990a560a8c99e`.

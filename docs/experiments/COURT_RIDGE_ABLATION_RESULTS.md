# Court ridge photometric ablations

All three frozen ablations complete all 64 saved case/model evaluations, for 192 total evaluations. Removing only the low-saturation whiteness condition gives the strongest aggregate label agreement: 386 versus 370 matches on the 32 selection images, and 67 versus 65 on the eight external-source-check images. Across those 40 images it gains 22 matches and loses four relative to the original refinement. This is the leading research variant for further validation, not a production-ready court mapper.

The experiment follows `COURT_RIDGE_ABLATION_PROTOCOL.md`, frozen before all three runs. Every variant keeps the same detector predictions, geometric initialization, canonical identities, inclusive seven-native-pixel scoring, line matching, optimizer and acceptance thresholds. The only changes are the two photometric conditions. Every variant still requires a ridge brighter than 100 and more than ten grayscale levels brighter than the brighter of its two neighboring surfaces. The best ridge-center shift is recomputed for each condition set. Original implementations, protocols and results are unchanged.

| Variant | Selection TP / FP / FN | Selection precision / recall | External-source TP / FP / FN | External-source precision / recall |
| --- | --- | --- | --- | --- |
| Original refinement | 370 / 64 / 77 | 85.25% / 82.77% | 65 / 33 / 47 | 66.33% / 58.04% |
| Remove whiteness only (`no_white`) | 386 / 48 / 61 | 88.94% / 86.35% | 67 / 31 / 45 | 68.37% / 59.82% |
| Remove similar-side condition only (`no_sides`) | 366 / 68 / 81 | 84.33% / 81.88% | 69 / 29 / 43 | 70.41% / 61.61% |
| Remove both (`ridge_only`) | 381 / 53 / 66 | 87.79% / 85.23% | 69 / 29 / 43 | 70.41% / 61.61% |

Selection has 447 eligible annotations and 434 available predictions at each compared stage; external-source has 112 eligible annotations and 98 available predictions. One initialization in each labeled group remains invalid, and the predeclared out-of-image annotation exclusion is unchanged. Wrong locations count FP plus FN. All counts and available-error statistics retain their original coverage. The no-white candidate reduces selection mean error from 4.69 to 4.31 native pixels and median error from 3.31 to 3.20; external-source mean remains 16.80 and median improves 3.78 to 3.64 pixels.

| Variant | Selection gained / lost vs original refinement | External-source gained / lost vs original | Selection gained / lost vs initial projection | Accepted selection / external / pilot |
| --- | ---: | ---: | ---: | ---: |
| Remove whiteness only | 20 / 4 | 2 / 0 | 100 / 7 | 23 / 5 / 6 |
| Remove similar-side condition only | 11 / 15 | 4 / 0 | 73 / 0 | 17 / 4 / 6 |
| Remove both | 24 / 13 | 4 / 0 | 90 / 2 | 21 / 5 / 6 |

These results do not establish that fewer photometric filters are always better. Removing only the similar-side condition loses four net selection matches, while removing both falls below removing whiteness alone. All 18 earlier pilot controls still accept no correction, but fifteen fail geometric initialization before refinement. The six previously accepted video-frame cases remain accepted for every variant. There are seventeen invalid initializations across the 64 cases. No new continuous-video or runtime integration was performed.

The preceding four-case diagnosis (`COURT_MISSING_BOUNDARY_DIAGNOSIS.md`) found geometrically compatible baseline segments in all four images, rejected by photometric conditions. Removing whiteness restores all nine final line identities in both Monte Carlo cases and in `5QObSWGBQB8_1200`. However, the latter still loses initial matches 1 and 6, despite now observing its far baseline. Missing boundary evidence was therefore a contributing mechanism in the earlier constrained-fit experiment, not a complete explanation of every localization error. The unchanged similar-side rule continues to exclude the near baseline in `ktiDOhLZIVs_3500`, preserving its two earlier losses.

All 21 changed cases for the no-white candidate were visually inspected in six paired boards, including its one changed pilot video frame. All four matches lost relative to the previous refinement were also inspected in native-resolution crops:

| Image / keypoint | Previous error | No-white error |
| --- | ---: | ---: |
| 26U1ie8eb7w_50 / 0 | 6.473 px | 7.512 px |
| bQiC1RcZOe4_500 / 3 | 3.296 px | 10.012 px |
| bQiC1RcZOe4_500 / 7 | 3.538 px | 8.754 px |
| bQiC1RcZOe4_500 / 11 | 6.406 px | 8.054 px |

The Delray Beach image loses one match. The indoor Murray/Nishikori image gains five and loses three, producing a net increase from seven to nine while introducing right-side alignment errors. It lacks the near baseline and both outer sidelines in its final matched evidence. The no-white candidate also retains the four previously known initial-point losses in the two diagnosis cases, giving seven losses relative to initialization overall. The aggregate improvement must not conceal these regressions.

Visual review shows a substantial correction in `5GFMhlMoqUs_100` (five to fourteen matches), residual misalignment in `5GFMhlMoqUs_3950` (two to six), and improved close alignment in `uXkAqALS0AA_100` (twelve to fourteen). `PuAPCalPLM4_1700` remains zero against known visibly misplaced publisher labels; its labels were neither corrected nor excluded. Several already-close courts change slightly without crossing the scoring threshold. These are visual observations, not independent measurements of court accuracy. The other two variants retain complete numerical results and rendered case panels but have not received a full visual review; neither is selected for runtime use.

The separate verifier `scripts/evaluate/verify_court_ridge_ablation.py` recomputes distances, eligibility, fixed identities, per-image counts, pooled metrics and every gained/lost identity without importing the benchmark scorer. It verifies unchanged labels, original replay tolerance, source reports, code/protocol hashes and rendered image hashes. All three reviews pass. The no-white inference report SHA-256 is `3eb5ee7b3e18103635b89e2ec731d8c8a6d4986a5d2b68035b7bdb739a0fb0e4`.

All 37 focused tests pass in 8.14 seconds, including photometric factor isolation, original-evidence parity on a synthetic fixture, rejection of a surface step without a bright ridge, projective constraints, the earlier real regression, scoring and calibration tests. This is not a full repository regression run. The completed benchmark additionally replays the original method on every usable saved case before comparing the variants.

These are reused development images with known annotation errors, prior selection exposure and unknown independence from the ResNet training set. The 64 cases include repeated views/models; they are not 64 independent cameras. Parameter selection using these results is development tuning. The next work should address the newly exposed local regressions, retain this frozen comparison, and validate on reliable unseen labels and continuous footage before physical court analytics or production promotion.

Artifacts are under `outputs/vision_upgrade_audit/court_ridge_ablation_no_white01`, `court_ridge_ablation_no_sides01`, and `court_ridge_ablation_ridge_only01`. Each contains `report.json`, `review.json` and all usable case panels. The selected no-white candidate also has `visual_review.json`, six changed-case boards and a native loss board. The scripts are `probe_court_ridge_ablation.py`, `benchmark_court_ridge_ablation.py`, `verify_court_ridge_ablation.py` and `render_court_ridge_review.py` under `scripts/evaluate`.

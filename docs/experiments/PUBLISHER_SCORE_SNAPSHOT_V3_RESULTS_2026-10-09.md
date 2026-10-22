# Private publisher score snapshot engineering evaluation — 2026-10-09

V3 evaluated directly observed pre-outcome score/server snapshots through the existing tennis probability engine. No hidden point winners, serve counts or missing history were reconstructed. The development-only logistic fit improves the retained calibration and test losses against the neutral score-only engine, but this is **unqualified private publisher-score engineering evidence**. The test has only 19 match groups; the publisher terminal score is the target and has no separate winner/result or human review. No production settings, physical camera gates or public claims changed.

## Frozen scope and lineage

The acquired [public sample](https://zenodo.org/records/22048731) is retained privately at `data/external/livetennisapi_public_sample_v1/`, with acquisition receipt, publisher checksums and SHA256 hashes for five publisher files. License and usage remain CC-BY-NC-4.0 / noncommercial academic research; raw data is not redistributed.

[V1](../../data/prediction/livetennisapi_2026_06_protocol_v1.json) remains zero eligible under its full-tape independent-result criteria. The original [V2 protocol](../../data/prediction/livetennisapi_2026_06_protocol_v2.json) and `data/external/livetennisapi_public_sample_v1/conditional_import_v2.json` remain immutable. V2 retains two inferred suffix tapes: development match11459 and test match15811, with zero calibration matches and no metrics executed. Its singles filter originally missed paired names from `players.csv` when tournament/round lacked “double”. The [V2b correction](../../data/prediction/livetennisapi_2026_06_conditional_player_audit_v2b.json) adds 456 explicit joined-player doubles exclusions; both retained tapes are singles, so eligibility stays two. V2b repairs this audit only and does not establish independently confirmed points or match-specific rule declarations.

[V3](../../data/prediction/livetennisapi_2026_06_protocol_v3.json) froze at `2026-10-09T11:16:09.934679Z`, before its import or dataset forecast metrics, SHA256 `d44cd3e316f67c9b16922e3678f36bcedd83a4f471f528c838deff8feca2a391`. Scheduled June1–14 development, June15–21 calibration and June22–30 test preserve publisher match groups. Selection is earliest scheduled UTC then match ID, cap60 per split; no eligible split reached the cap. No protocol, setting or selection changed after metrics.

Eligible families use exact named ATP/WTA regular grass events plus matching tour round prefix and explicit metadata BO3. Joined player names must be distinct, present and unpaired. Numeric tier keys, final scores and outcomes are never used to infer format. Slams, ITF and unknown event families stay excluded. The [2026 ATP rulebook](https://www.itftennis.com/media/15604/atp-2026-rulebook.pdf) and [December 2025 edition of the 2026 WTA rulebook](https://photoresources.wtatennis.com/wta/document/2025/12/24/b300b2a4-8d71-4346-969f-1f6b9399661f/2026-WTA-Rulebook-12-22-2025-.pdf) supply ordinary BO3 standard set conventions. The [official ATP calendar](https://www.atptour.com/en/news/what-is-the-2026-atp-tour-calendar/) and [WTA calendar](https://wtafiles.wtatennis.com/pdf/calendar/calendar.pdf) identify the declared families. The WTA calendar URL is mutable and the retrieved edition is later than these matches; it is family provenance, not a timestamped match declaration. Publisher metadata is a static release without archived prematch rule notices. Exceptional match-specific changes remain unverified. [ITF rules](https://www.itftennis.com/media/15546/2026-wtt-regulations.pdf) permit differing qualifying/modified formats, supporting their exclusion.

## Coverage

[The import](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3.json) accounts for every one of 951,064 captured rows across 5,380 original groups. The [corrected V3b coverage import](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3b.json), under the identical V3 protocol, has exactly the same group audit, row counts and selected match inputs/targets. It admits 114 groups (2.12%) and 12,185 unique nonterminal snapshots. This strict, sparse June grass-event subset cannot represent general professional tennis.

| Split | Eligible/selected match groups | Unique pre-outcome snapshots |
|---|---:|---:|
| Development | 57 | 6,330 |
| Calibration assessment | 38 | 3,876 |
| Test | 19 | 1,979 |

| Primary group exclusion | Groups |
|---|---:|
| Unknown/outside declared rules family | 4,322 |
| Doubles from joined names or event metadata | 456 |
| BO5/unknown format | 119 |
| Retirement/walkover/nonstandard terminal status | 148 |
| No observed rules-complete terminal target | 78 |
| Changed/reversed completed sets or games | 136 |
| Terminal score reverses prior observed progress | 1 |
| Captures cross reserved match split | 6 |
| Total excluded | 5,266 |

The selected groups contain 16,250 captured rows. Row-level exclusions include 366 missing servers, 49 six-all states missing the tiebreak flag, 6 invalid tiebreak counts, 95 completed tiebreaks still displayed active, 69 invalid standard point displays, 131 inconsistent set arrays and 179 other invalid rule states. There are 3,056 removed identical score/server duplicates and 114 terminal rows. Every remaining 12,185 row is a unique nonterminal snapshot. The other 934,814 rows belong to excluded groups. Polling gaps are unknown points; no point outcomes or serving performance counts are inferred. Current tiebreak first-server identity follows only from its direct current server, current count and documented rotation. Terminal scores and later rows never enter forecast features.

## Fixed model and fit

The raw model uses the unchanged `TennisWinProbability` engine with a fresh neutral Beta(2,2) serve prior and zero updates for each direct snapshot. It is mathematically identical to the primary neutral score-only engine: all raw paired improvements and intervals are exactly zero. The unconditional 0.5 winner probability is a secondary descriptive comparator only (Brier0.25, logloss0.693147 throughout); outperforming it does not establish added value over score-based forecasting.

The [development fit receipt](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3_development_fit.json) records the predeclared calibration `sigmoid(a+b*logit(raw))`, fitted only on 57 development groups with equal match weights. Fixed regularizationλ0.01 shrinks toward identity(a0,b1); logit input is clipped at1e-6; intercept bounds[-5,5], slope bounds[0.1,10]; L-BFGS-B uses an analytic gradient, max100 iterations, ftol1e-12, gtol1e-9. Fit succeeded once: `a=0.15629601140499222`, `b=1.34932608812652`. Calibration and test reuse this unchanged fit. No calibration-fold refit, tuning, threshold choice or test selection occurred.

Python3.13.5, NumPy2.3.1 and SciPy1.17.1 ran the bounded CPU-only fit/evaluation. Each receipt binds protocol, import, development fit and evaluated source hashes. Bootstrap resamples 1,000 whole match groups with seed20261009. Intervals condition on frozen fit parameters and **omit development-training uncertainty**. Repeated snapshots are not treated as independent matches.

## Losses and uncertainty

| Split | Raw/neutral-score Brier | Calibrated Brier | Raw/neutral-score log loss | Calibrated log loss |
|---|---:|---:|---:|---:|
| Development (training fit) | 0.174618 | 0.171450 | 0.522384 | 0.510966 |
| Calibration assessment | 0.147793 | 0.138387 | 0.460737 | 0.431412 |
| Test | 0.105016 | 0.090166 | 0.360291 | 0.312732 |

Positive improvement means neutral-score loss minus calibrated loss. Development figures are training performance.

| Split | Paired Brier improvement (95% match bootstrap) | Paired log-loss improvement (95% match bootstrap) |
|---|---|---|
| Development | 0.003168 [-0.004990,0.011634] | 0.011419 [-0.009681,0.034063] |
| Calibration assessment | 0.009405 [0.001105,0.017054] | 0.029325 [0.008597,0.049133] |
| Test | 0.014851 [0.006322,0.023062] | 0.047559 [0.027874,0.067088] |

The test raw Brier interval is[0.091141,0.118085], calibrated[0.075922,0.106672]; raw log loss[0.326849,0.391711], calibrated[0.276838,0.351870]. Full development/calibration/test per-model intervals and snapshot forecasts are retained in [development](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3_development_evaluation.json), [calibration](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3_calibration_evaluation.json) and [test](../../data/prediction/livetennisapi_2026_06_snapshot_import_v3_test_evaluation.json) receipts.

## Test reliability

Each cell shows mean forecast/observed P1-win frequency; counts are match groups/snapshots. Different raw/calibrated probabilities assign snapshots to different bins. Frequency uses equal match weights divided among that match's unique snapshots. All bins are descriptive, from the same small 19-group test subset.

| Probability bin | Raw counts | Raw mean/frequency | Calibrated counts | Calibrated mean/frequency |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 10/167 | 0.057/0.000 | 10/237 | 0.042/0.000 |
| 0.1–0.2 | 10/162 | 0.154/0.000 | 10/145 | 0.142/0.000 |
| 0.2–0.3 | 10/222 | 0.257/0.000 | 11/231 | 0.248/0.007 |
| 0.3–0.4 | 14/245 | 0.348/0.045 | 14/173 | 0.346/0.043 |
| 0.4–0.5 | 17/157 | 0.444/0.354 | 16/147 | 0.453/0.327 |
| 0.5–0.6 | 16/305 | 0.540/0.688 | 16/194 | 0.556/0.606 |
| 0.6–0.7 | 12/210 | 0.654/0.884 | 15/187 | 0.649/0.772 |
| 0.7–0.8 | 10/251 | 0.744/0.996 | 10/195 | 0.758/0.936 |
| 0.8–0.9 | 9/155 | 0.849/1.000 | 9/246 | 0.846/1.000 |
| 0.9–1.0 | 9/105 | 0.946/1.000 | 9/224 | 0.955/1.000 |

Substantial frequency gaps remain after the fit. Positive aggregate loss deltas do not establish reliable probabilities, independent reference quality or production readiness. Selected state weighting, normal-completion requirements and exclusion of inconsistent tapes induce sampling bias. Capture rows are polling observations, not guaranteed complete point history. Match-specific rules, results and timeline metadata have no human independent audit. No calibration or physical camera qualification gate is passed by this work.

## Validation

Initial `tests/test_publisher_snapshots.py` passed14 focused tests. Independent review found that terminal branching skipped the equal-timestamp conflict check. The implementation now detects conflicts before terminal handling and invalid-state filtering. Two added regressions verify that a differing nonterminal/terminal score at the same UTC rejects the group, while identical-state and terminal-display-cleanup duplicates remain permitted. The [corrected test receipt](../../artifacts/validation/yolo26_execution_2026-10-09/publisher_snapshots_tests_v3_same_time_fix.xml) records16 passing tests; the [original14-test receipt](../../artifacts/validation/yolo26_execution_2026-10-09/publisher_snapshots_tests_v3.xml) remains retained.

Before any metrics rerun, the [source impact audit](../../data/prediction/livetennisapi_2026_06_snapshot_v3_same_time_fix_audit.json) checked all114 selected groups and found zero affected groups, with exactly unchanged imported matches. The corrected full import then confirmed every5,380-group audit and row counter unchanged. Original metric receipts remain valid for these unchanged inputs and targets; no metrics were reexecuted. The [validation receipt](../../data/prediction/livetennisapi_2026_06_snapshot_v3_validation.json) verifies15 frozen receipts, common fit parameters, raw/baseline identity and zero fabricated updates. This is a protocol-compliance implementation repair; no eligibility choice, frozen setting, dataset selection or threshold was adjusted after inspecting performance.

The focused tests also exercise outcome-isolated forecasts, no fabricated counts across polling gaps, global duplicate removal, terminal exclusion, joined-player doubles, unknown format/event family rejection, invalid states, known third-set tiebreak service rotation, reversed/conflicting score exclusion, development-only fitting and equal match weighting. No production model, configuration or qualification gate was changed.

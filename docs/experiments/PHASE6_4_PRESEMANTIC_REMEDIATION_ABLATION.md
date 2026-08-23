# Phase 6.4 Pre-Semantic Remediation Ablation

## Scope and decision rule

All numbers are evaluator-v2.1 diagnostics on already-consumed `video_08`–`video_10`. Tracking receives only frozen raw candidates. `D` is not approximated; `F` is not justified because the selected config reaches the aggregate Stage-1 threshold and the residual proposal identity is unlabelled; `G` is gated by failure of the physical stream. `H` is the independently replayed selected `E` configuration.

## Aggregate A–H

| Variant | Status | Stage 1 R | Stage 2 R | Stage-3 TP/FP/FN | Stage-3 P/R/F1 | Covered candidates | Covered contacts | Timing MAE ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A preserved v2.1 | run | .6750 | .6250 | 22/47/18 | .3188/.5500/.4037 | 83 | 69 | 39.39 |
| B starting production 4/3/80 | run | .8750 | .6500 | 22/33/18 | .4000/.5500/.4632 | 79 | 55 | 66.67 |
| C class defaults 6/6/90 | run | .9000 | .6750 | 22/32/18 | .4074/.5500/.4681 | 75 | 54 | 50.00 |
| D historical 92.5% | `NOT_REPRODUCIBLE` | — | — | — | — | — | — | — |
| E reconciled config 6/6/90 | run/selected | .9000 | .6750 | 22/32/18 | .4074/.5500/.4681 | 75 | 54 | 50.00 |
| F targeted tracker fix | not justified | — | — | — | — | — | — | — |
| G physical FP-source fix | gated | — | — | — | — | — | — | — |
| H pre-semantic integrated | run/selected | .9000 | .6750 | 22/32/18 | .4074/.5500/.4681 | 75 | 54 | 50.00 |

## Per-video replay comparison

| Variant | Video | Stage 1 R | Stage 2 R | Stage-3 TP/FP/FN | P | R | F1 | Candidates | Contacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 08 | 1.0000 | 1.0000 | 7/10/3 | .4118 | .7000 | .5185 | 28 | 17 |
| A | 09 | 1.0000 | .9167 | 11/25/1 | .3056 | .9167 | .4583 | 36 | 36 |
| A | 10 | .2778 | .2222 | 4/12/14 | .2500 | .2222 | .2353 | 19 | 16 |
| B | 08 | 1.0000 | 1.0000 | 7/6/3 | .5385 | .7000 | .6087 | 25 | 13 |
| B | 09 | 1.0000 | .9167 | 11/20/1 | .3548 | .9167 | .5116 | 33 | 31 |
| B | 10 | .7222 | .2778 | 4/7/14 | .3636 | .2222 | .2759 | 21 | 11 |
| C/E/H | 08 | 1.0000 | 1.0000 | 6/7/4 | .4615 | .6000 | .5217 | 24 | 13 |
| C/E/H | 09 | 1.0000 | .9167 | 11/19/1 | .3667 | .9167 | .5238 | 30 | 30 |
| C/E/H | 10 | .7778 | .3333 | 5/6/13 | .4545 | .2778 | .3448 | 21 | 11 |

Candidate/minute, physical-contact/minute, per-video timing, state counts, longest prediction/interpolation runs, out-of-frame counts, above-speed transition indicators, reacquisition indicators, and deterministic trajectory hashes are in `phase6_4_tracker_replay_comparison.json` and `phase6_4_presemantic_ablation.json`.

## Selected causal chain

| Scope | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Stage 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| video_08 | 10 | 10 | 10 | 4 | 3 | 3 | 1 |
| video_09 | 12 | 12 | 11 | 11 | 1 | 0 | 0 |
| video_10 | 18 | 14 | 6 | 3 | 2 | 2 | 2 |
| Aggregate | 40 | 36 | 27 | 18 | 6 | 5 | 3 |

## Hard gate

| Criterion | Result |
|---|---|
| Stage 0 recall ≥ .95 | PASS (`1.0000`) |
| Stage 1 grounded recall ≥ .90 | PASS (`.9000`) |
| Stage 2 recall ≥ .85 | FAIL (`.6750`) |
| True Stage-3 precision ≥ .70 | FAIL (`.4074`) |
| True Stage-3 recall ≥ .80 | FAIL (`.5500`) |
| True Stage-3 F1 ≥ .75 | FAIL (`.4681`) |
| No candidate explosion | PASS (75 versus 79 starting-production) |
| Prediction bounded | PASS (maximum six frames) |
| No major wrong-association indicator regression | PASS (within 5% of starting-production count) |
| No held-out-video collapse | FAIL (`video_10` Stage 1 `.7778`) |
| All repository tests | PASS (`328 passed`) |

Earliest dominant blocker: `STAGE_2_PHYSICAL_CANDIDATE_RECALL`. No Stage-3 FP pack, human taxonomy, source-specific FP tuning, semantic tuning, or shot-classifier tuning was opened.

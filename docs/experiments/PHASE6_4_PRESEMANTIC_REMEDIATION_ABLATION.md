# Phase 6.4 Pre-Semantic Remediation Ablation

## Scope and decision rule

All numbers are evaluator-v2.1 diagnostics on already-consumed `video_08`–`video_10`. Tracking receives only frozen raw candidates. `D` is not approximated. `E` is rejected because its metric gain accompanies more out-of-frame points. `F` is the single-mechanism frame-bounds repair on `4/3/80`; `G` is gated by failure of the physical stream. `H` independently replays selected `F`.

## Aggregate A–H

| Variant | Status | Stage 1 R | Stage 2 R | Stage-3 TP/FP/FN | Stage-3 P/R/F1 | Covered candidates | Covered contacts | Timing MAE ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A preserved v2.1 | run | .6750 | .6250 | 22/47/18 | .3188/.5500/.4037 | 83 | 69 | 39.39 |
| B starting production 4/3/80 | run | .8750 | .6500 | 22/33/18 | .4000/.5500/.4632 | 79 | 55 | 66.67 |
| C class defaults 6/6/90 | run | .9000 | .6750 | 22/32/18 | .4074/.5500/.4681 | 75 | 54 | 50.00 |
| D historical 92.5% | `NOT_REPRODUCIBLE` | — | — | — | — | — | — | — |
| E reconciled config 6/6/90 | run/rejected | .9000 | .6750 | 22/32/18 | .4074/.5500/.4681 | 75 | 54 | 50.00 |
| F frame-bounds tracker fix | run/selected | .8750 | .6500 | 22/43/18 | .3385/.5500/.4190 | 84 | 65 | 45.45 |
| G physical FP-source fix | gated | — | — | — | — | — | — | — |
| H pre-semantic integrated | run/selected | .8750 | .6500 | 22/43/18 | .3385/.5500/.4190 | 84 | 65 | 45.45 |

## Per-video replay comparison

| Variant | Video | Stage 1 R | Stage 2 R | Stage-3 TP/FP/FN | P | R | F1 | Candidates | Contacts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 08 | 1.0000 | 1.0000 | 7/10/3 | .4118 | .7000 | .5185 | 28 | 17 |
| A | 09 | 1.0000 | .9167 | 11/25/1 | .3056 | .9167 | .4583 | 36 | 36 |
| A | 10 | .2778 | .2222 | 4/12/14 | .2500 | .2222 | .2353 | 19 | 16 |
| B | 08 | 1.0000 | 1.0000 | 7/6/3 | .5385 | .7000 | .6087 | 25 | 13 |
| B | 09 | 1.0000 | .9167 | 11/20/1 | .3548 | .9167 | .5116 | 33 | 31 |
| B | 10 | .7222 | .2778 | 4/7/14 | .3636 | .2222 | .2759 | 21 | 11 |
| C/E | 08 | 1.0000 | 1.0000 | 6/7/4 | .4615 | .6000 | .5217 | 24 | 13 |
| C/E | 09 | 1.0000 | .9167 | 11/19/1 | .3667 | .9167 | .5238 | 30 | 30 |
| C/E | 10 | .7778 | .3333 | 5/6/13 | .4545 | .2778 | .3448 | 21 | 11 |
| F/H | 08 | 1.0000 | .9000 | 6/8/4 | .4286 | .6000 | .5000 | 23 | 14 |
| F/H | 09 | 1.0000 | 1.0000 | 12/28/0 | .3000 | 1.0000 | .4615 | 41 | 40 |
| F/H | 10 | .7222 | .2778 | 4/7/14 | .3636 | .2222 | .2759 | 20 | 11 |

Candidate/minute, physical-contact/minute, per-video timing, state counts, longest prediction/interpolation runs, out-of-frame counts, above-speed transition indicators, reacquisition indicators, and deterministic trajectory hashes are in `phase6_4_tracker_replay_comparison.json` and `phase6_4_presemantic_ablation.json`.

## Selected causal chain

| Scope | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Stage 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| video_08 | 10 | 10 | 9 | 4 | 3 | 3 | 1 |
| video_09 | 12 | 12 | 12 | 12 | 3 | 1 | 0 |
| video_10 | 18 | 13 | 5 | 3 | 2 | 2 | 2 |
| Aggregate | 40 | 35 | 26 | 19 | 8 | 6 | 3 |

## Hard gate

| Criterion | Result |
|---|---|
| Stage 0 recall ≥ .95 | PASS (`1.0000`) |
| Stage 1 grounded recall ≥ .90 | FAIL (`.8750`) |
| Stage 2 recall ≥ .85 | FAIL (`.6500`) |
| True Stage-3 precision ≥ .70 | FAIL (`.3385`) |
| True Stage-3 recall ≥ .80 | FAIL (`.5500`) |
| True Stage-3 F1 ≥ .75 | FAIL (`.4190`) |
| No candidate explosion | FAIL (84 versus 79 starting-production) |
| Prediction bounded | PASS (maximum four frames) |
| No out-of-frame points | PASS (zero after fix) |
| No major wrong-association indicator regression | PASS (within 5% of starting-production count) |
| No held-out-video collapse | FAIL (`video_10` Stage 1 `.7222`) |
| All repository tests | PASS (`328 passed`) |

Earliest dominant blocker: `STAGE_1_GROUNDED_USABLE_OBSERVATION_RECALL`. No Stage-3 FP pack, human taxonomy, source-specific FP tuning, semantic tuning, or shot-classifier tuning was opened.

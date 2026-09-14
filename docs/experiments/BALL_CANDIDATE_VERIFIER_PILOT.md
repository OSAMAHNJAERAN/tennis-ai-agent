# Learned ball-candidate verifier pilot

Protocol recorded before preparing or training the verifier. The existing pixel-filter candidate remains available; this pilot is not integrated or promoted.

## Rationale and boundary

Original WASB threshold .05 contains a correct candidate on 526/543 visible expansion labels, but top-one ranking and hand-set appearance rejection cannot select it with sufficient precision. A small learned verifier will score **existing** candidates using local current/past appearance, detector confidence and rank. It cannot recover a ball that is absent from the candidate set, and it never invents positions or events. This tests a different objective from fine-tuning WASB's whole-frame heatmap.

## Data and preparation

Use the already acquired first twenty publisher training clips only. Every supervised example comes from an explicit CSV frame; unlabeled context supplies images only. For each labeled frame, run exactly the real overlapping WASB windows that contribute to that frame, at threshold .05, and keep at most the first 32 candidates by the existing mass ordering. There are no ground-truth proposals or label-directed candidate additions. Compare candidate coordinates with the explicit label at the existing 512x288 reference scale: distance at most four pixels is positive, at least eight is negative, intermediate candidates are excluded from training loss. Explicit absent frames label all proposed candidates negative. Evaluation still uses the strict four-pixel rule and includes all explicit labels, including frames with no candidates.

Save 32x32 current and .10-second-past RGB crops at reference resolution 960x540, centered on the detector observation. The classifier also receives peak detector confidence and reciprocal one-based rank. No absolute image coordinates, clip IDs, court identity, player identity or scoreboard masks enter the model. Preserve dataset/checkpoint/code/cache hashes and frame/candidate mappings. The preparation utility can also create separately labeled validation caches; the training utility must reject those caches.

## Predeclared training and selection

Reserve the last four of the twenty publisher training clips for internal model/threshold selection; train on the first sixteen. Use a small six-channel CNN (16/32/64 channels, 4x4 pooled appearance, two scalar features, 64-unit head), batch 64, AdamW learning rate .001, eight epochs, seed seven. Apply shared current/past horizontal flips and brightness/contrast perturbations. Balance positive/negative training draws without relabeling. Select the checkpoint and verifier-score threshold from .3/.5/.7/.9 using highest strict per-frame F1 on the four internal held-out training clips, with precision then recall as tie-breakers. Record all epochs/thresholds.

Then freeze that choice and evaluate on publisher validation separately, comparing against existing measured candidates. Threshold scores are uncalibrated model outputs. Training loss, binary candidate accuracy or an oracle ceiling cannot establish per-frame ball precision/recall. Publisher clip IDs are disjoint, but broadcast grouping and the official model's historical training overlap remain unverified. This is a bounded pilot, not final-test qualification.

## Completed preparation and internal selection

The training-only cache contains 5,379 real proposals from 1,000 explicit labeled frames: 875 positive, 4,465 negative and 39 ambiguous candidates. Preparation took 200.17 seconds excluding model construction. The first sixteen clips supply 703 positive and 4,016 negative loss examples. The reserved clips are `match187_000`, `match188_000`, `match189_000` and `match19_000`. Five targeted tests pass for sparse output-slot alignment, crop/channel centering, border crops, split rejection and strict per-frame scoring.

The 89,809-parameter CNN completed all eight epochs. Internal training-split selection chose epoch 8, score threshold .9, with P=97.65%, R=91.21%, F1=94.32% on the reserved clips. This is checkpoint/threshold selection evidence, not external accuracy. The choice is frozen before validation scoring. Cache and all checkpoints are preserved under `artifacts/training/vision_upgrade/verifier_train20_cache/` and `candidate_verifier_pilot01/`.

Validation preparation may reuse an existing .05/overlap candidate report instead of rerunning WASB. The preparer verifies exact dataset/checkpoint identities, original single-model settings, frame counts, geometry, FPS and every explicit CSV target. It then extracts crops from checksum-verified original video. This changes computation cost, not the proposals or labels, and the source report hash is retained.

## External result: first verifier rejected

Frozen epoch 8/.9 on the twelve validation clips gives **TP=473, FP=30, FN=70, TN=33; P=94.04%, R=87.11%, F1=90.44%**. Correct absence is 33/57. This improves raw .05 precision but loses too many true balls and is inferior to the existing .20/pixel-filter candidate (P=94.92%, R=93.00%, F1=93.95%). It is rejected as a replacement. The prepared cache's raw top-one baseline is TP=492, FP=104, FN=51, matching its original .05 proposal stream rather than the separate stationary-filter replay.

The validation cache contains 4,459 proposals from 600 labels, including 529 correct proposals (some labels have more than one) and eleven ambiguous proposals. This count is not frame recall. The deterministic rejected-true-candidate review includes up to two cases per clip, twenty total, showing blurred balls, low-contrast balls and balls near nets, legs or court lines. Images and metadata: `outputs/vision_upgrade_audit/verifier_rejected_true_candidates/`. It is qualitative error evidence, not proof of a single causal mechanism.

## Controlled second verifier pilot, declared before training

Reuse exactly the same cache, 16/4 split, architecture, eight epochs, seed and internal selection rule. Add only shared horizontal/vertical 3- or 5-pixel motion blur with probability .5. A separate random stream (seed 17) controls blur so the existing flip/photometric random stream is not consumed differently. This preserves labels and proposal centers. The hypothesis is that the first classifier over-relies on crisp local appearance and rejects blurred real balls. It may also lose precision by making distractors resemble balls. Freeze the second pilot's training-selected checkpoint/threshold before evaluating validation, and retain both results. No validation-directed threshold sweep is permitted in this comparison.

The blur pilot selects epoch 7/.9 internally, then gives **TP=470, FP=23, FN=73, TN=41; P=95.33%, R=86.56%, F1=90.73%** externally. It is also rejected. Correct validation proposals below base-detector confidence .20 are accepted only 1/22 times, versus 18/46 at .20–.50 and 452/461 at .50 or higher. Training-cache correct proposals are concentrated at high confidence: 40/61/774 in those bins. These are candidate-level diagnostics (including duplicated correct proposals), not frame recall or causal proof.

## Third controlled pilot: remove detector scalar inputs

Before the next training run, freeze the same blur-pilot settings but replace confidence/rank inputs with zeros during both training and evaluation. Keep the network parameterization, random seeds, data and selection rule identical. This tests whether shortcut learning from base-model strength impedes low-confidence recovery. The verifier can still learn correlated visual cues, and removing scalars may lose valuable ranking information. Select only on the four held-out training clips and report the external result without threshold retuning.

The appearance-only pilot selects epoch 8/.3 internally, with internal F1=78.31%, and gives **TP=435, FP=129, FN=108, TN=15; P=77.13%, R=80.11%, F1=78.59%** externally. This is substantially worse, so removing the scalars is not a successful fix. The observed weak-candidate rejection cannot be attributed solely to those inputs; crop representation, training coverage, optimization and candidate ambiguity remain possible causes. This ablation is rejected without trying validation-selected rescue thresholds.

## Comparison and decision

| Expansion candidate | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Existing .20 plus pixel-motion filter | 505 | 27 | 38 | 94.92% | 93.00% | 93.95% |
| Verifier pilot 1, epoch 8/.9 | 473 | 30 | 70 | 94.04% | 87.11% | 90.44% |
| Verifier pilot 2, blur, epoch 7/.9 | 470 | 23 | 73 | 95.33% | 86.56% | 90.73% |
| Verifier pilot 3, appearance only, epoch 8/.3 | 435 | 129 | 108 | 77.13% | 80.11% | 78.59% |

All three learned pilots are rejected as pipeline replacements. The strict model/threshold selections used only reserved training clips, but later hypotheses were informed by the reused expansion failures, so this is iterative validation research. No final test was downloaded or scored. The selected pixel-filter configuration is unchanged.

Evidence: `artifacts/validation/vision_upgrade/expansion12_candidate_verifier_pilot01.json`, `expansion12_candidate_verifier_pilot02_blur.json`, `expansion12_candidate_verifier_pilot03_appearance.json`; training histories and all checkpoints under `artifacts/training/vision_upgrade/candidate_verifier_pilot*/`. Six verifier tests cover alignment, real candidate scoring, split guards and shared augmentation. The full suite passed **470 tests**, one existing warning, in 12.66 seconds after pilot 2; the final appearance-only change was exercised by its actual train/evaluate runs. The verifier is experimental source code and is not connected to Phase 6 inference.

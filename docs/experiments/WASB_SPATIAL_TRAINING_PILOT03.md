# WASB spatial-view adaptation pilot 03

Protocol fixed before training or external scoring, 2026-09-09. Five-view inference increases ball recall but introduces false detections on the twelve-clip development expansion. Prior fine-tuning used only full-frame imagery. The hypothesis is that explicit supervision at the deployed crop scale, combined with more publisher training footage, can improve this operating point. Multiple changes are combined; this pilot cannot isolate the contribution of data quantity versus scale augmentation or learning rate.

## Data and isolation

Use the original 20 publisher training clips plus the next 20 entries (20:40) from RacketVision revision `85157ca21faa2abca96d837dd2b963738029bcc8`. The additional acquisition verifies publisher video SHA-256 values and preserves raw sparse ball CSVs. Both manifests must say TRAINING_ONLY, share the revision, contain no duplicate clips, and select actual publisher training members with no validation match-ID overlap. No final-test files are downloaded.

Reserve every fifth distinct publisher training match ID for internal checkpoint selection, keeping all rallies of each match together. For these 40 clips this gives 32 training and eight selection match IDs. This selection data is internal development evidence: original checkpoint pretraining overlap and source broadcast independence remain unverified. Earlier pilot training also used some of these original-20 clips, but this pilot starts from the untouched official checkpoint.

For each explicit label, cache the real full frame and four 60% overlapping corner crops at the original WASB affine input geometry. Preserve five actual neighboring context frames with boundary clamping as in the existing sparse training loader. Only the explicitly labeled frame supplies a target. Transform its active-ball coordinate into each crop; an active ball outside the crop is absent for that view. This does not label other objects as balls, interpolate trajectories, or use unlabeled video frames as negative targets. A zero heatmap means the annotated active ball is absent from the selected view, not that the scene contains no other physical balls.

## Frozen training and selection

- Start from official WASB tennis weights; train the last HRNet stage and output layers with frozen batch-normalization statistics.
- Three epochs, seed7, batch1, AdamW learning rate0.00001, weight decay0.0001, gradient clipping1.
- Each source labeled frame contributes once per epoch, sampling one of its five views uniformly. Keep existing random output-slot, horizontal flip, brightness/contrast and motion-blur augmentation. No synthetic scoreboard overlays or absent oversampling in this pilot.
- Internal selection evaluates every view of all reserved labels, with actual central-frame supervision, fixed chronological context and no image augmentation. Record focal loss before training and after every epoch. Preserve all checkpoint hashes.
- Select the lowest internal loss among the original checkpoint and the three epochs, before external inference. Loss is not precision, recall or a deployment acceptance metric.

## External evaluation and acceptance

If a trained checkpoint is selected, run the existing continuous five-view benchmark on all twelve expansion clips at unchanged threshold.20, stationary minimum.25 seconds and pixel-motion threshold12. Report raw, stationary and pixel-motion outputs separately. Apply the frozen similarity.98 / local-residual12 gate as a separate replay, preserving every explicit visible/absent publisher label. Compare paired errors with the original model at identical settings, including per-clip failures. Do not tune thresholds on these outcomes.

A promising candidate must then run on the six added validation clips and through a complete Phase6 video before considering any optional integration. Previously inspected validation sets remain development data. Production promotion requires broader independent footage and the full goal's player, racket, court, physical-speed and camera evidence. Event authority remains disabled.

Implementation: `scripts/train/finetune_wasb_spatial.py`. Training isolation, crop target geometry, deterministic selection context and loss stability are covered by `tests/test_spatial_ball_training.py`; 15 targeted tests pass together with the existing sparse-training tests. Dataset images and checkpoints remain ignored binary artifacts; manifests, code, protocol and evaluation reports retain provenance.

## Reproduction

```powershell
python scripts/train/finetune_wasb_spatial.py --datasets data/external/racketvision_training20 data/external/racketvision_training_additional20 --output artifacts/training/vision_upgrade/wasb_pilot03_spatial
```

Use the selected checkpoint recorded by the completed manifest with `benchmark_tiled_ball_stream.py --checkpoint`, a fresh report path and the unchanged expansion manifest. Never substitute the final epoch merely because it is newest. The benchmark records the checkpoint hash and rejects incompatible resume state.


## Data verification and launch

The two manifests contain **1,999 explicit labels**, not an assumed 2,000: training has **1,599 labels (1,466 visible,133 absent)** across32 match IDs; internal selection has **400 labels (378 visible,22 absent)** across8 IDs. These are original full-frame label counts; view sampling changes the probability that the active ball lies inside the input crop. The additional acquisition totals **95,931,929 bytes** including metadata and labels. Exact split lists and both manifest hashes are in `artifacts/validation/vision_upgrade/spatial_training_data_audit.json`.

Before training, the full regression suite passes **510 tests**, one existing warning,11.79 seconds (`outputs/vision_upgrade_audit/spatial_training_integration_tests.log`). The frozen pilot launched at2026-09-09T20:56:01+08:00. Runtime state is recorded separately in `outputs/vision_upgrade_audit/wasb_pilot03_spatial_process.json`; that record is not proof that the process remains alive. No training or validation outcome is claimed until the completed manifest and scored exports exist.


## Completed training and checkpoint selection

All three epochs completed. Internal selection focal loss changed from **0.0000738549** before adaptation to **0.0000582563**, **0.0000539921**, and **0.0000518925**. The predeclared rule selects epoch3. Its SHA-256 is `f5cbb95a6a7acab908e58ee02ff63fde2bfb94aa46693fde37642a6c6640e15f`, verified against the saved bytes after the process ended. All recorded training code hashes still match. The manifest is complete; every epoch remains preserved.

Actual absent/visible view draws were697/902,690/909 and729/870. These include real visible balls outside a sampled crop, so they are not the original full-frame absent-label prevalence. Epoch durations including internal selection were214.45,196.34 and187.52 seconds. They exclude preparation and initial selection; minor label-preview/file-inspection work overlapped, so they are observed training durations, not controlled throughput. The five-view label preview (`outputs/vision_upgrade_audit/pilot03_spatial_label_preview.jpg`) was visually inspected for coordinate alignment and correct outside-crop target absence.

The paired checkpoint evaluator (`scripts/evaluate/compare_tiled_checkpoints.py`) rejects changed data, settings, inference/filter/metric code, incomplete frame streams and inconsistent labels. Its six targeted tests pass. The earlier full suite passed510 tests; these six later comparison tests are separate and have not been represented as a new full-suite run.

The selected checkpoint's twelve-clip external evaluation is launched with the frozen protocol. Lower selection loss is not evidence of improved external precision/recall; results remain pending until all clips complete and paired scores are inspected. No default model or event setting changed.


## External result: reject this checkpoint at the frozen operating point

All12 clips and6,082 frames completed. At4 reference pixels over600 explicit labels, the trained model yields raw **513TP/84FP/30FN/2TN**; stationary **520/76/23/3**; pixel-motion **521/69/22/8**, precision **88.31%**, recall **95.95%**, F1 **91.97%**. Relative to original five-view plus pixel motion (518/35/25/33), it gains6 correct balls, loses3, gains1 correct absence and loses26. The net three-ball gain does not justify34 additional false detections.

The frozen similarity-plus-residual replay removes only two absent false detections: **521TP/67FP/22FN/10TN**, precision **88.61%**, recall **95.95%**, F1 **92.13%**. Primary.98 and diagnostic.95 thresholds are identical. The original checkpoint with the same residual gate scores518/32/25/36, precision94.18%, recall95.40%, F194.79%. The trained candidate is rejected; no further threshold tuning, six-clip promotion run or default replacement is justified by this outcome.

Evidence: `expansion12_pilot03_tiled_ball_stream.json`, `expansion12_pilot03_tiled_comparison.json`, and `expansion12_pilot03_tiled_similarity_residual.json` under `artifacts/validation/vision_upgrade`. The paired comparison verifies identical data/settings/inference/filter/metric hashes and complete frame streams; only the benchmark's explicit checkpoint argument was added. Elapsed times overlapped UVY archive metadata/acquisition and small CPU metric tests, recorded in `outputs/vision_upgrade_audit/expansion12_pilot03_execution_notes.txt`; these durations are not an isolated speed comparison.

This experiment establishes that improved internal focal loss can coexist with substantially worse false-positive behavior on different footage. More full/crop training exposure alone did not solve ball/distractor discrimination. Preserve all checkpoints and failure reports; the original checkpoint remains the experimental reference. The broader reliability goal remains incomplete.

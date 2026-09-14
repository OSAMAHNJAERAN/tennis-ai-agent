# WASB stationary-graphic training pilot 02

Prepared 2026-09-08 before scoring the new checkpoint. This is a validation experiment in the active vision upgrade, not production qualification.

## Evidence and hypothesis

On the twelve-clip expansion, original WASB step-one/.20 inference with a .25-second stationary-candidate filter achieves 92.49% precision and 93.00% recall. `match148_000` repeatedly selects a stationary yellow scoreboard service dot. The first training pilot and its ensemble reduce absent-ball specificity on broader footage. The next hypothesis is that explicit training exposure to stationary broadcast graphics and increased sampling of real absent-ball labels can reduce these errors without unacceptable loss of recall.

## Training fixed before validation

- Start from the untouched official WASB tennis checkpoint, not the first pilot's checkpoint.
- Reuse only the verified publisher-training split: 20 clips, 1,000 explicit labels, including 74 absent-ball labels. No validation image, coordinate, scoreboard crop or target is used for training.
- Train the last HRNet stage and output layers for three epochs, batch size one, learning rate 0.00005, seed seven. Preserve frozen pretrained batch-normalization statistics and the existing temporal blur/lighting/flip augmentation.
- With probability .50, synthesize a small broadcast graphic with text and a yellow service marker, shared exactly across all three input images. Its corner location, dimensions and marker vary. Skip any placement overlapping the supervised ball region and an eight-pixel margin. Leave the target heatmap unchanged.
- Give each explicit absent-ball sample four times the sampling weight of a visible-ball sample, with replacement and 1,000 draws per epoch. Report actual visible/absent draws. This changes training prevalence; it does not create new independent labels.
- Save every epoch and its SHA-256. Evaluate the final third epoch first; training loss alone cannot select or promote a checkpoint.

The augmentation's temporal identity, target protection, source immutability and abstention when placement is impossible are covered by tests in `tests/test_sparse_ball_training.py`.

## Evaluation and acceptance

First run the third-epoch checkpoint on the same twelve-clip expansion, overlapping step one at threshold .20. Report raw observations separately from the existing tracker. Replay the already selected .25-second stationary filter on its full candidate sequence. Compare TP/FP/FN, precision/recall/F1, absent specificity and per-clip results with the original model at identical settings. This is further validation selection on a previously inspected set, not a new independent test.

A result that only reduces false detections by missing more real balls does not qualify as an overall improvement. An improved candidate must subsequently run on the initial six clips and fresh held-out footage before any production claim. The requested >95% precision and recall, camera/generalization requirements, player/racket qualification, true ball-speed validation and event gate remain intact.

Do not promote the ensemble or new checkpoint solely for crossing 95% on the original 300-label selection set. Preserve all failed runs. Current defaults remain unchanged while this experiment is evaluated.

## Reproduction

From the repository root using the existing CUDA-capable Python:

```powershell
python scripts/train/finetune_wasb.py --epochs 3 --learning-rate 0.00005 --graphic-probability 0.5 --absent-sampling-factor 4 --output artifacts/training/vision_upgrade/wasb_pilot02_graphics
python scripts/evaluate/benchmark_video_ball.py --backend wasb --checkpoint artifacts/training/vision_upgrade/wasb_pilot02_graphics/epoch_03.pth.tar --dataset data/external/racketvision_validation_expansion12 --wasb-step 1 --wasb-threshold .20 --output artifacts/validation/vision_upgrade/racketvision_expansion12_pilot02_graphics020.json
python scripts/evaluate/replay_stationary_filter.py --report artifacts/validation/vision_upgrade/racketvision_expansion12_pilot02_graphics020.json --minimum-seconds .25 --output artifacts/validation/vision_upgrade/racketvision_expansion12_pilot02_stationary.json
```

All output paths must be fresh. The first launch was rejected by automatic approval review because of its usage limit; no training process or checkpoint was created. A subsequent account-capacity check allowed resubmission through the same approval mechanism, and training started successfully.

## Training outcome

All three epochs completed. Actual absent/visible draws were 234/766, 244/756 and 253/747. Mean focal losses were 0.00008726, 0.00007219 and 0.00005993, which are training diagnostics rather than detection accuracy. Epoch durations were 82.46, 68.38 and 63.12 seconds; small CPU-only checks/preview tasks overlapped training, so these are observed run durations rather than isolated performance benchmarks.

Final checkpoint SHA-256: `77c9c4c1e1c0d7c96e038ba59cef15d37842986bb17894f702f7b821c7561e40`. Manifest and every checkpoint are preserved in `artifacts/training/vision_upgrade/wasb_pilot02_graphics/`. The training augmentation preview at `outputs/vision_upgrade_audit/pilot02_augmentation_preview.jpg` was visually inspected: the synthetic graphics and markers are present, and visible target regions are preserved. Fifteen targeted augmentation, point-accounting and clip-summary tests passed; the final full suite passes 433 tests with one pre-existing dependency warning.

## Validation outcome: do not promote at the tested operating point

All 6,082 source frames were processed and the same 600 explicit labels were scored (543 visible, 57 absent). No threshold search was used to select these reported results.

| Candidate | TP | FP | FN | Precision | Recall | F1 | Correct absent / 57 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original, step one/.20 | 499 | 63 | 44 | 88.79% | 91.90% | 90.32% | 27 |
| Pilot 02, step one/.20 | 506 | 79 | 37 | 86.50% | 93.19% | 89.72% | 10 |
| Original + .25 s stationary filter | 505 | 41 | 38 | 92.49% | 93.00% | 92.75% | 36 |
| Pilot 02 + identical filter | 512 | 65 | 31 | 88.73% | 94.29% | 91.43% | 15 |

The pilot adds seven correct localizations but also 24 false positives relative to the selected filtered original. Correct absence handling falls from 36/57 to 15/57. This contradicts the intended improvement at the predeclared operating point. Lower training loss and modest hard-clip recall gains do not justify replacing the selected checkpoint. This does not prove synthetic augmentation is intrinsically ineffective: the bounded dataset, sampling change, learning rate and score calibration differ together. A causal ablation or larger training study is still needed.

Evidence: `artifacts/validation/vision_upgrade/racketvision_expansion12_pilot02_graphics020.json` and `racketvision_expansion12_pilot02_stationary.json`. The original checkpoint remains selected. Pilot weights remain research artifacts; no default configuration changed.

# Training-split operating-point results

Completed review: 2026-09-13. The saved inference run was already complete when this continuation began; inference and training were not repeated. The spatial epoch-three checkpoint remains rejected. Selecting each model's threshold on the same reserved training matches does not give the adapted model a higher pooled F1 than the original checkpoint.

The frozen protocol is `WASB_TRAINING_THRESHOLD_PROTOCOL.md`. Each model evaluates 400 unchanged labels from eight reserved publisher training match IDs: 378 visible and 22 absent. Both use five views and FPS-adjusted temporal spacing. The five thresholds are decoded from averaged heatmaps separately, preserving component-center changes. These are sparse internal selection measurements, not continuous pipeline results or independent qualification.

| Checkpoint | Threshold | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, historical threshold | 0.20 | 358 | 31 | 20 | 92.0308% | 94.7090% | 93.3507% |
| Original, training-selected | 0.35 | 354 | 17 | 24 | 95.4178% | 93.6508% | 94.5260% |
| Spatial epoch 03, historical threshold | 0.20 | 361 | 39 | 17 | 90.2500% | 95.5026% | 92.8021% |
| Spatial epoch 03, training-selected | 0.50 | 352 | 16 | 26 | 95.6522% | 93.1217% | 94.3700% |

The adapted model's best selected F1 is 0.1561 percentage points below the original's. Both selected pairs have nine absent false detections and thirteen correct absences. The adapted pair has seven wrong visible localizations and nineteen visible abstentions, versus eight and sixteen for the original. Wrong visible localization counts as both FP and FN. This is a small measured difference, not a statistical significance claim.

Paired correctness changes comprise one gained visible ball, three lost visible balls, one gained absence and one lost absence. Thus the adapted pair finds two fewer true balls with one fewer false detection. All eight per-clip results and all five threshold scores are preserved in `review.json`; aggregate performance must not conceal the harder clips. At the selected thresholds, match176 recall is only 76.09% for both models.

## Verification and visual evidence

`review_wasb_training_threshold.py` independently recomputes all ten pooled scores and both threshold selections. It verifies all 400 paired labels, source video and label hashes, checkpoint and training-manifest hashes, protocol and inference-code hashes, target-window alignment, top-proposal selection, and the sixteen recorded comparisons against the existing 0.20 inference path. Verification passes exactly.

Five examples were selected by the existing fixed rule: the first two per gained/lost visible/absence category, sorted by clip and frame. Both rendered boards were inspected:

- `match176_000:20`: the original false prediction is on static fence/background highlights; the adapted pair abstains.
- `match176_000:68`: a small bright object shifts across the background near the labeled point; the adapted pair recovers the label, while the original abstains.
- `match176_000:412`: the source context contains a moving bright ball against the blue advertising board; the adapted pair loses it.
- `match194_000:106`: the ball passes close to the far player's shorts/body; the adapted pair loses the labeled detection.
- `match202_000:578`: the adapted false prediction lies in spectator/flower background. The displayed crop does not establish a true active ball there; retain the publisher absence label.

These observations are assistant diagnostics of a selected subset, not new ground truth. Labels and measured scores remain unchanged. The third lost-visible case is included in the numeric comparison but not in the fixed five-example visual sample.

Eleven focused tests pass in 3.50 seconds: threshold decoding/selection, temporal alignment, and point metrics. The shell's default Python resolves to MSYS2 Python 3.12, which lacks OpenCV and pandas; its first collection attempt failed. Rendering and the successful tests use the established Python 3.13 installation explicitly. No package installation or environment replacement was needed. The existing sandbox prevented launching that interpreter; the approved execution succeeded.

Saved inference times sum to 310.37 seconds for the original and 331.96 seconds for the adapted model, excluding setup. These are previous sparse multi-threshold timings, not a fresh isolated speed comparison or full-pipeline FPS. No new GPU training was run in this continuation.

## Decision and next gate

Follow the frozen protocol: keep the adapted checkpoint rejected and skip its conditional eighteen-clip external comparison. Threshold selection improves internal F1 for both models but does not support advancing the adapted checkpoint. Neither selected threshold is promoted into runtime configuration. Testing a new model, threshold, epoch or training recipe requires a separately declared protocol and training-only selection; the repeatedly inspected 900 development labels must not become tuning data.

The next ball work should address the existing nineteen proposal failures and twenty-nine absent-label false detections with better training/proposal evidence. First resolve hard training-negative ambiguity and define a data/selection plan before spending on a larger training run. Independent match-level qualification, unfamiliar-view court/player reliability, and physical speed validation remain open. Broad production reliability remains **NO**.

## Artifacts and reproduction

Artifacts: `outputs/vision_upgrade_audit/wasb_training_threshold01/` contains the preserved `report.json`, new `review.json`, two comparison images, and the visually reviewed `visual_review.json`.

Preserved inference report SHA-256: `58753418dea6f0ac8cd92c309daba96258730c0b111621ebc4b6809b3988c09d`.

The existing review scripts refuse to overwrite completed outputs. Their commands below describe this completed review; preserve the artifacts rather than rerunning into the same directory.

```powershell
& 'C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe' scripts/evaluate/review_wasb_training_threshold.py
& 'C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe' scripts/evaluate/render_wasb_training_threshold_review.py
& 'C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe' -m pytest tests/test_wasb_training_threshold.py tests/test_ball_temporal_spacing.py tests/test_point_metrics.py -q -p no:cacheprovider
```

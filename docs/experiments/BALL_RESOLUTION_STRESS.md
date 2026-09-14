# Controlled ball-resolution validation

Protocol fixed before execution. The existing videos are all 1920x1080; software tests on synthetic frame dimensions do not establish real detection accuracy at lower resolutions.

Use the six added validation clips (`match154_000`–`match159_000`), all explicit labels, native frame rates and the original .20/step-one WASB plus .25-second stationary filter and pixel-motion threshold 12/radius three/lag .10 s. Decode original video and compare native 1080p, area-downsampled 720p, and area-downsampled 480p. Compute width by rounding original aspect ratio (1920, 1280 and 853 respectively). No re-encoding, label interpolation, FPS change or new thresholds. Labels are scaled from their original 1920x1080 coordinates; error remains four pixels at reference 512x288, so smaller images do not receive an easier matching tolerance.

Every model frame is processed chronologically. A short queue aligns the current frame with the model's two-frame lookahead, then applies the same causal appearance evidence. Per-clip and aggregate counts, native frame counts, input geometry, data/weight/code hashes and processing time are saved. The native run must reproduce the existing six-clip pixel-filter counts (TP=262, FP=13, FN=24, TN=9) before interpreting resolution differences.

This is controlled degradation of the same broadcasts, not independent footage captured at low resolution. It cannot establish amateur-camera, compression, lighting or lens generalization. If lower resolution unexpectedly improves accuracy, treat that as evidence about preprocessing sensitivity, not a universal recommendation to downsample.

## Completed results

All three runs completed on the same six clips and 300 labels (286 visible, 14 absent). The native control reproduces all 300 existing labeled predictions, including missing states, within .001 source pixel. The paired comparator rejects incomplete result sets, duplicate/mismatched labels and inconsistent target scaling. Five geometry/comparison regression cases pass.

| Input geometry | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1920x1080 | 262 | 13 | 24 | 9 | 95.27% | 91.61% | 93.40% |
| 1280x720 | 260 | 13 | 26 | 9 | 95.24% | 90.91% | 93.02% |
| 853x480 | 260 | 13 | 26 | 7 | 95.24% | 90.91% | 93.02% |

At 720p, two previously correct visible detections are lost and none gained; absence outcomes are unchanged. At 480p, three correct visible detections are gained but five lost, and two previously correct absent frames become false detections. Identical aggregate P/R at 720p and 480p therefore conceals different error distributions. At 480p, wrong visible locations fall from eight to six, but visible abstentions increase from sixteen to twenty and absent false detections rise from five to seven.

These results support a narrow conclusion: the candidate continues to operate on these downsampled broadcasts, with a small aggregate accuracy decline and individual-frame instability. It does not pass the recall target at any tested resolution. Native input remains selected; no preprocessing or model change is promoted from this experiment.

Full predictions, code/data/weight identities and timing scopes: `artifacts/validation/vision_upgrade/additional6_ball_resolution.json`. Paired changes and native-control verification: `additional6_ball_resolution_comparison.json`. Execution log: `outputs/vision_upgrade_audit/additional6_ball_resolution.log`. A short CPU-only regression check overlapped the native pass, so timing is diagnostic and not a controlled speed comparison between resolutions. These repeated labels must not be pooled as 900 independent examples.

Reproduction:

```powershell
python scripts/evaluate/benchmark_ball_resolution.py --dataset data/external/racketvision_validation_additional6 --output artifacts/validation/vision_upgrade/new_resolution_run.json
python scripts/evaluate/compare_ball_resolution.py --resolution-report artifacts/validation/vision_upgrade/new_resolution_run.json --baseline-report artifacts/validation/vision_upgrade/additional6_original020_pixel_motion.json --output artifacts/validation/vision_upgrade/new_resolution_comparison.json
```

# Higher-detail temporal crop proposal pilot

Protocol fixed before execution. The learned crop verifiers were rejected; this experiment changes proposal generation instead of training another classifier on the same proposals.

Use the original WASB checkpoint at threshold .20. Process four overlapping crops, each 60% of source width/height, anchored to the four corners. Together they cover the whole image with overlap and make the ball about 1.67 times larger within the model input. Do not choose crops from labels, players, court fit or scoreboards. For each explicit labeled frame, average exactly the real chronological three-frame windows that contribute to it. Convert crop detections back to source coordinates.

Use the six added validation clips and their existing .20/full-frame candidate report. Compare full-frame mass-ranked top one, full-frame peak-ranked top one, tile peak-ranked top one, and combined full/tile peak-ranked top one. Cross-view duplicates within four reference pixels are removed greedily by confidence without averaging coordinates. Peak confidence is not calibrated across crops. Report correct-candidate coverage before merging separately as an oracle diagnostic, not deployed recall.

This is a sparse proposal test: coordinate-stationary and pixel-motion filters cannot be evaluated by treating labeled frames as a continuous sequence, so they are not applied. Do not compare these raw outputs directly with filtered pipeline counts as if processing were identical. No events, gap filling or position interpolation are added. Three synthetic tests verify coverage, overlap output-slot alignment, coordinate offsets and identity-preserving duplicate removal.

The frozen model was trained on different image context/scale. Cropping may increase blur sensitivity, lose useful court context or amplify false detections. All explicit visible and absent labels remain in evaluation. If candidates improve enough to justify the cost, a later full-video experiment must measure actual filtering and runtime before integration. No default or pipeline setting changes are authorized by this sparse diagnostic alone.

## Sparse result

Completed on all 300 explicit labels (286 visible, 14 absent). Report: `artifacts/validation/vision_upgrade/additional6_tiled_ball_candidates.json`.

| Selection | TP / FP / FN / TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Full frame, mass | 262 / 14 / 24 / 8 | 94.93% | 91.61% | 93.24% |
| Full frame, peak | 261 / 15 / 25 / 8 | 94.57% | 91.26% | 92.88% |
| Four crops, peak | 268 / 15 / 18 / 6 | 94.70% | 93.71% | 94.20% |
| Full frame plus crops, peak | 272 / 15 / 14 / 6 | 94.77% | 95.10% | 94.94% |

Correct-candidate coverage is 264/286 for full-frame proposals, 270/286 for tiles, and 273/286 for their union (95.45% oracle ceiling). The combined selection recovers ten additional true positives in aggregate, but produces two more absent-frame false detections. It does not clear both thresholds and is not promoted.

## Continuous protocol

The sparse result justifies measuring the identical five-view detector on every frame of the same six clips. `benchmark_tiled_ball_stream.py` uses threshold .20, crop fraction .6 and peak-confidence merging within four reference pixels. Separate outputs retain raw selection, the existing .25-second stationary filter, and that filter followed by the existing pixel-motion filter (score 12, lag .1 seconds, radius 3). All settings are fixed before this run; labels are used only for scoring. One source decode feeds five synchronized temporal inference generators; the pipeline has two future-frame offline lookahead as before. No positions are interpolated.

Eight targeted tests pass, including continuous/sparse temporal averaging agreement at empty, one-, two-, three- and seven-frame boundaries. The continuous report stores every frame's selected coordinates, sparse scored rows, source/model/code hashes, completion state and elapsed ball-only processing time. It excludes model setup, player/pose/racket/court processing and rendering, so its throughput is not full-pipeline FPS. Raw labeled predictions must reproduce the sparse result before filtered gains are accepted. The same clips have informed this experiment: they are development evidence, not a final holdout.

## Completed continuous result

All 2,893 source frames and 300 explicit labels were evaluated. Raw continuous selection matches every sparse selected coordinate exactly (maximum difference 0.0 source pixel), including missing states. The fixed filters produce:

| Variant | TP / FP / FN / TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Raw five-view selection | 272 / 15 / 14 / 6 | 94.77% | 95.10% | 94.94% |
| Plus stationary filter | 272 / 14 / 14 / 7 | 95.10% | 95.10% | 95.10% |
| Plus stationary and pixel-motion filters | 272 / 13 / 14 / 8 | 95.44% | 95.10% | 95.27% |
| Existing filtered full-frame control | 262 / 13 / 24 / 9 | 95.27% | 91.61% | 93.40% |

Compared with the filtered full-frame control, the final tiled variant gains ten correct visible detections and loses none. It introduces one additional absent-frame error. Identical aggregate FP counts conceal one corrected visible localization and one newly wrong absence. Six of fourteen absent labels still trigger false detections; absence specificity is only 57.14%.

| Clip | TP / FP / FN / TN | Precision | Recall |
|---|---|---:|---:|
| match154_000 | 43 / 0 / 0 / 7 | 100.00% | 100.00% |
| match155_000 | 48 / 0 / 2 / 0 | 100.00% | 96.00% |
| match156_000 | 44 / 4 / 5 / 0 | 91.67% | 89.80% |
| match157_000 | 46 / 4 / 4 / 0 | 92.00% | 92.00% |
| match158_000 | 47 / 1 / 2 / 0 | 97.92% | 95.92% |
| match159_000 | 44 / 4 / 1 / 1 | 91.67% | 97.78% |

Only three of six clips individually pass both targets. This is a promising development result, not a reason to promote the pipeline default or enable authoritative events. Next detector validation must use the frozen five-view/filter settings on the twelve-clip expansion, including the difficult match148, before integration is considered. These reused development clips do not establish an untouched test result, amateur coverage, or source-broadcast independence.

### Execution interruption and timing

The initial process saved five complete clips, then ended before saving the sixth. Its old session handle was missing, the recorded process ID no longer existed, and the partial report explicitly had complete=false. No restart was inferred from a timeout. A fresh output resumed only the final unsaved clip after dataset/model/configuration and inference/filter/metric code checks. The first five clip objects were retained; the original partial report was preserved. The resume source checksum and inherited code hashes are recorded. The benchmark script changed solely to support this recovery; the inference/filter implementation stayed fixed.

Summed completed-clip processing time is 998.04 seconds. This includes an anomalous 434.91-second fourth clip: a tool wait requested for 30 seconds also returned after 263.73 seconds during that interval. The cause was not established. The sum excludes unsaved work from the interrupted sixth-clip attempt, model setup, time between processes and other pipeline components. The derived 2.90 ball frames/second is therefore neither uninterrupted end-to-end throughput nor a controlled overhead comparison. No concurrent test or image-rendering workload was intentionally started during either inference process.

### Verification and artifacts

The full regression suite passes **486 tests**, with one existing Starlette warning, in 18.86 seconds. This includes continuous/sparse real-window boundary tests and comparison checks for coordinate drift, missing states, duplicate labels and dimension changes. Actual raw agreement on all 300 labels supplies the real-data alignment check.

All eleven changed labeled outcomes were rendered with raw source patches. Three were visually inspected: match154 frame145 recovers a blurred ball beside the far player; match155 frame293 replaces a court-logo false localization with the real ball; match158 frame84 adds a detection near the far player's lower leg/foot despite an absent publisher label. These visual observations explain examples, not independent relabeling of the benchmark.

- Original partial run: artifacts/validation/vision_upgrade/additional6_tiled_ball_stream.json
- Completed run: artifacts/validation/vision_upgrade/additional6_tiled_ball_stream_resumed.json
- Sparse/control and paired checks: artifacts/validation/vision_upgrade/additional6_tiled_ball_stream_comparison.json
- Every changed labeled outcome: outputs/vision_upgrade_audit/tiled_ball_decision_changes/review.json and eleven JPGs
- Regression log: outputs/vision_upgrade_audit/tiled_ball_regression_tests.log

## Frozen twelve-clip expansion result

All 6,082 frames and 600 explicit labels completed using the same five-view/filter settings. Results:

| Variant | TP / FP / FN / TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Full-frame stationary/pixel control | 505 / 27 / 38 / 42 | 94.92% | 93.00% | 93.95% |
| Five views, raw | 514 / 64 / 29 / 17 | 88.93% | 94.66% | 91.70% |
| Five views, stationary | 518 / 49 / 25 / 24 | 91.36% | 95.40% | 93.33% |
| Five views, stationary/pixel | 518 / 35 / 25 / 33 | 93.67% | 95.40% | 94.53% |

Paired scoring gains fifteen correct visible detections and loses two, but introduces nine absent-frame errors. Seven of twelve clips individually pass both thresholds. The difficult match148_000 scores 31 TP / 9 FP / 7 FN / 6 TN: precision 77.50%, recall 81.58%, F1 79.49%, versus the full-frame pixel-filter F1 of 73.53%. More visible balls are recovered, but false detections prevent promotion. The passing six-clip aggregate does not establish generalization. No default detector or event-authority change is made.

Results: artifacts/validation/vision_upgrade/expansion12_tiled_ball_stream.json. Paired labels and source hashes: expansion12_tiled_ball_paired_changes.json in the same directory. The process completed without resume. COCO acquisition and dependency preparation overlapped this accuracy run; timing is diagnostic only, as recorded in outputs/vision_upgrade_audit/expansion12_tiled_execution_notes.txt.

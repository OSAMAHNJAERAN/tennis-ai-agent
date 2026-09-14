# Photometric patch persistence pilot

Observed motivation: all 35 false selected ball detections from the completed twelve-clip tiled run were rendered with real past/current/future source patches. Two inspected examples are a court-logo fragment (match143 frame112) and scoreboard service dot (match148 frame68). Both pass the existing raw pixel-change filter despite recurring appearance. This suggests that compression, brightness changes or small image shifts may defeat absolute differences; the visual check alone does not identify which cause dominates.

Frozen primary experiment: compare each selected point's current grayscale 11x11 patch at reference 960x540 to actual imagery .1 seconds earlier, using zero-mean normalized correlation and a +/-2-pixel translation search. Reject a selected point only if maximum similarity >=.98. A predeclared secondary .95 threshold is development diagnosis, not a new holdout. Unknown startup, border or flat-texture comparisons retain the existing observation. No predicted location, interpolation or label-driven crop is emitted. The score is not ball identity or true physical motion.

Risks: faint or stationary real balls can have a similar patch, and background texture can dominate. Translation search can follow a slowly moving real ball. Replay retains every explicit visible/absent label and must measure recall losses, not only removed false positives. It filters existing selected points and cannot expose a correct lower-ranked candidate. If losses exceed benefit, reject the experiment rather than silently filling positions.

Code: src/tracking/candidate_patch_similarity.py; scripts/evaluate/benchmark_ball_patch_similarity.py. Four targeted tests pass, covering positive photometric gain, brightness offset, translated texture, flat/border uncertainty and invalid inputs. First evaluate the existing twelve-clip development set. A passing or promising setting must be tested on the six added clips before any integration; neither subset is an untouched final holdout. No pipeline default change is made.

## Twelve-clip result

All 6,082 frames were processed in 27.11 seconds for the CPU decode/similarity replay. Threshold .98 changes 518 TP / 35 FP / 25 FN / 33 TN to **514 TP / 26 FP / 29 FN / 41 TN**, precision **95.19%**, recall **94.66%**, F1 **94.92%**. It removes eight absent-frame detections and one wrong visible localization, but also removes four correct balls. It does not pass both requirements. Threshold .95 removes twelve additional true positives while only removing one additional false positive: **502 / 25 / 41 / 42**, precision 95.26%, recall 92.45%, F1 93.83%; that setting is rejected.

The .98 true-ball losses are match144 frame590, match148 frame80, and match151 frames166 and481. These require qualitative diagnosis before any modified persistence rule. The two originally inspected distractors (match143 frame112 and match148 frame68) are correctly suppressed, but fixing these examples is insufficient to establish a safe detector change. No pipeline setting is changed. Result and every per-frame score: artifacts/validation/vision_upgrade/expansion12_tiled_patch_similarity.json.

## Six-clip failure and local-residual follow-up protocol

The frozen similarity-only .98 gate also fails on the six added clips: 269 TP / 13 FP / 17 FN / 8 TN, precision 95.39%, recall 94.06%. Three correct balls are lost with no false-positive reduction. At .95, counts are 265 / 12 / 21 / 9. The similarity-only gate is rejected for integration.

All four true-ball losses from the expansion were rendered. Two inspected cases, match148 frame80 and match151 frame166, show tiny balls beside strong static court-line/net structure. Background correlation overwhelms the small local change. This motivates an additional safeguard rather than another similarity threshold sweep.

Fixed follow-up: retain similarity threshold .98 (secondary .95 reported only as development diagnosis), radius5 and shift2 at reference960x540. Fit a positive gain in [.5,2] and additive brightness offset using the aligned patch's outer ring, excluding the central7x7 pixels. Compute mean top-three absolute residuals in that center. Reject only when similarity meets the threshold AND central residual <=12. Unsupported fit, border or texture conditions retain the observation. The center is excluded from photometric fitting so a new ball cannot be absorbed into that fit. This is appearance evidence, not identity or physical motion; a faint real ball can still be lost.

Six targeted tests pass, including a controlled local brightness change over an unchanged photometrically transformed background. Evaluate both development sets with these frozen settings before any pipeline integration. Prior reports remain unchanged.


## Residual safeguard and complete pipeline result

Both frozen thresholds (.98 primary and .95 diagnostic) give identical labeled outcomes with the residual safeguard. On the twelve-clip expansion: **518 TP / 32 FP / 25 FN / 36 TN**, precision **94.18%**, recall **95.40%**, F1 **94.79%**. Three absent-frame false detections are removed and no labeled true ball is lost relative to tiled plus pixel motion. On the six added clips: **272 / 13 / 14 / 8**, precision **95.44%**, recall **95.10%**, F1 **95.27%**, unchanged from tiled plus pixel motion. These reused development sets do not establish generalization; expansion precision still fails the goal.

Evidence: `artifacts/validation/vision_upgrade/expansion12_tiled_similarity_residual.json` and `additional6_tiled_similarity_residual.json`. Similarity-only failures remain preserved. The residual rule is available only as an explicit experimental configuration, with .98 similarity and maximum residual12. It has not replaced the default.

`wasb_tiled_residual_pose_small_validation.yaml` combines synchronized full-frame plus four 60% crops, stationary filtering, pixel motion, selected-patch persistence and YOLO11s-pose. Configuration guards require the evaluated WASB overlapping-window / model-top1 path and disabled event authority. Patch rejection emits missing, never a lower-ranked substitute or an invented point. Its per-frame evidence is exported in `patch_persistence_audit.json`.

The full Phase6 run on match148 decodes and exports all **600 frames at native 60 FPS**, with **31 TP / 9 FP / 7 FN / 6 TN**, precision **77.50%**, recall **81.58%**, F1 **79.49%**. Previous full-frame pixel-motion counts were 25 / 5 / 13 / 9: recall improves while precision worsens. All 600 positions, missing states and patch rejection decisions match the saved replay exactly (maximum coordinate difference **0.0 pixels**, no mismatches). One unlabeled frame is rejected; sparse label counts are unchanged by the residual gate on this clip.

Full runtime including construction/import is **181.53 seconds**, **3.31 processing FPS**, sampled peak RSS **1,939,177,472 bytes**. This is a single local run, not a throughput guarantee. The broadcast review was fully decoded (600 frames, 60 FPS); frame87 was visually inspected for player skeletons, racket box, ball trail and observed court occupancy. Physical ball speed and event authority remain unavailable. Review: `outputs/vision_upgrade_audit/broadcast_review_tiled_residual_match148/tracking_review.mp4`. Accuracy/invariant reports: `phase6_tiled_residual_match148_ball.json` and `phase6_tiled_residual_match148_invariants.json` under the validation directory.

Validation after integration: **503 tests passed**, one existing warning, 19.99 seconds (`outputs/vision_upgrade_audit/tiled_residual_integration_tests.log`). This checks software behavior, not production accuracy. Remaining errors require stronger learned discrimination and broader independent data; repeated filtering alone has not met both target metrics.

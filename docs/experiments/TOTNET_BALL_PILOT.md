# TOTNet tennis pilot: frozen protocol

Frozen before inference on 2026-09-12. Candidate is the official TOTNet tennis best checkpoint at revision `8a757f63391b262c14d18b4095486336852dbeef`, SHA-256 `36caadd2453cf1a37f26afb0024b861fd3285ca6e9a91e9e4a20d35e90a0a24a`. Its 196 tensors strictly match the author's 8,654,467-parameter architecture, five frames, 64 spatial channels, 288 by 512 input, causal last-frame target. No model parameters are filled from random initialization.

Use the six pre-existing clips in `data/external/racketvision_validation/manifest.json`, unchanged publisher annotations and manifest hashes. Decode every native frame, but infer only explicitly annotated frames using their five consecutive real frames. This is a point-detection pilot, not continuous tracking or end-to-end throughput qualification. Frames lacking four preceding frames abstain, remain in scoring, and are counted separately. No padding, interpolation, optical flow, temporal filtering, or tiled views.

Match author preprocessing: OpenCV BGR; Lanczos4 stretch to 512 by 288; division by 255 and mean `(0.485,0.456,0.406)` / standard deviation `(0.229,0.224,0.225)`; float32 tensor `[1,5,3,288,512]`. Verify pixel equality against the archived official Resize and Normalize implementations. Use full float32, evaluation mode and inference mode. Output is one flattened spatial softmax; decode its argmax. Primary raw scoring always returns the argmax location, including the origin. Report a separately named origin-as-absence interpretation without tuning: exact grid `(0,0)` becomes absent; all other coordinates remain unchanged. Neither interpretation uses a fitted confidence threshold.

Use existing visibility-aware `evaluate_points` with 4 pixels at reference 512 by 288, also report 2 and 8 pixels. Wrong visible locations count both FP and FN. Only explicit absent annotations count negatives. Preserve per-frame predictions, peak probability, dimensions, source frame IDs, labels and timings. Do not use paper PCK as a substitute for these precision/recall metrics. Dataset independence from training broadcasts remains unverified.

The standard restricted loader initially rejected EasyDict SETITEMS in Torch 2.13. The successful export uses the standard restricted loader with that metadata name mapped to built-in OrderedDict, so no EasyDict restoration methods execute. Exported plain tensors reload with the default restricted loader and compare exactly. The separately blocked RTMDet loader remains unchanged.


## Completed result

All 2,561 native frames decoded; all 300 publisher-labeled frames had five real input frames, so boundary abstentions were zero. Full float32 forward passes took 134.88 seconds for 300 predictions, excluding preprocessing and decoding. Peak allocated CUDA memory was 2,085,798,912 bytes. This sparse-label timing is not continuous or production FPS.

| Recipe on the same six clips | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TOTNet raw argmax | 261 | 39 | 33 | 0 | 87.00% | 88.78% |
| TOTNet exact-origin absence interpretation | 261 | 29 | 33 | 2 | 90.00% | 88.78% |
| Existing WASB overlap step 1, threshold 0.25 | 276 | 8 | 18 | 3 | 97.18% | 93.88% |

Raw TOTNet per-clip TP/FP/FN: match138 38/12/10; match139 40/10/10; match14 46/4/3; match140 47/3/1; match141 42/8/8; match142 48/2/1. The same labels, dimensions and frame IDs were verified exactly against the saved WASB report. Metrics were independently recomputed from the completed TOTNet rows and matched exactly. TOTNet finds 5 labeled balls that WASB misses but loses 20 that WASB finds; both find 256. This is complementarity analysis using ground truth, not an implemented or validated fusion rule.

Six diagnostic source-frame panels were rendered and inspected. Match139 frame275 places the prediction on the foreground player's head area while the publisher ball label is near the far player. Other examples show substantial displacements on court/background or an image border. Peak softmax values also remain appreciable for missed-ball outputs at top-left and top-right corners. These observations do not justify a fitted confidence threshold or relabeling ground truth.

The currently published training transform flips coordinates even when visibility is zero; the selected WBCE loss builds a normalized spatial Gaussian for every visibility class and clamps coordinates to the image. This is consistent with corner-coded absent targets, but the exact source revision used to train the released checkpoint is not established. The evidence supports a training-target hypothesis, not a proven causal diagnosis. The current loss factory also uses its default visibility weights rather than passing the saved configuration's weighting_list; do not assume recorded configuration proves effective training weights.

Decision: do not replace WASB with this checkpoint. Neither TOTNet interpretation meets the >95% precision and recall goal; the existing same-six-clip WASB control also misses the recall goal. Preserve the candidate for investigating missed-ball complementarity or improved training targets, without changing production defaults. This result does not replace the larger 18-clip WASB development evidence or establish unseen-camera performance.

Artifacts: `outputs/vision_upgrade_audit/totnet_ball_pilot01/report.json`, `paired_wasb_review.json`, `visual_review.json`, `localization_failures.jpg`. Report SHA-256 `9c6d5cc0ebbf082219c1e7734896504ba6cdc1e8305725705501fa1bb4e4decd`; paired WASB source report SHA-256 `e8ce7488e419720fd28ea2e98e67d4c91e86ee515ab1a858fccb9eb0c04dded6`. The goal remains active.

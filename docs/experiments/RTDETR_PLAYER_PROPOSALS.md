# RT-DETR-l person proposals and BoT-SORT comparison

Protocol frozen before inference. The original-racket BoT-SORT candidate improves selected-player precision and recall, but original YOLO11m person proposals still miss much of V02. Evaluate a distinct pretrained detector architecture before adding selection heuristics.

Use official Ultralytics release v8.4.0 RT-DETR-l asset ID 340060217, 66,511,432 bytes, SHA256 `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`. Verify the published digest and statically inspect its declared globals. Restricted weights-only loading permits only the explicitly reviewed installed architecture and standard PyTorch classes; never allow arbitrary pickle loading. Installed Ultralytics 8.4.36 remains fixed. This is the Ultralytics converted checkpoint, not a claim to run the original Apache-licensed implementation.

Run every one of the 2230 UVY frames at image size 640, confidence 0.10, class person (0), maximum 300, no augmentation. Preserve the installed RT-DETR recipe: scale-fill to 640x640, each query's argmax class, score sorting and no NMS. Clip xyxy coordinates to source image bounds for both raw evaluation and BoT-SORT input. Record degenerate boxes if any and exclude only zero-area boxes. The YOLO11m reference uses its earlier native rectangular preprocessing and NMS IoU 0.70. This is a comparison of complete detector recipes at the same nominal size, not equal pixel count, equal capacity or a causal architecture-only ablation. Do not add NMS to RT-DETR or tune confidence against the labels.

Compare raw person IoU-0.50 label-assisted coverage with the shared confidence-0.10 YOLO cache. Then apply the installed unchanged BoT-SORT YAML (sparse optical-flow compensation, appearance ReID disabled, tracker frame_rate 30 as in the existing adapter) to RT-DETR detections and clip its output to source bounds. Use native FPS for durations. Compare its coverage, per-GT fragmentation and official CLEAR association switches against the verified YOLO11m/BoT-SORT cache. Unmatched people are not person-detector false positives because labels cover active players only. Labels include known omissions, loose boxes and an umpire misclassification and remain unchanged.

Cache all detections/tracks, configurations, source hashes, checkpoint manifest, timing and per-GT diagnostics. Time decoding/detection separately from decoding/tracking and do not call either end-to-end speed. No production selection or active-player precision claim until a separately frozen downstream original-racket control and visual review are complete. Broad amateur/camera qualification remains outstanding.

## Completed results

The original process stopped after saving V01's complete raw cache. Its handle was missing and no corresponding OS process remained. `resume_rtdetr_player_proposals.py` verified the checkpoint, dataset, source code, installed implementation, configuration and completed cache hashes before retaining V01 and processing V02/V03 plus all tracking. `report_before_resume.json` preserves the partial state. No inference setting changed; timings exclude the stopped attempt's unfinished V02 work and are not total experiment wall time.

| Clip | YOLO raw matches | RT-DETR raw matches | YOLO + BoT matches | RT-DETR + BoT matches | YOLO / RT-DETR CLEAR switches |
|---|---:|---:|---:|---:|---:|
| V01 | 1416 | 1433 | 1341 | 1407 | 8 / 4 |
| V02 | 842 | 754 | 585 | 727 | 11 / 0 |
| V03 | 618 | 656 | 539 | 653 | 3 / 1 |

At the fixed operating point, RT-DETR raw coverage falls 2876 to 2843 matches, while BoT-SORT coverage improves 2465 to 2787, a gain of 322 matches. This is label-assisted person coverage, not active-player precision. V02 near-player matched coverage actually falls 338 to 322 while far-player coverage rises 247 to 405; the total must not hide the near-player loss.

RT-DETR/BoT-SORT has 176 / 68 / 96 distinct source IDs versus 50 / 49 / 34 for YOLO/BoT-SORT among all observed people. Its dominant matched IDs cover V01 near 444 and far 624 frames; V02 near 322 and far 404; V03 near 443 and far 194. CLEAR switches and framewise maximum-cardinality source counts use different assignment rules, so their differences are not contradictions. Longer associations are not verified re-identification.

The actual RT-DETR transformed image shape is 640x640x3 for all sequences. No clipped raw box became degenerate. Decode/detection times are 64.79 / 88.26 / 40.86 seconds; decode/tracking times 31.68 / 35.20 / 25.15 seconds. These single-run component measurements are slower than the earlier YOLO recipe and are not end-to-end timings. The clipped raw cache is preserved; pre-clipping RT-DETR coordinates were not separately saved in this pilot, a provenance limitation. The fixed checkpoint/source hashes permit a separate targeted verification if necessary.

All 2230 frames complete. The downstream original-racket selection is evaluated separately in `RTDETR_RACKET_PLAYER_SELECTION.md`; no production detector replacement or broad qualification follows from these proposal metrics alone.

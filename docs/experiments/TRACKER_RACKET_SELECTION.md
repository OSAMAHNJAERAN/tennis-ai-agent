# Paired tracker comparison through racket-supported player selection

Protocol frozen before downstream inference. Use the complete shared-confidence-0.10 person cache from `uvy_tracker_architecture_pilot01`, including its fresh ByteTrack baseline and BoT-SORT alternative. Do not reuse earlier source-ID mappings or racket assignments: both the tracker boxes and source IDs can differ.

Use the previously internally selected mixed-data racket checkpoint SHA256 `c4416708def99c60dc041aaa8fa5d6bbaf3d0d39f2b016876c61e2f87812495a`, restricted weights-only loading, confidence 0.25, crop size 640, nearest-person ownership and samples every 0.2 seconds at native FPS. This checkpoint remains research-only because its external racket precision regressed. All original person detection and tracker settings remain frozen.

Measure two prespecified selector variants for each tracker: (1) unchanged raw source identities; (2) unchanged image-flow linking plus brief duplicate-handoff reconciliation, independently recomputed from that tracker's observations. Both require at least two racket-support frames and 0.5 seconds of observed person boxes, rank by the existing opportunity-normalized confidence score and select at most two people. Do not interpolate boxes, copy annotations into detections, tune thresholds or use labels to choose identities.

Evaluate all 2230 frames using original class-1 publisher labels and pinned official TrackEval HOTA, CLEAR and Identity. Report TP/FP/FN, ID switches, IDF1 and HOTA per sequence and aggregate detection counts. Known label omissions, loose boxes and the mislabeled umpire remain; results are development diagnostics, not independent generalization evidence. The selector uses whole-sequence evidence and is offline. Time racket inference and linking separately from upstream detection/tracking; do not call these end-to-end speeds.

Save all sampled raw racket candidates and assignments, source mappings and linking diagnostics, both selected outputs, hashes and protocol. Confirm every selected box and confidence comes from the corresponding actual source observation. Review paired first/middle/last frames and inspect the third clip's recovered near-player boundary. No production promotion based solely on these three repeatedly inspected professional-match clips.

## Adapter correction before the second downstream run

The first direct-tracker pilot omitted image-boundary clipping performed by installed `Results.update`. Clipping the cached ByteTrack outputs reproduces every one of the 2230 original application frames exactly, including IDs, confidence, box coordinates and row order. Preserve that first pilot as an adapter diagnostic. The authoritative paired comparison uses `uvy_tracker_architecture_clipped01` with the same image clipping applied to both trackers; recompute racket crops, ownership, linking and selection for both. All thresholds and model weights remain frozen. Store the earlier downstream script with its original report for provenance.

## Completed authoritative results

All six model/sequence passes and both selector variants complete. With image clipping, the ByteTrack linked selection reproduces all 2230 prior trained-racket selected frames exactly. Every selected box and confidence is verified against the corresponding actual source observation.

| Clip | Tracker + flow/handoff | TP | FP | FN | IDSW | IDF1 | HOTA |
|---|---|---:|---:|---:|---:|---:|---:|
| V01 | ByteTrack | 767 | 102 | 678 | 3 | 38.20% | 36.07% |
| V01 | BoT-SORT | 769 | 100 | 676 | 2 | 37.94% | 39.55% |
| V02 | ByteTrack | 56 | 170 | 1880 | 0 | 5.18% | 7.24% |
| V02 | BoT-SORT | 64 | 164 | 1872 | 0 | 5.91% | 7.25% |
| V03 | ByteTrack | 469 | 1 | 353 | 0 | 72.60% | 61.28% |
| V03 | BoT-SORT | 498 | 49 | 324 | 0 | 72.75% | 61.05% |

Aggregate ByteTrack: 1292 TP / 273 FP / 2911 FN, precision 82.56%, recall 30.74%. BoT-SORT: 1331 / 313 / 2872, precision 80.96%, recall 31.67%. The candidate gains 39 matches and adds 40 false selections; it is not promoted.

Without flow/handoff, ByteTrack produces 1203 TP / 274 FP / 3000 FN (P81.45%, R28.62%, four switches). BoT-SORT produces the same selected output and metrics with and without linking; its only accepted flow edge joins an unselected V01 source. Thus the learned-racket BoT-SORT result is not a general precision improvement even though raw association is more stable.

### Visual diagnosis and validation

First/middle/last paired frames from all clips were inspected, along with every racket support and first/middle/last selected frames of V03 source 79. This is a seated foreground spectator. Its three trained-model supports at zero-based frames 228, 240 and 306 lie around spectators' heads/hands/glasses, not visible tennis rackets. The offline selector consequently emits 48 spectator boxes. The V03 near player contributes two additional matches and the far player 27 additional matches, while this false track contributes the 48 additional unmatched selections. This diagnosis is observational, does not relabel GT and does not become a hand-tuned source-ID exclusion.

All selected observations pass exact source verification. The paired V03 video fully decodes 443 frames at 29.97 FPS; a decoded frame 421 was visually inspected. Video SHA256: `72091b55a739ad3540e362692f550a6f8905263de80ec1431acbe0b08991a05d`. The five focused suites for flow linking, handoff, nearest ownership, racket selection and official metric adaptation pass 32 tests in 0.67 seconds. No production defaults changed.

Artifacts: `outputs/vision_upgrade_audit/tracker_racket_selection_clipped01/report.json`, `final_review.json`, `false_track79_review.json`, `paired_tracker_V03.mp4` and review images. Frozen protocol hashes refer to the pre-result copies in the output folders. The unclipped first pilot and its original benchmark-source snapshot remain intact. Racket decoding/inference takes 14.69 / 25.04 / 27.76 seconds for ByteTrack crops versus 13.59 / 26.03 / 13.81 seconds for BoT-SORT crops; linking takes 0.76 / 0.65 / 1.67 versus 0.72 / 0.39 / 1.23 seconds. These single-run timings depend on each tracker's crops and are not end-to-end speed claims.

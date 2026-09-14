# ByteTrack versus BoT-SORT on identical person detections

Protocol declared before inference. Recent racket-based player selection recovers more source fragments but leaves identity switches across source-ID changes. Compare the installed tracker architectures before adding further linking heuristics.

Use the unchanged YOLO11m checkpoint SHA256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`, verified source videos and all2230 native frames. Detect persons once at640, confidence0.10, NMS IoU0.70, maximum300, no augmentation. Save every untracked detector box and confidence. The old raw640 cache uses0.25 and is unsuitable for this paired comparison. Force restricted weights-only loading with the previously reviewed standard architecture classes.

Run trackers sequentially from fresh instances on this identical cache, so shared source-ID counters cannot contaminate simultaneous runs. Use the installed ByteTrack and BoT-SORT YAMLs unchanged, capture their contents/hashes and relevant installed source hashes. BoT-SORT uses sparseOptFlow camera compensation, with_reid false; this is a bundled architecture/configuration comparison, not a causal ablation of compensation alone. Keep the library adapter's frame_rate30 initialization for both to match the earlier ByteTrack baseline; use actual source FPS for timestamps and duration reporting. Document the small difference from native FPS; no physical speeds are inferred.

Measure person-proposal IoU0.50 coverage and per-GT matched source-ID fragmentation using identical maximum-cardinality matching. Original labels cover active players, not all people; unmatched spectators must not be presented as person-detector false positives. Trace association using the pinned official CLEAR assignment, but report ID-switch counts only with the caveat that these are labeled-player associations among unfiltered person proposals, not an active-player selector. Report number of source IDs, longest matched span, and dominant-ID coverage for each original GT identity. Known loose boxes, omissions and misclassified umpire remain unchanged.

Compare fresh ByteTrack output to the existing tracked640 cache and record any differences rather than assuming exact reproducibility. Record decode/detection and tracker timing separately; no comparison of startup-excluded times to prior end-to-end runs. Store source candidates, selected tracker boxes and provenance. No production promotion without downstream player/racket selection and visual review.

Primary reference: https://docs.ultralytics.com/modes/track/ . Installed code/configuration is authoritative for this version; current online docs may describe newer features.

## Completed measurements and adapter correction

The direct-tracker pilot completed all 2230 frames. Its boxes differed from the application because `Results.update` clips the tracker output to image boundaries. The adapter correction in `adapt_tracker_cache_to_image_bounds.py` reproduces all 819/968/443 old ByteTrack frames exactly, including IDs, confidence, box coordinates and row order. Both trackers receive identical clipping in the authoritative cache `uvy_tracker_architecture_clipped01`. The unmodified direct-tracker pilot is retained for provenance; it is not the application baseline.

| Clip | Matched proposals Byte / BoT | Source IDs Byte / BoT | CLEAR switches Byte / BoT |
|---|---:|---:|---:|
| V01 | 1329 / 1341 | 55 / 50 | 10 / 8 |
| V02 | 631 / 585 | 53 / 49 | 13 / 11 |
| V03 | 519 / 539 | 45 / 34 | 6 / 3 |

The proposal total decreases from 2479 to 2465 matches. These are label-assisted coverage measurements among all persons; unmatched spectators are not person-detector false positives. Fewer source IDs and switches do not establish better active-player precision.

For V03's near player, the dominant source covers 380 versus 442 matched frames and the longest uninterrupted matched span rises from 380 to 442 frames (12.68 to 14.75 seconds). V01 longest spans remain 382 near / 305 far frames under both trackers. V02's longest spans are 37 near / 75 far for ByteTrack versus 41 / 70 for BoT-SORT. All per-GT counts and spans are saved in the paired downstream final review. Original label defects remain unchanged.

Decode plus detection of the shared cache takes 10.53 / 10.32 / 5.45 seconds. Decode plus tracking takes 0.72 / 1.02 / 1.14 seconds for ByteTrack versus 7.31 / 8.54 / 6.68 seconds for BoT-SORT. These are single-run component measurements, exclude model startup and downstream processing, and do not establish end-to-end throughput.

Downstream studies: `TRACKER_RACKET_SELECTION.md` (trained racket) and `TRACKER_ORIGINAL_RACKET_SELECTION.md` (original-racket control). No production tracker default has been changed.

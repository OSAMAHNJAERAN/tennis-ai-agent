# Original-racket control for the paired tracker comparison

Frozen before inference. The trained-racket paired tracker experiment completes all 2230 frames, but BoT-SORT selects a foreground spectator for 48 frames in V03 after three false racket supports. The original racket detector had higher external precision. Compare both trackers with the original model before attributing the failure to tracking alone.

Reuse the verified image-clipped paired ByteTrack and BoT-SORT person caches. Change only racket weights from trained pilot04 to the unchanged original YOLO11m checkpoint SHA256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`. Retain confidence 0.25, crops at 640, samples every 0.2 seconds, nearest-person ownership, source durations, repeated racket support, ranking and maximum two selected people. Recompute all racket observations separately for each tracker. Evaluate both raw source identities and the unchanged flow-plus-handoff variant, independently recomputing identity links from each cache.

The ByteTrack linked control must reproduce the previous original-model duplicate-handoff selection (`uvy_duplicate_handoff_pilot01`) on every frame. Report all results without choosing per-clip trackers, models or thresholds. Compare original versus trained racket results as a prespecified control; these repeatedly inspected professional-match clips do not establish amateur or cross-camera qualification. Keep publisher labels unchanged. Inspect selected output and every unexpected spectator selection, preserve exact person observations and complete provenance. Report tracker and racket timings separately; no fabricated physical analytics or event claims.

## Completed results

All 2230 frames and both selector variants complete. The clipped ByteTrack control reproduces all 819 / 968 / 443 prior original-model selected frames exactly, with all current selected boxes and confidences verified against actual source observations.

| Clip | Tracker + flow/handoff | TP | FP | FN | IDSW | IDF1 | HOTA |
|---|---|---:|---:|---:|---:|---:|---:|
| V01 | ByteTrack | 551 | 102 | 894 | 1 | 42.14% | 34.36% |
| V01 | BoT-SORT | 641 | 102 | 804 | 1 | 40.13% | 38.32% |
| V02 | ByteTrack | 0 | 0 | 1936 | 0 | 0.00% | 0.00% |
| V02 | BoT-SORT | 0 | 0 | 1936 | 0 | 0.00% | 0.00% |
| V03 | ByteTrack | 440 | 1 | 382 | 0 | 69.68% | 61.13% |
| V03 | BoT-SORT | 498 | 1 | 324 | 0 | 75.40% | 62.77% |

ByteTrack linked totals: 991 TP / 103 FP / 3212 FN, precision 90.59%, recall 23.58%. BoT-SORT totals: 1139 / 103 / 3064, precision 91.71%, recall 27.10%. The improvement is 148 net additional matches with the same false-positive total. Frame-level diagnosis finds 94 gained and four lost matches in V01, plus 58 gained and none lost in V03. It is not a claim that every earlier true positive was preserved. V01 IDF1 decreases despite better coverage and HOTA; identity consistency remains incomplete.

BoT-SORT produces identical selected output with and without extra flow/handoff linking. Original-racket raw ByteTrack yields 872 TP / 103 FP / 3331 FN (P89.44%, R20.75%). Thus stable source tracks supply useful repeated racket evidence without requiring those additional linking heuristics on these clips.

The V03 BoT-SORT near player supplies 442 matched frames from 443 selected observations; far source 122 supplies 56 matched observations and receives two original-model racket supports. The trained-model spectator source 79 receives insufficient original-model support and is not selected. V01's longer source 178 carries racket evidence across its early interval, which the fragmented ByteTrack selector misses. These are offline whole-sequence selections, not proof of real-time recovery.

### Review, timing and decision

Reviewed all first/middle/last paired frames, V01 frame 663's recovered near player, V03 frame 382's recovered boundary and the far player's first/middle/last selected frames 285 / 318 / 355. The V03 video fully decodes 443 frames at 29.97 FPS, and decoded frame 421 was visually inspected. Video SHA256: `9fbbcf0f1f590ef64df5b1b7aff271bd1de786a248322654ca063c6049a23683`.

Racket decoding/inference takes 15.28 / 23.36 / 13.65 seconds for ByteTrack crops and 13.64 / 24.13 / 13.92 seconds for BoT-SORT crops. Linking takes 0.72 / 0.53 / 1.64 versus 0.52 / 0.35 / 1.20 seconds. Separate upstream timings are in `UVY_BOTSORT_COMPARISON.md`; none of these are end-to-end throughput measurements. The unchanged five focused core suites passed 32 tests before this control; exact baseline replay and artifact verification validate this benchmark run without repeating those unchanged tests.

Retain BoT-SORT plus the original racket model as a promising research candidate. It improves precision and recall relative to the original-model linked ByteTrack baseline, but V02 remains entirely empty, aggregate recall is only 27.10%, V01 still has false selections and an identity switch, and the clips are repeatedly inspected professional fan recordings. No production default is changed and no amateur/general-camera qualification is claimed.

Artifacts: `outputs/vision_upgrade_audit/tracker_original_racket_selection_clipped01/report.json`, `final_review.json`, `gain_diagnostics.json`, `paired_tracker_V03.mp4`, paired contact rows and targeted gain images. The frozen protocol in the output folder predates the results.

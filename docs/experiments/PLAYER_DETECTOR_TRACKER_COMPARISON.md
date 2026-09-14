# Measured player detector, tracker and racket comparison

Development comparison completed on all 2230 native frames of three UVY professional fan recordings, using the unchanged 4203 class-1 publisher boxes. These clips have been inspected repeatedly, contain known label defects, and do not provide independent amateur or general-camera qualification. Every row below uses the same offline repeated-racket selector, nearest-person ownership and at most two selected people. Image-flow and duplicate-handoff rules are applied consistently; they change ByteTrack output but do not change the listed BoT-SORT selected outputs.

| Person detector | Tracker | Racket detector | TP | FP | FN | Precision | Recall | ID switches |
|---|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11m | ByteTrack | Original YOLO11m | 991 | 103 | 3212 | 90.59% | 23.58% | 1 |
| YOLO11m | BoT-SORT | Original YOLO11m | 1139 | 103 | 3064 | 91.71% | 27.10% | 1 |
| YOLO11m | ByteTrack | Trained racket pilot04 | 1292 | 273 | 2911 | 82.56% | 30.74% | 3 |
| YOLO11m | BoT-SORT | Trained racket pilot04 | 1331 | 313 | 2872 | 80.96% | 31.67% | 2 |
| RT-DETR-l | BoT-SORT | Original YOLO11m | 1090 | 429 | 3113 | 71.76% | 25.93% | 1 |

Precision and recall aggregate counts across clips. ID switches are sums of official CLEAR sequence counts; they are not re-identification success rates. A system emitting no players can have zero switches, so switches alone cannot select a winner. HOTA and IDF1 remain per-sequence in the linked experiment reports rather than being averaged into an unsupported overall score.

## Interpretation

YOLO11m/BoT-SORT/original racket improves both precision and recall over the original-racket ByteTrack baseline, with 148 net additional true positives and the same false-positive count. V01 nevertheless loses four previously matched frame observations and its IDF1 decreases; the gain is not uniform. V02 emits no players under either original-racket recipe. This candidate is retained for further research, not promoted as a broadly reliable system.

The trained racket checkpoint trades precision for recall and introduces a foreground spectator in V03. RT-DETR improves tracked proposal coverage by 322 labeled boxes but worsens downstream selection: long nonplayer tracks receive repeated false racket supports. These results show why person coverage, stable identities and full player-selection accuracy must be evaluated separately.

All configurations use fixed checkpoints and confidence thresholds, not per-clip winning settings. RT-DETR uses square stretching and no NMS; YOLO uses its native rectangular recipe and NMS. The detector comparison is therefore of complete recipes at nominal size 640, not an isolated architecture effect or equal computation. RT-DETR also takes substantially longer in these measured component runs. Full timings and their exclusions are in the individual reports.

## Reproducibility and next evidence

The direct tracker benchmark initially omitted image-boundary clipping. Applying the installed application adapter's clipping to both trackers reproduces every original ByteTrack frame exactly. Both original- and trained-racket linked ByteTrack controls also reproduce all prior selected frames exactly. The initial unclipped pilot is preserved and is not the application baseline.

Raw person proposals, sampled racket candidates, assignments, selected observations, checkpoint/configuration/source hashes, frozen protocols and visual reviews are retained under `outputs/vision_upgrade_audit`. No person boxes were interpolated or generated from labels. No absolute ball speeds, court-contact events or biomechanical claims follow from these experiments.

Next evidence should address racket specificity and missed tiny rackets before extending repeated-support heuristics. The published RTMDet racket specialist is unmeasured: its official checkpoint was downloaded and verified, but default restricted loading rejects training metadata. The metadata and runtime prerequisites are under review. This does not establish whether its tennis-racket accuracy will improve the current result.

Experiment records:

- [Shared-detection tracker comparison](UVY_BOTSORT_COMPARISON.md)
- [Original-racket tracker control](TRACKER_ORIGINAL_RACKET_SELECTION.md)
- [Trained-racket tracker comparison](TRACKER_RACKET_SELECTION.md)
- [RT-DETR proposal evaluation](RTDETR_PLAYER_PROPOSALS.md)
- [RT-DETR downstream selection](RTDETR_RACKET_PLAYER_SELECTION.md)
- [Specialist runtime and checkpoint inspection](../research/RACKET_SPECIALIST_RUNTIME_2026_09_12.md)

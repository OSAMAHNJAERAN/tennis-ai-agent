# YOLO26 tennis annotations and independent evidence

Version 1, 8 October 2026. Scope: singles, stationary or nearly stationary elevated end-court/baseline broadcast view. This guide defines labels; it does not certify a dataset or report measured accuracy.

Each publisher/source record must retain owner, URL/path, acquisition UTC date, original checksum, tournament/match/camera-session identity, surface, view, resolution, actual FPS/time base and duration. Training, research evaluation, weights and public derivatives have separate permission records. An internet listing or code license is insufficient when a contradictory footage restriction is known. Retain restrictions and attribution; do not publish research-only footage as a public example.

Partition complete match/session groups before training. Initial target: 60% train, 20% validation/calibration, 20% frozen test, with harder new contexts reserved separately. Clips/replays/neighboring frames and exact/near duplicate groups stay in one split. Previously optimized project examples stay development data. A small pilot can establish valid labels and learning without claiming final-test qualification. Do not inspect frozen test outputs for selecting weights or thresholds.

Ball: tight visible box and center in original image pixels; states visible, occluded, absent or ambiguous; near/far court, blur severity, active/dead-ball status and frame timestamp. Label only the resolvable visible ball, never an imagined hidden center. A blur box encloses the visible smear consistently and marks uncertainty. Annotate spare balls, logos, rackets, white clothes, court lines and pre-serve bounces as negatives/ambiguities. Convert reviewed boxes to YOLO class 0 `tennis-ball`; empty label files are explicit negative images, not missing annotation files.

Players: two actual players, image boxes, stable match A/B identity, independent local track IDs, 17 COCO joints and visibility. Human pose YOLO class 0 is `person`; tennis participant identity is a separate reviewed field. Include jumps/bends/overlap/partial occlusion and non-player people as controls. Anatomical ankles are not proof of ground contact. Ground-reference labels record the contacting foot, jumping/occluded state, independent calibration source and spatial uncertainty.

Court: one instance, YOLO class 0 `court`, `kpt_shape: [14,3]`. Axis x runs left→right; y runs from the far baseline toward the near baseline, origin at far-left doubles corner. Canonical dimensions: doubles width 10.97 m, singles width 8.23 m, length 23.77 m and service line 6.40 m from net. Preserve this repository's indexed order:

| Index | Intersection | Canonical x,y (m) |
|---|---|---|
| 0,1 | Far baseline with left/right doubles sideline | (0,0), (10.97,0) |
| 2,3 | Near baseline with left/right doubles sideline | (0,23.77), (10.97,23.77) |
| 4,5 | Left singles sideline with far/near baseline | (1.37,0), (1.37,23.77) |
| 6,7 | Right singles sideline with far/near baseline | (9.60,0), (9.60,23.77) |
| 8,9 | Far service line with singles sidelines | (1.37,5.485), (9.60,5.485) |
| 10,11 | Near service line with singles sidelines | (1.37,18.285), (9.60,18.285) |
| 12,13 | Far/near service line with center service line | (5.485,5.485), (5.485,18.285) |

```text
0-----4--------------6-----1   far baseline, y=0
|     |              |     |
|     8------12------9     |   far service line
|     |       |      |     |
|=====|=======NET====|=====|   y=11.885
|     |       |      |     |
|    10------13-----11     |   near service line
|     |              |     |
2-----5--------------7-----3   near baseline, y=23.77
```

The existing coordinates describe regulation outside boundaries, whereas a visible intersection may be annotated at a line center. Record the exact convention and line width for each court. Do not infer centimeter line decisions until those offsets are reconciled. Visibility 0 means unlabelled/absent, 1 means labeled but occluded and 2 means visible. Occluded inferred court intersections must be distinguished from visible evidence and excluded from independent visible-localization denominators. Disable reflection unless the fixed semantic remapping is correct. Reserve distributed points/lines not used in fitting for independent calibration tests.

Events: original PTS/timestamp, frame, uncertainty interval, stable event ID, segment, point ID, event kind (point start/serve attempt/contact/bounce/net/rally end/outcome), player when known, observed ground contact when independently supported, and active play state. Preserve provisional/confirmed/corrected/rejected status and supporting source interval. Contact between frames has a timing range, not exact observed precision. First/second serves, lets, warmup, pre-serve bounces and dead-ball intervals are distinct. Do not count every bounce as a point.

Score: independently verified initial points/games/sets/server/format, every point winner or unknown, loss reason when visible, synchronization provenance and uncertainty. Unknown consequential points stop confirmed score propagation. Stable A/B identity survives end changes only with reviewed evidence. A filename, near/far role or end of video cannot establish match winner.

Physical references: radar or independently calibrated cameras must share source timing and the same speed definition (impact speed versus segment average). Retain reference error, sync tolerance, calibration, measurement windows and per-shot coverage. Ground displacement between bounces is a horizontal average, not flight or serve-impact speed. Homography of airborne pixels supplies no physical ball-speed ground truth.

Quality review: independent reviewers inspect all event/near-line test footage and a frozen sample of image labels without model predictions. Record disagreements, adjudicator, review date, excluded/unknown counts and reasons. Model proposals can help development annotation only after review; they cannot become final truth automatically. Record actual selected frame counts and independent session counts separately. Freeze tolerances and denominators before final evaluation. The [official ITF rules](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf) define dimensions, outside line boundaries, scoring and lets; the [official pose format](https://docs.ultralytics.com/datasets/pose/) defines training syntax.

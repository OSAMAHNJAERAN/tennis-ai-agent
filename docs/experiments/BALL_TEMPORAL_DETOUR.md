# Short temporal detour pilot

Protocol fixed before scoring this rule. The saved five-view WASB stream with pixel motion and guarded patch persistence still has 32 false detections on the twelve-clip development expansion. Inspection of their coordinate context shows short distant excursions as well as persistent moving distractors. This experiment addresses only excursions bracketed by a consistent observed track.

For each possible interval lasting at most 0.10 seconds, use two consecutive selected observations immediately before it and two immediately after it. All four anchors must exist. Fit a straight line in timestamp and reference image coordinates (512 x 288); every anchor must be within 4 reference pixels of that line. Each pair on the same side must move at least 2 reference pixels, excluding stationary anchor pairs. Reject an interval only when it contains at least one selected point and every selected point in it is at least 20 reference pixels from the fitted line. Missing points stay missing. Decisions use the original input stream simultaneously; rejected points cannot trigger another rejection pass. Overlapping rejected intervals form a union.

This is an offline image-space consistency heuristic, with up to 0.10 seconds plus two frame periods of future context. It does not estimate physical speed, infer ball identity, fill gaps, change coordinates, or establish events. A real hit, bounce, or camera cut may violate its assumptions; coherent false anchors may suppress a real ball. Those are measured failure risks, not reasons to relax the existing scoring convention.

Evaluate the twelve-clip expansion and then the six additional clips with these exact settings. Both are reused development evidence. Preserve every visible and absent label, report lost true detections, per-clip results and source hashes. No parameter sweep or pipeline-default change is authorized by a favorable pooled result alone.


## Completed result

| Development subset | Before TP / FP / FN / TN | After TP / FP / FN / TN | After precision | After recall | After F1 |
| --- | --- | --- | --- | --- | --- |
| Twelve expansion clips, 600 labels | 518 / 32 / 25 / 36 | 518 / 31 / 25 / 37 | 94.35% | 95.40% | 94.87% |
| Six additional clips, 300 labels | 272 / 13 / 14 / 8 | 272 / 12 / 14 / 8 | 95.77% | 95.10% | 95.44% |

The rule removes the scoreboard-dot detection on match148 frame68 and a wrong visible localization on match156 frame246. The latter becomes a visible abstention, so its false negative remains; recall does not improve. There are no lost labeled true balls. Seven of twelve and three of six clips pass both strict >95% gates. The difficult match148 still has precision79.49% and recall81.58% after this replay.

Across all8,975 saved frames,16 selected points are rejected (eight in each subset). Fourteen have no publisher ball label. All16 were rendered in four review boards, including actual past/current/future source patches. Visual inspection identifies court lettering/center marks, scoreboard service dots and player/racket details among the rejected points. Match156 frame51 initially looked ambiguous in an isolated patch; the unmarked45-56 sequence shows the ball higher in the image while the rejected position follows a racket artifact. Match15 frame410 remains uncertain near a moving racket/leg. These visual judgments are diagnostic, not independently adjudicated replacement ground truth. No unlabeled frame is silently counted as a success.

A separate remaining error, match145 frame80, has publisher Visibility=0 despite a visually apparent ball-like object at the selected point. The original CSV maps Visibility=0 to absent. The publisher label and all reported counts are unchanged. A prospective independent annotation audit should resolve such discrepancies; optimizing filters against possible omitted labels can suppress real balls.

The completed reports are `artifacts/validation/vision_upgrade/expansion12_tiled_temporal_detours_verified.json` and `additional6_tiled_temporal_detours_verified.json`. They verify parent-report hashes, reconstruct and exactly check the guarded-persistence input on all labeled frames, preserve every label, and record every rejection interval and anchor residual. Initial reports without the `_verified` suffix are preserved; final source additionally handles sub10FPS streams without exceeding the frozen duration and rejects duplicate residual labels. Outcomes on both datasets are identical.

Review artifacts are `outputs/vision_upgrade_audit/ball_temporal_detours_expansion12/` and `ball_temporal_detours_additional6/`. The renderer verifies the full source dataset manifest and file hashes before decoding. Its JSON inventories every reviewed frame. The additional contact sheet `match156_45_56_contact.jpg` uses native unmarked source crops at x1480:1820,y550:900.

Decision: retain as a bounded research replay, with no production integration. The tiny sparse-label gain and one unresolved unlabeled rejection do not establish a reliable filter or satisfy the precision goal. Persistent distractor tracks, proposal misses, ball-label quality, unseen-camera validation, court localization and physical analytics remain unresolved. Nine direct tests cover short excursions, preserved fast motion and a direction change, missing/static anchors, resolution scaling, invalid inputs, long intervals and low frame rates. These checks demonstrate software behavior, not ball identity.

Final validation:26 relevant tests passed in3.09 seconds. Both final report code/source hashes and complete16-frame review coverage were reverified. Provenance: `outputs/vision_upgrade_audit/ball_temporal_detours_provenance.json`.

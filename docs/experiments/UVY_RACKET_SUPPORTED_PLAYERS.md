# Court-independent racket-supported player selection

Protocol fixed before racket inference. Court-dependent selection emits zero tracks on all three UVY fan recordings despite useful person proposals. Test a semantic selection signal that does not require court geometry: repeated actual racket detections associated with existing person tracks.

Reuse the complete tracked640 person-proposal caches and original verified YOLO11m checkpoint. Sample source images every approximately0.20 seconds at their real frame rate. Apply the existing GlobalRacketTracking crop detector/IoU0.5 pooling/global assignment at confidence0.25 and input640 to all observed people, resetting its state between videos. Preserve raw crop proposals and assigned racket observations. Do not infer rackets from wrists or motion alone.

For each source person ID, require at least0.50 seconds worth of actual observed person boxes and racket detections on at least two distinct sampled frames. Rank eligible tracks by sum of assigned racket confidences divided by sampled person opportunities plus5. At each original frame, select at most two eligible observed boxes, breaking ties by support count, person confidence and source ID. This uses whole-video evidence offline and can select earlier observed boxes retrospectively. It never fills missing boxes or invents identities. Source IDs remain segment-local; near/far court roles and physical metrics remain unavailable.

Known risks: a source ID may switch people; repeated false rackets can validate a spectator; small or occluded rackets can leave real players unselected; crop provenance/proximity is not independently proven racket ownership. The goal is active-player discrimination, not simply returning any two people. Evaluate all2,230 frames against the unchanged publisher class1 player labels using existing TrackEval and IoU0.5 metrics, with known label omissions/misclassification retained and flagged. Compare against both the zero-output court selector and legacy player selector. No production promotion without actual results and visual review.


## Completed pilot01 and ownership diagnosis

All three sequences completed (2,230 frames, 4,203 unchanged class-1 labels). The first/middle/last review board was inspected on all three recordings. V01 selects the near player initially, misses both at the midpoint, and selects two people after the camera pans at the end. V02 emits no players and has zero sampled racket detections. V03 selects a foreground spectator throughout the review samples alongside the nearer player when available. Its spectator source ID3 receives16 racket supports: these racket boxes are above the spectator box, in the distant court area included by the enlarged crop. Crop origin is not racket ownership.

| Sequence | TP | FP | FN | IDF1 | HOTA |
|---|---:|---:|---:|---:|---:|
| V01 |492|104|953|37.53%|30.31%|
| V02 |0|0|1936|0%|0%|
| V03 |380|444|442|46.17%|43.16%|

The original pilot is rejected as a general fallback. Total precision61.41%, recall20.75% (872 TP,548 FP,3331 FN). Processing the cached people plus sampled racket inference and evaluation took137.88 seconds; this excludes original person tracking and initial model loading, so it is not an end-to-end throughput claim.

## Fixed follow-up protocol: nearest-person ownership replay

Declared before replaying or scoring the new rule. Reuse every raw crop proposal from pilot01. First run the unchanged global IoU0.5 pooling. For each retained racket, calculate the Euclidean rectangle gap in unnormalized image pixels to EVERY observed person. Restrict its crop-provenance owner set to the people achieving the minimum gap (numerical tie tolerance1e-9 pixels). If no closest person is a crop source, leave that racket unassigned. Keep the existing confidence, temporal memory, Hungarian assignment, two-support requirement and offline track ranking unchanged. No new inference, coordinates, interpolation, model or tuned distance threshold. Preserve the baseline and verify its assignments can be reproduced exactly before comparing.

This tests whether relative image proximity prevents a large foreground crop from claiming a distant racket. It is only an ownership heuristic; overlapping boxes, truncated people and actual outstretched rackets can defeat it. The same three recordings informed this diagnosis and are development data. V02 has no racket proposals and cannot be repaired by association alone. No production promotion or accuracy qualification follows from this replay.


## Completed nearest-person replay

The baseline assignments reproduce exactly on every sampled frame and every original selected frame. The candidate preserves the frame-level number of IoU0.5 player matches across all2,230 frames, not merely the aggregate total. It removes445 false selected boxes, including all443 observations of the foreground spectator in V03. V01 loses two false boxes; V02 remains empty.

| Sequence | Selector | TP | FP | FN | IDF1 | HOTA |
|---|---|---:|---:|---:|---:|---:|
|tennis_V01|Legacy|12|909|1433|1.01%|0.61%|
|tennis_V01|Court-dependent|0|0|1445|0.00%|0.00%|
|tennis_V01|Racket pilot01|492|104|953|37.53%|30.31%|
|tennis_V01|Nearest pilot02|492|102|953|37.57%|30.33%|
|tennis_V02|Legacy|204|1067|1732|10.48%|5.19%|
|tennis_V02|Court-dependent|0|0|1936|0.00%|0.00%|
|tennis_V02|Racket pilot01|0|0|1936|0.00%|0.00%|
|tennis_V02|Nearest pilot02|0|0|1936|0.00%|0.00%|
|tennis_V03|Legacy|440|63|382|57.36%|48.87%|
|tennis_V03|Court-dependent|0|0|822|0.00%|0.00%|
|tennis_V03|Racket pilot01|380|444|442|46.17%|43.16%|
|tennis_V03|Nearest pilot02|380|1|442|63.18%|53.37%|

Across all three sequences, nearest ownership yields872 TP,103 FP and3331 FN: precision89.44%, recall20.75%. The original racket selector yields61.41% precision at the same recall. Low or zero ID-switch counts do not establish identity reliability when most players are missing. The player labels retain documented omissions, loose boxes and a misclassified umpire; no labels or IoU thresholds were changed.

V01 retains source IDs2,311,324 and383; unsupported identity fragments still break coverage. V03 retains only source ID4 (381 observed boxes). Its far player remains missing. V02 has zero racket detections across162 sampled frames, so changing assignment cannot recover its players. Nearest-person gating removes a specific ownership failure but is rejected as a sufficient general fallback. Production defaults remain unchanged.

## Frozen racket controls

| Dataset | Association | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
|COCO,167 images|Original global|187|39|38|82.74%|83.11%|82.93%|
|COCO,167 images|Nearest owner|185|17|40|91.58%|82.22%|86.65%|
|RacketVision,92 frames|Original global|65|19|37|77.38%|63.73%|69.89%|
|RacketVision,92 frames|Nearest owner|65|18|37|78.31%|63.73%|70.27%|

The same ownership rule was used without retuning. Every original assignment reproduces exactly. Match source annotation hashes are verified; cached coordinates agree with original labels within1e-9 pixels (largest observed difference2.274e-13 from the old multiply/divide conversion). Original aggregate metrics reproduce exactly after JSON tuple/list normalization. The two initial partial control files are retained: one stopped at tuple/list comparison, one at strict floating-point coordinate equality; the final report is complete.

All COCO true-positive losses were reviewed: image121744 loses a background racket while retaining the active player racket; image323496 contains overlapping people and rackets and loses one genuine racket under the nearest-person restriction. These cases expose the mismatch between held-racket ownership and all-racket detection, plus ambiguity in proximity. The only changed match frame, match139/frame/000/0384.jpg, loses one false detection and remains a missed labeled racket. These are reused development sets with only racket-positive labels, not independent ownership or absence validation.

## Artifacts and validation

- Original report: `outputs/vision_upgrade_audit/uvy_racket_supported_players_pilot01/report.json`.
- Nearest report and paired counts: `outputs/vision_upgrade_audit/uvy_racket_supported_players_pilot02_nearest/report.json` and `paired_review.json`.
- Both first/middle/last nine-image boards inspected, plus both COCO loss cases in `coco_loss_review.jpg`.
- Complete racket controls: `outputs/vision_upgrade_audit/racket_nearest_ownership_controls_final.json`.
- Frozen pre-replay protocol snapshot: `uvy_racket_supported_players_pilot02_nearest/frozen_protocol.md`; its hash matches the replay report.
- Full paired video: `uvy_racket_supported_players_pilot02_nearest/ownership_review_V03.mp4`,443/443 frames decoded, source29.97002997 FPS/export29.97 FPS. This is a research comparison with actual observed boxes and explicit missing states.
-18 targeted ownership, global-racket, racket-tracking and player-selection tests passed. No claim of a newly rerun full regression suite.

The full system remains unqualified for arbitrary match videos. Next work must address tiny/distant racket detection, player identity fragmentation and camera-general court localization; relaxing ownership to restore the foreground spectator would undo a measured correction.

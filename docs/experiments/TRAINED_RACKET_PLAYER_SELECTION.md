# Trained-racket evidence for player selection

Protocol frozen before UVY inference. The original racket detector produces no observations on V02, so identity linking alone cannot select its players. The internally selected mixed-training racket pilot04 previously improved sparse broadcast racket recall63.73% to83.33%, but reduced precision77.38% to69.67% and regressed on COCO. It remains rejected as a general racket replacement. This experiment measures its downstream utility after the newer ownership and continuity constraints.

Use only the previously selected checkpoint SHA256 `c4416708def99c60dc041aaa8fa5d6bbaf3d0d39f2b016876c61e2f87812495a`, verifying the complete training report and original person-checkpoint hashes. Load using forced weights-only mode and the same explicitly known architecture/training classes used by the existing restricted evaluator. No retraining, epoch selection or confidence tuning based on these recordings.

Keep original tracked640 person caches, actual frame rate, approximately0.20-second sample indices, crop geometry, racket confidence0.25, input640, global pooling, nearest-person ownership,0.25-second memory and all existing source-ID chains unchanged. Use the already measured image-flow mapping, followed by the fixed brief-duplicate reconciliation. Change only the racket checkpoint. The trained model must never detect persons because its other classes were unlabeled during training.

Before new model inference, reproduce the complete original-checkpoint player selections from cached original racket evidence and frozen mappings. Evaluate all2230 frames of all three UVY recordings against unchanged publisher class1 labels. Save all raw racket candidates, assigned observations, final player boxes, code/input/model hashes and first/middle/last review images. Compare TP/FP/FN, IDF1 and HOTA with the original-model duplicate-handoff baseline. Examine false newly supported identities, not just total recall. These are reused development recordings with known publisher-label omissions, loose boxes and a misclassified umpire; no new ownership/absence qualification is implied.

Promotion requires a favorable measured tradeoff and wider validation. Repeated model false positives could now validate a nonplayer even after nearest-person assignment. No physical metrics, court roles, contact or bounce events are inferred by this experiment.


## Completed downstream benchmark

All2230 frames completed using the previously selected trained checkpoint; original-model baseline selections reproduced exactly before new inference. Person proposals, source identity mappings, confidence0.25, sampling, crop geometry and downstream selection rules remain fixed. No new training or epoch selection was performed.

| Recording | Original TP / FP / FN | Trained TP / FP / FN | Original IDF1 / HOTA | Trained IDF1 / HOTA |
|---|---|---|---|---|
|tennis_V01|551 /102 /894|767 /102 /678|42.14% /34.36%|38.20% /36.07%|
|tennis_V02|0 /0 /1936|56 /170 /1880|0.00% /0.00%|5.18% /7.24%|
|tennis_V03|440 /1 /382|469 /1 /353|69.68% /61.13%|72.60% /61.28%|

Aggregate original991 TP/103 FP/3212 FN becomes1292 TP/273 FP/2911 FN. Precision90.59% to82.56%; recall23.58% to30.74%. The trained model adds301 matches and170 unmatched selections under unchanged official scoring. It is not a qualified replacement.


## New identities and label uncertainty

V01 newly eligible canonical IDs216 and248 contribute126 and90 selected boxes, respectively; every one has a publisher player box at IoU>=0.50. V03 ID199 contributes29 selected boxes with matching player labels. Their first/middle/last appearances were visually inspected and follow real on-court players.

V02 ID222 contributes226 selected boxes, only56 meeting IoU0.50. Its median best publisher-player IoU is0.44210. First/middle/last reviewed examples show the selected box around the near player while publisher boxes extend beyond the visible body; this supports a box-label mismatch in those samples. It does not establish that all170 unmatched selections are correct. Every V02 racket support was reviewed: three on ID222 at zero-based724/813/819, and one on ID306 at885 (insufficient repeated support to select that ID). The small boxes lie near apparent racket/hand regions, but resolution and the absence of racket ground truth prevent confirming ownership or racket correctness. Publisher labels and all170 official unmatched counts remain unchanged. No corrected metric is reported.

All nine first/middle/last full-scene samples were inspected, followed by first/middle/last views of all four newly eligible identities and all four V02 racket supports. Every final selected box and confidence was verified against its exact original person detection. Source detections were not enlarged to match loose labels.

## Identity switches

The diagnostic `trace_player_identity_switches.py` observes the actual assignment calls inside the pinned official CLEAR implementation without modifying its files. It reproduces TP/FP/FN and IDSW for all six original/trained sequence results. Original V01 has one switch after the long missing interval, source150/canonical2 at frame441 to311 at710. The trained output has three switches for the same publisher player:2 to216 at464 after23 frames,216 to248 at617 after24 frames, and248 to311 at710 after2 frames (all zero-based).

The trained detector exposes more existing source-ID fragments; it does not link them. The increased coverage coincides with V01 IDF1 falling42.14% to38.20% despite HOTA rising34.36% to36.07%. These are unresolved identity breaks, not evidence of reliable re-identification. V02/V03 retain zero recorded ID switches, which does not validate their missing tracks.

## Verification and decision

Both paired videos completed full decode verification: V01 819/819 frames at29.97 FPS; V02 968/968 at29.904 FPS. One decoded frame in each recovered interval was visually inspected.38 targeted ownership, racket, selection, continuity and metric tests pass. This is not a newly rerun full regression suite. Checkpoint, input, frozen-protocol and code hashes were verified; loading used forced weights-only mode with explicitly known classes.

Recorded sequence processing times15.64/22.52/12.78 seconds include cached-person input, sampled racket inference and evaluation, excluding model startup and original person tracking. They must not be compared as end-to-end speedups against earlier model runs under different runtime/cache conditions.

Retain the checkpoint and combined selector as a research candidate; no production default change. The model improves player coverage but still yields low recall, a precision regression under the unchanged labels, and identity fragmentation. The second recording needs independent annotation adjudication, and longer recovery needs stronger image/appearance evidence. Universal match reliability remains NO.

Artifacts under `outputs/vision_upgrade_audit/uvy_trained_racket_players_pilot01`: `report.json`, `frozen_protocol.md`, `selection_review.jpg`, `new_identity_review.json`, `new_identity_and_support_review.jpg`, `identity_switch_trace.json`, `trained_review_V01.mp4`, `trained_review_V02.mp4`, corresponding video verification JSON, `decoded_video_review.jpg`, and `final_review.json`. Training provenance remains in `artifacts/training/vision_upgrade/racket_yolo11m_pilot04/report.json`.

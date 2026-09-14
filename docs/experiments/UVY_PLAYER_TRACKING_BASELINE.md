# UVY temporal player-tracking validation

Prepared2026-09-09 before model inference. The goal requires temporal identity and lost-track evidence beyond still-image person/pose AP. The [UVY authors' dataset](https://zenodo.org/records/21303900) provides a candidate source: three tennis sequences from user-generated videos, annotated with YOLO-World proposals and CVAT manual review/correction. The record API confirms CC BY4.0. User-generated does not itself establish amateur players, camera diversity or complete label quality.

## Acquisition and provenance

Public record21303900 lists archive `UVY.zip`,3,274,165,269 bytes, MD5 `f99594a1bd9f627ebe21219db86317a6`. A bounded64KB suffix read verifies ZIP central-directory location; the directory is3,203,273 bytes at offset3,270,961,974. Its reviewed tennis-only index is `artifacts/validation/vision_upgrade/uvy_tennis_archive_index.json`. All three tennis sequences were selected before any predictions:819,968 and443 image frames according to the publisher table.

The three archive ranges total about242MB. `scripts/data/acquire_uvy_tennis.py` extracts only expected tennis images, original sequence video, video metadata, labels and GT text. It checks exact HTTP content ranges, local header names, bounded decoded sizes and member CRC32, then records SHA-256 for every extracted file. It does not execute archive content or extract desktop configuration files. Partial archive extraction cannot verify the publisher's whole-archive MD5, and the manifest explicitly states that limitation.

The first metadata request timed out before a dataset directory existed. The next process obtained metadata but its first large archive response was truncated after about1.76MB; no sequence was extracted. Both failed attempts and the incomplete manifest are preserved. A fresh output uses1MB ranges with bounded retries to tolerate long-transfer truncation. Acquisition is not complete until its manifest says complete and all declared members have been verified. No UVY accuracy is claimed at this stage.

## Annotation audit required before scoring

Inspect publisher class names/IDs, GT coordinate format, frame-number origin, duplicate identities, positive box dimensions, image geometry, ignored/confidence/visibility semantics, source timing and label-frame alignment. Preserve all images, including frames lacking annotations; determine whether that means absence or missing supervision before choosing an evaluation denominator. Visually inspect annotated first/middle/last frames from each sequence and identity transitions. Record reviewed evidence and unresolved issues without silently correcting publisher labels from model predictions.

For a player-only tracker, explicitly annotated officials or other people may be distractors, while unannotated people make naïve all-person false-positive accounting unreliable. Freeze target and ignore handling after the schema audit and before inference. Do not mix person-box AP, role selection, persistent source IDs and near/far roles as interchangeable accuracy claims. Compare the existing detector/ByteTrack and the upgraded role-tracking path on identical frames where the input requirements are met; unsupported camera conditions must remain visible in coverage counts.

## Metric implementation and known limitation

Use the [official TrackEval implementation](https://github.com/JonathonLuiten/TrackEval), MIT source commit `12c8791b303e0a0b50f753af204249e622d0281a`, retained under `artifacts/research/TrackEval`. The adapter in `src/evaluation/tracking_metrics.py` validates every frame and identity, computes continuous-coordinate box IoUs, and passes aligned arrays to official HOTA, CLEAR and Identity classes. It loads metric modules in a private package namespace; a private NumPy compatibility proxy maps removed `np.int`/`np.float` aliases to their original built-in meanings. Upstream source and global NumPy remain unchanged, and source hashes are included in reports.

CLEAR and Identity use IoU0.5; HOTA retains its standard0.05–0.95 threshold sweep. Seven targeted tests cover perfect tracks, an explicit identity switch, false detections, invalid input and lost observations. A controlled empty-prediction interval exposed an upstream CLEAR limitation: `Frag` can be0 for an actual detected/missing/detected interval because the empty-frame branch does not reset its fragmentation state. Preserve and flag that raw upstream value; do not claim it measures all losses. A separate maximum-cardinality IoU0.5 diagnostic records misses, longest gaps and recovered gaps on explicitly annotated GT frames. Those are localization gaps, not independently labeled occlusion recovery or proof of re-identification.

No temporal accuracy outcome, model promotion or universal reliability conclusion is supported until the acquisition, annotation audit and actual inference/evaluation are complete.


## Acquired video and annotation audit

Large archive transfers repeatedly truncated or timed out, including a low-level urllib3 read timeout. Those attempts are preserved. Compact acquisition then completed: all three original videos, GT/class/timing metadata and five preselected publisher alignment images each, totaling **11,965,302 decoded file bytes** from **11,222,404 range bytes**, plus cached metadata. Manifest: `data/external/uvy_tennis_videos/manifest.json`. The extraction helper now catches the observed low-level timeout as well as requests errors.

All 2,230 frames decode at the expected geometry (640x360 for V01/V02,1280x720 for V03). The five JPEG comparisons per video have mean absolute BGR errors between1.77 and2.56 intensity levels, consistent with re-encoding differences. The original strict audit remains a failure because publisher metadata says30FPS. ffprobe confirms actual constant-rate streams at **29.97**, **29.904**, and **30000/1001 FPS**, with monotonically increasing PTS and only one-microsecond variation from printed decimal rounding. The frame-based benchmark uses those measured rates; it does not replace them with30 or claim speed accuracy. Reports: `outputs/vision_upgrade_audit/uvy_video_alignment/audit.json` and `artifacts/validation/vision_upgrade/uvy_tennis_timing_audit.json`.

The complete GT schema audit finds no invalid boxes, duplicate frame/identity rows or per-track class changes. Class1 is player,2 sports ball,3 goal (the net/umpire structure here),4 referee. Every frame has some annotation. Player boxes total **1,445 / 1,936 / 822**. V01 has no player annotation at frame605; inspection of the original decoded image shows visible players, establishing an omission. V03 contains player ID9 only at frame141; visual inspection shows its box covers the chair umpire rather than an active player. These are concrete annotation errors, not model disagreements. No publisher labels are silently rewritten, and other errors may exist. This dataset is unsuitable for unqualified production claims without further label review.

All three video titles identify fan recordings of professional Wimbledon2019 matches. They add side-on spectator cameras and foreground occlusions, **not amateur-match evidence**. The initial V01 and V02 annotations and the V01 frame605 / V03 frame141 problem frames were visually inspected; a complete manual review of every GT frame has not occurred.

The frozen diagnostic benchmark compares legacy PlayerTracker roles, upgraded CourtPlayerTracker roles with camera registration, and source IDs underlying the upgraded selection. All-person detections are recorded only as coverage, not scored as player-only predictions. Selected officials/spectators count as raw role false positives, since active-player selection is the task. Scores against the flawed original labels remain explicitly diagnostic. Full software suite: **535 passed**, one existing warning,24.95 seconds; no GPU benchmark overlapped that test run. Actual temporal inference results are pending.


## Completed temporal baseline: side-camera failure

All three original videos completed (819/968/443 frames). Legacy role-selector IDF1 is **1.01% / 10.48% / 57.36%**, HOTA **0.61% / 5.19% / 48.87%**. Its IoU0.5 counts are TP12/204/440, FP909/1067/63, FN1433/1732/382; ID switches0/1/1. These are raw scores against flawed publisher annotations, not verified true performance.

The upgraded court-dependent selector emits **zero player observations on all three videos**, so IDF1/HOTA are0 and all4,203 annotated player boxes are missed. Zero identity switches here means no tracking output, not consistency. Person detection itself returns5,789/10,087/4,736 boxes across the sequences, showing that person proposals exist; these counts alone do not locate the source of active-player coverage loss or establish active-player recall.

The court regressor places all14 initial keypoints on the spectator stands in each video. A three-image overlay was rendered and visually inspected: `outputs/vision_upgrade_audit/uvy_court_failure_review.jpg`. V02/V03 initial calibrations fail inlier support (10/14 and7/14). V01 incorrectly obtains a geometrically consistent supplied-landmark fit (11/14 inliers), and image-feature registration lasts122 frames before failing; the predicted geometry is still physically wrong. Fit residuals cannot establish that landmarks correspond to the actual court. Lowering the calibration rejection threshold would not repair this visual localization failure.

Complete metrics, per-frame role/source-ID predictions, camera evidence and code/checkpoint hashes: `outputs/vision_upgrade_audit/uvy_player_tracking_baseline/report.json`. First/middle/last output screenshots are preserved beside it. Runtime per sequence is10.12/10.99/6.12 seconds for the diagnostic player/court pass, not full ball/pose/racket pipeline timing. This is direct evidence that the current system cannot reliably analyze these side-on fan recordings. The next improvement must address court localization across camera viewpoints and player selection when geometry is unavailable; abstention alone does not satisfy the user's tracking objective.

The publisher labels are not silently repaired. Active-player omissions and the false umpire player label prevent qualification claims, but they cannot explain a complete absence of upgraded tracks. Broader independent annotation and camera-general court detection remain necessary. No ball-speed, bounce or event authority is enabled.


## Geometric-augmentation court candidate: partial coverage, incorrect geometry

A directly relevant [research court candidate](https://huggingface.co/Coddieharsh/tennis-court-keypoints) was acquired at revision `2b1dbe4f63339bfb61bd4a685e137515df147138`. Its94,584,537-byte checkpoint matches publisher SHA-256 `159a218246e01b56aa66626d63e02d9d3c61c43cc4225d9de33694c52a5d25d6`. The model card states research-use-only and describes geometric augmentation for vertically displaced predictions, with ground-level failure still a limitation. The card and provenance are retained in `artifacts/models/court/geoaug_research`; binary weights are ignored by Git. No production license or suitability is inferred.

The identical three-video benchmark completed with only the explicit court checkpoint changed. Data/alignment/timing hashes and person checkpoint are equal; all tracking, calibration and metric code hashes are equal. Person detection counts match, but exact equality of every raw person box was not verified because the earlier benchmark did not export those raw arrays. Paired artifact: `artifacts/validation/vision_upgrade/uvy_court_checkpoint_comparison.json`.

V01 registered-frame coverage rises122 to699 of819, and upgraded-role counts become298TP/0FP/1147FN, IDF1 **34.19%**, HOTA **22.40%**. Underlying detector IDs have IDF1 **13.08%** and3 identity switches, while role IDs have0 switches; these are different identity semantics. V02 and V03 remain entirely missing with invalid initial calibration (9/14 and8/14 inliers).

Visual inspection of `outputs/vision_upgrade_audit/uvy_geoaug_court_frame1.jpg` shows the new model still imposes an incorrect end-on court template onto the side-on scene. More player coverage and more internally consistent registration do not establish correct court geometry. **Reject promotion on these views.** Full candidate evidence: `outputs/vision_upgrade_audit/uvy_player_tracking_geoaug/report.json`. The original baseline and candidate source stay preserved. No production defaults change.

The next concrete gap is camera-general court orientation/localization and independent player association when geometry is wrong or absent. A fallback must demonstrate active-player discrimination from officials/spectators; merely returning any two people or lowering the calibration gate would not fulfill the goal. Independently reviewed temporal labels are also needed before declaring identity accuracy.


## Court input orientation diagnostic: rejected

The fixed geometric-augmentation checkpoint was evaluated on each first frame with all four input quarter-turns (0/90/180/270 degrees). Predictions were mapped back to original image coordinates, with inverse transforms checked against a synthetic marked pixel for every rotation. This is a diagnostic of the existing regressor, not a trained camera-general court detector. Script: `scripts/evaluate/audit_court_orientation.py`; complete report and three four-panel images: `outputs/vision_upgrade_audit/uvy_court_orientation/`.

All twelve mapped overlays were visually inspected. None correctly localizes the visible court. V01 passes the existing supplied-landmark fit at 0 and270 degrees despite incorrect locations. V02/V03 pass none. Rotating the image tends to rotate the learned end-on template into stands, spectators or umpire equipment, rather than identify the actual court lines.

The report's grayscale line-contrast diagnostic is not reliable evidence of court alignment: it can score crowd detail, and the highest contrast views (V01 90 degrees, V02 0 degrees, V03 90 degrees) are visibly wrong. No view-selection threshold or calibration gate is promoted from this test. Input rotation alone does not resolve the failure.

Next, measure active-player proposal coverage before court selection, preserving all person boxes and reporting oracle IoU matching recall only. Because publisher player annotations do not label every spectator, unmatched all-person proposals must not be called detector false positives. This separates missing proposals from failed role selection without inventing a deployable ground-truth-assisted selector.


## Person proposal coverage audit: completed

`scripts/evaluate/audit_uvy_person_coverage.py` runs all2,230 frames in each of three explicit modes with the same YOLO11m checkpoint and verified dataset. It exports every returned person box, score and source ID (frame-local row ID only for raw predictions), file hashes, predictor settings and IoU0.5 matching diagnostics. The existing tracker path resolves confidence0.1 with ByteTrack; raw640/raw1024 use confidence0.25, IoU NMS0.7 and maximum300 detections. The raw640-to-raw1024 comparison changes only input size. Tracking versus raw changes both method and input confidence, so it is not a controlled ByteTrack-only ablation. The raw predictor's stored `tracker=botsort.yaml` is an unused default: raw inference never invokes a tracker.

| Proposal path | V01 matched /1445 | V02 matched /1936 | V03 matched /822 | Total matched /4203 |
| --- | --- | --- | --- | --- |
| Existing detect_and_track640 |1329 (91.97%) |631 (32.59%) |519 (63.14%) |2479 (58.98%) |
| Raw detector640 |1376 (95.22%) |640 (33.06%) |551 (67.03%) |2567 (61.08%) |
| Raw detector1024 |1390 (96.19%) |638 (32.95%) |636 (77.37%) |2664 (63.38%) |

These are maximum-cardinality frame-local localization coverage measurements, not a usable player selector. Unmatched proposals include correctly detected spectators and officials and therefore are not counted as detector false positives. Identity consistency, re-identification and precision are not established. Full evidence and source hashes: `outputs/vision_upgrade_audit/uvy_person_{tracked640,raw640,raw1024}/`; paired report: `outputs/vision_upgrade_audit/uvy_person_coverage_comparison.json`. All reports completed and their prediction hashes were checked. Independent matching and gap accounting agree on every aggregate missed-box count.

The three elapsed sequence times sum to26.93 seconds for tracked640,25.95 seconds for raw640 and51.88 seconds for raw1024. These are diagnostic detector passes including decode and metric preparation, excluding model construction and output serialization; they are not full-pipeline throughput or controlled latency benchmarks. No other GPU inference ran concurrently. Raw1024 increases returned proposals from21,687 to34,445, largely adding candidate people that still need role discrimination. It improves V03 substantially but leaves V02 essentially unchanged and previously worsened overall COCO tennis-person AP. No production-resolution change is justified by these data alone.

Four fixed V02 frames (1,243,485,726) were rendered; frames1 and485 were visually inspected at twice source size. At frame1 the right player is detected with a tighter box than publisher GT (maximum IoU0.36), while the far-left player has no overlapping raw640 proposal. At frame485 both players have overlapping detections, but larger publisher boxes yield IoU0.41 and0.39. Thus the low V02 score combines actual missing proposals and bounding-box disagreement; it must not be paraphrased as all missed people. The original IoU0.5 result and labels remain unchanged, and no alternative threshold is selected from these observations. Images: `outputs/vision_upgrade_audit/uvy_person_coverage_tennis_V02_*.jpg`.

Court selection still discards useful proposals, particularly in V01. A geometry-independent active-player selector needs independent role supervision or validated court localization, and reliable box annotations are needed to qualify it. Resolution alone is insufficient. No physical speed or event authority is enabled.

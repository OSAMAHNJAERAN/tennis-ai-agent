# Candidate pixel-motion experiment

Protocol recorded before feature extraction and evaluation. This is reused-validation selection, not qualification.

The preceding .20 candidate replay reproduced TP=505, FP=41, FN=38 on the twelve expansion clips. The requested P/R gate failed at 92.49% / 93.00%. The failure is concentrated in saved labeled frames and is reproducible without GPU inference. Prior error crops show recurring stationary scoreboard markers. The .05 model threshold finds more correct candidates but produces 54 false detections on 57 absent frames. These existing replays and crops supply the minimized failure evidence; synthetic tests alone cannot prove that real video is fixed.

## Hypotheses

1. A stationary graphic may have stable image pixels despite jittering candidate centers. Rejecting candidates with little actual pixel change should reduce false detections and sometimes expose a lower-ranked true ball.
2. Compression, camera motion and changing text may create large pixel differences at false candidates, limiting this approach.
3. A held or slowly moving real ball may produce little difference and be rejected, reducing recall.

## Fixed measurement and comparison

Decode original checksum-verified expansion videos once and attach features to both saved .20 and .05 candidate sequences. Use grayscale images resized with area interpolation to 960x540. Compare the current image against the latest available image at least .10 seconds earlier. Within radius three pixels around each candidate, score the mean of the three largest absolute grayscale differences. Before history is available, evidence is unknown and candidates remain eligible. No labels, object masks, scoreboard coordinates or future frames enter the feature.

Apply the already selected coordinate-stationary filter (.25 s) first, then reject eligible candidates below pixel-difference thresholds 2, 4, 8 and 12 in the 0–255 grayscale range. Threshold zero reproduces the unchanged coordinate-filter baseline. Retain original heatmap-mass ordering among surviving candidates. Measure precision, recall, F1, visible abstentions, wrong locations and absent-frame errors, including per-clip results. This is a predeclared diagnostic grid; any choice still requires new independent validation.

The feature is image motion, not a physical speed or ball classifier. Camera cuts, pan, lighting changes, moving spectators and real stationary balls remain explicit risks. Do not promote a precision gain that hides a material recall loss.

## First measured results and controlled follow-up

At .20 model threshold, pixel threshold 12 gives TP=505, FP=27, FN=38: P=94.92%, R=93.00%, F1=93.95%, versus 92.49% / 93.00% / 92.75% without appearance rejection. Correct absence improves from 36/57 to 42/57. One true detection is gained in `match148_000` and one lost in `match144_000`; aggregate recall conceals that exchange. Pixel thresholds 2/4/8 give F1=93.27%/93.27%/93.53%. The lower .05 model stream remains worse: even pixel threshold 12 gives P=86.32%, R=94.11%, F1=90.04%, with 52 absent false detections. It is not selected.

The earlier six clips were rerun at the exact .20 model threshold to avoid mixing prior .25 reports. The frozen pixel threshold 12 leaves all counts unchanged: TP=276, FP=8, FN=18, TN=3; P=97.18%, R=93.88%. These clips were used in earlier selection and are a regression check, not an independent test.

Visual inspection confirms a scoreboard-marker rejection that reveals the true ball in `match148_000` frame 222. The lost true location in `match144_000` frame 459 lies near a white court line; little local appearance difference is visible. Radius three at 960x540 covers less than the benchmark's accepted localization error. A controlled follow-up enlarges only the patch radius to eight pixels, retaining threshold 12 and all other settings. This may recover a nearby moving ball but may also retain more false candidates by including unrelated motion. Record both outcomes; do not silently substitute this setting.

Decision reviews are saved under `outputs/vision_upgrade_audit/pixel_motion_decision_changes_v2/`. The first review's magnified current patches accidentally shared image memory with annotations; v2 copies patches before drawing so actual pixels remain visible. Measurement features were always extracted before annotation and are unaffected.

The radius-eight follow-up gives TP=504, FP=34, FN=39 (P=93.68%, R=92.82%, F1=93.25%) and is rejected in favor of radius three. Allowing more nearby motion does not rescue the problematic label overall and permits additional distractors to outrank true candidates.

## Frozen additional-clip check

Before acquiring further labels, freeze original WASB threshold .20, overlapping step one, coordinate-stationary .25 s, pixel radius three, lag .10 s and pixel threshold 12. Acquire publisher validation entries 18–23 at revision `85157ca21faa2abca96d837dd2b963738029bcc8`, which were not used to choose these settings. Compare the existing coordinate-filter baseline and the frozen added pixel filter on the same 300 expected sparse labels. This is an additional validation check, not the untouched publisher test split or proof of broadcast independence. Do not retune on the added set before reporting its frozen result.

The acquisition completed for `match154_000` through `match159_000`, with publisher video checksums and local label hashes verified. The six clips contain 300 explicit labels: 286 visible and 14 absent. The unchanged coordinate-filter baseline gives TP=262, FP=14, FN=24, TN=8 (P=94.93%, R=91.61%, F1=93.24%). The frozen pixel filter gives TP=262, FP=13, FN=24, TN=9 (**P=95.27%, R=91.61%, F1=93.40%**). This is one fewer absent false detection, with no change to correct visible detections. The gain is modest; it does not establish broad accuracy or satisfy the recall target. Results: `artifacts/validation/vision_upgrade/additional6_original020_pixel_motion.json`.

## Pipeline integration and limits

`configs/phase6_analytics/wasb_pixel_motion_validation.yaml` enables the optional feature after coordinate-stationary filtering. An extra bounded video pass computes appearance evidence, and `pixel_motion_audit.json` records scores, unavailable history and decisions. No interpolation is introduced. Existing baseline and product default configurations remain available/unchanged. Six synthetic feature tests and twelve pipeline smoke cases passed; the full suite then passed **463 tests**, one existing dependency warning, in 11.95 seconds. A subsequent nested-provenance change passed its three targeted summary tests.

The complete candidate run on `match148_000` decodes all 600 frames at 60 FPS. Actual exported predictions give **TP=25, FP=5, FN=13, TN=9; P=83.33%, R=65.79%, F1=73.53%**, matching the replay counts and improving on the old clip F1 of 65.75%. Run: `outputs/vision_upgrade_audit/wasb_pixel_motion_match148/`; score report: `artifacts/validation/vision_upgrade/phase6_pixel_motion_match148_ball.json`. Recorded execution time is 86.76 seconds including setup, sampled process RSS 1,875,304,448 bytes. Brief CPU provenance checks overlapped the run, so this is an execution/resource check, not a controlled filter-overhead benchmark.

The 24-clip diagnostic combines 1,200 labels (1,123 visible, 77 absent): TP=1043, FP=48, FN=80, TN=54; P=95.60%, R=92.88%, F1=94.22%. Only nine of 24 clips individually exceed both 95% targets. It must not be presented as independent qualification. All detector/filter parameters match, but feature source hashes differ because constructor input validation and the pipeline filtering wrapper were added between feature extractions; `pixel_motion_24clip_summary.json` explicitly flags the implementation difference. Its recursive provenance reader verifies each underlying report hash and retains the feature configuration/code identity. Existing immutable feature reports were not rewritten to hide that distinction.

The fresh `outputs/vision_upgrade_audit/broadcast_review_pixel_motion_match148/tracking_review.mp4` decodes to 600 frames at 60 FPS. Frame 87 was visually checked for pose, racket, ball and court overlays. A direct comparison of all 50 labeled-frame coordinates found zero prediction mismatches between replay and complete pipeline at 0.001 source-pixel tolerance, including missing-state agreement. The review continues to label physical ball speed and event authority as unverified.

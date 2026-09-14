# Image-flow fragment linking

Protocol fixed before dataset inference. Static boundary overlap and bidirectional box-velocity extrapolation did not recover the near-player fragment in V03. This experiment replaces box-velocity estimates with actual sparse image correspondences across each short source-track boundary.

Keep the maximum0.20-second strictly positive gap, non-overlapping source lifetimes, mutual unique maximum-score edges, numerical tie tolerance1e-9, and the unchanged offline racket-supported selection. No source labels participate in edge construction.

For each candidate boundary, detect at most80 Shi-Tomasi corners in the central90% of the source person box (quality0.01, minimum spacing2 native pixels, block3). Track them directly between the actual endpoint grayscale images using pyramidal Lucas-Kanade (15-pixel window,3 pyramid levels,30 iterations,epsilon0.01), then track backward to the source. Require at least4 observations with both status flags, finite coordinates, forward/backward error at most1.5 native pixels and LK photometric error at most20 in each direction. Estimate median translation, retain correspondences within2 pixels of that median, and require at least4 and at least50% of initially detected corners. Inlier source points must span at least25% of both original person-box width and height. Estimate median translation again from inliers.

Independently repeat from the successor image/box backward. Translate each boundary box using its measured displacement, without changing box size. Both translated-to-observed IoUs must be at least0.50; rank by their minimum. Any failed direction leaves the edge ineligible. Record all candidate-edge diagnostics, including failed feature/consistency gates. Image-derived translations are used only for association and never emitted as detections, physical speed or contact events.

Evaluate every UVY source pair satisfying the temporal interval and score all2230 original frames against unchanged labels. Compare to the verified nearest-racket baseline. Review every accepted link and the previously rejected source4-to230 edge. These are reused development recordings; clothing/background features, occlusion, nearby people, pose changes and scale/camera changes can still invalidate apparent flow. No threshold tuning after this run and no production promotion without broader evidence.


## Completed result

All2230 frames across the three UVY recordings completed. Image flow considers60 temporally eligible source pairs (30/13/17 by recording) and accepts four links. Every original nearest-racket selection reproduces exactly before linking. No labels, detected person boxes, confidences or frame times changed.

| Recording | Before TP / FP / FN | After TP / FP / FN | Before IDF1 / HOTA | After IDF1 / HOTA |
|---|---|---|---|---|
|V01|492 /102 /953|492 /102 /953|37.57% /30.33%|37.57% /30.33%|
|V02|0 /0 /1936|0 /0 /1936|0% /0%|0% /0%|
|V03|380 /1 /442|440 /1 /382|63.18% /53.37%|69.68% /61.13%|

Aggregate counts improve from872 TP/103 FP/3331 FN to932 TP/103 FP/3271 FN. Precision89.44% to90.05%; recall20.75% to22.17%. The gain is confined to one recording and is not broad qualification. V03 has zero recorded ID switches in both runs, while V01 retains one; sparse output and known label defects prevent interpreting this alone as high identity consistency.

The target V03 source4-to230 link spans zero-based frames380 to383. Forward image motion is[9.4503,0.4613] pixels with26 inliers from27 initial corners; backward motion is[-9.3040,-0.4507] with20/20. The features span approximately79%/83% of the original box width/height forward and72%/83% backward. Translated-box IoUs0.77414/0.76281 pass the unchanged0.50 gate. This recovers60 original source230 observations. The prior static and local box-velocity models did not explain the displacement sufficiently.

Every accepted boundary was visually inspected, including its supporting features: V01 source193-to200 appears to follow the same distant person; V03 source199-to208 the same distant white-clad person; source4-to230 the near player; source265-to286 the same foreground spectator. Only the near-player chain creates additional selected observations. These visual assessments are not independent identity annotations, and some corners can lie on clothing edges or adjacent background.

A complete observation-integrity check verifies every selected output box/confidence against its actual source detection on the same frame, finds60 additions and zero removals, and confirms V03 frames381 and382 remain empty. No predicted gap boxes are emitted. The full paired V03 review video decodes443/443 frames at source29.97002997 FPS/export29.97 FPS. An exported frame after the link was visually inspected and shows the recovered near player. No court roles or physical speed are inferred from this image-space linkage.

## Verification and decision

36 targeted tests pass, covering actual image translations, absent/inconsistent flow, incorrect successors, short/long gap behavior, no missing-box synthesis, static/motion linking, racket support, nearest ownership and the TrackEval adapter. This is not a newly rerun full regression suite. All report code hashes and the frozen pre-run protocol were verified after evaluation.

The image-flow rule remains a research candidate. V02 remains empty, V01 still misses many real players, source IDs that overlap or recur cannot be joined by this algorithm, and long gaps/occlusions require stronger identity evidence. Production defaults remain unchanged. Further work must improve distant-player/racket evidence and validate recovery beyond this reused development example.

Artifacts under `outputs/vision_upgrade_audit/uvy_flow_fragment_linking_pilot01`: `report.json`, `frozen_protocol.md`, `boundary_review.jpg`, `boundary_review.json`, `observation_integrity.json`, `flow_review_V03.mp4`, `flow_review_V03.json`, `flow_review_frame421.jpg` and `final_review.json`. The report preserves diagnostics for every eligible temporal pair, including rejected feature/consistency/overlap tests.

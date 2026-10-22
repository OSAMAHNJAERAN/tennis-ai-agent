# Original-image person evidence and bounded motion selector

The fresh diagnostic analysis `match_1bb12f2aa52a438ebce8a9895fa2bd4f` preserves every returned valid person proposal in `image_people.json` even when court calibration fails. The old completed public analysis and catalog entry remain unchanged. Detector source IDs are nullable, and image evidence does not gain named identities, court roles, metric positions or event authority merely by surviving export.

The previous [UVY review](UVY_PLAYER_TRACKING_BASELINE.md) showed a complete loss of useful person proposals after court-dependent role selection. The [paired racket selector](TRACKER_RACKET_SELECTION.md) also selected a seated spectator through false racket supports. Returning any two people, selecting the largest boxes, lowering court rejection or treating a tracker ID as verified active-player identity would preserve those failures.

## Labels before predictions

Eight exact, sequentially decoded original frames from the existing 600-frame Dimitrov–Tiafoe excerpt were inspected without prediction overlays before opening the new raw sidecar. Native video tools established the continuous rally and the near participant's temporary departure from the camera image. Exact decoded frames, image hashes and extraction provenance are saved in `artifacts/validation/product_execution/active_person_original_review/manifest.json`; labels are in the adjacent `labels.json`.

The frozen subset has 14 visible active-body boxes and 32 explicit crew/umpire negatives. At frame 270 the near participant is largely outside the right image boundary, so its partial envelope is an ignored localization case. At frame 300 it is outside the image, so promoting another visible person to keep two players would be wrong. The labels use clip-local appearance labels A and B without verified names or biometrics. Body envelopes are approximate visual estimates, excluding racket and shadow, and have only one agent reviewer. They are development evidence, not a qualified accuracy dataset or independent human ground truth. Spectators are not exhaustively labeled; unmatched all-person proposals must not be counted as detector false positives.

## Proposal diagnosis

All 14 visible active-body labels have an actual person proposal at IoU ≥ 0.5 on this small subset. The sampled frames return only two or three high-confidence proposals, including a persistent back-wall crew member. Eight of the 32 explicitly labeled negatives have an overlapping proposal. The detector therefore provides the needed visible players on these reviewed frames; the old zero active-player output comes from the geometry-dependent selection path. This does not establish complete detector recall across all 600 frames or other videos.

The near participant has source ID 1 before leaving the image and ID 8 after returning. The diagnostic does not merge them or claim successful automatic re-identification. Proposal protocol, script hash, raw prediction hash and per-label matches are retained in `proposal_protocol.json` and `proposal_coverage.json`.

## Developer-only intervention

`src/tracking/motion_supported_people.py` uses observed image travel after independent image-feature camera compensation. It has no court calibration input, source-ID whitelist, top-two ranking, identity merging, box interpolation or gap filling. It requires at least 0.5 seconds of actual observations, four supported camera samples spanning 0.5 seconds, and image travel of 0.6 median body heights. It emits all eligible observed moving fragments when at most two coexist and abstains when more than two coexist or duplicate IDs make a frame ambiguous. Unknown IDs and unsupported camera evidence abstain.

The bounded audit registers 51 sparse original frames to the first image using distributed ORB correspondences and a robust similarity transform, masking detected body regions. This compensates camera pan without using the rejected court regressor. Residuals and fit acceptance are internal consistency evidence, not independent camera accuracy. Sparse registration supports offline fragment selection only; it does not supply per-frame speed, physical position or geometry.

All 51 sampled registrations pass the frozen internal criteria. Source fragments 1, 2 and 8 show 3.195, 2.376 and 3.176 body heights of compensated travel. The back-wall crew fragment 3 shows 0.064 and is rejected. Two very short fragments have no camera support and abstain. At the eight frozen-label frames, the motion candidate produces 14 TP / 0 FP / 0 FN at IoU 0.5. Frame 300 emits only the far participant's actual source observation. All 1,138 selected boxes, confidence values and IDs across the video exactly match the raw source sidecar; no role or known participant status is added.

Final reproducible revision: `artifacts/validation/product_execution/active_person_original_review/motion02/`. It contains protocol, source snapshots, full camera audit, candidate observations, labeled-subset matches and exact-source integrity checks. The earlier first diagnostic remains preserved. No production default or API selection changes were made.

Motion does not prove active participation: moving officials, ball crew or spectators can satisfy it; static players can be missed; source IDs can switch; nonuniform camera motion or background mismatch can corrupt compensation. More than two plausible moving people must remain an abstention case. The next promotion gate requires independently reviewed moving-distractor negatives, inactive/stationary-player positives, camera cuts and occlusions across separate sessions, with identity semantics frozen before evaluation. This one already reviewed professional rally is insufficient for promotion.

## Software checks

The raw-person addition passes 23 detector/result tests and 40 API/match/raw-person compatibility tests. The frontend passes 13 targeted repository/player tests, lint, typecheck and production build. Five pure motion-selector tests cover stationary crew under camera drift, missing observations, three-person ambiguity, unavailable cameras, unmerged identities and malformed/duplicate observations. These are software checks, not scientific qualification.

Future annotation renders use plain consumer copy and omit the mini-court on invalid calibration frames. Original video geometry and completed historical outputs are preserved. The owned pipeline file uses LF line endings and its reported trailing-whitespace blank line was removed with an identical Python syntax tree.

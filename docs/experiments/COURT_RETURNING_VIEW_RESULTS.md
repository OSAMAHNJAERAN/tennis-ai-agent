# Original-view court registration recovery — 2026-09-14

An opt-in wrapper can now resume the original court registration after an incompatible view ends. The existing tracker permanently latched its first failure. Recovery always starts from the original anchor image and calibration, uses the existing image-correspondence gates, and requires consecutive successful registrations before publishing coordinates. Failed and pending frames remain missing.

## Controlled before/after evidence

The frozen protocol is `COURT_RETURNING_VIEW_PROTOCOL.md`. Inputs reuse development clips match148 and match143; frame maps, source hashes, per-frame audits and code hashes are saved in `outputs/vision_upgrade_audit/returning_view_registration01/report.json`.

| Controlled case | Frames | Existing valid | Recovery valid | First recovery frame |
| --- | ---: | ---: | ---: | ---: |
| Healthy original view | 120 | 120 | 120 | — |
| Different court, then original returns | 150 | 60 | 114 | 96 |
| Blank interval, then original returns | 150 | 60 | 114 | 96 |
| Permanent different court | 120 | 60 | 60 | — |
| Original returns for only two frames | 122 | 60 | 60 | — |

The original view returns at frame 90. Both sustained returns publish at frame 96, a 0.10-second delay on this controlled 60 FPS timeline. Six consecutive confirmations are required; the retry cadence schedules the first successful probe at frame 91. No unrelated or blank frame is accepted. Healthy matrices remain exactly equal to the existing path. Maximum returned-versus-uninterrupted landmark ground-coordinate disagreement is 0.0106093 m; this is numerical agreement with the existing calibration, **not physical accuracy**.

Independent saved-artifact verification checks source/code/protocol/image hashes, recomputes valid counts and confirmation streaks, verifies missing transforms, and recomputes coordinate agreement with homogeneous matrix arithmetic. The 150-frame comparison video fully decodes at 1280×400 and 60 FPS. Screenshots 60, 95 and 119 were visually inspected: different court withheld, original return awaiting confirmation, and restored landmarks aligned visually with the original court lines. See `completion_review.json` beside the report.

## Implementation and use

`src/court/returning_view_registration.py` wraps the unchanged `CourtCameraRegistration`. The Phase 6 pipeline accepts `court_detection.camera_registration.returning_view.enabled: true`, with `retry_seconds: 0.5` and `confirmation_seconds: 0.1`. Frame counts derive from source FPS, with at least three confirmation frames. Recovery adds frame-aligned status, confirmation count, segment ID and original-anchor-only scope to the audit. It requires camera registration enabled and retains the existing prohibition on authoritative events with dynamic geometry.

The opt-in recipe is `configs/phase6_analytics/wasb_returning_view_validation.yaml`. Existing recipes retain their previous behavior. No pretrained model, court-fit threshold or ball-filter setting changes. Pipeline edits intentionally change its source hash; historical run hashes remain preserved and describe their historical code, not this revision.

Thirty-five focused tests pass in 15.35 seconds using the project Python runtime: synthetic known translations, healthy equivalence, unrelated views, interrupted confirmations, bounded retries, invalid anchors/timing, and actual orchestration/export at 320×240 and 1920×1080. Synthetic pipeline tests verify withheld detections during the gap and restoration only after confirmation; they do not establish neural model accuracy.

## Real-model integration

`scripts/evaluate/run_returning_view_pipeline_check.py` re-encodes the frozen 150-frame different-court splice at 1920×1080 and 60 FPS and runs the opt-in recipe with the existing court, player, ball, pose and racket models on the available CUDA runtime. Source map, input, config, model, full Python source and output hashes are preserved in `outputs/vision_upgrade_audit/returning_view_pipeline01/integration_review.json`.

The check completes in 48.8064 seconds including model setup (47.28 seconds reported inside the pipeline). This is about 3.07 processed frames/second including setup on this short run; it does not establish a speed improvement or real-time operation. Court validity is 114/150 with restoration at frame 96, reproducing the controlled wrapper result despite video re-encoding and real foreground detections. Every invalid frame has missing selected-player detections, ground positions and speeds. The first restored frame has no velocity or acceleration bridging the gap. Physical ball speed and authoritative events remain withheld.

All 150 annotated output frames fully decode at their native dimensions/FPS. Pipeline screenshots 75 and 119 were visually inspected: the alternate court shows the registration-loss notice and empty player mini-court; the returned court restores player boxes, pose skeletons, landmarks and player mini-court positions. Ball image tracking remains visible independently of court registration. These checks verify orchestration and presentation, not ball/pose/identity accuracy.

## Limits

This is development evidence from controlled splices, not natural-cut or independent-camera qualification. Saved benchmark masks contain only the two selected players; other people are not masked. Anchor landmarks for the controlled comparison derive from the saved initial homography. The unrelated source is retimed onto the synthetic 60 FPS grid. Re-encoding in the separate real-model integration check can change feature support.

The wrapper cannot repair an incorrect initial court, register a new court, prove player identity across camera cuts, or validate physical ball speed. Ball temporal inference and filtering are not reset by this wrapper. Court/player generalization, ball proposal misses and physical-event validation still prevent production readiness. **Readiness: NO.**

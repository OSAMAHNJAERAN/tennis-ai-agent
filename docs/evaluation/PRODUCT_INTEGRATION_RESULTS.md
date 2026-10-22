# Upload-to-dashboard integration verification

19 September 2026. Branch: `codex/vision-system-upgrade`; starting commit: `f854804`.

## Outcome and scope

The local upload-to-analysis product workflow is implemented and verified with the existing Phase 6 pipeline. A user can upload a tennis video, follow persistent processing status, reopen the completed analysis, play source and annotated video, inspect player and ball tracks, court mapping, player statistics, speed charts and heatmaps, and download the original, annotated video and report.

This completes the accepted production-integration scope. It does **not** qualify the broader professional-accuracy objective: the configured inference recipe remains experimental, and ball speed, bounce/shot placement, rally statistics and scoring remain unavailable when the pipeline withholds authority. No model was replaced or trained. Long-match stress testing and public multi-user deployment were not performed.

## Requirements and evidence

| Requirement | Implementation and verification |
| --- | --- |
| Current-system audit | `docs/plans/UPLOAD_INTEGRATION.md` records the frontend, API, inference, data-contract and storage audit and implementation sequence. |
| Video upload | Drag/drop and file selection; byte progress and cancellation; MP4/MOV/AVI acceptance; extension, signature and bounded-size validation; UUID-isolated storage. API/UI tests cover valid formats, invalid files, path isolation and streamed limits. |
| Existing inference | Separate worker invokes `Phase6Pipeline(config_path).run(video, output_dir)` using the existing WASB/pose/racket configuration. The real uploaded MP4 completed inference twice under separate IDs. |
| Job management | SQLite persists UPLOADED, PROCESSING, COMPLETED and FAILED, stages, timestamps and sanitized errors. OS file locking serializes workers; recovery and queue transitions are tested. Completed results persisted through actual API restarts. |
| Dynamic dashboard | `/analysis/{id}` loads the requested result; uploaded sessions do not inherit fictional showcase statistics. Source/annotated playback, frame stepping, 2D/3D court views, player statistics, charts and heatmap were checked in the browser. |
| Data reliability | Required metadata/trajectory validation; frame alignment checks; nullable positions and camera gaps; confidence retention; optional-section isolation. Missing/unqualified analytics display unavailable states. |
| Failure handling | Real corrupted-container job failed clearly without publishing media/results. Tests cover missing configuration/weights, general inference failure, memory failures and malformed optional results. |
| Regression protection | Final Python suite: **825 passed**, one existing Starlette/httpx deprecation warning. `pnpm check`: lint, TypeScript, **71 tests in 15 files**, and production build passed. `git diff --check` passed with line-ending notices only. |

## Real-video evidence

Browser-uploaded analysis: `match_dc93224eebea4312bc6effa23badf26c`.

- Input: `data/sample_videos/input_video.mp4`, 12,939,389 bytes.
- Original SHA-256: `5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d`.
- Timeline: **214 frames**, **30 FPS**, **1920 × 1080**, **7.133333 seconds**. Both browser videos fully decoded and matched the analysis frame count.
- Uploaded at `10:36:10.699060 UTC`, started at `10:36:11.282639 UTC`, completed at `10:37:49.967376 UTC`.
- Reported pipeline throughput: 2.291489 FPS. This is one short run with other checks running, not a controlled performance benchmark.
- Both browser players loaded without media errors; source stepping advanced by 1/30 second, playback advanced, and annotated playback was available.
- HTTP range requests returned 206 for both media files; original-download hash matched input; downloaded report matched the result endpoint.
- Both player position arrays contained 214 mapped frames. Ball output contained 210 DETECTED and 4 MISSING frames; those are availability counts, **not accuracy scores**.

| Exported player estimate | Player 1 | Player 2 |
| --- | ---: | ---: |
| Distance | 16.905845 m | 19.259743 m |
| Average speed | 8.571978 km/h | 9.765504 km/h |
| Maximum speed | 28.728218 km/h | 30.000295 km/h |

Player IDs represent tracked court roles, not recognized identities. Motion estimates are not independently labeled ground truth. The 0.918642-pixel court reprojection residual measures fit consistency, not independent calibration accuracy.

Machine-readable evidence is in `artifacts/validation/product_integration/browser_upload.json`. The final adapter was also checked against these saved inference outputs without rerunning models; only summary publication metadata changed (`final_adapter_check.json`).

After restarting the final production server, the persisted analysis, all 214 frames, report/original downloads and both ranged videos passed verification again (`server_restart_check.json`). The rebuilt analytics page displayed the same real player statistics in the browser.

Corrupted-input analysis: `match_98bbadb29082430a849171b99f62a986`. A 44-byte file with a plausible container header reached worker validation and failed with a clear decode/encode message. Its result endpoint returned 409, media returned 404, and the website remained available. Evidence: `corrupt_video_check.json`.

## Reproduction

Follow `docs/PRODUCT_WORKFLOW.md` to build and start the local product. On the verified host, use the installed Python 3.13 ML environment rather than the MSYS `python` shim.

```powershell
# From the repository root, using the existing ML Python:
python -m pytest tests -q -p no:cacheprovider
python scripts/run/verify_product.py --analysis-id match_dc93224eebea4312bc6effa23badf26c --input data/sample_videos/input_video.mp4 --report artifacts/validation/product_integration/recheck.json
# To submit a NEW real inference job, omit --analysis-id.

# From frontend/:
pnpm check
```

MOV and AVI container acceptance is tested at the API boundary. The recorded real-model end-to-end sample is MP4; it does not establish every codec/container combination, variable-frame-rate camera or two-hour match as tested.

## Changes and remaining boundaries

The integration adds API/job/worker/result modules under `src/api/`, startup and verification scripts under `scripts/run/`, upload and live analytics UI, dynamic repository loading and focused tests. Existing pipeline behavior is preserved; its court export additionally includes existing keypoints. A checkpoint-export script's import-time environment mutation was moved into its CLI entry point to avoid contaminating unrelated model tests; its existing safety guard remains intact.

The product remains local and single-host with one inference worker. Storage is retained explicitly; there is no automatic retention quota. Authentication, remote deployment, full-match capacity qualification and new model-accuracy claims require separate work. The existing configurable AI coach API is retained; no paid-provider/live-coaching claim is made by these integration checks.

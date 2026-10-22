# Upload-to-analysis integration

19 September 2026. Continue branch `codex/vision-system-upgrade`, baseline `f854804`.

## Current architecture and gaps

- React 19/Vite uses `AnalysisContext` and a Zod-validated static repository. The existing telemetry workspace already renders synchronized source video, player/ball paths, courts, charts, heatmaps, events and coaching. Preserve it and the historical demo runs.
- FastAPI currently exposes only the coach adapter. No upload endpoint, job queue, analysis history API or media-serving API exists.
- `Phase6Pipeline(config_path).run(input_video_path, output_dir)` is the existing multipass inference entry point. Its bounded decoder, detectors and models must remain intact. Use the existing `wasb_tiled_residual_pose_small_validation.yaml` recipe; its experimental qualification and withheld event/ball-speed authority remain explicit.
- Phase 6 exports nullable player metrics, `player_motion.json`, camera registration and calibration v2. These differ from the snapshot contract. The adapter must preserve missing positions and derive player paths from the exported motion samples, without running another model or interpolating missing evidence.
- Outputs live in `outputs/`; research artifacts and demo media already exist. New uploads need isolated UUID directories and persistent metadata. Never operate on existing research output directories.

## Implementation sequence

1. Add SQLite-backed job records and bounded streaming uploads under `outputs/product/uploads/<id>`. Queue a separate, exclusive worker process, keeping HTTP responsive and serializing GPU work. Retain timestamps, stage, sanitized errors and local logs. Recover interrupted jobs and queued work on service restart.
2. Validate/decode and normalize uploads to browser-playable constant-rate H.264, invoke the existing pipeline, transcode annotated output for browser playback, validate exports and publish a normalized result. Model paths are checked before loading; missing assets never trigger a replacement/download.
3. Serve job history, status, validated results, original/annotated playback and downloads through explicit ID-scoped routes. Keep the coach endpoints. Reuse the same-origin Vite proxy.
4. Add drag/drop and file selection, actual upload progress, polling with recovery, persisted job history and `/analysis/:id`. Load dynamic results into the existing dashboard; preserve nullable scientific evidence and section-level unavailable states.
5. Test upload rejection, job transitions/recovery, inference failure, result validation and media ranges. Run existing checks and a real tennis video through HTTP upload, actual models, completed artifacts and browser dashboard.

## Verification and scope

Model accuracy is unchanged by integration. Missing/withheld speed, bounce, shot and scoring measurements must remain unavailable. This is a single-host product with one GPU worker; public multi-tenant hosting requires a separate authentication/deployment scope.

Implementation sources: [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/), [heavy background work](https://fastapi.tiangolo.com/tutorial/background-tasks/), [Starlette ranged file responses](https://www.starlette.io/responses/). Streaming the request body permits enforcing a byte limit before accepting the whole upload.

Initial shell checks: the default Python resolves to MSYS rather than the documented ML environment; the global npm shim is broken. Use explicit verified runtime paths when testing, without changing machine configuration.

# Connected authenticated correction lifecycle: next implementation handoff

Read-only source mapping on 2026-10-18. This is an internal plan for the next active goal turn, not implementation or completion evidence. Only this artifact was written. Current preparation code is being tested by the parent; it must not be treated as frozen or qualified merely because it appears here. Source pins below identify the inspected files.

## What exists and what is still disconnected

The private chain is concrete: `rebuild_corrected_rule_timeline` → `relink_corrected_shots` → `rebuild_corrected_forecasts` → `write_corrected_exports` → `render_corrected_video` → `prepare_corrected_result`. It rebuilds semantics, current line calls, shots, point/serve/rally counts, owned causal history, source-based videos and actual JSON/HTML/CSV. Private helpers retain `published=False` and do not authenticate their reviewer strings. They are not yet a website correction feature.

There is no uploaded-analysis correction route, durable correction command/queue table, correction worker branch, owner review editor, correction progress/retry UI, authenticated complete-correction publisher, uploaded native-forecast product response or frontend forecast schema. Existing `ReviewWorkflow.tsx` and `PointTimelineEvidence.tsx` display evidence only. Manual score correction exists for manual matches and explicitly rejects uploads.

## Reuse the current authentication and ownership boundary

- `src/api/auth.py`: `AuthStore.install` adds `AccessMiddleware`, opaque cookie sessions and `/api/auth/*`; `current_owner(request)` reads the middleware-derived owner. The middleware already rejects unauthenticated private mutations, cross-origin changes, inappropriate remote/local mode and applies private caching. Reuse it; no new auth flow is required.
- `src/api/app.py:create_app`: `job_or_404(analysis_id,owner_id)` checks the analysis owner. Correction mutations must use this **owner-only** path, never `public=True` or curated-demo permission. Public curation allows specific GET/HEAD results, not modifications.
- `src/api/product.py:ProductStore.command`: existing `BEGIN IMMEDIATE`, command-id/payload idempotency and expected revision checks are useful patterns, but the route `/api/matches/{id}/score-commands` is manual-only and replays an unrelated manual point ledger. Do not send uploaded corrections there.
- Derive reviewer identity and authorization receipt from the authenticated request/session and analysis owner, not a caller-supplied `reviewer_id`. Keep reason/evidence reference as recorded review inputs. Preserve separate product metadata revision, analysis revision and native scoring revision0; they are different domains.

## Minimal connected store/API additions

Add a focused correction store/router module, mounted in `create_app` with the current `JobStore`, settings and `ensure_worker`. Suggested owner-only endpoints are POST `/api/analyses/{id}/corrections`, GET its correction status/list, GET one bounded review-source page, and POST cancellation. These are proposed routes, not present APIs.

Use the existing SQLite database/connection context. Add a separate durable correction table keyed by correction ID, analysis ID, owner and idempotent command ID; record canonical command/hash, expected base generation/revision/replay digest, source digest, stage/status, attempt/heartbeat/cancel flag, timestamps, private-stage seal if retained, committed generation and failure. A same-ID/same-command retry returns its durable state; a changed command under the same ID is409. Only the same completed owner analysis and a current expected base can admit a new request. Bound raw body size before parsing and use the native request's1MiB/128-amendment limits and strict primitive checks.

Keep the analysis job itself `COMPLETED` while a correction is queued or processing, so its old committed replay/report remains available. Existing `analyses.status` only supports UPLOADED/PROCESSING/COMPLETED/FAILED; reusing it would make `saved_document` reject old result reads until completion and could mark a valid old analysis failed when a correction fails. Correction status/progress/cancellation belongs in its separate table.

Migration must coordinate both current version readers: `JobStore.__init__` currently writes `PRAGMA user_version=3`, and `product.migrate_product` rejects versions greater than3. Do not bump one side alone or let initialization silently downgrade a new migration. Keep new schema changes SQLite-specific; no external database is required.

The review-source endpoint should derive editable representations from hash-bound native replay inputs, returning bounded pages and expected fingerprints, not all match arrays. The server should reconstruct complete replacement records while preserving immutable source measurements/context. Current `_replacement` only allows semantic/current reviewed-binding changes and a full reviewed initial state; it forbids moving events or changing timestamps, pixels, confidence, trajectory, evidence, source clock/segment or changing a physical record into a reviewed signal. Coordinate/timing/line-call overrides or insertion/deletion require a separately designed evidence contract. The UI must not imply those edits already work or ask users to edit raw codec JSON/provenance fields.

## Worker and resource seams

`src/api/worker.py:main` owns the OS-backed `worker.lock`, invokes `recover_interrupted` and repeatedly claims one analysis job. Extend this locked worker to claim correction jobs fairly in a separate branch; do not run the private native pipeline in the HTTP process. `JobStore.claim` already uses `BEGIN IMMEDIATE`; its analysis stage/finish functions cannot represent correction progress without corrupting old job state.

The correction worker resolves the sealed expected base and builds the complete private chain in unique owned scratch. It must recheck ownership, cancellation, command/base/source/model identity before expensive work and before commit. Existing recovery only supports completed original inference checkpoints; `.pipeline_checkpoint.json` is not a corrected semantic/prepared checkpoint. On process loss, replay a still-current command from its sealed base or validate a complete prepared descriptor under a defined new protocol. A stale base becomes a correction conflict; do not silently apply it to whichever generation is now current.

When a reachable configured shot branch requires crop pose features, the worker must bind the callback to the already selected YOLO26s Pose role/checkpoint and actual source/window settings. The private trusted callback does not establish that identity itself. No fourth model, silent CPU fallback, lowered memory floor or skipped configured classifier branch is authorized by this integration. Empty handedness/orientation maps legitimately take the native unavailable branch without loading crop inference.

## Critical two-phase publication boundary

**Do not call `publish_result(prepared.storage,...)` inside the current nested private contexts.** Publication closes that ResultStorage and advances the current pointer. Preparation exit checks its open storage/root/report equality; rule/shot/forecast contexts also recheck the unchanged base on exit. A successful publication would then make those guards fail.

The clean next seam is:

1. Complete preparation and all native contexts while the base remains current. Retain only a trusted producer-issued descriptor containing private stage path, exact preparation-manifest digest, command/auth binding, base/source/new revision and inventory. No lazy `prepared.result` or old owner escapes.
2. Outside those native contexts, enter a new publication-reader context. Verify the descriptor type/seal/inventory/byte hashes and authenticated queued command; reopen `report.json` through a **new** writable `ResultStorage` and strictly validate its required metadata and preserved native evidence. Never use the closed prepared storage or accept an arbitrary caller-resealed directory as producer authority.
3. Call `publish_result(new_storage,result,analysis_directory,generation_files=exact_complete_inventory,analysis_revision=new_revision,expected_generation=base,before_commit=final_check)`. The existing `publish_analysis_generation` checks expected base on entry and again under `publication.lock`, independently copies files, seals DB/result and atomically replaces `current.store`. The callback runs after copies/seals and immediately before commit. Do not mutate result arrays in that callback: the publisher has already closed its DB.
4. Record the committed generation in the correction job. If the process dies after pointer replacement but before this acknowledgement, inspect the authoritative generation's correction ID/request/command binding and reconcile it as committed, rather than rerunning blindly or marking the completed analysis failed. Current publisher code deliberately preserves a generation whose pointer commit succeeded despite lost acknowledgement.

There is another explicit contract decision: current prepared `analysisCorrection.authorizationStatus` is `PRIVATE_LOCAL_REVIEW_NOT_AUTHENTICATED`, its status is PRIVATE_PENDING_PUBLICATION and `published=False`. `analysis_correction_report_chunks` currently **requires those private flags**. An authenticated publication stage must not silently label these bytes authenticated or delete the warning. Introduce a clearly defined authenticated-review receipt before report sealing, or retain the immutable private review report as private-stage evidence and produce separate authenticated delivery metadata/reports. Authentication proves who authorized the correction, not independent tennis/CV/prediction accuracy. Keep `nativeForecast.status=UNQUALIFIED`, calibration pending and independent accuracy false in either case.

## Persistence, dynamic downloads and native forecast integration

`ResultStorage` and the generation publisher preserve arbitrary extra result keys, so validated `nativeForecast` and `analysisCorrection` survive DB/result serialization. `ResultDocument.presented` delegates to `media_revision.presentation_urls`, pinning source/annotation and six owned download URLs to the committed generation. Do not pre-pin corrected links to the old base token. Historical generations remain readable and curated public snapshots remain explicitly pinned.

Existing `/download/report` streams the saved normalized result JSON. `/download/readable-report` streams `readable_report_chunks(saved,document.asset_directory)` and extracts optional evidence JPEGs from that generation's annotated video. `/download/csv` streams telemetry from the same result. `ResultStreamingResponse` closes its owner on completion/disconnect. The new native-forecast and private correction report sections already integrate after the point timeline when those saved result keys are present. Static prepared `report.json`/HTML/CSV are reviewable candidate files; the current public routes regenerate them dynamically and do not serve those static filenames directly.

`ProductStore.serialize` currently loads only `result_summary` for completed uploads and always returns unavailable upload predictions. For product current estimates, open one pinned `ResultDocument` in a context, validate native source/revision association and detach only the small `nativeForecast.current` and generation metadata. Avoid copying the complete history into every library/list response. Full uploaded history can come from the pinned analysis result or a separately paginated owner endpoint. Never route it through manual `replay_predictions`.

## Frontend connection points

- `frontend/src/data/contracts.ts:AnalysisRunSchema` currently has no `nativeForecast` or `analysisCorrection`. Zod object parsing strips unrecognized fields, so a backend result containing them is presently lost in frontend decoding. Add optional strict schemas with exact raw history row/source/revision metadata and unqualified statuses; preserve legacy runs without those sections.
- `frontend/src/data/repository.ts:parseLiveRun` independently validates optional sections and cross-checks point-analysis ID/revision. Add the two new sections to that validation and require matching analysis ID/revision/source associations; malformed optional forecast evidence should become unavailable without discarding playable replay. `LiveAnalysisRepository.getRun` currently fetches latest `/result` only. A historical comparison UI needs an explicit optional generation selector so a selected old snapshot never silently refreshes to the latest.
- `frontend/src/app/AnalysisContext.tsx` owns selected run/event and a reload key (`retry`). After a correction becomes committed, refresh to the returned generation, clear/reconcile stale selected event state and replace the entire coherent run. Keep the old run usable during correction progress and show conflicts without discarding the user's draft.
- `frontend/src/pages/MatchAnalysis.tsx` re-exports `SpatialAnalysis.tsx`; the latter consumes existing pinned export links and `SynchronizedTelemetryView`. `components/PointTimelineEvidence.tsx` already limits point/execution rendering, and `ReviewWorkflow.tsx` is read-only. Add an owner review editor adjacent to the selected source event/initial score with human-readable allowed semantics, reason/evidence input, idempotent submission, status and safe same-command retry. Hide mutation controls for curated guest/historical read-only views. Authentication display state comes from existing auth context/API.
- Existing `components/WinProbability.tsx` consumes **manual** flattened history, filters unavailable rows away and builds one connected sampled SVG path. Do not plug raw native history into that code unchanged: it would lose phase/source clocks and bridge unavailable gaps. Build a bounded native selector/chart using at most600 conservative bins, preserving gaps and selecting the latest causal prefix at/before the replay frame. Show prior-only, uncalibrated, unavailable and confirmed actual result distinctly. `productApi.ts:WinPrediction` is an interface, not a runtime validator; `predictionService.get` is still an unavailable placeholder. ProductInsights currently gates its forecast component to manual matches, while ProductWorkspace can display the currently unavailable upload forecast.
- Use existing `VideoEvidencePlayer`/telemetry frame selection and normalized source clock; exact rational original PTS remains evidence. Review controls should never imply that the offline original render is live inference.

## Required connected verification and remaining master scope

Before calling this lifecycle complete, exercise actual two-account/guest HTTP isolation, forged reviewer rejection, same-command idempotency, stale base/revision409, old-generation replay/download availability while processing and after failed/cancelled corrections, concurrent commands/CAS, restart before preparation and after pointer commit, and preserved public curation. Run the actual full synthetic native fixture through worker preparation, real codecs, strict readers, complete CAS and dynamic JSON/HTML/CSV; demonstrate the changed double fault/dead receiver and causal forecast history at the website. Check browser schema retention, unknown/gap rendering, accessible form/status errors and refresh to one committed generation. These are software integration tests, not independent accuracy evidence.

Even once that is connected, independent held-out detector/identity/ground/event/speed/winner/line-call/prediction gates, deferred training admission, real10/30-minute then2–3-hour/crash-resume inference and sustained actual live FPS/P95-latency operation remain required by the master prompt. A correction lifecycle, a test video or manual live scoring cannot replace those deliverables.

## Inspected source pins

Final-freeze refresh: the preparation source pin below was updated to `3acaebe4141b7e9eb17f43e131beef34293eda60b911fa440ce080b1380cf5a3` after reviewing its current fresh derivative-credit handling. Preparation writes the new modification description into `result['mediaSource']` before JSON, HTML and CSV serialization, and binds the same description to the new annotated-video credit sidecar. This keeps those review outputs consistent about the fresh corrected overlays, score timeline and uncalibrated forecasts. It adds no authenticated correction lifecycle, product endpoint or frontend forecast contract; the implementation gaps and two-phase publication recommendations above remain applicable. All other pins were rechecked without modification. This is a source handoff refresh, not a new qualification receipt.

| Source | SHA-256 |
| --- | --- |
| src/api/auth.py | 7f6f5cef47d969cfda672a566cf3069777486d19258ea2d372f91cfef742fd5d |
| src/api/app.py | 5c517e6bdd705bdb0586807aea6563fcd3c835b2fde6c204a8cea0fa4e662667 |
| src/api/jobs.py | a2721b9359cb439f2d27895a149ae8dc3b6290f4a68b26e2abb8f8efd0e46936 |
| src/api/product.py | 9f0216fdbcba2f685e53d7abcd55642e874bfbe6a0b985ca93900bf99a709750 |
| src/api/worker.py | aefc2008a171ac689948fcbc961b603bd57fe5860c81b06218472c363b92196e |
| src/api/result_document.py | 27595098ae059ab81b659b86e362af41cbe10ed53bf87be09dab0a963d3a0fc8 |
| src/api/result_storage.py | 4eb1d7a2dfeb33d93a6ca5fcee536ec34cbe17eaa793a55b9ac824cc02de1213 |
| src/api/analysis_generation.py | ffcaf8a2e4a2ae40121a47a7204e9444eb9611da291f7b0021d1afa7a17c894a |
| src/api/media_revision.py | dad47ca4054f826c3aacbd8fa249c3b106ca0e9863064ec495a8ef928dcdde47 |
| src/api/reporting.py | 7302594a9d8872b68ecaf48c62c272589203d6e5ac88e44c7190d30d6d236c40 |
| src/scoring/corrected_rule_rebuild.py | 7801e15a5538406b1e21c7944046920626ae71ad03b4cbbe055cc8df65eb1dd2 |
| src/scoring/corrected_shot_rebuild.py | e19605d359e86b5c39185ad8b1ec931041a38855c4b4006783aec19bc5b46f1f |
| src/scoring/corrected_forecasts.py | f6bde158752b4f0607fc2cef3ca7626555bb40733ee662e14572b0ce824104e4 |
| src/scoring/corrected_exports.py | 719fa7a6556b3d082189cee9373ff3dc4b312d196dedcfd8bf60f4e950064ae2 |
| src/scoring/corrected_rendering.py | 01affb6493caa71ffe40ecbb398b512d614e748d5dd27af29f7d2e23e5454e85 |
| src/scoring/corrected_preparation.py | 3acaebe4141b7e9eb17f43e131beef34293eda60b911fa440ce080b1380cf5a3 |
| src/scoring/corrected_forecast_presentation.py | eae419797e0449cfb7fe0f95016d7c76e4fe78a7a995afe581076c637991a2ad |
| frontend/src/data/contracts.ts | 79a3543446e62217a8943c75bc63cb1c09b768f22879c6d2885e0681908a1dc1 |
| frontend/src/data/repository.ts | 8bb8d6057507cba34456f927ae6590102762392e7d7d717fa8fa8060bf97c96f |
| frontend/src/data/productApi.ts | 4b2358bc697ee045ef8aa1e40fc5eb288c44a848325c8abe184e6349b28d4e2a |
| frontend/src/data/analysisApi.ts | 6a0d4ceff9de1339c82b48186c2c8f4a061f24eec5d1e8c743f0953f6d5c43a8 |
| frontend/src/app/AnalysisContext.tsx | 8c7d3ed63e1e727ae9074e755e0bbb42146894fa83cbead178e766d6d4c94dde |
| frontend/src/pages/SpatialAnalysis.tsx | c60204eb43016ed732f8593fe810023ec9daa8a9f80bbfb68e8b32d572ac0e85 |
| frontend/src/pages/ManualLive.tsx | 9b4b9c75d6fabd0325a4138cbaced439903069b6d5c4213ee4eb9700a99c94ab |
| frontend/src/pages/ProductInsights.tsx | 9fc40ed1bd120608ac02c9937ee000e227ff57e4669557818e20b4cfedd5c43c |
| frontend/src/pages/ProductWorkspace.tsx | 5397359a1e9a6cf8aa93db38fbf62433cef920ec873a07f2bbbb27270c7767d9 |
| frontend/src/components/ReviewWorkflow.tsx | 405576a7738ab6e57e0c11f759f18c38540efee035b05fafd1788cd9ec62693c |
| frontend/src/components/PointTimelineEvidence.tsx | 88d262ba2d62289e1b15bde7d2cf628385e5dfb247cb6cbf875c5e831007eea6 |
| frontend/src/components/WinProbability.tsx | 257b721486cca6f8c691ed97bdf0275e7cfe9f2262e4290299ab7c5a73262327 |

# Private complete-correction preparation: current source handoff

Read-only mapping captured on 2026-10-18. This document proposes the next private integration stage; it is not an executed complete-result, report, publication, model, accuracy, long-match, or live qualification. No source, test, or product documentation file was changed for this mapping. The final section contains SHA-256 pins of the inspected sources and reusable fixture helpers.

The concrete next module can be `src/scoring/corrected_preparation.py`. Its purpose is to assemble the complete new revision in private scratch, strictly normalize it with the real result producer, attach a separately validated causal forecast presentation, and stream actual JSON/HTML/CSV reports while the result owner remains open. It must not call a publication function or change any current generation.

## Ownership and a proposed interface

```python
@contextmanager
def prepare_corrected_result(rebuilt, shot_analysis, forecast, exports, render,
                             scratch, *, job):
    # Admit all existing native owners, immutable base, command and stage seals.
    # Compose a unique private directory, strict-load, then build the real result.
    with ResultStorage(result_scratch) as storage:
        strict_native = load_point_analysis(stage, analysis_id,
            frame_count=frames, authoritative=authority, storage=storage)
        result = build_result(stage, job, _storage=storage)
        # Require exact new revision, counts and source; add a validated forecast
        # bridge without manufacturing qualified probabilities or new inference.
        storage.set_root(result)
        # Stream result/report writers now, before closing storage.
        yield PreparedCorrection(directory=stage, storage=storage,
            result=result, files=exact_read_only_inventory,
            base_generation=base, analysis_revision=revision,
            manifest_sha256=private_seal_digest)
        # Recheck native owners/base/stage bytes before releasing the context.
```

The returned context must own all lazy arrays through result consumption, strict JSON streaming and report generation. Returning `result` after closing `ResultStorage` is invalid: large arrays are `DiskSeries`, not detached lists. Frozen metadata, paths and digests may outlive the context. A frozen prepared-result object should bind the exact native contexts, original `job['id']`, command hash, base generation, source digest, new analysis revision, export seal and render seal. Context entry and exit must reject source/base/seal mutation; failures retain diagnostics without a final preparation seal. Do not accept resident caller lists as substitutes for fresh native owners.

`corrected_exports._admission(rebuilt, shots, forecast, scratch)` already checks exact current native owner associations and forecast scratch. `corrected_rendering._guard_exports(...)` verifies the export seal, hashes, exact inventory and equality of the live physical event owner with its sealed presentation. The next stage needs the corresponding **render** seal admission: validate the actual `CorrectedRenderResult`, exact three-file inventory and manifest digest, common source/base/revision, matching export-manifest digest, and native/browser artifact hashes and sizes. Merely finding `annotated_web.mp4` is insufficient. Before accepting a prepared result, keep the current base rechecks and actual source/native/browser full-decode evidence from rendering.

## Complete inventory: immutable observations versus fresh semantic outputs

`analysis_resume.REQUIRED_PIPELINE_FILES` currently contains 37 filenames. Fourteen semantic files below and `annotated.mp4` must be regenerated; the other 22 are allowed byte-identical observation/configuration inputs for this **semantic-only** correction:

```
source.mp4
media_metadata.json
inference_config.yaml
player_motion.json
racket_tracking.json
event_candidates.json
event_verification_trace.json
player_metrics.json
ball_metrics.json
court_geometry.json
detections.json
temporal_spacing_audit.json
temporal_detour_audit.json
trajectories.json
run_config.yaml
camera_registration.json
camera_intervals.json
match_identity.json
image_people.json
stationary_candidate_audit.json
pixel_motion_audit.json
patch_persistence_audit.json
```

Require each mandatory name in the committed base snapshot. Resolve with `snapshot.resolve_file(name)`, verify its trusted hash/size, independently copy in bounded chunks, fsync/close, and verify source and destination again. Never substitute a mutable root alias, hardlink, symlink or junction. `source.mp4` already exists in the semantic export set: join it by exact digest equality rather than overwriting it with another authority. The new helper should compute the exact union of inventories, rejecting conflicting overlapping names. Do not infer completeness from a hardcoded count alone.

Four optional unchanged observations may be copied when present in the sealed snapshot: `source_frame_mapping.jsonl`, `normalization_frame_stats.txt`, `media_source.json`, `pose_tracking.json`. The fifth existing optional file, `landing_zones.json`, must come from the fresh semantic exports. Missing optional evidence remains unavailable. Mandatory originals additionally require the sealed `original` plus the original job extension for the existing download contract; do not read the mutable upload directory as fallback. A fresh annotation-credit record, if used, must bind the new browser-video hash. Old `annotated_web_credit.json` cannot describe a changed annotation.

The current `corrected_exports.EXPORT_FILES` is an exact 19-file set; its separate seal makes the returned inventory 20 files:

```
point_analytics.json        rallies.json             match_analytics.json
serve_executions.json       shot_play_scope.json      scoring_events.json
score_history.json          match_state.json          shot_events.json
line_calls.json             match_events.json         metrics.json
native_rule_replay.json     point_timeline_manifest.json
corrected_forecasts.json    correction_request.json   correction_audit.json
source.mp4                  landing_zones.json
```

`corrected_export_manifest.json` binds that set. `render_corrected_video(...)` returns the separate exact set `annotated.mp4`, `annotated_web.mp4`, `corrected_render_manifest.json`. Every semantic file and both rendered videos must come from these fresh outputs; copying the old 37-file generation and replacing only the visible report is not a valid correction. The original inference checkpoint remains evidence about old inference, not a checkpoint of new semantic preparation.

## Real normalized result producer: required and optional behavior

Use `src.api.results.build_result(stage: Path, job, _storage=storage)`. Do not call its materializing default. The job needs at least `id`, `filename` and `started_at`; `completed_at` supplies `summary.analyzedAt` when present. Keep the original analysis ID and source label. Avoid writing a fabricated completion time or new processing throughput into retained inference provenance.

The builder strictly requires these three raw documents:

| Input | Actual current checks and normalization |
| --- | --- |
| `detections.json` | Object with `metadata.frames` positive integral, finite FPS in [.1, 240], positive integral width/height; exact frame count/order. Selected boxes must be finite, positive extent and within image bounds; confidence in [0,1]. Camera registration gates frame metric availability. |
| `trajectories.json` | Exact `ball_trajectory` count/order, supported tracking state, finite timestamp within .01 seconds of index/FPS. Invalid/missing/occluded image measurements become unavailable. Physical speed and world xyz are forcibly withheld. |
| `metrics.json` | Required object. `authoritative_events_enabled is True` controls semantic event/call exposure; new correction metrics intentionally have no inference FPS. |

Other actual reads are `camera_registration.json`, `court_geometry.json`, `media_metadata.json`, `player_motion.json`, `match_events.json`, `line_calls.json`, `pose_tracking.json`, `racket_tracking.json`, `image_people.json`, `media_source.json`, optional native point artifacts and `landing_zones.json`. A complete publication inventory must still contain the remaining mandatory observation/audit files even when `build_result` does not parse them.

Key compatibility details:

- Court presentation additionally requires valid `homography_matrix` **and finite nonnegative `reprojection_error_px`**. The current small correction fixture's `court_geometry.json` lacks that residual and canonical/image14 arrays, so it is valid for rule-source tests while its web court will be unavailable. A complete synthetic fixture must supply these explicit fields if it is intended to test actual court and landing web presentation.
- `player_motion.json` can hold null ground positions/speeds with complete aligned samples. Native candidate exports set `metric_measurements_available=False`; retain those unknowns and the qualifications. The builder currently normalizes motion from positions/sample availability; do not introduce positive ground positions merely to make reports look populated.
- `pose_tracking.json` is preferred over `player_motion.pose_frames`; an optional stale or malformed legacy pose file cannot be silently ignored while a different unsealed pose source is substituted. All optional JSON is parsed through EOF before use. Native pose visuals are image coordinates, not qualified ankle contacts or world geometry.
- When `media_metadata.timeMapping.status` records a captured/partial/unavailable source map, `validMeasurementTransitionIntervals` controls timing admission. Preserve exact original PTS and time-base files and unknown transitions. Normalized frame/FPS time is not physical source timing authority.
- `provenance(stage, metrics)` prefers `inference_config.yaml` to `run_config.yaml` and hashes configured local checkpoint files without importing models. This recomputes a bundle signature from files **currently** at their configured paths. For a delayed correction, verify those dependencies still match the retained inference identity or explicitly preserve/bind original provenance; do not claim newly hashed replacement checkpoints produced the sealed old observations. The new correction's metrics must not present old inference FPS as correction throughput.

### Strict native validation must precede the forgiving web builder

Use `load_point_analysis(stage, analysis_id, frame_count=frames, authoritative=authority, storage=storage)` directly, require its source digest/new analysis revision/scoring revision0 and expected counts, then call `build_result`. The builder intentionally catches invalid optional point bundles and returns `pointAnalysis=None` plus a warning. It similarly catches invalid landing evidence, returns `landingZones=[]`, and warns. Therefore successful normalization alone does not establish native consistency. Strictly validate landing zones first with `validate_landing_zones(raw, analysis_id, events, frame_calibrations=_CameraRecords(rebuilt.sources), expected_revision=new_revision)`, retaining indexed calibration roots and owned arrays. Then assert the normalized point/landing presentation matches the verified new evidence.

`point_timeline_manifest.json` must keep its exact eight-file set. Prepared/report/forecast/whole-stage metadata belong in a separate preparation seal.

## Forecast bridge is new work, not an existing builder feature

Current `build_result` never reads `corrected_forecasts.json`. It returns `matchState=None`, empty top-level `scoringEvents`/`scoreHistory`, and `summary.capabilities.scoring=False`; reviewed native scoring is separately represented by `pointAnalysis`. Current `AnalysisRunSchema` in `frontend/src/data/contracts.ts` has no prediction field. Product `WinPrediction` in `frontend/src/data/productApi.ts` is only a TypeScript interface, and uploaded `ProductStore.serialize` still returns unavailable predictions. These gaps remain even when all fresh forecast JSON files are correct.

For the private preparation stage, introduce an explicit native forecast presentation object with source/base/new revision, model signature, uncalibrated prior/model assumptions, detached current value, and an owned bounded history. Each native history row has phase INITIAL/BEFORE_EVENT/AFTER_EVENT, exact source PTS/timebase/frame, event/point IDs, raw-state digest, uses-through event ID and a forecast payload. Do not flatten away these fields to mimic manual score history. Validate the native forecast seal and owner association, chronology/causal prefixes, event joins and `1+2*accepted_inputs` count before attaching it to the normalized result. Keep `status=unavailable|prior_only|uncalibrated|confirmed_result`, nullable probabilities and separate actual winner. Uncalibrated values must not become a confident winner claim.

Large forecast history should be copied one record at a time into this result owner's `storage.series(...)`, preserving prior-only and unavailable rows. Do not `list(forecast.history)`, reuse manual `replay_predictions` with its resident history/seen map, or store an owner from another context inside the result. A report forecast section and frontend Zod schema/product wiring will need explicit implementation; current HTML/CSV writers do not render forecasts simply because a new field is attached.

## Actual JSON, HTML and CSV APIs

| Output | Existing bounded writer / semantics |
| --- | --- |
| Result/report JSON | `storage.dump(result, text_stream)` or iterate `storage.json_chunks(result)`. Strict finite JSON is checked while streaming. `storage.set_root(result)` retains the owned tree for a later private read-only database or eventual generation publisher. Do not `json.dumps` a whole DiskSeries tree. |
| Readable HTML | Iterate `src.api.reporting.readable_report_chunks(result, stage)` directly into a UTF-8 atomic destination. This includes current rule timeline and landing tables, fixed-size charts, escaping and optional embedded evidence JPEGs. Its `readable_report` compatibility wrapper joins the complete string and should not be used for long preparation. |
| Telemetry CSV | Iterate `telemetry_csv_chunks(result, chunk_chars=64*1024)`. This is per-frame image/ground telemetry with credit columns; it does not include point or forecast history. Source credit formula-like cells are neutralized. Its `telemetry_csv` wrapper materializes the whole CSV. |

`reporting.evidence_images(stage,result)` reads **new `annotated_web.mp4` in that stage**, samples first/middle/last, scales width to at most960, and embeds JPEG quality80. It catches selected decode/image errors and can produce no images. The coordinator must separately require already verified video readability; an empty optional image section is not proof of actual decoding. Native HTML can be generated from the prepared stage, but public browser navigation remains unavailable until an authorized complete generation exists.

Use fresh names such as `report.json`, `readable_report.html`, `telemetry.csv` plus one last-written `corrected_preparation_manifest.json`, with exact filenames/hash/size and common binding. The existing HTTP routes currently generate JSON/HTML/CSV from `ResultDocument` on demand, rather than serving these static names; private artifacts make the complete candidate reviewable and can be verified byte-for-byte against those same streamed APIs. This design does not itself add HTTP/auth routes.

## Browser links and future publication boundary

`build_result` generates unpinned analysis-relative `/api/analyses/{id}/...` links. Do not pin them to the **base generation**: those URLs would show old source/annotation/reports while the candidate contains new semantics. Do not invent the future token before sealing; the generation digest depends on its result/asset manifest.

Keep stored candidate links as base-relative templates with explicit private status. After a future complete generation is authorized and committed, `media_revision.presentation_urls(result, snapshot.files_directory, analysis_id, generation=token, snapshot=snapshot)` pins source/annotated URLs and all six owned download kinds to `?generation=token`, and sets `analysisGeneration`/`analysisRevision`. `ResultDocument.presented(analysis_id)` already calls this. It also admits annotation credit only when its input/source/annotation hashes match. Reports generated during preparation should be self-contained with embedded images and record the base generation separately as provenance, never present base-linked downloads as the corrected artifact.

Future publication, deliberately outside this private stage, is `publish_result(storage,result,analysis_directory,generation_files=exact_inventory,analysis_revision=new_revision,expected_generation=base_generation,before_commit=final_owner_source_cancel_model_revision_check)`. The existing complete publisher then copies independently, seals DB/result and CAS-swaps the pointer. Its JSON and summary reserved filenames must not be passed as conflicting caller asset aliases. Do not call `publish_result` here, mutate the job to completed, or substitute normal retry for a correction revision.

## Reusable actual fixture foundations and the missing complete fixture

No existing helper inspected supplies every required native observation plus fresh correctable replay **and** playable source. A new test helper should extend the current synthetic fixtures; it should not use checkpoint publication stubs as semantic evidence.

1. `tests.test_corrected_shot_rebuild.build_shot_base(tmp_path, frame_count=12, playable_source=True, repeated_hits=False)` is the best actual foundation. It creates real native rule/line/shot journals, exact rational source frames, reviewed A/B bindings, independent immutable generation copies, full-size1280x96030FPS encoded synthetic source, and a decisive legal second serve → corrected double-fault/dead receiver. Its generation currently has only source, detections, camera registration, court geometry, run config, native replay, plus generated result/summary; **not all37**.
2. Extend it before the final base-generation publish with explicit native-shaped trajectories derived from its retained measured detections, complete null metric motion samples and image-pose policy, disabled racket tracking, aligned raw image people, source media/time mapping/configuration, and missing observation/audit envelopes. Keep all synthetic/model-unqualified labels. Require the mandatory inventory and use actual `publish_analysis_generation` to retain it. Set `reprojection_error_px` and semantically ordered canonical/image14 only when exercising valid court presentation; do not borrow the100x100 normalized fixture's incompatible dimensions or identity matrix. Preserve exact source clock intervals and selected track IDs already bound to replay.
3. `tests.test_correction_sources.write_observation_assets(...)` supplies actual native-shaped detection/camera/court documents and current interval joins. Its default width1920x1080 is replaced by `build_shot_base(playable_source=True)` to1280x960; its basic court lacks a residual/canonical14. It is reusable as a document template, not complete production output.
4. `tests.test_bounded_result_contract.artifacts(tmp_path,count,fps)` supplies the **real `build_result` normalizer's** minimum raw document shape and aligned samples. It has100x100 synthetic calibration and invented positive movement for compatibility tests, no valid correctable rule replay or playable video. Use its schemas, not its measurements, to fill the new helper.
5. `tests.test_point_timeline_web_contract.fixture_bundle`/`reviewed_fixture_bundle` and `basic_artifacts` exercise actual strict point/report readers from retained native fixtures with verified hashes. Their69-byte source is intentionally nonplayable and `basic_artifacts` is only minimal normalizer input; these cannot satisfy new render preparation alone.
6. `tests.test_generation_http_contract.publish` uses the real bounded builder/publisher/pinned HTTP routes but synthetic nonplayable media and no corrected rule source. Its URL and ownership assertions can be reused for later route tests.
7. `tests.test_worker_result_publication.native_checkpoint_fixture` fills missing mandatory files with generic `{"synthetic_no_inference":true}` and replaces `worker.build_result`; `tests/helpers/checkpoint_worker_child.py` also stubs both media and normalized result. They prove queue/checkpoint/crash boundaries only. Do not reuse their placeholders as a complete native-schema fixture.

Decisive private preparation tests should run the **real** strict point/landing readers, real bounded `build_result`, new causal forecast bridge and actual JSON/HTML/CSV streams from the fresh full candidate; verify corrected double-fault stats, point denominators, dead receiver, no stale polygons/line calls, causal prefix history and new revision in all outputs. Inject changed/missing mandatory observations, old or substituted render seals, source/job/frame mismatch, halfway report serialization failure, closed owner and exact base-generation advance; no final preparation seal/current pointer should appear. Grow completed points/history separately from video-frame capacity. Preserve model import tripwires and identify every synthetic source/measurement. These tests establish integrated software behavior only.

## Inspected-source SHA-256 pins

The following table was computed directly from current files after this mapping was written. Hashes bind this handoff to its inspected implementation, not completion of the proposed stage.

| Source | SHA-256 |
| --- | --- |
| src/scoring/corrected_exports.py | 719fa7a6556b3d082189cee9373ff3dc4b312d196dedcfd8bf60f4e950064ae2 |
| src/scoring/corrected_rendering.py | 01affb6493caa71ffe40ecbb398b512d614e748d5dd27af29f7d2e23e5454e85 |
| src/scoring/corrected_forecasts.py | f6bde158752b4f0607fc2cef3ca7626555bb40733ee662e14572b0ce824104e4 |
| src/scoring/corrected_landing_exports.py | 5c0d47e9b9a1eee9a39f029c5c70ab4e5d8f38e179d352f0f7da0cf009fd7437 |
| src/api/analysis_resume.py | 732a3bdca9d62c7b72913cdd6e3cd4b992734c06192c16b45e8526ed25b128b9 |
| src/api/results.py | e5c32937658ab21b6dbc83689fdf4f282230c38eb22cfb35125e03078ab1f4e3 |
| src/api/reporting.py | d037d43ba6fd5b3d3ea77c37f3a0e4dcd4a948f0e882a04342a1d2249be7cf33 |
| src/api/result_storage.py | 4eb1d7a2dfeb33d93a6ca5fcee536ec34cbe17eaa793a55b9ac824cc02de1213 |
| src/api/result_document.py | 27595098ae059ab81b659b86e362af41cbe10ed53bf87be09dab0a963d3a0fc8 |
| src/api/media_revision.py | dad47ca4054f826c3aacbd8fa249c3b106ca0e9863064ec495a8ef928dcdde47 |
| src/api/analysis_generation.py | ffcaf8a2e4a2ae40121a47a7204e9444eb9611da291f7b0021d1afa7a17c894a |
| src/api/point_timeline_contract.py | 5b0a8d18019efe5df1f54f6fc1c91e941b492c0cb208fef1cdf85d295245413a |
| src/api/landing_zone_contract.py | 4f970da685f504563cad2243943a17c2746458bb463064155b8e00f96691d0cb |
| src/api/product.py | 9f0216fdbcba2f685e53d7abcd55642e874bfbe6a0b985ca93900bf99a709750 |
| src/api/app.py | 5c517e6bdd705bdb0586807aea6563fcd3c835b2fde6c204a8cea0fa4e662667 |
| src/api/worker.py | aefc2008a171ac689948fcbc961b603bd57fe5860c81b06218472c363b92196e |
| src/prediction/tennis.py | e3a93c22c4866858468cbf9e19890015c550e109fc01139a08820337e3c780c1 |
| frontend/src/data/contracts.ts | 79a3543446e62217a8943c75bc63cb1c09b768f22879c6d2885e0681908a1dc1 |
| frontend/src/data/productApi.ts | 4b2358bc697ee045ef8aa1e40fc5eb288c44a848325c8abe184e6349b28d4e2a |
| tests/test_corrected_shot_rebuild.py | 66e1ed1d3f6597b41dafc11a04b1a1c185e2f40e7a8b0946094dd05da88deb9b |
| tests/test_corrected_rule_rebuild.py | d589eee5154d1a6ac028aa8fc93a82cf96261f62122a7536bad05f1807650c84 |
| tests/test_correction_sources.py | 905085e912e958d2f39f4c8e89a26a5e15514a11f166bc10b241a3dd78a0a590 |
| tests/test_bounded_result_contract.py | 28268bdd9aaf3a9fc4f8c3015f5c5ffadcb89699406629b89305fff92179dba9 |
| tests/test_point_timeline_web_contract.py | 1b97c106dbd809f8c62e0d24faf44e94961c39bc5b86159c0fb66df6f7f788ea |
| tests/test_point_timeline_http_contract.py | 056323ebf6aebf93c4d288d0ab8b2e5dd966b7e084d55b4b7f89f8d630de350a |
| tests/test_generation_http_contract.py | f2ee9b36900b3e13c5a89525937cf57d395611018a58b9b814d442e2e953f5d1 |
| tests/test_worker_result_publication.py | 58dc675338c6906d21bddec2b0fb3cabd084198a085b8dfcaa0806de3bee3560 |
| tests/helpers/checkpoint_worker_child.py | ea468341210d76a96d039728d657006e95779f7b0a0442ebca5fa64bbd9677f2 |
| tests/test_corrected_exports.py | 4013dc691093d283a488e6c9aada99164091447f7eb32e33772ceaff7bc484e3 |
| tests/test_corrected_forecasts.py | 998a9ff156a8a0244b4e53cda00ce9d0a28746d1cda7aea6a62e9f298db4efe2 |
| tests/test_bounded_reporting.py | b9f00f841bc72dc7dbcbfa32e45ee553050a01b3e6ff4c16ddc3319926247447 |

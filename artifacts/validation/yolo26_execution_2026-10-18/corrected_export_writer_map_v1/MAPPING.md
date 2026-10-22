# Current native writer map for the next private corrected export coordinator

Captured 2026-10-18 after the actual SQLite predictor-index regression run. This is a read-only source mapping, not an executed export, renderer, publication or independent-accuracy receipt. `source_pins.json` binds 20 selected current source files. No application source or documentation was changed by this mapping.

The immediately implementable next module can be `src/scoring/corrected_exports.py`, with a private entry point accepting the live `rebuilt`, `shot_analysis`, `forecast` and their already guarded scratch directory. All three contexts must remain open through export and verification. It should return a small exact inventory of its generated files, hashes and byte counts only after all writes and checks succeed. It must leave the current generation untouched. Rendering and owner-authenticated compare-and-swap publication remain later work.

## Concrete input ownership and admission

1. Reuse the current corrected-forecast `_admit(rebuilt, shot_analysis)` checks at entry and after writes, plus `forecast.owner._open()`, `forecast.owner.sealed`, exact `forecast.history is forecast.owner.history`, exact `forecast._rebuilt is rebuilt` and `forecast._shot_analysis is shot_analysis`, and matching `forecast.snapshot` / `forecast.base_generation`. The actual forecast result is a frozen `_ForecastResult`; its `current` and `audit` properties return detached JSON values.
2. Use `rebuilt.owner`, `rebuilt.engine`, `shot_analysis.shots`, and `shot_analysis.event_owner.events`. Check `shot_analysis.events is shot_analysis.event_owner.events`. Do not let a caller replace these with resident lists or old generation arrays.
3. Read correction command and raw initial/final states from the actual `rebuilt.audit['request']`, `initial_raw_state` and `final_raw_state`. The request already contains reviewer, evidence, reason, expected base generation/revision/replay hash and exact amendments. The eventual HTTP handler must derive reviewer/ownership from its authenticated owner; these private request fields do not provide authentication.
4. Recompute `native_match_analytics(owner, shots, engine.get_current_state(), authoritative=...)` for the export, or validate the supplied analytics against that reducer. Do not blindly trust the mutable `shot_analysis.match_analytics` dictionary after another consumer could have modified it.
5. Export while all SQLite owners are open. The returned inventory contains paths and primitive hash/size metadata, never a lazy array or owner. Cleanup may close all readers after the private output set has been verified.

## Fresh output inventory and direct writer calls

These 14 existing native files can all be regenerated from the fresh owners before rendering. Keep the established schemas so the current point-timeline reader can validate them.

| File | Actual bounded producer / writer | Required metadata |
| --- | --- | --- |
| `point_analytics.json` | `write_membership_analytics(path, 'points', owner.rows('points'), sanitize, schema_version='2.0')` | Writer preserves incremental `shot_ids` membership arrays. |
| `rallies.json` | Same writer with `'rallies'`, `owner.rows('rallies')`, schema `2.0` | Includes partial/review records as current native producer does. |
| `match_analytics.json` | `write_match_analytics(path, fresh_analytics, sanitize)` | Fresh reducer returns schema `2.0`; its `ball_landing_heatmap` is a lazy eligible-shot view. |
| `serve_executions.json` | `write_json_array_artifact(path, metadata, 'serve_executions', owner.rows('attempts'), allow_nan=False)` | `schema_version='2.0'`, exact new `analysis_revision`. |
| `shot_play_scope.json` | Same array writer with `'shots'`, `owner.member_rows()` | `schema_version='2.0'`, exact new `analysis_revision`. |
| `scoring_events.json` | Same array writer with `'scoring_events'`, `owner.rows('transitions')` | `schema_version='2.0'`. |
| `score_history.json` | `engine.save_history_json(path)` | Real journal delegate atomically streams schema `2.0`, rule `revision=0`, history and corrections. Do not use `history.to_list()`. |
| `match_state.json` | Atomic small-object writer around `engine.get_current_state().to_dict()` | `schema_version='1.0'`, `authoritative_events_enabled`, and native `semantic_role`; raw complete state remains separately in the replay. |
| `shot_events.json` | Array writer over `(shot.to_dict() for shot in shot_analysis.shots)` | `schema_version='1.0'`, array key `'shot_events'`. |
| `line_calls.json` | Array writer over `(call.to_dict() for call in engine.native_line_calls())` | `schema_version='1.0'`, array key `'line_calls'`. These are the new accepted calls. |
| `match_events.json` | Array writer over fresh `shot_analysis.event_owner.events`, using the established event wire fields below | `schema_version='1.0'`, `analysis_id`, new `analysis_revision`, and exact physical `events_count`. Reviewed rule signals remain in the rule replay and scoring timeline, not this physical-event array. |
| `metrics.json` | Atomic small-object writer; preserve/bind original inference timing separately and replace event-dependent counts | Required existing fields include `total_frames`, `shots_linked`, `rallies`, `authoritative_events_enabled`, `production_qualified=False`. Do not pass original inference throughput off as correction/render throughput. |
| `native_rule_replay.json` | `write_rule_replay(path, engine, owner, rebuilt.sources.source_frames, authoritative_events_enabled=...)` | Existing writer accepts the new `owner.revision` and the fresh rule-journal `engine.revision == 0`. It validates accepted input/history/transition joins and emits the new complete raw initial/final state, raw/linked shots, memberships and all source frames. |
| `point_timeline_manifest.json` | `publish_point_timeline_manifest(...)` **after the preceding eight point files have closed successfully** | `analysis_id=owner.analysis_id`, `analysis_revision=owner.revision`, source hash, `scoring_revision=0`, exact authority flag. |

The current `POINT_TIMELINE_FILES` tuple contains exactly: point analytics, rallies, match analytics, serve executions, shot play scope, scoring events, score history, match state. Do not add the replay or forecasts to that exact-eight manifest: the reader requires exact set equality. A separate private export-set inventory binds all regenerated outputs together.

Three additional private files make the corrected evidence reviewable:

| File | Producer |
| --- | --- |
| `corrected_forecasts.json` | Existing `write_corrected_forecasts(scratch / 'corrected_forecasts.json', forecast)`; it requires that exact filename and exact forecast scratch parent. Streams the fresh history and detached current value. |
| `correction_request.json` | Atomic canonical finite JSON from `rebuilt.audit['request']`, with a bounded size already enforced by the rule builder. This records the exact original command, rather than a shortened textual summary. |
| `correction_audit.json` | Atomic bounded aggregate of rule, shot and forecast audits, their common analysis/source/base/new revision bindings, hashes of the 17-file private set where noncircular, and `published=False`, `human_review_dependent=True`, `independent_accuracy_validated=False`. The inventory cannot include its own hash inside its own hash; put aggregate file hashes in a separate final seal. |

Recommended final seal: `corrected_export_manifest.json`, written last with schema, common binding, exact generated filename-to-{bytes,sha256} mapping, status `PRIVATE_PENDING_RENDER_AND_PUBLICATION`, no self-hash. A complete generation later binds this manifest and every export. Neither seal is an alternate current-generation pointer.

### Physical-event wire helper

The existing `phase6_pipeline.event_export_records` is small and bounded, but importing `phase6_pipeline` eagerly imports Torch and many inference classes. Factor the helper into a lightweight shared module, or implement the established encoding in the new private module without importing Phase6. Its fields are:

```python
{
    'event_id': event.event_id,
    'event_type': event.event_type.value,
    'frame': event.frame_index,
    'timestamp_s': event.timestamp_s,
    'player_id': event.player_id,
    'ball_position_px': [round(x, 2) for x in event.ball_position_px]
                        if event.ball_position_px else None,
    'court_position_m': list(event.court_position_m) if event.court_position_m else None,
    'confidence': round(event.confidence, 3),
    'trajectory_state': event.trajectory_state,
    'evidence': event.evidence,
}
```

Keep the lossless unrounded input in `native_rule_replay.json`. The public event encoding is a presentation and must not become the next correction's measurement source.

### A minimal execution sequence for the next implementation

```python
# Within live corrected-rule, fresh-shot and fresh-forecast contexts:
admit_exact_owners_and_base(rebuilt, shot_analysis, forecast)
analytics = native_match_analytics(rebuilt.owner, shot_analysis.shots,
    rebuilt.engine.get_current_state(),
    authoritative=rebuilt.authoritative_events_enabled)

# Each producer is streamed directly from its owner; no list(...) of points,
# shots, events, transitions, line calls, memberships, landings or history.
write_the_eight_native_point_files(...)
write_fresh_shots_calls_physical_events_and_metrics(...)
write_rule_replay(...)
write_corrected_forecasts(forecast.scratch / 'corrected_forecasts.json', forecast)
write_bounded_request_and_stage_audits(...)
publish_point_timeline_manifest(...)

# Source video may be copied from the sealed snapshot into the private stage
# for the existing validator. Never use mutable analysis-root aliases.
with ResultStorage(scratch) as proof_storage:
    proof = load_point_analysis(scratch, rebuilt.owner.analysis_id,
        frame_count=rebuilt.snapshot.frames,
        authoritative=rebuilt.authoritative_events_enabled,
        storage=proof_storage)
    require(proof['analysisRevision'] == rebuilt.owner.revision)
    # Validate counts, source hash, exact source/event/revision joins before
    # closing proof_storage; detached summary only may leave this context.

admit_exact_owners_and_base(rebuilt, shot_analysis, forecast)
write_exact_hash_bound_private_export_manifest_last(...)
# Return only small primitive metadata. No whole-generation publication here.
```

`write_the_eight...` above is a dispatch to the concrete writer calls in the table, not a suggested new aggregate algorithm. Small-object atomic writing can use the existing analytics `_destination` pattern (same-directory temporary file, finite JSON, flush, fsync, replace). It should have the same guarded destination policy as the other corrected helpers. The exact `corrected_forecasts.json` destination restriction means a new nested stage directory must also be the forecast context's scratch directory; otherwise keep this coordinator in that context's existing guarded scratch and guard its explicit filename set.

## Existing validation, web result and publication seams

`load_point_analysis(directory, analysis_id, frame_count=..., authoritative=..., storage=ResultStorage)` already admits the new nonnegative **analysis revision** while insisting on fresh scoring **revision 0**. It checks all eight hashes before and after parsing, source.mp4 hash, physical match-events ID/revision/clock bindings, point/attempt/rally membership joins and totals/denominators. Its no-storage default materializes JSON; the coordinator must supply owned storage.

`build_result(directory, job, _storage=storage)` is the real bounded presentation producer. It requires frame-complete `detections.json`, `trajectories.json` and `metrics.json`; it reads other sealed artifacts into owned DiskSeries. It calls `load_point_analysis` with the supplied storage, but deliberately catches an invalid optional native timeline and emits a warning plus `pointAnalysis=None`. Therefore the coordinator must call the strict validator itself and reject before result construction; a superficially successful `build_result` is insufficient proof. After building, assert the expected new `analysisRevision` and non-null point evidence.

Current `build_result` does not yet consume `corrected_forecasts.json` into the web prediction contract. It still returns `matchState=None`, empty top-level scoring events/history and unqualified capabilities, while the native point evidence is in `pointAnalysis`. That independent product-contract work remains necessary before claiming uploaded correction completion.

The complete-generation writer is already capable of atomic publication:

```python
publish_result(storage, fresh_result, analysis_directory,
    generation_files=exact_complete_inventory,
    analysis_revision=rebuilt.owner.revision,
    expected_generation=rebuilt.base_generation,
    before_commit=final_owner_source_cancel_model_and_revision_check)
```

It copies assets independently in bounded chunks, seals the database and result JSON, then compares and swaps the pointer. Existing owner-authenticated handlers must remain the only path that authorizes that future call. Do not call this from the private export coordinator. Do not substitute ordinary retry or `engine.correct_event` for an analysis revision.

## Exact source-copy versus fresh-output inventory

`REQUIRED_PIPELINE_FILES` has 37 files. Fourteen semantic/diagnostic outputs in the fresh table above plus `annotated.mp4` account for 15; the other 22 can remain byte-identical sealed observation/configuration inputs for this semantic-only correction:

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

Copy each by explicit `base_snapshot.resolve_file(name)` with source hash/size verification before and after copy. No mutable root fallback, symlink/junction or hardlink. These are original inference inputs/observations; the new correction request and replay record the changed semantics. Do not rewrite the original model configuration while claiming it is byte-identical inference evidence. If future correction changes measurements/camera/identity, this copy policy must be replaced by the corresponding recomputation and qualification rules.

Of the five current optional files, `source_frame_mapping.jsonl`, `normalization_frame_stats.txt`, `media_source.json`, `pose_tracking.json` can be copied from the sealed generation when present. **`landing_zones.json` cannot be copied**: it binds event semantics, active play, calibration and the old analysis revision.

For regenerated landing zones, actual APIs are `LandingZoneSequence(scratch, analysis_id, revision)`, `landing_record(event_dict, current_event_H, analysis_id=..., revision=..., calibration_id=..., partition=..., duration_seconds=...)`, `owner.append(record, calibration)`, then `owner.seal().write_json(destination)`. The builder returns None unless bounce review is confirmed/corrected, active_play is true, position_type is ground_contact, source and finite uncertainty are supplied. Never invent review/uncertainty from a line-call confidence. Skip freshly dead events; use the event's accepted new geometry and fresh outcome. The calibration dictionary needs analysis ID, new analysis revision, frame index and exact event-frame homography. Produce an empty valid new-revision record set when no contact qualifies; do not reuse the old highlighted polygons. Existing Phase6 additionally updates the corresponding fresh event evidence with matching analysis/revision/calibration/contact-pixel binding before exporting match_events.

Future complete publication additionally requires newly rendered `annotated.mp4` and `annotated_web.mp4`, a fresh annotation-credit record if used, and the sealed original video asset (`original` plus its preserved extension). Existing annotations or their credit cannot stand in for corrected replay. Start rendering from sealed `source.mp4`, using this new point/shot/forecast/landing state; never overlay onto the old annotation. The existing completed-inference checkpoint remains evidence of the original inference stage only, and cannot certify new semantics/rendering.

## Decisive next integration evidence

- Existing double-fault fixture: write all fresh native outputs and strict-load them with `ResultStorage`; original legal serve/rally membership must become corrected fault/dead receiver, one completed point and one posterior update.
- Same output set plus a reviewed let/unknown boundary, authority off, game-ending preaward server, tie-break and complete-match cases; preserve unavailable probabilities and all count denominators.
- Inject a serialization failure mid-way and an exact base-generation advance; no private export seal or new current-generation pointer may appear. Existing previous destinations and the immutable base stay readable.
- Close owner / replace event alias / mismatch forecast owner / edit fresh audit/current values: reject or regenerate from authoritative fresh owners, with no fallback to old arrays.
- Grow actual completed native points, accepted inputs, posterior IDs **and** exported history. Existing `build_shot_base(..., repeated_hits=True)` grows later dead contacts but still only one completed point, so it cannot establish bounded growth of the posterior ID index by itself. Existing point-timeline-only capacity uses supplied outcomes and is not an actual corrected forecasting end-to-end substitute.
- Verify private exact generated inventory hashes and strict point reader alongside a Torch/Ultralytics import tripwire. Export correctness is software evidence only; no speed, detector, calibration, long-match or live gate is established.

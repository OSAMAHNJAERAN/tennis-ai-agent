# Next implementation seam: corrected causal forecasts

Read-only mapping, 18 October 2026. This artifact implements no forecast feature, changes no production source, and does not qualify prediction accuracy. Source hashes and inspected symbol lines are in `receipt.json` beside this file. The complete master goal remains unfinished.

## Actual reusable predictor

`src/prediction/tennis.py:171` defines `TennisWinProbability`: an actual pure statistical tennis-rules calculator with session-local Beta serve-point estimates, game/set/match probabilities, analytic deuce/tie-break/advantage-set tails, and no deep model. `ServePointPrior` at line 21 defaults to Beta(2,2), explicitly an engineering prior. `observe_point` at line 182 accepts a stable point ID, actual server, winner, and causal timestamp. `snapshot` at line 199 takes a real `MatchState`, analysis revision, and optional source-playback timestamp. It returns `prior_only`, `uncalibrated`, `unavailable`, or `confirmed_result`; a completed match has a winner and no probability. Preserve those distinctions and `calibration_status=not_calibrated`.

Do not call the existing convenience `replay_predictions` (line 237) on an uploaded full match. It retains `history=[]` and `seen={}` for every point. The predictor itself retains `point_ids={}` (line 179), so merely streaming its snapshots does not fix duration growth. `PointLedger` also retains points/audit in Python; it is the manual compatibility implementation, not a bounded native input source.

The predictor's `_RuleProbabilities` caches belong to one snapshot, deliberately avoiding cross-match global retention. Preserve that ownership. No training or model import is required for the native forecast stage.

## Exact raw state, rather than presentation reconstruction

`PointTimeline.observe` (`src/scoring/point_timeline.py:265`) records accepted physical/rule transitions in `owner.rows('transitions')`. Each row has stable `uid`, `event_id`, global `frame`, normalized `timestamp_s`, `context` with exact PTS/time-base/source-frame tokens, `pre_state`, `post_state`, `outcome_type`, `outcome`, and point/attempt/rally UIDs. `owner.event(event_id)` is indexed (`src/utils/point_analysis_sequence.py:86`). Completed point rows have `status=COMPLETED`, `point_id`, `server_id`, `winner_id`, `anchor_event_id`, and `terminal_event_id`; `pre_point_state` is also presentation data.

These state dictionaries came from `MatchState.to_dict` (`src/scoring/match_state.py:183`). `player_N.points_raw` means ordinary counters OR tie-break counters depending on active mode, and does not retain both. Do not decode them as raw checkpoints or infer the historical tie-break first server from current server. The lossless codec is `encode_match_state` / `decode_match_state` (`match_state_codec.py:134,157`), retaining all 37 raw fields and typed completed sets.

The strongest existing seam is a fresh bounded rules replay of NEW corrected accepted inputs:

1. Start from a detached raw copy of `rebuilt.engine.initial_state`, not the original generation's initial score.
2. Iterate `rebuilt.engine.iter_rule_inputs()` (`scoring_engine.py:88`), closing its iterator on every exit. Pair with native accepted replay rows/transitions in order, checking counts, exact IDs/timestamps/fingerprints and each newly accepted input.
3. Decode each NEW accepted input with `TennisScoringEngine._decode_input` (line 122), and process it through a fresh `TennisScoringEngine` backed by a caller-owned `ScoringJournal`. This provides real raw pre/post `MatchState` at every event. These are newly corrected accepted line calls, not the old generation's recorded decisions. Do not recompute score by awarding every physical bounce or shot.
4. Compare replayed outcome/history/pre/post presentation records to the already accepted native transition; compare final `encode_match_state` exactly to `rebuilt.engine`'s raw final state. Keep rule revision zero, analysis revision `owner.revision`.
5. Only update the posterior once for a native COMPLETED point at its exact terminal transition, with the pre-award server. Cross-check the point row's server/winner and terminal event against the actual replay outcome. First faults, lets, postpoint hits/bounces, unknown events, and reviewed signals do not count as completed points by themselves. Never use the server after a game-ending award as that point's server.

`POINT_REVIEW_PENDING` freezes subsequent rules (`scoring_engine.py:189`); its snapshots must become unavailable immediately. A partial/unanchored point must not silently become a qualified posterior observation. Explicitly decide and test the conservative forecast-withholding policy when a rule outcome awards a point but the native point is not anchored/COMPLETED. Historical reviewed initial score may be used as score context, but cannot fabricate pre-clip serve-point sample counts.

## Proposed bounded API and ownership

Proposed new context (not yet implemented):

    with rebuild_corrected_forecasts(rebuilt, shot_analysis, scratch) as forecast:
        forecast.history  # owned SQLite-backed ordered sequence
        forecast.current  # one detached snapshot
        forecast.audit

Require the same analysis/source/base-generation/revision, an open sealed point owner after fresh shot attachment, zero scoring revision, and the base generation still current at entry and exit. A future publisher must CAS against that same base. No root alias or old forecast artifact is an input.

Add an explicit optional point-index seam to `TennisWinProbability` while preserving the default manual dictionary API. A caller-owned SQLite mapping supplies the operations actually used: membership, lookup, insertion, length, and truth. Store exact tuple `(server_id,winner_id,timestamp_seconds)` under a UNIQUE stable point UID. Identical duplicate is idempotent; different payload rejects. Preserve tuple comparison semantics after JSON decoding. Strictly validate primitive types to avoid True/1 or 1.0/1 aliases. Keep only two participant win/total counters, one last timestamp, and current snapshot in Python. Bound SQLite cache to 2 MiB, disable mmap, commit at most 256 writes, and close/remove only its own ephemeral files and sidecars on failures/cancellation.

Store forecast history in owned SQLite, including sequence, phase, current global frame/time, exact source PTS/time base, event/point UIDs, analysis/source/revision/base-generation bindings, model/prior version/hash, and `uses_points_through_event_id`. Enforce monotonically nondecreasing timestamps and stable equal-time ordering. Do not convert huge rational source times into the normalized playback clock. Keep a sample for the initial state and every accepted transition (or formally specify/test a point-boundary plus state-availability-change cadence). For a pre-outcome row, take the snapshot BEFORE processing/updating that event; an AFTER_EVENT row may use only outcomes through that event and must say so. Final confirmed-result rows are result displays, excluded from forecast accuracy metrics.

Avoid making the sealed point owner writable merely to store forecasts: use a separate forecast owner with the same revision bindings. Never set `history` to a materialized Python list in API/report/export paths. Use existing `ResultStorage.series` (`result_storage.py:224`) or a dedicated typed SQLite sequence, and `JsonArrayArtifact` (`json_array_artifact.py:9`) with `allow_nan=False` for fresh atomic exports. Preserve existing completed destinations if serialization fails. Reuse the complete generation publisher only after forecasts, exports, zones and fresh rendering all succeed; the existing eight-file point manifest intentionally does not cover a new forecast artifact.

## Calibration and available artifacts

There is no qualified prediction-weight artifact selected in production config and no forecast connection in Phase6. `ProductStore.serialize` (`src/api/product.py:140`) serves manual `replay_predictions` results; upload forecasts are explicitly unavailable (lines 152-155). `frontend/src/data/productApi.ts:15` and `WinProbability.tsx:6` support the existing four evidence states, but the component filters/materializes all supplied history before sampling 300 display points. A future native endpoint needs pagination or server-side bounded chart reduction; current browser thinning does not bound incoming payload/RAM.

The retained statistical fit exists at `data/prediction/livetennisapi_2026_06_snapshot_import_v3_development_fit.json`: logistic intercept 0.15629601140499222 and slope 1.34932608812652, fitted on 57 development groups/6,330 snapshots. Its sidecar digest is verified in the receipt. Its status is `unqualified_same_publisher_engineering_only`. The frozen direct-snapshot protocol permits only known regular BO3 grass-event families and explicitly uses fresh neutral Beta(2,2) with ZERO point updates. Its outcomes are same-publisher terminal scores, not independently reviewed video outcomes. A suffix calibration split was assessed separately; there was no independent production promotion.

Do not automatically apply that calibrator to the proposed posterior-updating video forecaster: the source/feature/model scope differs. Do not refit or select parameters on reserved test. The safe initial implementation is real raw statistical output labeled not calibrated, while independent qualification remains required. A separately exposed private neutral-score engineering comparison would need explicit scope/version bindings and still cannot claim qualification.

`src/prediction/evaluation.py:112` is the existing independent point-ledger evaluator. It requires chronological source-group splits, independently reviewed complete outcomes, continuous confirmed human/manual points, and explicit initial rules/server. It uses equal match weight, grouped bootstrap and Brier/log loss/reliability. Frozen preliminary gates at lines 175-192: at least 20 test matches; 95% lower bound of Brier improvement over neutral score-only baseline greater than zero; log loss no worse; supported bins (at least five matches) reliability gap at most 0.10. These gates remain unqualified and cannot be satisfied by software fixtures or same-publisher terminal targets.

## Decisive next tests

1. Extend the existing original legal second-serve versus corrected AD double-fault fixture: the initial corrected score changes the prior score context, one terminal double fault increments the receiver's posterior exactly once, and the later receiver hit contributes zero. Earlier legal-rally forecasts are not copied.
2. Prefix causality: extending a source with future points/results leaves every earlier forecast payload unchanged (compare excluding revision/audit fields only where intentionally different). Changing one later reviewed outcome affects that point's AFTER_EVENT and subsequent rows, never its BEFORE_EVENT row.
3. Let and first-fault controls; duplicate retry; conflicting duplicate; scoring game/server rollover; ordinary deuce, seven-/ten-point tie-break and advantage-set raw-state equivalence; final result has no probability.
4. Unknown initial state, consequential gap, unanchored partial point, authority off, stale base generation, mismatched source/revision, malformed/bool identifiers, and failing serializer/provider all withhold/reject without publication; previous generation remains readable.
5. Fresh independent comparisons against existing small `replay_predictions` fixtures demonstrate identical mathematics on an identical confirmed prefix, while the new path keeps histories/deduplication on disk.
6. Growing COMPLETED point/history/dedup cardinality with duration—not only more frames and two points—demonstrates stable retained Python memory; report numerical/SQLite native memory exclusions explicitly. Do not repeat the already measured shot-only capacity as prediction evidence.

Next actual work is implementing this bounded forecast stage and its coupled corrections tests, then fresh exports/zones/rendering and owner-only complete-generation correction workflow. This mapping is not a substitute for those implementations or the independent/long/live gates.

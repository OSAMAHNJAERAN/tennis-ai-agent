# Independent read-only review of the native point timeline

Four small probes executed the current actual scoring/resolver definitions and current timeline methods. They retained `receipt.json` and did not edit production source or tests. Bound sources were unchanged during the probe. Torch, Ultralytics, NumPy, OpenCV and PIL imports were refused before loading. All evidence is synthetic; no perception accuracy, long-match capacity or production qualification is inferred.

## P1 — active rally accepts a new identity/calibration revision without re-anchoring

`src/scoring/point_timeline.py:216` checks the elapsed source interval between successive events, but only `service_target()` at line 238 compares a binding's continuity key to the opening attempt. After a legal service bounce has moved the resolver into `LIVE`, return contacts and rally bounces can use a different current identity/calibration revision without a review transition.

The probe supplies serve/contact 1, legal service bounce 2, return contact 3 and out bounce 4. From contact 3 onward, the current confirmed identity audit and binding both advance identity revision 0 → 1, and the binding changes calibration revision. Source PTS, camera and anchor stay continuous. All per-event binding checks pass. The real engine completes a point with `score_confirmed=True`; transition 3 remains `PLAYER_2_HIT`. No explicit correction/re-anchoring revision was supplied.

The active point/rally needs an explicit continuity policy for identity epoch/revision, participant-map reference, calibration revision, geometry qualification and anchor. A change must withhold the old point's continuation or enter an explicit reviewed new revision; it cannot silently join new proof to old rally membership. The same continuity policy must apply to contacts and rally bounces, not only the first service bounce.

## P2 — conflicting duplicate event payload is accepted as an idempotent retry

`src/scoring/point_timeline.py:398` compares an existing event's frame, physical kind and a few source fields. It omits the physical player/role, event timestamp/confidence, reviewed signal evidence and the participant-binding payload.

The probe first processes event ID 1 as player 1's serve at timestamp .1/confidence .99. It retries the same ID/frame/kind/source PTS as player 2's serve at timestamp .9/confidence .1, with a current confirmed binding for the other participant. The helper returns cached `SERVE_IN_PLAY` and raises no conflict.

Persist and compare an immutable canonical input fingerprint that includes every scoring-relevant field and reviewed context/evidence reference. Exact retries can return the cached outcome; changed input must require a new revision or be rejected. An event ID must not hide a changed actor or reviewed ruling.

## P2 — direct native handoff accepts a context belonging to another owner/source

`candidate_event_context()` validates its provider against the caller-supplied analysis/source arguments, while `PointTimeline.scoring_input()` / `process_timeline_event()` never compare that context to `timeline.owner.analysis_id` and `source_fingerprint`.

The probe opens an owner for `different-analysis` / `different-source`, then supplies an otherwise valid context bound to `fixture` / `synthetic-source-v1`. The real engine returns `SERVE_IN_PLAY` and the ledger opens an official attempt. The current Phase6 adapter passes matching arguments, so this is a missing invariant at the new reusable native handoff rather than a demonstrated cross-source mix in the current Phase6 caller.

Enforce the owner/context analysis, source fingerprint and revision binding at the native processing boundary itself. Reject or withhold a context from another owner. Tests must use consistent owner fingerprints rather than relying on the optional empty default.

## P2 — same-frame reviewed signals can be emitted in reverse declared time order

`merge_rule_inputs()` at line 141 chooses a reviewed signal only when its frame is strictly earlier than the next physical event. It does not resolve or reject same-frame ordering.

For serve event 1 at frame 0/time .1, physical bounce event 3 at frame 1/time .3 and explicitly reviewed let event 2 at frame 1/time .2, the merged order is `[1, 3, 2]`, with declared times `[.1, .3, .2]`. The real scoring engine can then reject source-time reversal; processing a legal bounce before its earlier let also changes the point phase.

Use an explicit source-bound order for reviewed signals and physical events, or reject ambiguous same-frame combinations before mutating the engine. Do not infer subframe physical order from an arbitrary display-FPS time. Tests should cover both explicitly ordered same-frame signals and unresolved ties.

The previously acknowledged scoring-history/correction memory growth and whole-analysis revision atomicity remain outside these findings. This review does not claim those are fixed. The root's separate `finish()` performance work was not modified.

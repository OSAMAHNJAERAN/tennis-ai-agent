# Phase 6.4 False-Negative Physical-Event Audit

## Purpose and scope

This is the required event-by-event false-negative audit for the Phase 6.4
semantic physical-event recovery. It covers all 40 manual diagnostic GT events
in `video_08`–`video_10`. The clips are already-consumed development data and
the results are not qualification or generalization evidence.

Matching uses global-nearest one-to-one assignment within each video. A
prediction is eligible when it falls inside the manual `[frame_min, frame_max]`
interval expanded by 0.2 seconds at native FPS. All available clips are 30 FPS,
so the effective expansion is six frames. Event-type equality is required for
final-event matches but not for untyped candidate recall.

GT timestamps in the forensic table are explicitly derived as
`frame_best / native_fps`; they were not present in the GT file. Unavailable
pose/camera evidence is not fabricated.

## Before and after

| Detector state | TP | FP | FN | Candidate recall | Final recall |
|---|---:|---:|---:|---:|---:|
| Integrated pre-recovery A | 4 | 29 | 36 | 22/40 = 0.5500 | 4/40 = 0.1000 |
| Current recovery H | 12 | 67 | 28 | 25/40 = 0.6250 | 12/40 = 0.3000 |

H recovered eight prior FNs, but the candidate layer still misses 15 GT events
under the evaluator window and the final detector misses 28. The precision cost
is material: FP rose by 38.

## Pre-recovery forensic taxonomy

The legacy detector did not persist candidate IDs, refined timestamps,
verification scores, or rejection traces. Its 36-FN taxonomy is the frozen
result of the one-time artifact lineage review in
`phase6_4_false_negative_audit_pre_recovery.json`:

| Root cause | Count | Percentage |
|---|---:|---:|
| `CANDIDATE_OUTSIDE_MATCH_WINDOW` | 17 | 47.22% |
| `WRONG_EVENT_TYPE` | 7 | 19.44% |
| `PREDICTED_OR_INTERPOLATED_STATE_REJECTION` | 5 | 13.89% |
| `PLAYER_REACH_OR_RACKET_REGION_REJECTION` | 3 | 8.33% |
| `DEAD_BALL_REJECTION` | 3 | 8.33% |
| `EVENT_TIME_REFINEMENT_OUTSIDE_MATCH_WINDOW` | 1 | 2.78% |
| Total | 36 | 100.00% |

This explains the 4-TP integrated failure: half of all GT events were lost at
candidate generation, then strict provenance/reach/serve semantics and a
scoring-derived dead-ball filter removed or mistyped many surviving contacts.

## Current H false-negative taxonomy

The staged detector now supplies candidate IDs and explicit rejection traces.
The current 28-FN taxonomy is:

| Root cause | Count | Percentage |
|---|---:|---:|
| `TRACKING_GAP` | 12 | 42.86% |
| `WRONG_EVENT_TYPE` | 9 | 32.14% |
| `PLAYER_CONTACT_MULTI_CUE_REJECTION` | 4 | 14.29% |
| `CANDIDATE_OUTSIDE_MATCH_WINDOW` | 2 | 7.14% |
| `BOUNCE_CONTACT_SIGNATURE_REJECTION` | 1 | 3.57% |
| Total | 28 | 100.00% |

### By rejection stage

| Rejection stage | Count | Interpretation |
|---|---:|---|
| Raw ball trajectory | 12 | No observed or defensible short-gap ball support in the GT window |
| Event candidate generation | 2 | A trajectory exists, but no candidate matches the GT interval+tolerance |
| Physics verification | 5 | Four player-contact multi-cue failures and one bounce-signature failure |
| Event-type classification | 9 | A nearby physical event exists but has the wrong external semantic type |

### By GT class and cause

| GT class | Candidate outside | Tracking gap | Player multi-cue | Bounce signature | Wrong type | Total FN |
|---|---:|---:|---:|---:|---:|---:|
| SERVE_CONTACT | 0 | 0 | 0 | 0 | 3 | 3 |
| PLAYER_HIT | 0 | 6 | 1 | 1 | 3 | 11 |
| BOUNCE | 2 | 6 | 3 | 0 | 3 | 14 |
| Total | 2 | 12 | 4 | 1 | 9 | 28 |

### By video

| Video | Support | Candidate matches | Final matches | False negatives | Main causes |
|---|---:|---:|---:|---:|---|
| video_08, 720p | 10 | 10 | 5 | 5 | player multi-cue 2, wrong type 2, bounce signature 1 |
| video_09, 720p | 12 | 11 | 5 | 7 | wrong type 4, player multi-cue 2, candidate timing 1 |
| video_10, 1080p | 18 | 4 | 2 | 16 | tracking gap 12, wrong type 3, candidate timing 1 |

The video_10 result is decisive: no final-threshold adjustment can recover the
12 contacts for which the input trajectory has no observed/short-gap-supported
ball location. Generating long interpolated paths would improve apparent recall
by inventing evidence and is explicitly rejected.

## Candidate generation versus verification

| Stage | Serve | Hit | Bounce | Overall |
|---|---:|---:|---:|---:|
| H candidates | 3/3 = 1.0000 | 11/17 = 0.6471 | 11/20 = 0.5500 | 25/40 = 0.6250 |
| H final events | 0/3 = 0.0000 | 6/17 = 0.3529 | 6/20 = 0.3000 | 12/40 = 0.3000 |

Serve candidate recall of 1.0 proves the candidate generator sees a nearby
kinematic event for every serve; all three are lost by semantic typing. Serve
must therefore be fixed after physical player-contact preservation, not by
reintroducing first-event or early-frame rules.

Bounce candidate recall remains 0.55, and only six bounces survive typing. The
current contact test intentionally requires a down-to-up local image-y rebound,
trajectory change, continuity/provenance support, and player-distance context
without requiring every cue simultaneously. The remaining bounce losses divide
between tracking, candidate timing, player-contact confusion, and wrong typing.

Hit candidate recall improved to 0.6471, but six of 11 hit FNs are tracking gaps
and four surviving physical candidates are rejected or typed incorrectly.

Candidate recall and final recall are separate global matchings. Local time
refinement can move a discovery candidate into another GT window, so simple
subtraction is not a per-event causal lineage. The 40-row forensic table, stable
candidate IDs, discovery/refined time pairs, and rejection traces are the
authoritative per-event evidence.

## Provenance around outcomes

| Outcome | DETECTED | TRACKED | PREDICTED | INTERPOLATED | MISSING | Total |
|---|---:|---:|---:|---:|---:|---:|
| True positive prediction | 8 | 3 | 1 | 0 | 0 | 12 |
| False positive prediction | 37 | 10 | 15 | 5 | 0 | 67 |
| False negative at GT frame | 9 | 1 | 0 | 4 | 14 | 28 |

Provenance weighting is therefore necessary but insufficient. Twenty FPs are
supported by predicted/interpolated states, while 47 FPs are still
`DETECTED`/`TRACKED`. Conversely, categorical rejection of all non-detected
states would discard a real predicted-state TP and would not solve the detected
post-rally noise.

## False-positive counterweight

Recovery must not optimize recall alone. H's 67 false positives are:

| Cause | Count | Percentage |
|---|---:|---:|
| `POST_RALLY_BALL_MOTION` | 43 | 64.18% |
| `TIMING_DRIFT_OR_DUPLICATE` | 11 | 16.42% |
| `WRONG_EVENT_TYPE` | 7 | 10.45% |
| `UNMATCHED_KINEMATIC_MOTION` | 6 | 8.96% |

The vision-side low-energy dead-ball gate produced no E→F change because none
of the provisional false events met both low-speed and low-kinematic criteria.
The remaining post-rally artifacts look kinematically active to the corrupted
tracker, so scoring state must not be used to delete them and a broader
observable activity model is still required.

## Forensic record completeness

Each current record contains:

```text
video/GT identity and derived GT time
nearest and matched candidate frame/time/offset
candidate score and discovery/refined times
ball state/confidence
pre/post velocity vectors and normalized kinematics
direction, acceleration, curvature, rebound and continuity evidence
time-local normalized distances to both players
selected player and bbox scale
internal physical type and external candidate event type
verification scores, stage-pass flags, rejection stage/reason
final emission status
```

`pose_support` is null because compatible pose evidence was unavailable and the
feature is disabled. Camera motion is recorded as
`UNAVAILABLE_NO_PER_FRAME_STATIC_SCENE_TRANSFORM`, not fabricated as zero
motion. The static homography is not used to claim a stabilized airborne
trajectory.

The prior diagnostic review manifest includes contact sheets for the 20
serves/hits but not for the 20 bounce rows. The GT protocol asserts raw manual
inspection, but future diagnostic work should add prediction-free bounce review
packs before treating bounce-specific visual explanations as independently
auditable.

## Root blocker and stop condition

The remaining blocker is now more precise than “the verifier is strict”:

```text
The 1080p trajectory is absent at many real contacts, while later tracker
motion appears kinematically active and creates 43 post-rally false events.
```

The candidate target (`>=0.90`) and final event readiness targets remain unmet.
Serve is still completely broken and bounce F1 is only `0.1935`.

```text
PHASE 6.4 EVENT RECOVERY: FAIL
READY FOR PRODUCTION/EVALUATOR FREEZE: NO
```

No `video_11+` is acquired, no freeze is created, and Phase 7 is not started.

## Reproduction and artifacts

```powershell
python scripts/create_phase6_4_pre_recovery_audit.py
python scripts/evaluate_phase6_4_event_recovery.py --variant all --input-root outputs/phase6_4_qualification/cross_match_diagnostic_final
```

```text
artifacts/validation/phase6_4_false_negative_audit_pre_recovery.json
artifacts/validation/phase6_4_false_negative_audit.json
artifacts/validation/phase6_4_event_forensic_table.json
artifacts/validation/phase6_4_event_stage_metrics.json
artifacts/validation/phase6_4_event_recovery_results.json
data/benchmarks/cross_match_final_holdout/ground_truth_events.json
outputs/phase6_4_qualification/cross_match_diagnostic_final/
```

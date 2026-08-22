# Phase 6.4 Semantic Physical-Event Recovery Audit

## Status and scientific boundary

This audit continues Phase 6.4. It is not Phase 6.5, a production freeze, a
pristine qualification, or Phase 7 work. Every result in this document comes
from the already-consumed `video_08`–`video_10` cross-match development set and
is therefore **development diagnostic evidence only**.

```text
Scientific split: CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED
Qualification evidence: false
Pristine video_11+ consumed: no
Production freeze created: no
Evaluator freeze created: no
Readiness verdict: FAIL
```

The experiment reruns event reasoning over preserved `trajectories.json` and
`detections.json`. It does not rerun ball/player inference, mutate scoring, or
use GT inside the detector. Ground truth enters only the fixed evaluator and
forensic reporting after predictions exist.

## Metric provenance: three results that must not be conflated

| Result | Evaluator/provenance | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Historical corrected report | Requested historical before-state; predates the current interval-aware fields | 11 | 113 | 29 | 0.0887 | 0.2750 | 0.1341 |
| Legacy preserved-output rescore | Current fixed evaluator applied to the older legacy output root | 13 | 111 | 27 | 0.1048 | 0.3250 | 0.1585 |
| Integrated pre-recovery A | Fixed evaluator applied to `cross_match_diagnostic_final` | 4 | 29 | 36 | 0.1212 | 0.1000 | 0.1096 |
| Current recovery H | Fixed evaluator applied to the staged detector over the same preserved inputs | 12 | 67 | 28 | 0.1519 | 0.3000 | 0.2017 |

The historical `11/113/29` line is retained because it is the requested
published before-state. The `13/111/27` rescore is shown only to make the later
interval-aware re-evaluation transparent. Variant A is the correct direct
comparison for this recovery implementation.

Relative to A, H recovered eight TPs and removed eight FNs, but added 38 FPs.
It improved F1 from `0.1096` to `0.2017`; it did not solve the precision/recall
balance. This is a measured recovery, not diagnostic readiness.

## Why the previous integrated result collapsed from 11 TP to 4 TP

The pre-recovery forensic audit found two equally large losses:

```text
Candidate generation loss: 18 / 40 GT events
Verification or semantic loss after a candidate: 18 / 40 GT events
Final false negatives: 36 / 40 GT events
```

The frozen pre-recovery FN lineage was:

| Cause | Count | Share of 36 FNs |
|---|---:|---:|
| Candidate outside match window | 17 | 47.2% |
| Wrong event type | 7 | 19.4% |
| Predicted/interpolated-state rejection | 5 | 13.9% |
| Player reach or racket-region rejection | 3 | 8.3% |
| Dead-ball rejection | 3 | 8.3% |
| Refined time outside match window | 1 | 2.8% |

The mechanism was not merely “thresholds too strict.” It combined:

- candidate recall of only `0.5500` overall;
- raw-pixel kinematic thresholds and a broad/fixed reach model;
- categorical rejection of useful short-gap provenance;
- a serve branch that depended on context the candidate generator never set;
- player alignment, physical verification, semantic typing, and suppression
  being combined without candidate IDs or rejection traces; and
- scoring outcomes deleting physical observations from `match_events.json`.

The earlier batch evaluator also reused one `Phase6Pipeline` object for all
three videos. Its `TennisScoringEngine` retained event IDs and match state
across video boundaries, producing duplicate-ID outcomes in video_09 and
video_10. Those scoring outcomes are not defensible vision-side activity
evidence. The recovery experiment bypasses scoring, and the pipeline now keeps
verified physical observations authoritative even when scoring marks an event
dead for downstream score/shot consumers. Scorer lifecycle reset remains a
separate consumer-integrity concern; scoring was not redesigned here.

## Recovered event architecture

The detector now exposes this explicit sequence:

```text
preserved ball trajectory
  -> untyped EventCandidate
  -> normalized kinematic feature record
  -> physical verification
       BALL_CONTACT_WITH_PLAYER
       BALL_CONTACT_WITH_COURT
       UNKNOWN_PHYSICAL_EVENT
  -> time-local player attribution
  -> external semantic type
       SERVE_CONTACT / PLAYER_n_HIT / BOUNCE
  -> vision-side low-energy activity filter
  -> time-based duplicate suppression
  -> authoritative TennisEvent
```

Every candidate has a stable candidate ID plus discovery frame/time. A separate
refined frame/time is selected locally; the authoritative timestamp now follows
the refined frame rather than retaining the stale discovery time. Each trace
records feature values, physical and semantic scores, stage-pass flags,
rejection stage/reason, and final emission status. Unavailable evidence remains
`null` or an explicit unavailable state.

Internal physical types are additive. External Analytics Contract V1 event
names remain unchanged. `event_verification_trace.json` is an additive audit
artifact and is not an authoritative consumer event stream.

## Feature definitions and configuration

Event kinematics are normalized by the frame diagonal
`sqrt(width_px^2 + height_px^2)`:

- normalized speed = mean pre/post image speed divided by frame diagonal;
- normalized velocity change = `|v_post - v_pre| / frame_diagonal`;
- normalized acceleration = velocity change per second divided by frame
  diagonal;
- player distance = point-to-bbox distance divided by the time-local player
  bbox height; and
- curvature is an image-plane local derivative normalized for frame scale.

This compensation is limited to event features. The upstream ball tracker still
contains pixel/frame gating parameters and is the main observed 1080p failure.
No claim is made that an airborne ball becomes a metric court-plane point.

Canonical time parameters are seconds, with native trajectory timestamps used
for feature windows, temporal player proximity, contact-time refinement,
departure evidence, serve evidence, candidate spacing, and final suppression.
The active diagnostic values include:

```text
feature half-window:             0.067 s
candidate minimum interval:      0.120 s
player proximity window:         0.140 s
event refinement window:         0.120 s
departure window:                0.120 s
serve evidence window:           0.400 s
final event minimum interval:    0.180 s
```

All three diagnostic videos happen to be 30 FPS, so these media do not
empirically demonstrate 24/30/60 FPS transfer. FPS independence is an
implementation and unit-regression property here, not a cross-source result.

Camera-motion compensation is independently configurable, but no per-frame
court/static-scene offsets exist in the preserved artifacts. Variant G therefore
records `UNAVAILABLE_NO_PER_FRAME_STATIC_SCENE_TRANSFORM` and uses raw image
coordinates. Enabling G is an honest measured no-op. A static first-frame
homography is deliberately not presented as camera-motion compensation.

Pose support is also configurable but disabled because the recovery inputs do
not persist suitable event-time pose evidence. No pose value was fabricated.

## GT-event survival by stage

Matching is global-nearest one-to-one within each video. The GT uncertainty
interval is expanded by 0.2 seconds at native FPS. Candidate/physics/player
rows are type-agnostic; semantic and final rows require the correct event type.

| Stage | Serve recall | Hit recall | Bounce recall | Overall recall | Timing MAE |
|---|---:|---:|---:|---:|---:|
| Raw candidate generator | 3/3 = 1.0000 | 11/17 = 0.6471 | 11/20 = 0.5500 | 25/40 = 0.6250 | 2.76 f / 92.0 ms |
| Physics verification | 3/3 = 1.0000 | 9/17 = 0.5294 | 8/20 = 0.4000 | 20/40 = 0.5000 | 3.40 f / 113.3 ms |
| Player attribution | 3/3 = 1.0000 | 9/17 = 0.5294 | 8/20 = 0.4000 | 20/40 = 0.5000 | 3.40 f / 113.3 ms |
| Event-type classification | 0/3 = 0.0000 | 7/17 = 0.4118 | 6/20 = 0.3000 | 13/40 = 0.3250 | 3.62 f / 120.5 ms |
| Temporal suppression | 0/3 = 0.0000 | 6/17 = 0.3529 | 6/20 = 0.3000 | 12/40 = 0.3000 | 4.42 f / 147.2 ms |
| Final authoritative events | 0/3 = 0.0000 | 6/17 = 0.3529 | 6/20 = 0.3000 | 12/40 = 0.3000 | 4.42 f / 147.2 ms |

Stage flags are cumulative after the raw-candidate row. The detailed
candidate-ID lineage is in the forensic table; aggregate nearest matching can
still assign a nearby refined event differently from its discovery candidate,
so the table should be used for individual root-cause review.

## Current diagnostic metrics

| Class | TP | FP | FN | Precision | Recall | F1 | Timing MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SERVE_CONTACT | 0 | 4 | 3 | 0.0000 | 0.0000 | 0.0000 | N/A |
| PLAYER_HIT | 6 | 27 | 11 | 0.1818 | 0.3529 | 0.2400 | 3.67 f / 122.2 ms |
| BOUNCE | 6 | 36 | 14 | 0.1429 | 0.3000 | 0.1935 | 5.17 f / 172.2 ms |
| OVERALL | 12 | 67 | 28 | 0.1519 | 0.3000 | 0.2017 | 4.42 f / 147.2 ms |

Candidate recall improved from `0.5500` to `0.6250`, led by hit recall moving
from `0.4706` to `0.6471`. Serve candidate recall remained `1.0000`, proving
serve contacts reach the candidate layer, but serve semantic recall remains
zero. Bounce candidate recall remained `0.5500`.

False positives are dominated by post-rally motion:

| FP cause | Count | Share of 67 FPs |
|---|---:|---:|
| Post-rally ball motion | 43 | 64.2% |
| Timing drift or duplicate | 11 | 16.4% |
| Wrong event type | 7 | 10.4% |
| Unmatched kinematic motion | 6 | 9.0% |

The low-energy dead-ball gate had no eligible provisional events and therefore
did not change E. It does not yet suppress the high-kinematic false trajectory
segments responsible for most post-rally FPs.

## Resolution, provenance, and grouped checks

| Resolution group | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 720p: video_08 + video_09 | 10 | 27 | 12 | 0.2703 | 0.4545 | 0.3390 |
| 1080p: video_10 | 2 | 40 | 16 | 0.0476 | 0.1111 | 0.0667 |

At final outcomes, ball-state distributions are:

| Outcome | DETECTED | TRACKED | PREDICTED | INTERPOLATED | MISSING |
|---|---:|---:|---:|---:|---:|
| TP | 8 | 3 | 1 | 0 | 0 |
| FP | 37 | 10 | 15 | 5 | 0 |
| FN at GT frame | 9 | 1 | 0 | 4 | 14 |

Half of current FNs are `MISSING` at the GT frame. The FN audit identifies 12
explicit tracking gaps, all in video_10. Event verification cannot recover a
physical contact from an absent trajectory without inventing a long path, which
this work correctly refuses to do.

Grouped leave-one-video-out reporting uses the same H configuration in every
group; thresholds were **not** refit using only the two development groups.
These are grouped slices, not an unbiased nested cross-validation claim:

| Validation group | Development groups | Resolution | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| video_08 | video_09, video_10 | 720p | 0.3125 | 0.5000 | 0.3846 |
| video_09 | video_08, video_10 | 720p | 0.2381 | 0.4167 | 0.3030 |
| video_10 | video_08, video_09 | 1080p | 0.0476 | 0.1111 | 0.0667 |

Only three videos, one serve per group, one 1080p source, and a common US Open
broadcast setting are available. Resolution is confounded with match, year,
and source. All thresholds have already seen these diagnostic clips.

## Shot diagnostic boundary

The H experiment is event-only and does not rerun production shot linking.
The latest measured integrated production-conditional shot result remains
macro F1 `0.0000`, UNKNOWN rate `1.0000`, and coverage `0.0000` on seven matched
hits. It must not be silently replaced by an oracle result.

The separate oracle-event diagnostic supplies only GT hit frame, player, and
event type to the existing classifier. It is **NOT PRODUCTION**, **NOT
END-TO-END**, and **GT-ASSISTED DIAGNOSTIC ONLY**:

```text
Records: 20
FOREHAND F1: 0.0000 (support 10)
BACKHAND F1: 0.0000 (support 7)
SERVE F1: 1.0000 (support 3; event-type passthrough)
Macro F1: 0.3333
UNKNOWN rate: 0.8500
Coverage: 0.1500
```

Serve is not an independent mechanics result because permitted oracle input
already says `SERVE_CONTACT`. All 17 forehand/backhand rows abstain. This makes
stroke classification a documented secondary future blocker, while physical
event verification remains the focus of this iteration.

## Performance and contract regression

H event analysis took about `0.51 s` for 2,672 exported frames (about 5,251
offline event-analysis FPS). This excludes decode, detectors, tracker, pose,
rendering, and serialization and is not end-to-end throughput.

The last full diagnostic pipeline measurements remain:

| Video | Resolution | End-to-end FPS | Status |
|---|---:|---:|---|
| video_08 | 1280x720 | 26.663 | pre-recovery full run |
| video_09 | 1280x720 | 28.011 | pre-recovery full run |
| video_10 | 1920x1080 | 8.956 | pre-recovery full run |

No full H inference rerun was performed, so an end-to-end performance-regression
claim would be unsupported. The event stage itself is not the measured
bottleneck; the earlier video_10 ball-proposal stage was about 84.14 seconds.

External Analytics Contract V1 event names and nullable/UNKNOWN behavior are
preserved. The physical intermediate enum and audit trace are additive. Scoring
and line-call algorithms were not redesigned.

## Readiness gate and stop decision

| Diagnostic readiness check | Direction | H result | Status |
|---|---:|---:|---|
| Candidate recall overall | >= 0.90 | 0.6250 | FAIL |
| Overall precision | >= 0.70 | 0.1519 | FAIL |
| Overall recall | >= 0.75 | 0.3000 | FAIL |
| Overall F1 | >= 0.70 | 0.2017 | FAIL |
| PLAYER_HIT recall | >= 0.80 | 0.3529 | FAIL |
| Serve no longer broken | required | 0/3 TP | FAIL |
| Bounce no longer nearly broken | required | recall 0.3000, F1 0.1935 | FAIL |

```text
PHASE 6.4 EVENT RECOVERY: FAIL
READY FOR PRODUCTION/EVALUATOR FREEZE: NO
PRIMARY REMAINING BLOCKER: the 1080p ball trajectory is absent across many real contacts and surviving post-rally tracker motion still generates high-kinematic false events
```

No production/evaluator freeze is created, no `video_11+` is acquired or run,
and Phase 7 remains blocked.

## Reproduction commands and authoritative paths

Run from the repository root with the project Python environment:

```powershell
python scripts/create_phase6_4_pre_recovery_audit.py
python scripts/evaluate_phase6_4_event_recovery.py --variant all --input-root outputs/phase6_4_qualification/cross_match_diagnostic_final
python scripts/evaluate_phase6_4_oracle_shots.py --diagnostic-output-root outputs/phase6_4_qualification/cross_match_diagnostic_final
python -m pytest
```

The first three commands consume existing diagnostic artifacts and do not run
raw-video inference.

Authoritative evidence:

```text
data/benchmarks/cross_match_final_holdout/ground_truth_events.json
data/benchmarks/cross_match_final_holdout/ground_truth_shots.json
data/benchmarks/cross_match_final_holdout/videos.json
outputs/phase6_4_qualification/cross_match_diagnostic_final/video_08/
outputs/phase6_4_qualification/cross_match_diagnostic_final/video_09/
outputs/phase6_4_qualification/cross_match_diagnostic_final/video_10/
artifacts/validation/phase6_4_false_negative_audit_pre_recovery.json
artifacts/validation/phase6_4_event_recovery_results.json
artifacts/validation/phase6_4_event_stage_metrics.json
artifacts/validation/phase6_4_event_forensic_table.json
artifacts/validation/phase6_4_false_negative_audit.json
artifacts/validation/phase6_4_oracle_shot_diagnostic.json
```

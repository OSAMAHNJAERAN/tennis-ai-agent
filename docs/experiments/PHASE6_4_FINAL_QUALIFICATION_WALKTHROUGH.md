# PHASE 6.4 FINAL QUALIFICATION WALKTHROUGH

This is the authoritative close-out for the Phase 6.4 final qualification
correction. It reports a failed readiness gate honestly. The work stopped
before production/evaluator freeze and before acquiring `video_11+` because
the corrected diagnostic system is not ready for a scientifically useful
pristine attempt.

## A. Git State

```text
Starting SHA: 3ae90c909d81a5c4afade61cf6e1edd5e5339027
Diagnostic refinement SHA: 415af410b4d818f7a35bfcf85231e7353cdb8dce (core); bd42421d4a055da626259e62621d02ce76b2dfca (contract-integrity follow-up)
Production freeze SHA: NOT CREATED — diagnostic readiness failed
Evaluator freeze SHA: NOT CREATED — diagnostic readiness failed
GT freeze SHA: NOT CREATED — pristine acquisition stage not reached
Final SHA: recorded in the final Codex handoff (this file is part of that commit)
Branch: codex/phase6-4-final-correction
Working tree: user-owned pre-existing frontend/theme/.gitignore changes preserved and unstaged
Remote status: origin/master = 644808f2130ece14dabd2a502db5ff4bc78eeea6; starting SHA is a legitimate local descendant
```

No reset, overwrite, or destructive checkout was used. Git LFS tracking for
models and media remained intact.

## B. Baseline Reproduction

The baseline below is a corrected re-score of preserved historical artifacts,
not pristine qualification evidence.

```text
Tests: historical 107/107; corrected branch final 120/120
Event Precision: 0.0887
Event Recall: 0.2750
Event F1: 0.1341
Conditional Shot Macro F1: 0.1667
End-to-End Shot F1: 0.0000
720p FPS: video_08 26.64; video_09 25.73
1080p FPS: video_10 8.57
RAM: historical measured ~1.85 GB; not re-profiled in this correction
VRAM: not recorded by the historical runner
```

The historical hit-only report (precision 0.3902, recall 0.8000, F1 0.5246,
conditional macro F1 0.1783, end-to-end F1 0.0656) is retained for provenance
but superseded because it did not evaluate all physical event classes under the
semantics its report claimed.

## C. Scientific Problems Found

### Problem 1 — hardcoded serve/frame logic

```text
Problem: first/early detections were coerced into serves and early bounces were treated as serve context.
Evidence: event_detector used idx == 0 and frame <= 60; shot_linker used first hit <= 75; pipelines used frame < 81/100 and a frame 81–95 overlay.
Root cause: behavior was encoded from one short reference clip rather than tennis state/evidence.
Fix: removed absolute frame checks; serve requires explicit verified context; overlays and scoring use state-machine outcomes.
Verification: regression tests cover first event, mid-rally starts, delayed contact, and pre-serve motion.
```

### Problem 2 — candidate and authoritative events were conflated

```text
Problem: geometric inflections immediately became semantic match events.
Evidence: preserved artifacts contained 113 corrected false positives, 72 of them bounce predictions.
Root cause: no explicit high-recall candidate object and multi-cue verification boundary.
Fix: added EventCandidate records, separate candidate export, player-scale/time-local verification, continuity/inversion cues, and authoritative-event export.
Verification: post-correction false positives fell from 113 to 29, and candidate/authoritative behavior is regression tested.
```

### Problem 3 — player identity encoded handedness/orientation

```text
Problem: both players defaulted to right-handed and player ID stood in for court orientation.
Evidence: classifier and pose feature paths inferred geometry from ID 1/2 without explicit metadata.
Root cause: identity, handedness, and court end were incorrectly coupled.
Fix: explicit handedness and court-orientation maps; missing metadata safely abstains to UNKNOWN.
Verification: tests cover unknown handedness and left-handed inversion.
```

### Problem 4 — FPS/time assumptions

```text
Problem: rally duration used /30 and evaluator timing used 33.33 ms/frame with a fixed frame tolerance.
Evidence: source inspection of rally_analyzer and the historical evaluator.
Root cause: a 30 FPS development clip was treated as universal.
Fix: native FPS is required where timestamps are unavailable; evaluator uses 0.2 s converted with actual FPS; rally bounds use frame/FPS time.
Verification: 24/30/60 FPS equivalence, tolerance, millisecond, and rally-bound tests.
```

### Problem 5 — evaluator scope and matching

```text
Problem: the evaluator compared shot records with shot GT but called the result overall physical event detection; duplicate and cross-video matching semantics were unsafe.
Evidence: corrected preserved-artifact score changed the historical interpretation to TP=11, FP=113, FN=29.
Root cause: hit-only input, greedy/local assumptions, hardcoded time conversion, and no semantic split guard.
Fix: global nearest one-to-one same-type matching within video, duplicate predictions as FP, all-event per-class tables, and separate conditional/end-to-end metrics.
Verification: evaluator unit tests and forensic audit.
```

### Problem 6 — GT leakage risk and contradictory documentation

```text
Problem: video_08–10 were still described in places as a final holdout after their outputs had been inspected and used for engineering.
Evidence: git/docs/data forensics and the existing post-freeze audit.
Root cause: intended split role was not reclassified when scientific status changed.
Fix: authoritative registry, manifests, protocol, plan, results, and audits now label them CROSS_MATCH_DIAGNOSTIC with qualification_evidence=false.
Verification: repository search and manifest regression test; pristine_final_holdout is empty.
```

### Problem 7 — output contract defects

```text
Problem: not every JSON had a top-level schema, bool values could serialize as 0/1, and rally end_time added duration twice.
Evidence: contract inspection and final artifact validation; pre-fix video_10 end_time exceeded its frame-derived bound.
Root cause: NumPy sanitizer checked integer before bool and rally analyzer mixed timestamps/duration.
Fix: schema 1.0 envelopes, bool-first sanitizer, and authoritative frame/FPS rally bounds.
Verification: 48/48 final diagnostic JSON artifacts carry schema 1.0; bool and rally-bound regression tests pass.
```

## D. Event Detector Improvements

Candidate architecture: `EventCandidate` is a high-recall trajectory-inflection
record. `detect_events` independently verifies semantic evidence and exports
only verified events to `match_events.json`; candidates remain in
`event_candidates.json` for audit.

Verification cues include pre/post speed, direction change, acceleration,
curvature, vertical inversion, track provenance/continuity, ball confidence,
time-local player distance, player-scale reach, attribution separation, and
explicit serve context. Post-point scoring state marks dead-ball records and
excludes them from live analytics.

False-positive taxonomy after the correction:

| Cause | Count |
|---|---:|
| Post-rally ball noise | 20 |
| Timing drift | 5 |
| Bounce as hit | 3 |
| Unknown cause | 1 |
| Total | 29 |

Required event ablation status:

| Variant | Precision | Recall | F1 | Timing MAE | Status |
|---|---:|---:|---:|---:|---|
| A — corrected preserved baseline | 0.0887 | 0.2750 | 0.1341 | 142.4 ms | measured |
| B — candidate/final separation | — | — | — | — | not independently isolated |
| C — multi-cue hit verification | — | — | — | — | not independently isolated |
| D — bounce verification | — | — | — | — | not independently isolated |
| E — pre-serve suppression | — | — | — | — | not independently isolated |
| F — dead-ball/rally gating | — | — | — | — | not independently isolated |
| G — integrated event system | 0.1212 | 0.1000 | 0.1096 | 158.3 ms | measured |

Measured A→G change: precision `+0.0325`, recall `-0.1750`, F1 `-0.0246`,
timing MAE `+15.9 ms`; FP count improved `113→29`, but TP fell `11→4`.
Because the intermediate changes were not independently switchable, no causal
claim is made for B–F and the system was not frozen. The integrated result does
not meet the diagnostic direction of approximately 0.70 precision / 0.80 recall.

## E. Shot Classifier Improvements

The classifier no longer assumes handedness or court orientation. Pose and
geometry classification require explicit metadata and weak/ambiguous evidence
returns `UNKNOWN`. Serve passthrough requires an authoritative serve event.

| Tier / slice | Final diagnostic result |
|---|---|
| Pose tier | no classified matched hits; safe abstention |
| Geometry tier | no classified matched hits; safe abstention |
| Consensus tier | no classified matched hits; safe abstention |
| UNKNOWN | 7/7 matched hits; rate 1.000; coverage 0.000 |
| Near/far | not separately measurable after full abstention |
| Left/right | no diagnostic labels/config supplied; unit normalization passes |
| Unknown handedness | safe UNKNOWN by policy |
| 720p/1080p | classification remained all UNKNOWN in both resolution groups |

Shot ablation:

| Variant | Conditional Macro F1 | UNKNOWN rate | Coverage | Status |
|---|---:|---:|---:|---|
| A — corrected preserved baseline | 0.1667 | 0.2222 | 0.7778 | measured |
| B — player-relative geometry correction | — | — | — | not independently isolated |
| C — pose normalization | — | — | — | not independently isolated |
| D — temporal pose | — | — | — | not independently isolated |
| E — calibrated UNKNOWN | — | — | — | not independently isolated |
| F — integrated classifier | 0.0000 | 1.0000 | 0.0000 | measured |

The abstention is safer than unsupported left/right inference but is not useful
classification performance and is far below the 0.80 pristine gate.

## F. Performance Audit

Exact end-to-end final diagnostic measurements:

| Video | Resolution | Frames | Total wall time | FPS |
|---|---:|---:|---:|---:|
| video_08 | 1280x720 | 469 | 17.590 s | 26.663 |
| video_09 | 1280x720 | 408 | 14.566 s | 28.011 |
| video_10 | 1920x1080 | 1,795 | 200.419 s | 8.956 |

The runner prints coarse step timings. The final video_10 run measured ball
proposal inference at 84.14 s; temporal association was 0.05 s. Decode,
player detection/tracking, court inference/homography, event detection,
pose/classification, analytics, rendering, and serialization are not persisted
as separate structured fields. CPU utilization, GPU utilization, data-transfer,
GC, peak RAM, and peak VRAM were not captured by a profiler in this correction.
Accordingly, the requested fine-grained performance audit is **INCOMPLETE**;
no bottleneck beyond the measured ball-proposal stage is guessed. Historical
streamed rendering reduced RAM to about 1.85 GB, but that number was not
re-measured here. No coverage-reducing optimization was made.

## G. Tests

```text
Before: 107 passed historically (the first local invocation used the wrong Python and produced 13 collection errors; it is not counted as a code failure)
After: 120 passed in 6.50s
Passed: 120
Failed: 0
Skipped: 0
```

## H. Analytics Contract

```text
schema_version: 1.0
files validated: 48 JSON files (16 per diagnostic video)
breaking changes: none to the six Contract V1 consumer artifacts; event_candidates is additive
null handling: unavailable speed/classification values remain null/UNKNOWN, never fabricated zero; bools remain JSON booleans
status: CONTRACT-CORRECT DIAGNOSTIC ARTIFACTS; production interface not qualification-frozen
```

Terminology is constrained to “2D Court-Projected Ball Speed Estimate.” Line
calls remain uncertainty-aware research decision support; the empirical contact
patch is not represented as the full ball radius.

## I. Diagnostic Media Status

```text
video_01–10:
DIAGNOSTIC / DEVELOPMENT

video_08–10:
NOT PRISTINE HOLDOUT
```

The legacy `cross_match_final_holdout` directory name is retained only for path
compatibility. Its authoritative semantic split is `CROSS_MATCH_DIAGNOSTIC`.

## J. Pristine Media Provenance

No `video_11+` was acquired. The production/evaluator freeze stage was not
reached because diagnostic event and shot performance regressed below readiness.
Creating a pristine set at this point would consume unseen evidence before the
system is ready and would not cure the technical failure.

```text
Video ID: NONE
Source: N/A
Usage status: pristine acquisition not started
SHA256: N/A
Resolution: N/A
FPS: N/A
Frames: N/A
Duration: N/A
GT commit: NOT CREATED
Review pack: NOT CREATED
```

## K. Freeze Integrity

```text
Production frozen before inference: FAIL — no production freeze created
Evaluator frozen before inference: FAIL — no evaluator freeze created
GT frozen before inference: FAIL — no pristine GT exists
Inference run once: FAIL — only reusable diagnostic media was rerun; pristine inference did not occur
Post-result tuning: YES on diagnostic data; NO pristine result exists
```

This is not a failed one-shot pristine attempt; the qualification attempt was
stopped before freeze/acquisition because readiness failed.

## L. Final Metrics

All numbers below are from the final `CROSS_MATCH_DIAGNOSTIC` run and are not
unseen-test claims.

### Player tracking and throughput

| Video | P1 raw coverage | P2 raw coverage | FPS |
|---|---:|---:|---:|
| video_08 | 1.0000 | 1.0000 | 26.663 |
| video_09 | 0.9975 | 0.9730 | 28.011 |
| video_10 | 0.8295 | 0.8563 | 8.956 |
| Frame-weighted aggregate | 0.8851 | 0.8993 | not averaged across resolutions |

Visible-frame coverage, ID switches, fragmentation, ball proposal recall,
ball localization percentiles, longest gaps, and court failure rate do not have
independent GT/profiler measurements in this run and are reported as `N/A`.
Court reprojection errors were 0.4442 px, 4.0318 px, and 0.0843 px; all three
runs emitted `is_valid=true`.

### Event detection

| Class | TP | FP | FN | Precision | Recall | F1 | Timing MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SERVE_CONTACT | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | N/A |
| PLAYER_HIT | 3 | 19 | 14 | 0.1364 | 0.1765 | 0.1538 | 3.67 frames / 122.2 ms |
| BOUNCE | 1 | 10 | 19 | 0.0909 | 0.0500 | 0.0645 | 8.00 frames / 266.7 ms |
| OVERALL | 4 | 29 | 36 | 0.1212 | 0.1000 | 0.1096 | 4.75 frames / 158.3 ms |

Matching uses global nearest one-to-one same-event-type semantics within each
video and a 0.2 s tolerance converted using actual FPS.

### Conditional shot classification

| Class | TP | FP | FN | Precision | Recall | F1 | Matched support |
|---|---:|---:|---:|---:|---:|---:|---:|
| FOREHAND | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | 4 |
| BACKHAND | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | 2 |
| SERVE | 0 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | 1 |

```text
Macro F1: 0.0000
Coverage: 0.0000
UNKNOWN rate: 1.0000 (7/7 matched hits)
```

### End-to-end shot recognition

```text
TP: 0
FP: 32
FN: 20
Precision: 0.0000
Recall: 0.0000
F1: 0.0000
```

Rally metrics against independent manual rally GT were not implemented by the
corrected evaluator, so exact stroke-count matches, MAE, and segmentation errors
are `N/A`, not inferred from self-produced rally artifacts.

## M. Gate Table

| Requirement | Required | Measured | PASS/FAIL |
|---|---:|---:|---|
| Evaluation Integrity | PASS | diagnostic evaluator corrected; no frozen qualification evaluator | FAIL |
| Pristine Different-Match Holdout | PASS | none | FAIL |
| Manual GT | PASS | diagnostic manual GT only | FAIL |
| No GT Leakage | PASS | PASS for code path; no pristine attempt | FAIL (qualification absent) |
| Evaluator Frozen | PASS | not frozen | FAIL |
| Production Frozen | PASS | candidate status only | FAIL |
| Player Visible-Frame Coverage | ≥ 0.90 | independent visible-frame denominator unavailable; raw P1 0.8851/P2 0.8993 | FAIL |
| Overall Event F1 | ≥ 0.75 | 0.1096 diagnostic | FAIL |
| PLAYER_HIT Recall | ≥ 0.80 | 0.1765 diagnostic | FAIL |
| Conditional Shot Macro F1 | ≥ 0.80 | 0.0000 diagnostic | FAIL |
| End-to-End Shot Recognition F1 | ≥ 0.70 | 0.0000 diagnostic | FAIL |
| Rally Stroke MAE | ≤ 1.0 | N/A | FAIL |
| All Tests | PASS | 120/120 | PASS |
| Analytics Contract V1 | PASS | 48/48 schema/type checks; unit regressions pass | PASS |

## N. Remaining Failure Analysis

The single primary remaining blocker is **semantic physical-event verification
does not preserve real contacts while rejecting non-play ball motion**.
Evidence: the integrated detector reduced FP from 113 to 29 but also reduced TP
from 11 to 4; overall event F1 is 0.1096, PLAYER_HIT recall is 0.1765, serve
recall is zero, and 20/29 remaining false positives are post-rally noise.
Because shot classification depends on matched authoritative hits, this failure
also leaves only seven matched hits and forces 100% UNKNOWN abstention.

The next correction should make rally/serve state explicit before semantic event
verification and calibrate multi-cue thresholds with independently switchable
B–F ablations on diagnostic media. After it reaches a credible precision/recall
balance, create a new production/evaluator freeze and only then acquire a new,
independent `video_11+` pristine holdout.

## O. Artifact Paths

```text
Diagnostic GT:
  data/benchmarks/cross_match_final_holdout/ground_truth_events.json
  data/benchmarks/cross_match_final_holdout/ground_truth_shots.json
  data/benchmarks/cross_match_final_holdout/ground_truth_rallies.json

Review packs:
  artifacts/validation/phase6_4_diagnostic_review/manifest.json
  artifacts/validation/phase6_4_diagnostic_review/*_contact_sheet.jpg

Configs:
  configs/phase6_4_qualification/final.yaml
  configs/phase6_analytics/pipeline.yaml

Metrics/evaluation:
  outputs/phase6_4_qualification/aggregate_cross_match_diagnostic_corrected.json
  outputs/phase6_4_qualification/aggregate_cross_match_diagnostic_final.json
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_08/metrics.json
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_09/metrics.json
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_10/metrics.json

Failure reports:
  artifacts/validation/phase6_4_false_positive_audit_final.json
  artifacts/validation/phase6_4_false_positive_audit_final.csv
  docs/audit/PHASE6_4_FALSE_EVENT_AUDIT.md
  docs/experiments/PHASE6_4_FAILURE_ANALYSIS.md

Diagnostic production outputs / annotated videos:
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_08/
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_09/
  outputs/phase6_4_qualification/cross_match_diagnostic_final/video_10/

Hash/split manifests:
  data/benchmarks/cross_match_final_holdout/videos.json
  data/benchmarks/cross_match_final_holdout/splits.json
  data/data_registry.yaml

Pristine GT/review/output/hash artifacts: NOT CREATED
```

Current recorded hashes (SHA256):

```text
Evaluator: 8F686AE0432AD825FB96928BDFD22340FF853DD1338DDE9D44E47E56A21A0890
Candidate config: 68332C186AA557C60B23DCACC4B0893C9C10391BFCCA29A593B0E3732F647B46
Ball weights: 24691B50AAC718689B1716A9BE9683D9DF29D10A565303E2E332C0E2FC80B34E
Court weights: 16EBB7E46DC88247440C86B388E4F07F0D4ABB76CE0A01A22925D3163F7FB7F3
Phase 6 pipeline: C2C3C65406CB8BCC6EDCC78A7F46E2A585831BB7BABCA3580B97BFF014596173
```

These are provenance hashes, not freeze declarations.

## P. Final Verdict

```text
PHASE 6.4 FINAL QUALIFICATION: FAIL
PHASE 7 DASHBOARD: BLOCKED
PRIMARY REMAINING BLOCKER: semantic physical-event verification does not preserve real contacts while rejecting non-play ball motion
```

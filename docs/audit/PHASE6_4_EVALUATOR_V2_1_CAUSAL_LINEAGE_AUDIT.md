# Phase 6.4 Evaluator v2.1 Causal Lineage Audit

## Scope and scientific status

- Starting SHA: `29cd39af013e6498599e19dd0ced97d00247c917`
- Branch: `codex/phase6-4-final-correction`
- Evaluator-only correction: yes
- Model tuning: none
- Scientific split: `CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED`
- Coverage review status: `MODEL_ASSISTED_PROVISIONAL`
- Metric authority: `DIAGNOSTIC_ONLY`
- Qualification evidence: false
- Human approved: false

Evaluator v2.0 same-video isolation, explicit media identity, native timing,
200 ms tolerance, OUTSIDE_SCOPE exclusion, per-video count aggregation,
Analytics Contract V1, and production GT isolation remain intact.

## Optimal matching correction

The v2.0 canonical matcher sorted valid edges by temporal error and greedily
accepted unused endpoints. Consider point GT timestamps `G1=1.00 s` and
`G2=0.90 s`, predictions `P1=1.00 s` and `P2=1.15 s`, and tolerance 0.200 s.
The valid graph is `P1->{G1,G2}` and `P2->{G1}`. Greedy chooses the zero-cost
`P1->G1`, leaving only one TP. The optimal assignment chooses `P1->G2` and
`P2->G1`, producing two TP.

Evaluator v2.1 uses deterministic minimum-cost maximum bipartite flow. Its
lexicographic objective is:

1. maximize one-to-one match cardinality;
2. among maximum-cardinality assignments, minimize total temporal error;
3. apply stable prediction/GT input-index ordering for deterministic ties.

Every edge still requires the same `video_id`, compatible requested semantic
and player constraints, and GT-interval temporal distance no greater than an
inclusive 0.200 seconds. Native timestamps/FPS remain per video.

The adversarial regression changes one TP to two. On the current diagnostic
videos, optimal matching itself does not change final-output counts; the v2.1
count change is entirely due to coverage-boundary repair.

| Counterfactual | TP | FP | FN |
|---|---:|---:|---:|
| v2.0 greedy, old coverage | 20 | 25 | 20 |
| greedy, tolerance-safe v2.1 coverage | 21 | 25 | 19 |
| optimal, tolerance-safe v2.1 coverage | 21 | 25 | 19 |

## Coverage-boundary integrity

A GT event is symmetric-evaluation eligible only when a reviewed interval
contains its complete uncertainty interval plus the 0.200-second match margin,
truncated only at a physical video boundary. Otherwise the GT is
`BOUNDARY_CENSORED`, and covered predictions eligible for that GT are censored
symmetrically rather than converted to FP.

Existing dense raw-video review sheets already include the small required
extensions. Metadata-derived extensions were therefore made without changing
GT or claiming human approval.

| Video | Coverage | First GT interval | Last GT interval | Required margin | Safe GT | Censored GT | Extension |
|---|---|---|---|---:|---:|---:|---|
| `video_08` | 0-206 | 40-44 | 196-200 | 0.200 s each side | 10 | 0 | 202 -> 206 |
| `video_09` | 0-244 | 33-37 | 234-238 | 0.200 s each side | 12 | 0 | 240 -> 244 |
| `video_10` | 0-399 | 53-57 | 368-372 | 0.200 s each side | 18 | 0 | none |

The final right margins are 0.200 s for `video_08`, 0.200 s for `video_09`,
and 0.900 s for `video_10`. All 40 included GT events are now coverage-safe.
Human approval remains pending; these remain provisional diagnostic metrics.

## V2.0 stage-lineage defect

V2.0 independently rematched GT against each stage-like record set and reused
final emitted events as stages 3-6. A preserved example is `video_08` GT event
8, whose historical record says `stage_3=false` but
`stage_4=stage_5=stage_6=true`. `video_10` GT event 8 similarly says
`stage_2=stage_3=false` and stages 4-6 true. These are impossible causal
histories. The old stage metrics and lineage are retained but marked
`SUPERSEDED_BY_EVALUATOR_V2_1_CAUSAL_STAGE_SEMANTICS`.

## True stages and stable candidate identity

- Stage 0: raw detector proposal evidence.
- Stage 1: usable trajectory observation.
- Stage 2: `EventCandidate` generated and assigned stable `candidate_id`.
- Stage 3: that candidate passed physical-contact verification, before semantic correctness or final suppression.
- Stage 4: the Stage-3 survivor received a tennis semantic type; causal GT survival additionally requires the type to be correct.
- Stage 5: the Stage-4 survivor received correct player attribution where applicable; BOUNCE requires no player.
- Stage 6: the same candidate survived actual final temporal/dead-ball/export filtering and was emitted.

One detector analysis is run per video. From Stage 2 onward, `candidate_id`
links discovery frame/time, refined frame/time, physical result, semantic
result, player result, suppression reason, and final event ID. Temporal
rematching is not used to infer later causal stages.

## Causal monotonicity and first failure

For every record, `S0 >= S1 >= S2 >= S3 >= S4 >= S5 >= S6` is asserted.
Aggregate monotonicity is also asserted, and artifact generation fails on any
violation.

| Causal stage | video_08 | video_09 | video_10 | Total |
|---|---:|---:|---:|---:|
| Stage 0 | 10 | 12 | 18 | 40 |
| Stage 1 | 10 | 12 | 5 | 27 |
| Stage 2 | 10 | 11 | 4 | 25 |
| Stage 3 | 5 | 11 | 3 | 19 |
| Stage 4 | 4 | 4 | 0 | 8 |
| Stage 5 | 4 | 2 | 0 | 6 |
| Stage 6 | 2 | 2 | 0 | 4 |

- Per-record invariant: PASS
- Aggregate invariant: PASS
- Impossible transitions: 0

| First failure | video_08 | video_09 | video_10 | Total |
|---|---:|---:|---:|---:|
| NO_RAW_PROPOSAL | 0 | 0 | 0 | 0 |
| TRACK_NOT_USABLE | 0 | 0 | 13 | 13 |
| NO_EVENT_CANDIDATE | 0 | 1 | 1 | 2 |
| PHYSICAL_CONTACT_REJECTED | 5 | 0 | 1 | 6 |
| SEMANTIC_TYPE_WRONG | 1 | 7 | 3 | 11 |
| PLAYER_ATTRIBUTION_WRONG | 0 | 2 | 0 | 2 |
| FINAL_SUPPRESSION | 2 | 0 | 0 | 2 |
| FULLY_CORRECT | 2 | 2 | 0 | 4 |

For `video_10`, the 18 GT events localize to 13 failures at Stage 1, one at
Stage 2, one at Stage 3, three at Stage 4, none at Stages 5/6, and zero fully
correct. No tuning was performed.

## Final-output physical match

This metric honestly names the v2.0 concept: final emitted events matched
type-agnostically. It is not true Stage-3 verification.

| Video | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `video_08` | 6 | 5 | 4 | 0.5455 | 0.6000 | 0.5714 |
| `video_09` | 11 | 14 | 1 | 0.4400 | 0.9167 | 0.5946 |
| `video_10` | 4 | 6 | 14 | 0.4000 | 0.2222 | 0.2857 |
| Aggregate | 21 | 25 | 19 | 0.4565 | 0.5250 | 0.4884 |

There are 46 covered final predictions and 82 OUTSIDE_SCOPE predictions.

## True Stage-3 physical verification

Stage 3 is extracted directly from `physics_verification=true` traces and is
matched type- and player-agnostically.

| Video | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `video_08` | 7 | 10 | 3 | 0.4118 | 0.7000 | 0.5185 |
| `video_09` | 11 | 25 | 1 | 0.3056 | 0.9167 | 0.4583 |
| `video_10` | 4 | 12 | 14 | 0.2500 | 0.2222 | 0.2353 |
| Aggregate | 22 | 47 | 18 | 0.3188 | 0.5500 | 0.4037 |

| GT class | GT | Physical matches | Stage-3 recall |
|---|---:|---:|---:|
| SERVE_CONTACT | 3 | 3 | 1.0000 |
| PLAYER_HIT | 17 | 9 | 0.5294 |
| BOUNCE | 20 | 10 | 0.5000 |

## Independent stage capability

This is explicitly not causal survival. Each actual stage output is evaluated
independently to describe capability even when upstream causal association
differs.

| Stage | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 | 40 | 716 | 0 | 0.0529 | 1.0000 | 0.1005 |
| Stage 1 | 27 | 447 | 13 | 0.0570 | 0.6750 | 0.1051 |
| Stage 2 | 25 | 58 | 15 | 0.3012 | 0.6250 | 0.4065 |
| Stage 3 | 22 | 47 | 18 | 0.3188 | 0.5500 | 0.4037 |
| Stage 4 | 13 | 47 | 27 | 0.2167 | 0.3250 | 0.2600 |
| Stage 5 | 11 | 49 | 29 | 0.1833 | 0.2750 | 0.2200 |
| Stage 6 | 9 | 37 | 31 | 0.1957 | 0.2250 | 0.2093 |

## Semantic and player metrics

End-to-end exact semantics: 11 TP / 35 FP / 29 FN, precision 0.2391,
recall 0.2750, F1 0.2558.

Conditional on the 22 true Stage-3 physical matches, semantic classification
has 11 correct, accuracy 0.5000, macro F1 0.3961, and two UNKNOWN abstentions
(0.0909). Conditional class F1 is 0.0000 SERVE_CONTACT, 0.5882 PLAYER_HIT,
and 0.6000 BOUNCE.

End-to-end player-aware recognition: 9 TP / 37 FP / 31 FN, precision 0.1957,
recall 0.2250, F1 0.2093. Conditional attribution excludes BOUNCE and requires
a Stage-3 match plus a correct eligible semantic type: support 5, correct 3,
accuracy 0.6000.

## Final suppression

Two valid causal Stage-5 survivors were suppressed before export. Both are
`video_08` candidates (`candidate_id` 14 and 24) with reason
`DUPLICATE_VERIFIED_EVENT`. The lineage retains their candidate IDs.

## Production, analytics, and leakage regression

- Production predictions before: 128
- Production predictions after: 128
- Changed prediction frames: 0
- Changed event labels/players: 0
- Analytics schema: 1.0, unchanged
- Null-to-zero fabrication: none
- Production access to GT, coverage, human labels, evaluation status, or GT IDs: none
- Model tuning performed: no

## Evaluator provenance

- Version: `2.1`
- Evaluator SHA256: `84342716bc2b217aab10b047b71b9f7c0e99e99ca848c241994c0e47b3de99ef`
- GT SHA256: `57559bec15f7d3df7e55c09d421825b4f19947c40fa82c9f0ffcc8da6538b2ee`
- Coverage SHA256: `ddd04996a4cbfb243b1343b13ed40b613a6bda0ca02b3a9e10e0e42454c461ae`
- Video manifest SHA256: `1974c6af6744ecb5140de42dd35fa9eb197b882b87bad54bd58096d06a3db702`
- Config SHA256: `19653074723994b9591ad4fdd7d3e954130e1ef8b2ebb8c46fb1a7f45179f852`
- Matching: `PER_VIDEO_NATIVE_TIMESTAMP_GT_INTERVAL_MAX_CARDINALITY_MIN_COST_ONE_TO_ONE`
- Coverage policy: `TOLERANCE_NEIGHBORHOOD_MUST_BE_FULLY_OBSERVABLE_OR_SYMMETRICALLY_CENSORED`
- Causal lineage schema: `1.0`

## Verification

- Baseline: 279 passed
- V2.1 full suite: 315 passed, 0 failed, 0 skipped
- V2.1-specific tests: 36 passed
- `git diff --check`: PASS

## Verdict

PHASE 6.4 EVALUATOR V2.1 INTEGRITY: PASS

CAUSAL STAGE LINEAGE: PASS

OPTIMAL ONE-TO-ONE MATCHING: PASS

COVERAGE BOUNDARY HANDLING: PASS

MODEL TUNING PERFORMED: NO

QUALIFICATION EVIDENCE: NO

NEXT PRIMARY MODEL BLOCKER: TRACK_NOT_USABLE is the largest causal first-failure category (13/40 GT events), all localized to video_10; no corrective tuning is performed in this run.

# Phase 6.4 Evaluator Integrity and Annotation Coverage Audit

## Scientific status

- Starting local and remote SHA: `d15de4fcc8b0479070f408a024d98b46eef2be7c`
- Branch: `codex/phase6-4-final-correction`
- Split: `CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED`
- Qualification evidence: **false**
- Model tuning in this correction: **none**
- Evaluator version: `2.0`
- Evaluation scope: `CROSS_MATCH_FINAL_HOLDOUT_DIAGNOSTIC_V2_RALLY_INTERVALS`

## Cross-video defect reproduction and correction

The starting implementation in `src/events/event_evaluator.py` considered only temporal frame distance. A prediction with `video_id=video_A, frame=55` matched GT with `video_id=video_B, frame_best=55`; the old result was one match. The regression in `tests/test_phase6_4_physical_precision.py` explicitly expected that invalid result.

The corrected canonical matcher:

1. normalizes `video_id` once at the evaluator boundary (legacy `_video_id` is accepted only there);
2. fails closed when identity is missing in an explicitly identified/multi-video call;
3. rejects different video identities before temporal comparison;
4. converts each record using its video's native FPS and prefers a valid prediction `timestamp_s`;
5. measures distance to the GT uncertainty interval in seconds;
6. uses an inclusive deterministic tolerance of 0.200 seconds;
7. performs greedy one-prediction/one-GT matching, leaving duplicates unmatched.

The corrected fixture produces zero matches and the test now expects zero. The old implementation fails the corrected invariant.

## Matcher architecture

The starting repository had two materially different matchers:

- `src/events/event_evaluator.py::canonical_one_to_one_matches`, which ignored video identity;
- `scripts/evaluate_phase6_4_cross_match.py::one_to_one_matches`, which conditionally checked legacy `_video_id`.

`src/events/event_evaluator.py::canonical_one_to_one_matches` is now the sole matching implementation. The script-level function remains only as a compatibility adapter that normalizes legacy records and delegates to the canonical timestamp matcher. Evaluators match each video independently and aggregate TP, FP, and FN counts.

## Evaluator call-site audit

| File/artifact | Status | Resolution |
|---|---|---|
| `src/events/event_evaluator.py` | FIXED_CANONICAL | Same-video timestamp/GT-interval matcher plus coverage-aware evaluation and count invariants. |
| `scripts/evaluate_phase6_4_cross_match.py` | FIXED | Compatibility adapter delegates to canonical matcher; aggregate report sums per-video counts. |
| `scripts/evaluate_phase6_4_event_recovery.py` | FIXED | Per-video native timing, coverage filtering, count aggregation, regenerated provenance. |
| `scripts/evaluate_phase6_4_semantic_event_recovery.py` | FIXED | Removed source-specific truncation; full inference followed by metadata-driven coverage filtering and per-video count aggregation. |
| `scripts/evaluate_phase6_4_event_pipeline.py` | FIXED | Exact and per-class metrics are matched per video and aggregated as counts. |
| `scripts/evaluate_phase6_4_oracle_shots.py` | SAFE | Joins and validates records within each explicit video bucket; no global temporal matcher. |
| `scripts/audit_false_positives.py` | FIXED | Only covered predictions can enter the FP audit; outside predictions are counted separately. |
| `scripts/generate_pre_semantic_recovery_artifacts.py` | FIXED | Per-video matches and count/timing aggregation; no global mixed timeline. |
| `scripts/generate_pipeline_docs.py` | INVALIDATED_OUTPUT GENERATOR | It embeds legacy metrics but performs no matching. Its generated historical conclusions are not authoritative. |
| `scripts/create_phase6_4_pre_recovery_audit.py` | LEGACY_DIAGNOSTIC, MATCHER FIXED | Delegates through the canonical adapter; hard-coded historical fixture classifications are retained as forensic data, not corrected metrics. |

## Source-specific evaluator rule audit

The executable branch in `scripts/evaluate_phase6_4_semantic_event_recovery.py` that truncated only `video_10` to 400 frames was an `EVALUATOR_SOURCE_HACK`. It and the associated total-frame calculation were removed. The evaluator now analyzes complete inputs and obtains legal evaluation scope from `annotation_coverage.json`.

Remaining `video_08`/`video_09`/`video_10` occurrences are classified as:

- `DATA_REFERENCE_OK`: diagnostic manifest iteration, report labels, historical artifact prose;
- `TEST_FIXTURE_OK`: benchmark membership and no-production-branching tests;
- `EVALUATOR_SOURCE_HACK`: **none remaining**;
- `PRODUCTION_SOURCE_HACK`: **none found**.

## Raw-video annotation coverage

Coverage was reviewed from the raw MP4 files using playback plus 5-frame-interval raw contact sheets. The review generator reads no predictions, tracks, or FP labels; its manifest records `prediction_inputs_used=false`. This correction does not claim human approval: reviewer provenance is `CODEX_MODEL_ASSISTED_RAW_VIDEO_REVIEW`, while the original annotation protocol records the earlier frame-level raw-video annotation process.

| Video | FPS | Frames | Inclusive exhaustive interval | Covered frames | Coverage | Review status |
|---|---:|---:|---:|---:|---:|---|
| `video_08` | 30 | 469 | 0–202 | 203 | 43.28% | Raw-video reviewed; not newly human-approved |
| `video_09` | 30 | 408 | 0–240 | 241 | 59.07% | Raw-video reviewed; not newly human-approved |
| `video_10` | 30 | 1795 | 0–399 | 400 | 22.28% | Raw-video reviewed; not newly human-approved |

Frames after those intervals are `UNANNOTATED`, not negative GT. The raw review visibly shows additional tennis action after the covered region, particularly in `video_10`; those predictions are `OUTSIDE_SCOPE`.

No missing physical event was identified inside the defined covered intervals. GT revision: **NONE**. The benchmark remains at 40 physical events. No GT was generated from model predictions.

## Historical result invalidation

The following are retained for forensic value but invalidated as authoritative evidence because they predate same-video matching and/or legal negative coverage:

- `phase6_4_event_pipeline_ablation.json`
- `phase6_4_event_pipeline_confusion.json`
- `phase6_4_physical_precision_ablation.json`
- `phase6_4_physical_fp_before_after.json`
- `phase6_4_physical_prediction_lineage.json`
- `phase6_4_physical_prediction_lineage_final.json`
- `phase6_4_fp_manual_review_manifest.json`
- `phase6_4_fp_source_taxonomy.json`
- walkthrough/report values derived from `30 TP / 75 FP / 10 FN`

The two `d15de4` FP artifacts now carry `scientific_status=INVALIDATED_BY_EVALUATOR_INTEGRITY_CORRECTION`, `authoritative=false`, `label_source=AUTOMATED_HEURISTIC_UNVERIFIED`, and `human_reviewed=false`. Their legacy category counts must not be used for training or conclusions.

## Executed failure history

All historical results below were executed in detached temporary Git worktrees using the same provisioned Python runtime. The worktrees were removed afterward.

| Test | `270dfa7` | `3c69c37` | `d15de4` | Corrected HEAD | Cause |
|---|---|---|---|---|---|
| Predicted trajectory remains possible with reduced confidence | PASS | FAIL | FAIL | PASS | `3c69c37` changed `enable_provenance_gating` to a hard-rejecting default. The corrected default is opt-in (`false`), while provenance confidence weighting remains. |
| Recovery input hashes equal current sources | FAIL | FAIL | FAIL | PASS | The artifact was already stale at `270dfa7`. The authoritative generator recomputed hashes after all source corrections. |
| Cross-video isolation regression | NOT PRESENT | PASS (bad expectation) | PASS (bad expectation) | PASS (correct expectation) | The old test expected one cross-video match and therefore codified the defect. |

## Corrected metrics (no tuning)

### Stage 3 physical contact, type-agnostic

| Video | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `video_08` | 5 | 5 | 5 | 0.5000 | 0.5000 | 0.5000 |
| `video_09` | 11 | 14 | 1 | 0.4400 | 0.9167 | 0.5946 |
| `video_10` | 4 | 6 | 14 | 0.4000 | 0.2222 | 0.2857 |
| **Aggregate** | **20** | **25** | **20** | **0.4444** | **0.5000** | **0.4706** |

Count proof: `20 = 5+11+4`, `25 = 5+14+6`, and `20 = 5+1+14`. Timing MAE is 0.04833 seconds (48.33 ms). Frame MAE is intentionally not aggregated across potentially heterogeneous FPS.

### Stages 0–6

| Stage | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Stage 0 raw proposal | 40 | 708 | 0 | 0.0535 | 1.0000 | 0.1015 |
| Stage 1 usable observation | 27 | 443 | 13 | 0.0574 | 0.6750 | 0.1059 |
| Stage 2 physical candidate | 25 | 58 | 15 | 0.3012 | 0.6250 | 0.4065 |
| Stage 3 physical contact | 20 | 25 | 20 | 0.4444 | 0.5000 | 0.4706 |
| Stage 4 semantic type | 10 | 35 | 30 | 0.2222 | 0.2500 | 0.2353 |
| Stage 5 player attribution | 8 | 37 | 32 | 0.1778 | 0.2000 | 0.1882 |
| Stage 6 authoritative event | 8 | 37 | 32 | 0.1778 | 0.2000 | 0.1882 |

Stage 0/1 precision is proposal/observation density, not a production event precision claim.

### Exact semantic type

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `SERVE_CONTACT` | 1 | 13 | 2 | 0.0714 | 0.3333 | 0.1176 |
| `PLAYER_HIT` | 3 | 6 | 14 | 0.3333 | 0.1765 | 0.2308 |
| `BOUNCE` | 6 | 11 | 14 | 0.3529 | 0.3000 | 0.3243 |
| **All exact types** | **10** | **35** | **30** | **0.2222** | **0.2500** | **0.2353** |

Exact type plus player attribution is `8 TP / 37 FP / 32 FN`, precision 0.1778, recall 0.2000, F1 0.1882. Bounce matching does not require player attribution.

## Corrected FP population and provenance

- Covered Stage-3 FP: **25**
- Outside-scope predictions: **83**
- Human reviewed in this correction: **0**
- Automated/unverified: **25**
- Pending human review: **25**

Every review row contains frame, timestamp, coverage status, match identity, full frame, ball crop, context crop, automated prelabel/confidence, `human_reviewed=false`, `human_label=null`, and `label_source=AUTOMATED_HEURISTIC_UNVERIFIED`.

## Integrity, leakage, and analytics checks

- Aggregate TP/FP/FN equal the sums of per-video counts: **PASS**
- Covered GT denominator equals TP+FN: **PASS**
- FP originates only inside exhaustive coverage: **PASS**
- Outside-scope predictions do not affect P/R/F1: **PASS**
- Cross-video match count: **0**
- All corrected metric artifacts share evaluator version/hash: **PASS**
- Production tracker/detector/semantic/shot code receives no GT, coverage, or review label inputs: **PASS**
- Analytics schema remains `1.0`; no production analytics fields were changed and no null-to-zero fabrication was introduced: **PASS**

## Evaluator provenance

- Evaluator SHA256: `4309a6fa08230807bea8f9cef5764f852cb6f5897cca50494a3f00fe52b9ca57`
- GT SHA256: `57559bec15f7d3df7e55c09d421825b4f19947c40fa82c9f0ffcc8da6538b2ee`
- Coverage SHA256: `4659682bdcd1ecf350598ada05513de0ffe5679a2ac796704520bc14d4f05a91`
- Video manifest SHA256: `1974c6af6744ecb5140de42dd35fa9eb197b882b87bad54bd58096d06a3db702`
- Config SHA256: `19653074723994b9591ad4fdd7d3e954130e1ef8b2ebb8c46fb1a7f45179f852`

## Verification

- `python -m pytest tests/ -q`: **279 passed, 0 failed, 0 skipped**
- `git diff --check`: **PASS**

## Verdict

PHASE 6.4 EVALUATOR INTEGRITY: PASS

ANNOTATION COVERAGE: DEFINED

METRICS RECONSTRUCTED: YES

MODEL READINESS: NOT ASSESSED IN THIS RUN

NEXT BLOCKER: Corrected Stage-3 physical-contact performance is 0.4444 precision / 0.5000 recall / 0.4706 F1, with `video_10` recall at 0.2222; optimization is intentionally deferred pending review.

# Phase 6.4 Event-Recovery Ablation

## Experiment boundary

This experiment measures independently configurable event-verifier components
on preserved `video_08`–`video_10` diagnostic trajectories and player boxes.
All three clips are already-consumed development data. Results are not
qualification, pristine-holdout, or unseen-generalization evidence.

The runner never reads GT inside candidate generation or verification. It first
produces candidates/events for each variant, then applies the unchanged fixed
evaluator. Matching is global-nearest one-to-one within video, same-type for
semantic events, with the GT frame interval expanded by 0.2 seconds at native
FPS. Candidate recall is type-agnostic and then stratified by GT class.

## Baseline naming

Three “before” numbers exist for different provenance reasons:

| Label | TP | FP | FN | Precision | Recall | F1 | Use |
|---|---:|---:|---:|---:|---:|---:|---|
| Historical corrected report | 11 | 113 | 29 | 0.0887 | 0.2750 | 0.1341 | Required published historical reference |
| Legacy output fixed-evaluator rescore | 13 | 111 | 27 | 0.1048 | 0.3250 | Shows effect of current interval-aware rescore on the legacy output root |
| A: integrated pre-recovery | 4 | 29 | 36 | 0.1212 | 0.1000 | Direct fixed-evaluator comparison for B–H |

The A–H table does not substitute the historical line with a more favorable
rescore. Variant A is the fixed-evaluator integrated system whose severe recall
collapse triggered this recovery.

## Runnable variants

| Variant | Increment relative to the preceding variant |
|---|---|
| A — `A_INTEGRATED_PRE_RECOVERY` | Preserved integrated pre-recovery events; no mutable detector execution |
| B — `B_HIGH_RECALL_CANDIDATES` | Normalized high-recall candidates; temporal proximity, refinement, bounce/serve semantics, activity gate, camera compensation, and provenance weighting disabled |
| C — `C_TEMPORAL_PLAYER_PROXIMITY` | B + time-local player proximity |
| D — `D_EVENT_TIME_REFINEMENT` | C + source-independent local event-time refinement |
| E — `E_EVENT_SPECIFIC_VERIFICATION` | D + bounce-contact and serve-semantic verification |
| F — `F_VISION_DEAD_BALL_GATING` | E + observable low-energy roll/jitter gate |
| G — `G_CAMERA_MOTION_COMPENSATION` | F + camera-compensation switch |
| H — `H_FINAL_INTEGRATED` | G + trajectory-provenance weighting |

Direction-change and velocity-discontinuity features remain independently
configurable in the central settings and are enabled in these measured variants.
Pose support remains disabled because no suitable persisted pose input exists.

## Candidate and overall metrics

| Variant | Candidate recall S/H/B/O | TP | FP | FN | Precision | Recall | F1 | Timing MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.000 / 0.471 / 0.550 / 0.550 | 4 | 29 | 36 | 0.1212 | 0.1000 | 0.1096 | 4.75 f / 158.3 ms |
| B | 1.000 / 0.647 / 0.550 / 0.625 | 6 | 25 | 34 | 0.1935 | 0.1500 | 0.1690 | 3.33 f / 111.1 ms |
| C | 1.000 / 0.647 / 0.550 / 0.625 | 6 | 28 | 34 | 0.1765 | 0.1500 | 0.1622 | 2.67 f / 88.9 ms |
| D | 1.000 / 0.647 / 0.550 / 0.625 | 7 | 30 | 33 | 0.1892 | 0.1750 | 0.1818 | 3.29 f / 109.5 ms |
| E | 1.000 / 0.647 / 0.550 / 0.625 | 11 | 66 | 29 | 0.1429 | 0.2750 | 0.1880 | 4.45 f / 148.5 ms |
| F | 1.000 / 0.647 / 0.550 / 0.625 | 11 | 66 | 29 | 0.1429 | 0.2750 | 0.1880 | 4.45 f / 148.5 ms |
| G | 1.000 / 0.647 / 0.550 / 0.625 | 11 | 66 | 29 | 0.1429 | 0.2750 | 0.1880 | 4.45 f / 148.5 ms |
| H | 1.000 / 0.647 / 0.550 / 0.625 | 12 | 67 | 28 | 0.1519 | 0.3000 | 0.2017 | 4.42 f / 147.2 ms |

`S/H/B/O` means serve, player hit, bounce, and overall candidate recall.
Candidate timing for B–H is `2.76 frames / 92.0 ms`; A is `4.09 frames /
136.4 ms`.

## Per-class precision, recall, and F1

| Variant | Serve P/R/F1 | Player-hit P/R/F1 | Bounce P/R/F1 |
|---|---:|---:|---:|
| A | 0 / 0 / 0 | 0.1364 / 0.1765 / 0.1538 | 0.0909 / 0.0500 / 0.0645 |
| B | 0 / 0 / 0 | 0.1935 / 0.3529 / 0.2500 | 0 / 0 / 0 |
| C | 0 / 0 / 0 | 0.1765 / 0.3529 / 0.2353 | 0 / 0 / 0 |
| D | 0 / 0 / 0 | 0.1892 / 0.4118 / 0.2593 | 0 / 0 / 0 |
| E | 0 / 0 / 0 | 0.1786 / 0.2941 / 0.2222 | 0.1429 / 0.3000 / 0.1935 |
| F | 0 / 0 / 0 | 0.1786 / 0.2941 / 0.2222 | 0.1429 / 0.3000 / 0.1935 |
| G | 0 / 0 / 0 | 0.1786 / 0.2941 / 0.2222 | 0.1429 / 0.3000 / 0.1935 |
| H | 0 / 0 / 0 | 0.1818 / 0.3529 / 0.2400 | 0.1429 / 0.3000 / 0.1935 |

Serve counts make the failure especially clear:

```text
A–D: TP 0, FP 0, FN 3
E–G: TP 0, FP 7, FN 3
H:   TP 0, FP 4, FN 3
```

Serve semantic evidence reduces false serve labels under provenance weighting,
but never recognizes a true serve. Candidate recall is 3/3, so the failure is
semantic verification, not serve candidate generation.

## Increment interpretation

### A to B: high-recall normalized candidate path

B improves candidate recall from `0.5500` to `0.6250`, mainly through hit
candidate recall (`0.4706 -> 0.6471`). It recovers two TPs while reducing four
FPs, raising F1 to `0.1690`. Bounce and serve semantic branches are disabled,
so B's authoritative output is not a complete production alternative.

### B to C: temporal player proximity

C leaves TP/FN unchanged and adds three FPs. Timing MAE improves from 111.1 ms
to 88.9 ms, showing that time-local geometry helps alignment but does not by
itself establish physical contact.

### C to D: local time refinement

D recovers one additional hit TP and raises hit recall to `0.4118`, with two
additional FPs. The refined event owns a coherent refined timestamp; discovery
time remains separately auditable.

### D to E: event-specific verification

E activates bounce and serve semantics. It recovers six bounces and raises
overall TP from 7 to 11, but adds 36 FPs. Serve produces seven FPs and zero TPs.
This is the main precision failure, not an acceptable readiness result.

### E to F: vision-side dead-ball gate

F is exactly equal to E. This is a real enabled no-effect ablation, not a
missing run: none of E's provisional events satisfied both the low normalized
speed and low kinematic-support conditions. The 43 final post-rally false
events look active because the tracker follows false motion.

### F to G: camera-motion switch

G is exactly equal to F. The preserved inputs contain no per-frame static-scene
motion offsets, so the detector records camera compensation as unavailable and
uses raw image coordinates. A static homography is not substituted, and no
metric claim is made for airborne ball coordinates.

### G to H: provenance weighting

H gains one TP and one FP, reduces FN by one, and raises F1 from `0.1880` to
`0.2017`. Serve FPs fall from seven to four; hit TP rises from five to six while
hit FP rises from 23 to 27. This is the best measured F1, but it remains far
below readiness.

## Stage analysis for H

| Stage | Serve recall | Hit recall | Bounce recall | Overall recall |
|---|---:|---:|---:|---:|
| Raw candidate generator | 1.0000 | 0.6471 | 0.5500 | 0.6250 |
| Physics verification | 1.0000 | 0.5294 | 0.4000 | 0.5000 |
| Player attribution | 1.0000 | 0.5294 | 0.4000 | 0.5000 |
| Event-type classification | 0.0000 | 0.4118 | 0.3000 | 0.3250 |
| Temporal suppression | 0.0000 | 0.3529 | 0.3000 | 0.3000 |
| Final authoritative events | 0.0000 | 0.3529 | 0.3000 | 0.3000 |

The largest stage loss is semantic typing: all serve recall disappears and
overall recall falls from `0.5000` after physical/player verification to
`0.3250`. Candidate recall is already too low, especially on video_10, so
semantic threshold loosening alone cannot reach the target.

## Resolution and grouped diagnostic checks

| Validation video | Development labels | Resolution | TP/FP/FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| video_08 | video_09 + video_10 | 720p | 5/11/5 | 0.3125 | 0.5000 | 0.3846 |
| video_09 | video_08 + video_10 | 720p | 5/16/7 | 0.2381 | 0.4167 | 0.3030 |
| video_10 | video_08 + video_09 | 1080p | 2/40/16 | 0.0476 | 0.1111 | 0.0667 |

These are grouped slices under one common H configuration.
`threshold_refit_on_development_groups=false`; therefore they are not a claim
of nested leave-one-video-out model selection. Every clip has already been used
for development. The sole 1080p group is also a different match/year/source,
so the resolution effect is confounded.

## Runtime

The offline event-only passes over 2,672 preserved frames took approximately
0.46–0.51 seconds per B–H variant. H measured about 5,251 event-analysis FPS.
This excludes raw video decode, neural detectors, tracking, pose, rendering,
and serialization. It demonstrates that the staged verifier is not expensive
relative to the known full pipeline, but it is not an end-to-end performance
benchmark.

No full H video inference was run. The most recent full diagnostic throughput
remains 26.663 FPS and 28.011 FPS on the two 720p clips and 8.956 FPS on the
1080p clip.

## Selection and readiness decision

H is retained as the final integrated diagnostic variant because it has the
highest measured overall F1 and preserves explicit provenance/abstention. It
is not selected for production freeze:

| Requirement | Target | H |
|---|---:|---:|
| Candidate recall | >= 0.90 | 0.6250 |
| Overall precision | >= 0.70 | 0.1519 |
| Overall recall | >= 0.75 | 0.3000 |
| Overall F1 | >= 0.70 | 0.2017 |
| PLAYER_HIT recall | >= 0.80 | 0.3529 |
| Serve operational | non-zero true performance | 0/3 TP |
| Bounce operational | preferably per-class F1 near 0.70 | 0.1935 |

```text
PHASE 6.4 EVENT RECOVERY: FAIL
READY FOR PRODUCTION/EVALUATOR FREEZE: NO
PRIMARY REMAINING BLOCKER: upstream 1080p tracking gaps coexist with high-kinematic post-rally false motion, preventing a usable candidate and authoritative-event precision/recall balance
```

No production/evaluator freeze is created, no pristine `video_11+` is consumed,
and Phase 7 is not started.

## Reproduction and artifacts

```powershell
python scripts/create_phase6_4_pre_recovery_audit.py
python scripts/evaluate_phase6_4_event_recovery.py --variant all --input-root outputs/phase6_4_qualification/cross_match_diagnostic_final
```

```text
scripts/evaluate_phase6_4_event_recovery.py
configs/phase6_analytics/pipeline.yaml
artifacts/validation/phase6_4_event_recovery_results.json
artifacts/validation/phase6_4_event_stage_metrics.json
artifacts/validation/phase6_4_event_forensic_table.json
artifacts/validation/phase6_4_false_negative_audit.json
outputs/phase6_4_qualification/cross_match_diagnostic_final/
```

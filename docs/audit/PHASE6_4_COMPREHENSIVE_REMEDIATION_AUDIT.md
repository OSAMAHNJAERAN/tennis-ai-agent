# Phase 6.4 Comprehensive Remediation Audit

## Outcome

Tracker provenance is resolved and current raw-candidate performance is no longer represented by stale trajectories. The selected pre-semantic diagnostic pipeline fails at Stage 2, so all downstream gates remain closed.

## Investigation and implementation

1. Verified local and GitHub branch state at SHA `918f876479ec0d59aa166116658977f3d3837e76`, ahead/behind `0/0`.
2. Froze evaluator v2.1 semantics and added a randomized exhaustive ≤5×5 matching oracle. The preserved baseline reproduced exactly.
3. Audited production YAML, class defaults, duplicated Phase 6.4 constructors, preserved run configs, source files, and historical claims.
4. Replayed preserved, starting-production, class-default, reconciled, and selected variants from identical frozen raw candidates. Historical 92.5% settings were not approximated.
5. Classified the initial cause as `MIXED`: stale preserved artifact plus configuration drift, followed by residual tracker-quality failures.
6. Performed grouped video diagnostics. The apparent `6/6/90` recall win was rejected because it increased out-of-frame trajectories. Tracker construction was centralized and a single switchable frame-bounds safety fix was ablated on the original `4/3/80` config.
7. Regenerated versioned trajectories, manifests, comparison metrics, all-40-GT lineage, video_10 18-event forensics, loss taxonomy, and pre-semantic ablation.
8. Recomputed the gate and stopped before FP-source, semantic, and shot stages as required.

## Before and after

| Measure | Preserved v2.1 | Starting production replay | Selected replay |
|---|---:|---:|---:|
| Stage 1 recall | .6750 | .8750 | .8750 |
| Stage 2 recall | .6250 | .6500 | .6500 |
| True Stage-3 TP/FP/FN | 22/47/18 | 22/33/18 | 22/43/18 |
| True Stage-3 P/R/F1 | .3188/.5500/.4037 | .4000/.5500/.4632 | .3385/.5500/.4190 |
| Out-of-frame trajectory points | 50 | 197 | 0 |
| Causal Stage 0→6 | 40→27→25→19→8→6→4 | 40→35→26→17→7→6→3 | 40→35→26→19→8→6→3 |

The config reconciliation is valid only as diagnostic development evidence. It does not satisfy the pre-semantic gate and is not a qualification claim.

## Integrity constraints

- Evaluator version, tolerance, matching, coverage, and semantics were not changed.
- GT was loaded after trajectories were produced and was used only for post-hoc evaluation/taxonomy.
- Coverage, scoring, game/set state, and review labels were not tracker inputs.
- No video-specific tracker branches or resolution/FPS literals were introduced.
- `video_11+` was not accessed.
- Existing frontend/theme and user FP-review files were not modified or staged.
- Existing preserved trajectories were not overwritten.
- No automated label is described as human-reviewed.

## Deepest gate

`PHASE 6.4 TRACKER PROVENANCE: PASS`

`PRE-SEMANTIC PIPELINE: FAIL`

`NEXT BLOCKER: STAGE_1_GROUNDED_USABLE_OBSERVATION_RECALL (35/40 = .875, required ≥ .90; video_10 = 13/18)`

The exact machine-readable evidence is under `artifacts/validation/phase6_4_tracker_*` and `artifacts/validation/phase6_4_presemantic_ablation.json`.

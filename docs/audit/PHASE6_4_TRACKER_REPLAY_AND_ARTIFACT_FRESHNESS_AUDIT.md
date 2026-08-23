# Phase 6.4 Tracker Replay and Artifact Freshness Audit

## Scientific status

This audit uses only `video_08`–`video_10`, a diagnostic split already consumed during development. It is not qualification evidence. Annotation coverage remains model-assisted provisional and is not human-approved. No `video_11+` media, GT, coverage interval, score state, or review label entered tracker inference.

## Audited state

- Starting branch: `codex/phase6-4-final-correction`
- Starting local and remote SHA: `918f876479ec0d59aa166116658977f3d3837e76`
- Evaluator: v2.1, unchanged semantics, deterministic maximum-cardinality/minimum-total-timing-error matching
- Preserved input: `outputs/phase6_4_qualification/cross_match_diagnostic_final/video_08`–`video_10`
- Frozen raw proposals: `artifacts/validation/raw_candidates/video_08_candidates.json`–`video_10_candidates.json`

## Provenance finding

The preserved `trajectories.json` files contain trajectory values but omit the generating Git SHA, tracker source hash, complete tracker settings, raw-candidate hash, generation command, and generation timestamp. Co-located `run_config.yaml` files establish partial `4/3/80` evidence but do not identify the source revision or feature-toggle state. The historical `37/40 = 92.5%` recovery configuration is therefore `NOT_REPRODUCIBLE`; no approximation was run.

The v2.1 reconstruction was confirmed to load these preserved trajectories, not replay HEAD. It reproduced the documented baseline exactly: causal `40→27→25→19→8→6→4`, with all 13 Stage-1 losses in `video_10`.

## Executable configuration paths

At the starting SHA, production YAML resolved to `high=.08`, `low=.01`, prediction/interpolation gaps `4/3`, speed bound `80 px/frame`, base gate `45 px`, multi-candidate/adaptive/reacquisition enabled, camera compensation disabled, and resolution scaling enabled. Class defaults differed at `6/6/90`. Phase 6 scripts also contained duplicated `4/3/80` constructors.

The replay constructs `BallObservation` directly from frozen JSON candidates. All trajectories are completed and hashed before GT is loaded. Player boxes and the unchanged event detector are applied only after tracking.

## Case verdict

Initial root-cause classification: `MIXED`.

- Staleness component: fresh starting-production replay raises aggregate Stage-1 recall from `27/40` to `35/40`.
- Configuration component: class defaults raise it to `36/40`, exactly the Stage-1 threshold, while Stage-2 recall improves from `26/40` to `27/40` and covered Stage-3 predictions fall from 55 to 54.
- Residual component: four `video_10` Stage-1 windows still fail. Post-hoc frames show that candidate existence does not establish true-ball proposal identity; without point-localized proposal labels, those four causes remain `UNKNOWN`. Tracker association was not relaxed to manufacture a match.

## Grouped diagnostic validation

The fixed `4/3/80` and `6/6/90` configurations were compared on every held-out video grouping. This is a development check, not qualification.

| Train groups | Held out | 4/3/80 Stage 1 | 6/6/90 Stage 1 | 4/3/80 Stage 2 | 6/6/90 Stage 2 | 4/3/80 Stage-3 F1 | 6/6/90 Stage-3 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 08+09 | 10 | .7222 | .7778 | .2778 | .3333 | .2759 | .3448 |
| 08+10 | 09 | 1.0000 | 1.0000 | .9167 | .9167 | .5116 | .5238 |
| 09+10 | 08 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | .6087 | .5217 |

`6/6/90` is Stage-1/2 non-inferior on all three held-out groups and materially improves the collapsing `video_10` group. However, the required quality audit found 252 out-of-frame trajectory points versus 197 under starting production. It therefore did not win honestly under the no-hallucination rule and was rejected, despite its better temporal-match metric.

## Corrective action

`src/tracking/tracker_factory.py` now resolves all tracker settings through one strict typed path. Missing keys fail instead of silently invoking class defaults. `Phase6Pipeline` and active Phase 6.4 evaluators use the factory. The active config retains `4/3/80` and explicitly enables one targeted safety repair: candidates and synthesized points outside the declared frame are rejected. This reduces selected out-of-frame points to zero. Pre-fix behavior remains independently replayable with the toggle disabled.

Fresh, versioned replay trajectories and their hashes are recorded under `artifacts/validation/phase6_4_tracker_replay/`. Preserved trajectories remain intact as historical diagnostic evidence.

## Remaining blocker

The safety-correct selected replay does not reach Stage 1: independent grounded recall is `35/40 = .875`, below `.90`, with `video_10` at `13/18`. Stage 2 is `26/40 = .65`; true Stage-3 is `22 TP / 43 FP / 18 FN`, `P=.3385 / R=.55 / F1=.4190`. Downstream FP-source remediation, semantic tuning, and shot-classifier work remain closed.

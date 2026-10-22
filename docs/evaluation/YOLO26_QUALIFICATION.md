# YOLO26 qualification evidence contract

9 October 2026. **No YOLO26 specialist bundle is independently qualified.**
The candidate manifest keeps `qualification_receipt: null`. Pilot losses, COCO
metrics, agent-reviewed labels, and passing software tests cannot supply this
receipt. The retained [check record](YOLO26_QUALIFICATION_CHECK_2026-10-09_V2.json)
reports missing production ball weights and `production_eligible: false`; its
adjacent `.sha256` file pins the exact record bytes. It did not execute a vision
evaluation. Later code/configuration changes require a new check record.

## Production boundary

`validate_yolo26_config(..., check_assets=True)` checks the three actual artifact
hashes and provenance, then invokes
`src/evaluation/yolo26_qualification.py::validate_qualification_receipt` before
model loading. The old `independent_accuracy_validated` fields are informational
and cannot promote a model, even when set to `true`. The successful runtime return
contains `verified_qualification` with receipt hash and individual qualified tasks.
Configuration-only inspection remains available with `check_assets=False`.
Historical configurations retain their existing behavior.

The deployment owner obtains evidence from an independent evaluator and pins
the receipt in `configs/production/yolo26.artifacts.json`:

```json
"qualification_receipt": {
  "path": "data/yolo26/qualification/version-id/receipt.json",
  "sha256": "<lowercase SHA256 of the retained receipt bytes>"
}
```

This example is an incomplete contract fragment, not supplied qualification
evidence. All evidence paths are relative to the repository retention root;
absolute paths, path traversal, and symlink escapes are rejected. Private retained
evidence must stay private. Hashes prove file consistency, not a person's identity
or the truth of their statements. The owner must verify evaluator independence,
source rights, completeness of the development registry and the authenticity of
human review. The checker cannot discover an omitted development match or
authenticate an invented reviewer. No arbitrary pass receipt is created here.

## Required retained documents

Every document descriptor is `{ "path": "relative/path", "sha256": "..." }`.
The receipt uses schema `tennis.yolo26.qualification.v1` and records `receipt_id`,
`evidence_origin: EXTERNAL_INDEPENDENT_EVALUATION`, timezone-aware
`evaluation_started_at` and `created_at`, `developer_ids`, the exact three-role
`model_sha256` map, `config_sha256`, and `evidence` descriptors for `dataset`,
`protocol`, `results`, `review`, and a list of `references`.

The configuration hash is produced by `qualification_config_sha256(config,
root=...)`. It pins model operating settings, temporal processing, geometry,
camera/identity settings, events and line settings, and actual tracker bytes.
Artifact and receipt paths, manifest status and weight path aliases are omitted;
exact weight bytes are separately pinned. Per-video manual review/correction
inputs remain in the analysis audit, separately from operating settings.
Any inference or tracker-setting change needs evaluation under
the new fingerprint. The evaluator must retain its evaluated implementation and
environment; source revision is a required protocol field.

1. **Dataset:** `frozen: true`, `frozen_at`, `scope:
   elevated_end_court_singles`, `duplicate_audit_complete: true`, a documented
   `duplicate_audit_method`, `sources` and `samples`. Source records contain
   `source_id`, `match_id`, `session_id`, `original_source_video_id`, original
   `sha256`, explicit `split` (`train`, `val`, `development`, `test`), owner,
   URL/original path, rights evidence, `evaluation_allowed`, acquired UTC date,
   surface, view, original resolution, FPS and duration. Test sources require a
   reviewed camera and `used_for_development: false`.
2. **Samples:** Each sample identifies its source, split, `sample_id`, model
   role, original frame index/timestamp, retained `media` and `annotation`
   descriptors, and audited `near_duplicate_group`. Test samples require
   `reviewer_type: HUMAN`, independent reviewer IDs,
   `independent_human_reviewed: true`, and annotation provenance
   `HUMAN_MANUAL` or `PUBLISHER_LABEL_HUMAN_REVIEWED`. Agent/model labels cannot
   freeze final truth. Frames or temporal clips can be retained; the frozen
   denominator must state what each evaluates. The training conversion manifest
   is input to the external audit, not automatically this qualification dataset.
3. **Protocol:** `frozen_at`, source scope, code revision, hashed
   `environment_lock` and `evaluator_implementation` files, frozen model/config
   hashes, `minimum_independent_test_matches`, and `tasks`. Each task freezes
   measurement, denominator and exclusion definitions, a nonempty tolerance
   object, its independent `reference_id`, and numerical thresholds written as
   `{ "metric": { "operator": ">=", "value": 0.95 } }`. Freeze before the
   final evaluation, and preserve split reservation before training in the
   retained source audit. A lower master target cannot be relabeled as a pass by
   this checker; a changed approved protocol needs a versioned code/policy review.
4. **Results:** `tasks` with the fields described below, pinned to every relevant
   frozen test sample. Every core task covers all final samples for its role.
   Split grouping checks match, camera session, original video, original file
   hash, sample media hash and near-duplicate group. Sources seen in development
   must stay development even if their IDs or manifest version change.
5. **Review:** An independent `HUMAN_EXTERNAL` reviewer identity and review date,
   hashes of the exact dataset/protocol/results, declarations
   `source_registry_complete` and `no_test_development_reuse`, and one
   `task_decisions` record per evaluated task. Each decision requires
   `decision: ACCEPTED`, `reference_sha256`, rationale, sample-size rationale and
   unknown/coverage rationale. Acceptance alone cannot override failed numbers.
6. **References:** Each reference receipt identifies its `reference_id`, tasks,
   complete task source IDs, reference method, measurement definitions per task,
   independent external human reviewer, and `independent_of_model: true`.
   Physical ball speed additionally requires a synchronized reference matching
   the precise speed definition. Fit residuals, publisher camera calibration or
   trajectory agreement alone do not establish an independent measurement.

At least two independent matches per task are needed to support the declared
match-grouped interval. This is a minimum consistency requirement, **not** a
claim that two matches establish 95% performance. The evaluator must predeclare
a sufficient sample size and review its adequacy; the collection plan calls for
considerably more diversity. The software does not invent a confidence-bound
acceptance threshold that the master did not specify.

## Task results and numerical gates

Each result repeats the exact frozen definitions and tolerance, lists unique
final `sample_ids`, and reports integer `total`, `evaluable`, `measured`, `unknown`,
and `excluded`. The invariants are `total = evaluable + excluded` and
`evaluable = measured + unknown`; denominators are nonempty. Publish numerical
`metrics`, reconciled `match_groups`, a match-grouped `confidence_interval` with
method/bounds/independent match count, and subgroup `breakdowns` with their
evaluable/measured/unknown counts and numerical metrics. Near/far diagnostics are
required for identity; ball additionally requires blur diagnostics.

Every task retains an `unknown_cases` JSON file with `task`, `unknown_count` and
`cases`. Each case or counted interval has a unique `case_id`, registered match,
`count`, reason and original `source_interval`; counts reconcile with the task.
This prevents quietly discarding unresolved observations. Event/point truth and
the review must still include every point, not just automatically decided points.

| Task | Checked target and retained measurements |
| --- | --- |
| `visible_ball` (required) | Recomputed TP/(TP+FP) and TP/(TP+FN), each >=0.95. FN includes every visible miss/unknown. Require visible/absent/occluded/ambiguous frame counts, no-valid-ball frames and duration, original-image pixel units, median/P95 pixel localization error, false alarms per evaluated minute, and blur/far diagnostics. |
| `player_identity` (required) | Recomputed `2*idtp/(2*idtp+idfp+idfn)` >=0.95. Unidentified ground truth belongs in IDFN. Publish HOTA as a fraction, identity switches, interruptions and match/near/far breakdowns. |
| `player_ground_position` (required) | One retained independent-reference absolute error per measured observation, in meters. Recomputed median <=0.20 m and P95 <=0.50 m. |
| `calibration` (required) | Independent verification points **and lines excluded from fitting**, errors in meters in separated regions, valid-time fraction and coverage. The external protocol must supply concrete thresholds and reference acceptance; no meter or coverage target is invented. |
| `strokes`, `bounces` | Recomputed precision >=0.95 and recall >=0.90, one-to-one matching within at most +/-80 ms, dead-ball review and declared timing tolerance. |
| `player_speed` | Recomputed MAE <=0.5 m/s from retained errors, with bias, P95, window definition and unknown/coverage accounting. |
| `physical_ball_speed` | Recomputed median absolute error <=5 km/h and P95 <=10 km/h, synchronized independent reference and stroke-type coverage. |
| `point_winner` | Recomputed precision >=0.95 of automatic decisions and coverage >=0.80 of evaluable points. Incorrect decisions remain in the measured denominator; unknowns remain visible. |
| `scoring_rules` | All declared cases pass; zero failed cases, duplicate points and score drift; complete-match review and external reference decision. |
| `line_decisions`, `win_prediction`, `long_live_video`, `multi_user_isolation` | Separate independently reviewed reference and externally frozen numerical policy. The registry does not substitute a scalar vision target for these acceptance methods. |

Error P95 uses linear interpolation at `(n-1)*0.95`; freeze this method with the
protocol. Recomputed metrics disagreeing with a supplied summary fail (comparison
tolerance 1e-9). These checks verify arithmetic and consistency of the retained
external evaluation, not the authenticity or correctness of every reference.

Core bundle approval covers only its individual validated vision/geometry tasks.
It does not qualify speed, authoritative events/scoring, line calls, prediction or
live capacity. Call the receipt checker with additional `required_tasks`, or the
command with `--require-task`, when enabling those capabilities. The wider master
definition of done remains incomplete until their separate evaluations pass.

## Run and retain a check

```powershell
python -m scripts.evaluate.check_yolo26_qualification --output outputs/yolo26_qualification/new-check.json
python -m scripts.evaluate.check_yolo26_qualification --output outputs/yolo26_qualification/new-speed-check.json --require-task player_speed --require-task physical_ball_speed
python -m pytest tests/test_yolo26_qualification.py tests/test_yolo26_models.py -q --basetemp artifacts/test_tmp/new-qualification-test
```

Create the basetemp parent directory first when it does not exist. Check output
is immutable, includes configuration/manifest/checker hashes, has an adjacent
SHA256 sidecar, and exits nonzero for unqualified candidates. It does not create
labels, measurements, references or a pass receipt. New evaluations and evidence
corrections belong to new retained versions, not overwrites of frozen files.

The checker tests use explicitly named `SYNTHETIC_SOFTWARE_FIXTURE` receipts and
nonmodel toy bytes. Only the direct test API can allow these fixtures; it still
returns `production_eligible: false`. The bundle rejects that evidence origin.
Software validation is separate from the outstanding independent human labels,
qualified specialist checkpoints and physical reference measurements.

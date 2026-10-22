"""Verify retained external qualification evidence; never generate model accuracy.

Receipts are integrity records, not signatures or a substitute for human review.
The deployment owner must obtain the receipt from an independent evaluator and
pin its hash in the candidate manifest. Synthetic fixtures cannot promote models.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = "tennis.yolo26.qualification.v1"
CORE_TASKS = frozenset({"visible_ball", "player_identity", "player_ground_position", "calibration"})
TARGETS = {
    "visible_ball": {"precision": (">=", .95), "recall": (">=", .95)},
    "player_identity": {"idf1": (">=", .95)},
    "player_ground_position": {"median_error_m": ("<=", .20), "p95_error_m": ("<=", .50)},
    "strokes": {"precision": (">=", .95), "recall": (">=", .90)},
    "bounces": {"precision": (">=", .95), "recall": (">=", .90)},
    "player_speed": {"mae_m_s": ("<=", .5)},
    "physical_ball_speed": {"median_absolute_error_km_h": ("<=", 5.), "p95_absolute_error_km_h": ("<=", 10.)},
    "point_winner": {"precision": (">=", .95), "coverage": (">=", .80)},
}
EXTERNAL_POLICY_TASKS = frozenset({"calibration", "line_decisions", "win_prediction", "long_live_video", "multi_user_isolation"})
KNOWN_TASKS = frozenset(TARGETS) | EXTERNAL_POLICY_TASKS | {"scoring_rules"}
ROLE_TASK = {"ball": "visible_ball", "players": "player_identity", "court": "calibration"}
ERROR_TASKS = {
    "player_ground_position": ("median_error_m", "p95_error_m", "m"),
    "physical_ball_speed": ("median_absolute_error_km_h", "p95_absolute_error_km_h", "km/h"),
}


class QualificationError(ValueError):
    """Missing, altered, leaked or failing qualification evidence."""


def _require(condition, message):
    if not condition:
        raise QualificationError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Require lowercase SHA256")
    return value


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {name}")
    return value


def _number(value, name, minimum=0):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum,
             f"Invalid finite {name}")
    return float(value)


def _count(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"Invalid count {name}")
    return value


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(result.tzinfo is not None, "Evidence dates require a timezone")
        return result.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QualificationError("Invalid qualification evidence date") from exc


def _load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key {key}")
            result[key] = value
        return result
    def constant(value):
        raise QualificationError(f"Nonfinite JSON value {value}")
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"Unreadable retained JSON evidence: {path}") from exc
    _require(isinstance(result, dict), f"Evidence must be a JSON object: {path}")
    return result


def _retained(ref, root, *, as_json=True):
    _require(isinstance(ref, dict), "Missing retained evidence path/hash")
    relative = Path(_text(ref.get("path"), "retained evidence path"))
    # Receipts are portable, and cannot resolve a path outside their retention root.
    _require(not relative.is_absolute(), "Retained evidence paths must be relative to the repository root")
    path = (Path(root) / relative).resolve()
    _require(path.is_relative_to(Path(root).resolve()), "Retained evidence escapes its retention root")
    _require(path.is_file(), f"Missing retained evidence: {relative}")
    _require(sha256_file(path) == _digest(ref.get("sha256")), f"Retained evidence SHA256 mismatch: {relative}")
    return _load_json(path) if as_json else path


def qualification_config_sha256(config, *, root):
    """Pin operating settings and exact tracker bytes; ignore promotion metadata.

    Paths to the receipt and artifacts do not affect inference and are omitted.
    Artifact bytes are independently pinned by the receipt's model_sha256 map.
    """
    sections = ("model_system", "player_detection", "ball_detection", "court_detection", "pose_estimation",
                "temporal_tracking", "player_motion", "event_detection", "line_calling", "shot_classification",
                "camera_intervals", "match_identity")
    selected = json.loads(json.dumps({key: config.get(key, {}) for key in sections}, allow_nan=False))
    for key in ("artifact_manifest", "qualification_receipt", "status"):
        selected["model_system"].pop(key, None)
    for section, key in (("player_detection", "model"), ("ball_detection", "model"), ("court_detection", "model_path"), ("pose_estimation", "model_path")):
        selected[section].pop(key, None)
    # Reviews/corrections are per-video evidence, like the uploaded media. Their
    # records belong to the versioned analysis audit, not model operating settings.
    for section, keys in (("camera_intervals", ("reviews", "reviews_path")), ("match_identity", ("anchors", "anchors_path"))):
        for key in keys:
            selected[section].pop(key, None)
    tracker = Path(config["player_detection"]["tracker"])
    tracker = tracker if tracker.is_absolute() else Path(root) / tracker
    selected["player_detection"]["tracker"] = {"sha256": sha256_file(tracker)}
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _dataset(document, root, evaluation_start, developer_ids):
    _require(document.get("frozen") is True and _time(document.get("frozen_at")) < evaluation_start, "Dataset must be frozen before final evaluation")
    _require(document.get("scope") == "elevated_end_court_singles", "Qualification is limited to elevated end-court singles")
    _text(document.get("duplicate_audit_method"), "duplicate/near-duplicate audit method")
    _require(document.get("duplicate_audit_complete") is True, "Duplicate/near-duplicate audit is incomplete")
    sources, samples = document.get("sources"), document.get("samples")
    _require(isinstance(sources, list) and bool(sources) and isinstance(samples, list) and bool(samples), "Require retained sources and annotated samples")
    by_id, groups = {}, {}
    for source in sources:
        _require(isinstance(source, dict), "Invalid source record")
        identifier = _text(source.get("source_id"), "source ID")
        _require(identifier not in by_id, "Duplicate source ID")
        split = source.get("split")
        _require(split in {"train", "val", "test", "development"}, "Every source needs an explicit grouped split")
        for key in ("match_id", "session_id", "original_source_video_id", "sha256"):
            value = _digest(source.get(key)) if key == "sha256" else _text(source.get(key), key)
            group = (key, value)
            _require(group not in groups or groups[group] == split, f"Split leakage through {key}")
            groups[group] = split
        _text(source.get("rights_evidence"), "source rights evidence")
        for key in ("owner", "surface"):
            _text(source.get(key), key)
        _require(bool(source.get("url") or source.get("source_path")), "Source requires an original URL or path")
        _time(source.get("acquired_at"))
        _number(source.get("fps"), "original source FPS", minimum=1e-12)
        _number(source.get("duration_seconds"), "source duration", minimum=1e-12)
        resolution = source.get("resolution")
        _require(isinstance(resolution, list) and len(resolution) == 2 and all(_count(value, "resolution") > 0 for value in resolution), "Source requires original resolution")
        _require(source.get("evaluation_allowed") is True, "Source rights do not authorize evaluation")
        if split == "test":
            _require(source.get("view") == document["scope"] and source.get("camera_reviewed") is True, "Final source has no reviewed supported view")
            _require(source.get("used_for_development") is False, "Final source has been used for development")
        by_id[identifier] = source
    _require(any(item["split"] in {"train", "development"} for item in sources) and any(item["split"] == "val" for item in sources), "Declare all training/development and validation sources")
    test_sources = {key: value for key, value in by_id.items() if value["split"] == "test"}
    _require(bool(test_sources), "Missing independent final test sources")
    sample_by_id, duplicates = {}, {}
    for sample in samples:
        _require(isinstance(sample, dict), "Invalid annotated sample")
        identifier = _text(sample.get("sample_id"), "sample ID")
        _require(identifier not in sample_by_id, "Duplicate sample ID")
        source = by_id.get(sample.get("source_id"))
        _require(source is not None and sample.get("split") == source["split"], "Sample and source split disagree")
        _require(sample.get("role") in ROLE_TASK, "Unknown annotation role")
        _count(sample.get("frame_index"), "source frame index")
        timestamp = _number(sample.get("timestamp_seconds"), "original source timestamp")
        _require(timestamp <= source["duration_seconds"], "Sample timestamp exceeds source duration")
        for key in ("media", "annotation"):
            _retained(sample.get(key), root, as_json=False)
        for key, value in (("media_sha256", sample["media"]["sha256"]), ("near_duplicate_group", sample.get("near_duplicate_group"))):
            _text(value, key)
            group = (key, value)
            _require(group not in duplicates or duplicates[group] == source["split"], f"Sample split leakage through {key}")
            duplicates[group] = source["split"]
        if source["split"] == "test":
            _require(sample.get("annotation_provenance") in {"HUMAN_MANUAL", "PUBLISHER_LABEL_HUMAN_REVIEWED"}, "Model/agent labels cannot be final ground truth")
            reviewers = sample.get("reviewer_ids")
            _require(sample.get("reviewer_type") == "HUMAN" and sample.get("independent_human_reviewed") is True
                     and isinstance(reviewers, list) and bool(reviewers) and all(isinstance(value, str) and value for value in reviewers)
                     and not (set(reviewers) & developer_ids), "Final samples require independent human review")
        sample_by_id[identifier] = sample
    for role in ROLE_TASK:
        _require(any(sample["split"] == "test" and sample["role"] == role for sample in samples), f"Missing reviewed final {role} samples")
    return test_sources, sample_by_id


def _match_percentile(errors, percentile=.95):
    # Frozen linear interpolation, matching NumPy's default percentile method.
    values = sorted(errors)
    position = (len(values) - 1) * percentile
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def _metrics(task, row):
    total, evaluable, measured, unknown, excluded = (_count(row.get(key), key) for key in ("total", "evaluable", "measured", "unknown", "excluded"))
    _require(evaluable > 0 and measured > 0 and total == evaluable + excluded and evaluable == measured + unknown, "Task denominator/unknown/exclusion accounting is inconsistent")
    _text(row.get("denominator_definition"), "task denominator definition")
    _text(row.get("exclusion_definition"), "task exclusion definition")
    _text(row.get("measurement_definition"), "task measurement definition")
    _require(isinstance(row.get("tolerance"), dict) and bool(row["tolerance"]), "Missing frozen task tolerance")
    _require(isinstance(row.get("metrics"), dict), "Task metrics must be a numerical object")
    metrics = {key: _number(value, key, minimum=-float("inf")) for key, value in row["metrics"].items()}
    metrics["coverage"] = measured / evaluable
    if task in {"visible_ball", "strokes", "bounces", "point_winner"}:
        counts = row.get("counts", {})
        tp, fp, fn = (_count(counts.get(key), key) for key in ("tp", "fp", "fn"))
        _require(tp + fp > 0 and tp + fn > 0, "Empty precision/recall denominator")
        if task != "point_winner":
            _require(tp == measured and tp + fn == evaluable and fn == unknown, "Detection denominator must include missed/unknown ground truth")
        else:
            _require(tp + fp == measured, "Automatic point decisions must include incorrect decisions")
        metrics.update(precision=tp / (tp + fp), recall=tp / (tp + fn))
        if task == "visible_ball":
            states = row.get("ground_truth_states", {})
            _require(_count(states.get("visible"), "visible") == evaluable and _count(states.get("absent"), "absent") > 0, "Ball test must include visible and no-valid-ball frames")
            for key in ("occluded", "ambiguous"):
                _count(states.get(key), key)
            _require(sum(states.values()) == total, "Ball denominator must include every annotated visibility state")
            duration = _number(row.get("evaluated_duration_minutes"), "evaluated duration", minimum=1e-12)
            negative_duration = _number(row.get("negative_duration_minutes"), "negative duration", minimum=1e-12)
            _require(negative_duration <= duration and row.get("unit") == "original_image_px", "Ball localization/timing must use original source units")
            for key in ("median_localization_error_px", "p95_localization_error_px", "false_alarms_per_minute"):
                _number(metrics.get(key), key)
            metrics["false_alarms_per_minute"] = fp / duration
        if task in {"strokes", "bounces"}:
            _require(row["tolerance"].get("matching") == "one_to_one" and 0 < _number(row["tolerance"].get("milliseconds"), "event tolerance") <= 80, "Event tolerance must be one-to-one and at most +/-80 ms")
            _require(row.get("dead_ball_controls_reviewed") is True, "Events need dead-ball controls")
    if task == "player_identity":
        counts = row.get("counts", {})
        tp, fp, fn = (_count(counts.get(key), key) for key in ("idtp", "idfp", "idfn"))
        _require(tp == measured and tp + fn == evaluable and fn == unknown, "Identity denominator must include unidentified ground truth")
        metrics["idf1"] = 2 * tp / (2 * tp + fp + fn)
        _require(0 <= _number(metrics.get("hota"), "HOTA") <= 1, "HOTA must be a fraction")
        for key in ("identity_switches", "interruptions"):
            _count(row.get(key), key)
    if task in ERROR_TASKS or task == "player_speed":
        errors = row.get("absolute_errors")
        _require(isinstance(errors, list) and len(errors) == measured, "Retain one independent-reference error per measured observation")
        errors = [_number(value, "absolute error") for value in errors]
        if task == "player_speed":
            _require(row.get("unit") == "m/s", "Player speed requires m/s")
            metrics["mae_m_s"] = statistics.mean(errors)
            for key in ("bias_m_s", "p95_error_m_s"):
                _number(metrics.get(key), key, minimum=-float("inf") if key == "bias_m_s" else 0)
            metrics["p95_error_m_s"] = _match_percentile(errors)
        else:
            median_key, p95_key, unit = ERROR_TASKS[task]
            _require(row.get("unit") == unit, "Incorrect reference error units")
            metrics[median_key], metrics[p95_key] = statistics.median(errors), _match_percentile(errors)
    if task == "calibration":
        _require(row.get("unit") == "m" and row.get("verification_points_excluded_from_fit") is True and row.get("verification_lines_excluded_from_fit") is True, "Calibration needs independent points AND lines excluded from the fit in meters")
        regions = row.get("verification_regions")
        _require(isinstance(regions, list) and len(set(regions)) >= 2, "Calibration must report widely separated regions")
        for key in ("median_error_m", "p95_error_m", "valid_time_fraction"):
            _number(metrics.get(key), key)
        _require(metrics["median_error_m"] <= metrics["p95_error_m"], "Calibration error percentiles are inverted")
        _require(metrics["valid_time_fraction"] <= 1, "Invalid calibration valid-time fraction")
    if task == "scoring_rules":
        _require(_count(row.get("specified_cases"), "specified scoring cases") > 0, "Missing scoring cases")
        for key in ("failed_cases", "duplicated_points", "score_drift_points"):
            _require(_count(row.get(key), key) == 0, f"Scoring failed: {key}")
        _require(row.get("complete_match_reviewed") is True, "Scoring requires complete reviewable matches")
    for key, value in metrics.items():
        if key in row.get("metrics", {}):
            _require(math.isclose(row["metrics"][key], value, rel_tol=1e-9, abs_tol=1e-9), f"Reported {task} {key} disagrees with retained counts/errors")
    return metrics


def _apply_thresholds(task, metrics, policy):
    thresholds = policy.get("thresholds")
    _require(isinstance(thresholds, dict), "Missing predeclared thresholds")
    if task in EXTERNAL_POLICY_TASKS:
        _require(bool(thresholds), f"{task} requires external predeclared acceptance thresholds; no default is invented")
    for metric, (operator, limit) in TARGETS.get(task, {}).items():
        declared = thresholds.get(metric, {})
        _require(declared.get("operator") == operator, f"Missing/wrong {task} {metric} threshold")
        value = _number(declared.get("value"), metric)
        _require(value >= limit if operator == ">=" else value <= limit, f"Cannot lower master target for {task} {metric}")
    for key, threshold in thresholds.items():
        _require(isinstance(threshold, dict) and threshold.get("operator") in {">=", "<="}, "Unknown threshold operator")
        value = _number(threshold.get("value"), key, minimum=-float("inf"))
        measured = _number(metrics.get(key), key, minimum=-float("inf"))
        passed = measured >= value if threshold["operator"] == ">=" else measured <= value
        _require(passed, f"Numerical qualification failed: {task} {key}={measured} {threshold['operator']} {value}")


def validate_qualification_receipt(manifest, config, *, root, required_tasks=CORE_TASKS, allow_synthetic=False):
    """Return integrity-checked evidence summary or raise QualificationError.

    `allow_synthetic` exists exclusively for isolated software fixtures; the
    production bundle never enables it. This function does not change the manifest.
    """
    _require(manifest.get("qualification_receipt") is not None, "Production needs independent tennis qualification receipt; pilot weights are not promoted")
    receipt = _retained(manifest["qualification_receipt"], root)
    _require(receipt.get("schema_version") == SCHEMA, "Unsupported qualification receipt schema")
    origin = receipt.get("evidence_origin")
    synthetic = origin == "SYNTHETIC_SOFTWARE_FIXTURE"
    _require(origin == "EXTERNAL_INDEPENDENT_EVALUATION" or (synthetic and allow_synthetic), "Synthetic/development evidence cannot promote production")
    _text(receipt.get("receipt_id"), "receipt ID")
    _require(receipt.get("model_sha256") == {role: _digest(record.get("sha256")) for role, record in manifest["models"].items()}, "Receipt does not bind the exact selected three weight hashes")
    _require(receipt.get("config_sha256") == qualification_config_sha256(config, root=root), "Inference/tracker settings differ from qualification")
    evaluation_start = _time(receipt.get("evaluation_started_at"))
    _require(evaluation_start < _time(receipt.get("created_at")), "Receipt must follow evaluation")
    developer_ids = receipt.get("developer_ids")
    _require(isinstance(developer_ids, list) and bool(developer_ids) and all(isinstance(value, str) and value for value in developer_ids), "Declare development team identities")
    developer_ids = set(developer_ids)
    evidence = receipt.get("evidence", {})
    dataset = _retained(evidence.get("dataset"), root)
    protocol = _retained(evidence.get("protocol"), root)
    results = _retained(evidence.get("results"), root)
    review = _retained(evidence.get("review"), root)
    _require(_time(protocol.get("frozen_at")) < evaluation_start, "Metric definitions/thresholds must be frozen before evaluation")
    _require(protocol.get("scope") == dataset.get("scope"), "Protocol and source scope disagree")
    _text(protocol.get("code_revision"), "evaluated code revision")
    _retained(protocol.get("environment_lock"), root, as_json=False)
    _retained(protocol.get("evaluator_implementation"), root, as_json=False)
    _require(protocol.get("model_sha256") == receipt["model_sha256"] and protocol.get("config_sha256") == receipt["config_sha256"], "Protocol does not freeze selected models/settings")
    test_sources, samples = _dataset(dataset, root, evaluation_start, developer_ids)
    matches = {source["match_id"] for source in test_sources.values()}
    minimum_matches = _count(protocol.get("minimum_independent_test_matches"), "minimum test matches")
    _require(minimum_matches >= 2 and len(matches) >= minimum_matches, "Insufficient independent matches for match-grouped qualification")
    _require(review.get("reviewer_type") == "HUMAN_EXTERNAL" and review.get("reviewer_id") not in developer_ids, "Acceptance requires an independent external human reviewer")
    _text(review.get("reviewer_id"), "external reviewer identity")
    _require(_time(review.get("reviewed_at")) >= evaluation_start, "Review predates final evaluation")
    _require(review.get("dataset_sha256") == evidence["dataset"]["sha256"] and review.get("results_sha256") == evidence["results"]["sha256"]
             and review.get("protocol_sha256") == evidence["protocol"]["sha256"], "External review does not bind retained dataset/results/protocol")
    _require(review.get("source_registry_complete") is True and review.get("no_test_development_reuse") is True, "External source completeness/leakage review is missing")
    references = {}
    for ref in evidence.get("references", []):
        reference = _retained(ref, root)
        identifier = _text(reference.get("reference_id"), "independent reference ID")
        _require(identifier not in references, "Duplicate reference ID")
        _require(reference.get("independent_of_model") is True and reference.get("reviewer_type") == "HUMAN_EXTERNAL"
                 and reference.get("reviewer_id") not in developer_ids, "Reference must be independently human verified")
        _text(reference.get("reviewer_id"), "reference reviewer")
        _text(reference.get("method"), "independent reference method")
        references[identifier] = (reference, ref["sha256"])
    tasks, gates = results.get("tasks", {}), protocol.get("tasks", {})
    _require(isinstance(tasks, dict) and CORE_TASKS <= set(tasks) and set(required_tasks) <= set(tasks) and set(tasks) <= KNOWN_TASKS, "Qualification is missing mandatory tasks or declares an unknown task")
    _require(set(tasks) == set(gates) == set(review.get("task_decisions", {})), "Results, frozen gates and external task decisions differ")
    checked = {}
    for task, row in tasks.items():
        _require(isinstance(row, dict) and isinstance(gates[task], dict), "Invalid task result or frozen gate")
        policy = gates[task]
        for key in ("measurement_definition", "denominator_definition", "exclusion_definition", "tolerance"):
            _require(row.get(key) == policy.get(key), f"Final {task} definition differs from frozen protocol: {key}")
        sample_ids = row.get("sample_ids")
        _require(isinstance(sample_ids, list) and bool(sample_ids) and len(sample_ids) == len(set(sample_ids))
                 and all(identifier in samples and samples[identifier]["split"] == "test" for identifier in sample_ids), "Task evaluation contains development/unregistered samples")
        relevant_roles = [role for role, name in ROLE_TASK.items() if name == task]
        if task == "player_ground_position":
            relevant_roles = ["players"]
        if relevant_roles:
            required_samples = {key for key, sample in samples.items() if sample["split"] == "test" and sample["role"] in relevant_roles}
            _require(set(sample_ids) == required_samples, f"{task} omits frozen test samples")
        task_matches = {test_sources[samples[identifier]["source_id"]]["match_id"] for identifier in sample_ids}
        _require(len(task_matches) >= minimum_matches, f"Insufficient independent matches for {task}")
        group_rows = row.get("match_groups")
        _require(isinstance(group_rows, list) and len(group_rows) == len(task_matches) and {group.get("match_id") for group in group_rows} == task_matches, "Publish results per independent match")
        for key in ("total", "evaluable", "measured", "unknown", "excluded"):
            _require(sum(_count(group.get(key), key) for group in group_rows) == row.get(key), "Per-match denominators do not reconcile")
        for group in group_rows:
            _require(group["total"] == group["evaluable"] + group["excluded"] and group["evaluable"] == group["measured"] + group["unknown"], "Per-match denominator/unknown accounting is inconsistent")
            _require(isinstance(group.get("metrics"), dict) and bool(group["metrics"]), "Missing per-match numerical diagnostics")
            for key, value in group["metrics"].items():
                _number(value, key, minimum=-float("inf"))
        ci = row.get("confidence_interval", {})
        _require(ci.get("grouping") == "match" and ci.get("independent_matches") == len(task_matches), "Confidence interval must group by independent match")
        _text(ci.get("method"), "confidence interval method")
        _require(_number(ci.get("lower"), "CI lower", minimum=-float("inf")) <= _number(ci.get("upper"), "CI upper", minimum=-float("inf")), "Inverted confidence interval")
        unknowns = _retained(row.get("unknown_cases"), root)
        _require(unknowns.get("task") == task and unknowns.get("unknown_count") == row["unknown"], "Unknown-case ledger does not match task denominator")
        cases = unknowns.get("cases")
        _require(isinstance(cases, list) and sum(_count(case.get("count"), "unknown interval count") for case in cases) == row["unknown"], "Retain all unknown cases or counted intervals")
        for case in cases:
            _require(case.get("match_id") in task_matches, "Unknown case has an unregistered match")
            for key in ("case_id", "reason", "source_interval"):
                _text(case.get(key), key)
        _require(len({case["case_id"] for case in cases}) == len(cases), "Repeated unknown case ID")
        _require(isinstance(row.get("breakdowns"), dict) and bool(row["breakdowns"]), "Missing task subgroup diagnostics")
        required_breakdowns = {"near", "far", "blur"} if task == "visible_ball" else {"near", "far"} if task == "player_identity" else set()
        _require(required_breakdowns <= set(row["breakdowns"]), f"Missing {task} near/far/blur breakdowns")
        for breakdown in row["breakdowns"].values():
            _require(isinstance(breakdown, dict) and isinstance(breakdown.get("metrics"), dict) and bool(breakdown["metrics"]), "Subgroup diagnostics must contain numerical metrics")
            evaluable, measured, unknown = (_count(breakdown.get(key), key) for key in ("evaluable", "measured", "unknown"))
            _require(evaluable == measured + unknown and evaluable <= row["evaluable"], "Subgroup denominator/unknown accounting is inconsistent")
            for key, value in breakdown["metrics"].items():
                _number(value, key, minimum=-float("inf"))
        reference_item = references.get(policy.get("reference_id"))
        _require(reference_item is not None, f"{task} requires a retained independent reference receipt")
        reference, reference_hash = reference_item
        _require(task in reference.get("tasks", []) and set(reference.get("source_ids", [])) == {samples[item]["source_id"] for item in sample_ids}, "Independent reference does not cover task/source scope")
        _require(reference.get("measurement_definitions", {}).get(task) == row["measurement_definition"], "Independent reference has a different measurement definition")
        if task == "physical_ball_speed":
            _require(reference.get("synchronized") is True, "Physical ball speed needs a synchronized independent reference")
        decision = review["task_decisions"][task]
        _require(decision.get("decision") == "ACCEPTED" and decision.get("reference_sha256") == reference_hash, f"{task} has no independent reference acceptance")
        for key in ("rationale", "sample_size_rationale", "unknown_coverage_rationale"):
            _text(decision.get(key), key)
        metrics = _metrics(task, row)
        _apply_thresholds(task, metrics, policy)
        checked[task] = {"metrics": metrics, "scope": protocol["scope"], "unknown": row["unknown"], "evaluable": row["evaluable"]}
    return {"receipt_id": receipt["receipt_id"], "receipt_sha256": manifest["qualification_receipt"]["sha256"],
            "evidence_origin": origin, "qualified_tasks": checked, "independent_test_matches": len(matches),
            "production_eligible": not synthetic}

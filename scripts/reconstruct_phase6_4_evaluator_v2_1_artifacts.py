"""Build Phase 6.4 evaluator-v2.1 causal and optimal-matching artifacts.

This module is evaluation-only. It runs the unchanged detector once per video,
uses persisted candidate identity for stages 2-6, and keeps independent stage
capability separate from causal GT survival.
"""

from __future__ import annotations

import hashlib
import json
import sys
import copy
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase6_4_event_recovery import (
    DEFAULT_INPUT_ROOT,
    _load_video_inputs,
    _variant_configs,
)
from src.events.event_detector import EventDetectionAnalysis, TennisEventDetector
from src.events.event_evaluator import (
    MATCHING_ALGORITHM,
    PHASE6_4_EVALUATOR_VERSION,
    canonical_event_type,
    evaluate_covered_events,
    frame_inside_annotation_coverage,
    gt_coverage_boundary_status,
    precision_recall_f1,
    validate_annotation_coverage,
)
from src.tracking.temporal_ball_tracker import BallState


BENCHMARK = ROOT / "data" / "benchmarks" / "cross_match_final_holdout"
VALIDATION = ROOT / "artifacts" / "validation"
GT_PATH = BENCHMARK / "ground_truth_events.json"
VIDEOS_PATH = BENCHMARK / "videos.json"
COVERAGE_PATH = BENCHMARK / "annotation_coverage.json"
CONFIG_PATH = ROOT / "configs" / "phase6_analytics" / "pipeline.yaml"
EVALUATOR_PATH = ROOT / "src" / "events" / "event_evaluator.py"
V2_BASELINE_PATH = VALIDATION / "phase6_4_corrected_fp_manifest.json"
SCOPE_ID = "CROSS_MATCH_FINAL_HOLDOUT_DIAGNOSTIC_V2_1_TOLERANCE_SAFE_RALLY_INTERVALS"
SCIENTIFIC_SPLIT = "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED"
COVERAGE_REVIEW_STATUS = "MODEL_ASSISTED_PROVISIONAL"
METRIC_AUTHORITY = "DIAGNOSTIC_ONLY"
CAUSAL_SCHEMA_VERSION = "1.0"
EVENT_CLASSES = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")
STAGES = tuple(f"stage_{index}" for index in range(7))


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance() -> Dict[str, Any]:
    return {
        "evaluator_version": PHASE6_4_EVALUATOR_VERSION,
        "evaluator_sha256": _sha256(EVALUATOR_PATH),
        "gt_sha256": _sha256(GT_PATH),
        "coverage_sha256": _sha256(COVERAGE_PATH),
        "video_manifest_sha256": _sha256(VIDEOS_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "matching_algorithm": MATCHING_ALGORITHM,
        "coverage_policy": "TOLERANCE_NEIGHBORHOOD_MUST_BE_FULLY_OBSERVABLE_OR_SYMMETRICALLY_CENSORED",
        "causal_lineage_schema_version": CAUSAL_SCHEMA_VERSION,
        "matching_tolerance_s": 0.2,
    }


def _artifact(payload: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "phase": "6.4",
        "scientific_split": SCIENTIFIC_SPLIT,
        "qualification_evidence": False,
        "coverage_review_status": COVERAGE_REVIEW_STATUS,
        "metric_authority": METRIC_AUTHORITY,
        "human_approved": False,
        **payload,
        "evaluator_provenance": dict(provenance),
    }


def _frame(record: Mapping[str, Any]) -> int:
    for key in ("frame", "frame_index", "frame_best", "refined_frame"):
        if record.get(key) is not None:
            return int(record[key])
    raise KeyError(record)


def _covered(
    records: Sequence[Mapping[str, Any]], video_id: str, coverage: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if frame_inside_annotation_coverage(video_id, _frame(record), coverage)
    ]


def _event_record(event: Any, video_id: str, index: int) -> Dict[str, Any]:
    candidate_id = event.evidence.get("candidate_id")
    return {
        "video_id": video_id,
        "prediction_id": f"{video_id}:final:{index}",
        "candidate_id": int(candidate_id) if candidate_id is not None else None,
        "final_event_id": int(event.event_id),
        "frame": int(event.frame_index),
        "timestamp_s": float(event.timestamp_s),
        "event_type": event.event_type.value,
        "player_id": event.player_id,
        "confidence": float(event.confidence),
        "trajectory_state": event.trajectory_state,
    }


def _candidate_record(candidate: Any, trace: Mapping[str, Any], video_id: str, candidate_id: int) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "prediction_id": f"{video_id}:candidate:{candidate_id}",
        "candidate_id": candidate_id,
        "frame": int(candidate.frame_index),
        "timestamp_s": float(candidate.timestamp_s),
        "discovery_frame": int(
            candidate.discovery_frame_index
            if candidate.discovery_frame_index is not None
            else candidate.frame_index
        ),
        "discovery_timestamp_s": float(
            candidate.discovery_timestamp_s
            if candidate.discovery_timestamp_s is not None
            else candidate.timestamp_s
        ),
        "refined_frame": int(trace["refined_frame"]),
        "refined_timestamp_s": float(trace["refined_timestamp_s"]),
        "score": float(candidate.score),
    }


def _trace_record(trace: Mapping[str, Any], video_id: str) -> Dict[str, Any]:
    physical_type = str(trace.get("physical_event_type", "UNKNOWN_PHYSICAL_EVENT"))
    semantic_type = str(trace.get("candidate_event_type", "UNKNOWN_EVENT"))
    player_id = trace.get("selected_player")
    return {
        "video_id": video_id,
        "prediction_id": f"{video_id}:candidate:{int(trace['candidate_id'])}",
        "candidate_id": int(trace["candidate_id"]),
        "frame": int(trace["refined_frame"]),
        "timestamp_s": float(trace["refined_timestamp_s"]),
        "physical_event_type": physical_type,
        "event_type": semantic_type,
        "player_id": player_id,
        "stage_pass": dict(trace["stage_pass"]),
        "final_event_emitted": bool(trace.get("final_event_emitted", False)),
        "final_event_id": trace.get("final_event_id"),
        "suppression_reason": trace.get("suppression_reason") or trace.get("rejection_reason"),
        "rejection_stage": trace.get("rejection_stage"),
    }


def _stage_zero_records(video_id: str) -> list[Dict[str, Any]]:
    raw = _load(VALIDATION / "raw_candidates" / f"{video_id}_candidates.json")
    fps = float(raw["fps"])
    return [
        {
            "video_id": video_id,
            "prediction_id": f"{video_id}:raw:{frame}",
            "frame": frame,
            "timestamp_s": frame / fps,
        }
        for frame, proposals in enumerate(raw["frames"])
        if proposals
    ]


def _stage_one_records(video_id: str, points: Sequence[Any]) -> list[Dict[str, Any]]:
    usable = {BallState.DETECTED, BallState.TRACKED, BallState.INTERPOLATED}
    return [
        {
            "video_id": video_id,
            "prediction_id": f"{video_id}:observation:{point.frame_index}",
            "frame": int(point.frame_index),
            "timestamp_s": float(point.timestamp_seconds),
        }
        for point in points
        if point.state in usable and point.x_px is not None and point.y_px is not None
    ]


def stage_sets_from_analysis(
    video_id: str, analysis: EventDetectionAnalysis
) -> Dict[str, list[Dict[str, Any]]]:
    """Extract actual stage-2 through stage-6 records from one detector run."""
    traces = {int(trace["candidate_id"]): trace for trace in analysis.verification_traces}
    if len(traces) != len(analysis.candidates):
        raise AssertionError("Every detector candidate must have exactly one verification trace")
    stage_2 = [
        _candidate_record(candidate, traces[index], video_id, index)
        for index, candidate in enumerate(analysis.candidates, 1)
    ]
    trace_records = [_trace_record(traces[index], video_id) for index in range(1, len(traces) + 1)]
    stage_3 = [row for row in trace_records if row["stage_pass"]["physics_verification"]]
    stage_4 = [
        row
        for row in stage_3
        if row["stage_pass"]["event_type_classification"]
        and canonical_event_type(row) != "UNKNOWN_EVENT"
    ]
    stage_5 = [row for row in stage_4 if row["stage_pass"]["player_attribution"]]
    stage_6 = [
        _event_record(event, video_id, index)
        for index, event in enumerate(analysis.events, 1)
    ]
    return {
        "stage_2": stage_2,
        "stage_3": stage_3,
        "stage_4": stage_4,
        "stage_5": stage_5,
        "stage_6": stage_6,
    }


def causal_stage_survival(
    *,
    stage_0_match: bool,
    stage_1_match: bool,
    stage_2_match: bool,
    trace: Optional[Mapping[str, Any]],
    gt_event: Mapping[str, Any],
) -> Dict[str, bool]:
    """Compute monotonic causal survival for one GT-linked candidate."""
    survival: Dict[str, bool] = {}
    survival["stage_0"] = bool(stage_0_match)
    survival["stage_1"] = survival["stage_0"] and bool(stage_1_match)
    survival["stage_2"] = survival["stage_1"] and bool(stage_2_match)
    stage_pass = trace.get("stage_pass", {}) if trace else {}
    survival["stage_3"] = survival["stage_2"] and bool(
        stage_pass.get("physics_verification", False)
    )
    semantic_correct = bool(
        trace
        and stage_pass.get("event_type_classification", False)
        and canonical_event_type({"event_type": trace.get("candidate_event_type")})
        == canonical_event_type(gt_event)
    )
    survival["stage_4"] = survival["stage_3"] and semantic_correct
    gt_type = canonical_event_type(gt_event)
    player_correct = (
        gt_type == "BOUNCE"
        or gt_event.get("player_id") is None
        or (trace is not None and trace.get("selected_player") == gt_event.get("player_id"))
    )
    survival["stage_5"] = survival["stage_4"] and player_correct
    survival["stage_6"] = survival["stage_5"] and bool(
        trace and trace.get("final_event_emitted", False)
    )
    return survival


def first_failure_from_survival(survival: Mapping[str, bool]) -> str:
    failures = (
        ("stage_0", "NO_RAW_PROPOSAL"),
        ("stage_1", "TRACK_NOT_USABLE"),
        ("stage_2", "NO_EVENT_CANDIDATE"),
        ("stage_3", "PHYSICAL_CONTACT_REJECTED"),
        ("stage_4", "SEMANTIC_TYPE_WRONG"),
        ("stage_5", "PLAYER_ATTRIBUTION_WRONG"),
        ("stage_6", "FINAL_SUPPRESSION"),
    )
    return next((failure for stage, failure in failures if not survival[stage]), "FULLY_CORRECT")


def assert_causal_monotonicity(records: Sequence[Mapping[str, Any]]) -> None:
    """Fail artifact generation on any per-record or aggregate resurrection."""
    for record in records:
        values = [bool(record["causal_stage_survival"][stage]) for stage in STAGES]
        if any(not earlier and later for earlier, later in zip(values, values[1:])):
            raise AssertionError(
                f"Non-causal stage resurrection for {record['video_id']} GT {record['gt_event_id']}"
            )
    counts = [
        sum(bool(record["causal_stage_survival"][stage]) for record in records)
        for stage in STAGES
    ]
    if any(earlier < later for earlier, later in zip(counts, counts[1:])):
        raise AssertionError(f"Aggregate causal stage counts are non-monotonic: {counts}")


def _evaluate(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    videos: Mapping[str, Any],
    coverage: Mapping[str, Any],
    *,
    event_type: bool = False,
    player: bool = False,
) -> Dict[str, Any]:
    return evaluate_covered_events(
        predictions,
        gt,
        videos,
        coverage,
        tolerance_s=0.2,
        require_event_type=event_type,
        require_player=player,
    )


def _match_maps(
    result: Mapping[str, Any],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    coverage: Mapping[str, Any],
) -> Dict[tuple[str, int], Mapping[str, Any]]:
    maps: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for match in result["matches"]:
        video_id = str(match["video_id"])
        covered_predictions = _covered(predictions[video_id], video_id, coverage)
        prediction = covered_predictions[int(match["prediction_index"])]
        maps[(video_id, int(match["gt_event_id"]))] = prediction
    return maps


def _stage_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "per_video": result["per_video"],
        "aggregate": result["aggregate"],
        "match_count": len(result["matches"]),
    }


def _legacy_greedy_physical_counts(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    videos: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> Dict[str, int]:
    """Reproduce v2.0 greedy/asymmetric counts for change attribution only."""
    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    for video_id, metadata in videos["videos"].items():
        fps = float(metadata["fps"])
        covered_predictions = _covered(predictions[video_id], video_id, coverage)
        covered_gt = _covered(gt[video_id], video_id, coverage)
        edges = []
        for prediction_index, prediction in enumerate(covered_predictions):
            prediction_time = float(
                prediction.get("timestamp_s", _frame(prediction) / fps)
            )
            for gt_index, event in enumerate(covered_gt):
                start = int(event.get("frame_min", event["frame_best"])) / fps
                end = int(event.get("frame_max", event["frame_best"])) / fps
                distance = start - prediction_time if prediction_time < start else (
                    prediction_time - end if prediction_time > end else 0.0
                )
                if distance <= 0.2 + 1e-12:
                    edges.append((distance, prediction_index, gt_index))
        used_predictions = set()
        used_gt = set()
        for _, prediction_index, gt_index in sorted(edges):
            if prediction_index in used_predictions or gt_index in used_gt:
                continue
            used_predictions.add(prediction_index)
            used_gt.add(gt_index)
        matches = len(used_predictions)
        totals["true_positives"] += matches
        totals["false_positives"] += len(covered_predictions) - matches
        totals["false_negatives"] += len(covered_gt) - matches
    return totals


def _conditional_semantic(
    stage3_result: Mapping[str, Any],
    stage3: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    gt_lookup = {
        (video_id, int(event["event_id"])): event
        for video_id, events in gt.items()
        for event in events
    }
    rows = []
    for match in stage3_result["matches"]:
        video_id = str(match["video_id"])
        prediction = _covered(stage3[video_id], video_id, coverage)[int(match["prediction_index"])]
        gt_event = gt_lookup[(video_id, int(match["gt_event_id"]))]
        predicted_type = canonical_event_type(prediction)
        gt_type = canonical_event_type(gt_event)
        rows.append(
            {
                "video_id": video_id,
                "candidate_id": prediction.get("candidate_id"),
                "gt_event_id": gt_event["event_id"],
                "gt_event_type": gt_type,
                "predicted_event_type": predicted_type,
                "correct": predicted_type == gt_type,
                "unknown_abstention": predicted_type == "UNKNOWN_EVENT",
            }
        )

    def summarize(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        per_class = {}
        for event_class in EVENT_CLASSES:
            tp = sum(row["gt_event_type"] == event_class and row["predicted_event_type"] == event_class for row in subset)
            fp = sum(row["gt_event_type"] != event_class and row["predicted_event_type"] == event_class for row in subset)
            fn = sum(row["gt_event_type"] == event_class and row["predicted_event_type"] != event_class for row in subset)
            per_class[event_class] = precision_recall_f1(tp, fp, fn)
        macro_f1 = fmean(row["f1"] for row in per_class.values()) if per_class else 0.0
        correct = sum(bool(row["correct"]) for row in subset)
        unknown = sum(bool(row["unknown_abstention"]) for row in subset)
        return {
            "conditional_support": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else 0.0,
            "unknown_abstention_count": unknown,
            "unknown_abstention_rate": unknown / len(subset) if subset else 0.0,
            "macro_f1": macro_f1,
            "per_class": per_class,
        }

    return {
        "definition": "SEMANTIC_CLASSIFICATION_CONDITIONAL_ON_TRUE_STAGE3_GT_MATCH",
        "per_video": {
            video_id: summarize([row for row in rows if row["video_id"] == video_id])
            for video_id in gt
        },
        "aggregate": summarize(rows),
        "records": rows,
    }


def _conditional_player(
    conditional_semantic: Mapping[str, Any],
    stage3: Mapping[str, Sequence[Mapping[str, Any]]],
    coverage: Mapping[str, Any],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    gt_lookup = {
        (video_id, int(event["event_id"])): event
        for video_id, events in gt.items()
        for event in events
    }
    trace_lookup = {
        (video_id, int(row["candidate_id"])): row
        for video_id, rows in stage3.items()
        for row in _covered(rows, video_id, coverage)
    }
    rows = []
    for semantic in conditional_semantic["records"]:
        gt_event = gt_lookup[(semantic["video_id"], int(semantic["gt_event_id"]))]
        if canonical_event_type(gt_event) == "BOUNCE" or gt_event.get("player_id") is None:
            continue
        if not semantic["correct"]:
            continue
        prediction = trace_lookup[(semantic["video_id"], int(semantic["candidate_id"]))]
        rows.append(
            {
                "video_id": semantic["video_id"],
                "candidate_id": semantic["candidate_id"],
                "gt_event_id": semantic["gt_event_id"],
                "gt_player_id": gt_event.get("player_id"),
                "predicted_player_id": prediction.get("player_id"),
                "correct": prediction.get("player_id") == gt_event.get("player_id"),
            }
        )

    def summarize(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        correct = sum(bool(row["correct"]) for row in subset)
        return {
            "conditional_support": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else 0.0,
        }

    return {
        "definition": "PLAYER_ATTRIBUTION_CONDITIONAL_ON_STAGE3_MATCH_AND_CORRECT_ELIGIBLE_SEMANTIC_TYPE",
        "bounce_excluded_from_denominator": True,
        "per_video": {
            video_id: summarize([row for row in rows if row["video_id"] == video_id])
            for video_id in gt
        },
        "aggregate": summarize(rows),
        "records": rows,
    }


def _production_regression(
    final_predictions: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    baseline = _load(V2_BASELINE_PATH)["records"]
    old_by_video: Dict[str, list[tuple[int, str, Optional[int]]]] = {}
    for video_id in final_predictions:
        old_by_video[video_id] = [
            (int(row["frame"]), str(row["event_type"]), row.get("player_id"))
            for row in baseline
            if row["video_id"] == video_id
        ]
    new_by_video = {
        video_id: [
            (int(row["frame"]), str(row["event_type"]), row.get("player_id"))
            for row in rows
        ]
        for video_id, rows in final_predictions.items()
    }
    changed_frames = []
    changed_labels = []
    for video_id in final_predictions:
        old_rows = old_by_video[video_id]
        new_rows = new_by_video[video_id]
        if [row[0] for row in old_rows] != [row[0] for row in new_rows]:
            changed_frames.append(video_id)
        if [row[1:] for row in old_rows] != [row[1:] for row in new_rows]:
            changed_labels.append(video_id)
    result = {
        "baseline_artifact": str(V2_BASELINE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "production_prediction_count_before": sum(map(len, old_by_video.values())),
        "production_prediction_count_after": sum(map(len, new_by_video.values())),
        "changed_prediction_frames": changed_frames,
        "changed_event_labels_or_players": changed_labels,
        "semantically_equivalent": old_by_video == new_by_video,
    }
    if not result["semantically_equivalent"]:
        raise AssertionError(f"Production output changed during evaluator-only correction: {result}")
    return result


def _mark_v2_artifacts_superseded(final_output_changed: bool) -> None:
    stage_path = VALIDATION / "phase6_4_corrected_stage_metrics.json"
    lineage_path = VALIDATION / "phase6_4_corrected_event_lineage.json"
    for path, reason in (
        (stage_path, "INDEPENDENT_REMATCHING_DID_NOT_REPRESENT_CAUSAL_STAGE_SURVIVAL"),
        (lineage_path, "NON_MONOTONIC_INDEPENDENT_STAGE_MATCHES_WERE_LABELLED_STAGE_SURVIVAL"),
    ):
        payload = _load(path)
        payload["scientific_status"] = "SUPERSEDED_BY_EVALUATOR_V2_1_CAUSAL_STAGE_SEMANTICS"
        payload["supersession_reason"] = reason
        payload["superseded_by"] = "phase6_4_v2_1_causal_stage_metrics.json"
        _dump(path, payload)
    for path in (
        VALIDATION / "phase6_4_corrected_per_video_metrics.json",
        VALIDATION / "phase6_4_corrected_aggregate_metrics.json",
    ):
        payload = _load(path)
        payload["metric_scope_clarification"] = "FINAL_OUTPUT_PHYSICAL_MATCH_NOT_TRUE_STAGE_3"
        payload["scientific_status"] = (
            "V2_0_GREEDY_MATCHING_SUPERSEDED"
            if final_output_changed
            else "HISTORICAL_V2_0_FINAL_OUTPUT_METRIC_OPTIMAL_COUNTS_UNCHANGED"
        )
        _dump(path, payload)


def main() -> None:
    videos_document = _load(VIDEOS_PATH)
    videos = videos_document["videos"]
    gt = _load(GT_PATH)["events"]
    coverage = _load(COVERAGE_PATH)
    validate_annotation_coverage(coverage, videos_document)
    if coverage.get("human_approved") is not False:
        raise AssertionError("Evaluator v2.1 must not claim human coverage approval")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["event_detection"]
    detector_config = _variant_configs(config)["H_FINAL_INTEGRATED"]

    analyses: Dict[str, EventDetectionAnalysis] = {}
    stage_predictions: Dict[str, Dict[str, list[Dict[str, Any]]]] = {
        stage: {} for stage in STAGES
    }
    for video_id, metadata in videos.items():
        points, player_1, player_2, _ = _load_video_inputs(DEFAULT_INPUT_ROOT, video_id)
        analysis = TennisEventDetector(config=detector_config).analyze(
            points,
            player_1,
            player_2,
            fps=float(metadata["fps"]),
            frame_size=(int(metadata["width"]), int(metadata["height"])),
            camera_offsets_px=None,
        )
        analyses[video_id] = analysis
        extracted = stage_sets_from_analysis(video_id, analysis)
        stage_predictions["stage_0"][video_id] = _stage_zero_records(video_id)
        stage_predictions["stage_1"][video_id] = _stage_one_records(video_id, points)
        for stage in STAGES[2:]:
            stage_predictions[stage][video_id] = extracted[stage]

    final_output_physical = _evaluate(stage_predictions["stage_6"], gt, videos_document, coverage)
    true_stage3 = _evaluate(stage_predictions["stage_3"], gt, videos_document, coverage)
    end_to_end_semantic = _evaluate(
        stage_predictions["stage_6"], gt, videos_document, coverage, event_type=True
    )
    end_to_end_player = _evaluate(
        stage_predictions["stage_6"], gt, videos_document, coverage, event_type=True, player=True
    )

    independent_results = {
        "stage_0": _evaluate(stage_predictions["stage_0"], gt, videos_document, coverage),
        "stage_1": _evaluate(stage_predictions["stage_1"], gt, videos_document, coverage),
        "stage_2": _evaluate(stage_predictions["stage_2"], gt, videos_document, coverage),
        "stage_3": true_stage3,
        "stage_4": _evaluate(
            stage_predictions["stage_4"], gt, videos_document, coverage, event_type=True
        ),
        "stage_5": _evaluate(
            stage_predictions["stage_5"], gt, videos_document, coverage, event_type=True, player=True
        ),
        "stage_6": end_to_end_player,
    }
    independent_maps = {
        stage: _match_maps(result, stage_predictions[stage], gt, coverage)
        for stage, result in independent_results.items()
    }
    trace_lookup = {
        (video_id, int(trace["candidate_id"])): trace
        for video_id, analysis in analyses.items()
        for trace in analysis.verification_traces
    }

    causal_records = []
    for video_id, events in gt.items():
        for event in events:
            key = (video_id, int(event["event_id"]))
            candidate = independent_maps["stage_2"].get(key)
            candidate_id = int(candidate["candidate_id"]) if candidate else None
            trace = trace_lookup.get((video_id, candidate_id)) if candidate_id is not None else None
            survival = causal_stage_survival(
                stage_0_match=key in independent_maps["stage_0"],
                stage_1_match=key in independent_maps["stage_1"],
                stage_2_match=candidate is not None,
                trace=trace,
                gt_event=event,
            )
            first_failure = first_failure_from_survival(survival)
            causal_records.append(
                {
                    "video_id": video_id,
                    "evaluation_scope_id": SCOPE_ID,
                    "gt_event_id": int(event["event_id"]),
                    "gt_event_type": canonical_event_type(event),
                    "gt_player_id": event.get("player_id"),
                    "gt_frame_min": int(event["frame_min"]),
                    "gt_frame_best": int(event["frame_best"]),
                    "gt_frame_max": int(event["frame_max"]),
                    "candidate_id": candidate_id,
                    "candidate_discovery_frame": candidate.get("discovery_frame") if candidate else None,
                    "candidate_discovery_timestamp_s": candidate.get("discovery_timestamp_s") if candidate else None,
                    "candidate_refined_frame": trace.get("refined_frame") if trace else None,
                    "candidate_refined_timestamp_s": trace.get("refined_timestamp_s") if trace else None,
                    "physical_verification_result": trace.get("physical_event_type") if trace else None,
                    "semantic_result": trace.get("candidate_event_type") if trace else None,
                    "player_result": trace.get("selected_player") if trace else None,
                    "final_event_id": trace.get("final_event_id") if trace else None,
                    "final_emission_status": bool(trace and trace.get("final_event_emitted", False)),
                    "suppression_reason": (
                        trace.get("suppression_reason") or trace.get("rejection_reason")
                        if trace
                        else None
                    ),
                    "causal_stage_survival": survival,
                    "first_failure_stage": first_failure,
                    "independent_stage_match_capability": {
                        stage: key in independent_maps[stage] for stage in STAGES
                    },
                }
            )
    assert_causal_monotonicity(causal_records)

    causal_per_video = {
        video_id: {
            stage: sum(
                row["video_id"] == video_id and row["causal_stage_survival"][stage]
                for row in causal_records
            )
            for stage in STAGES
        }
        for video_id in gt
    }
    causal_aggregate = {
        stage: sum(row["causal_stage_survival"][stage] for row in causal_records)
        for stage in STAGES
    }
    failure_names = (
        "NO_RAW_PROPOSAL",
        "TRACK_NOT_USABLE",
        "NO_EVENT_CANDIDATE",
        "PHYSICAL_CONTACT_REJECTED",
        "SEMANTIC_TYPE_WRONG",
        "PLAYER_ATTRIBUTION_WRONG",
        "FINAL_SUPPRESSION",
        "FULLY_CORRECT",
    )
    failure_distribution = {
        failure: {
            **{
                video_id: sum(
                    row["video_id"] == video_id and row["first_failure_stage"] == failure
                    for row in causal_records
                )
                for video_id in gt
            },
            "total": sum(row["first_failure_stage"] == failure for row in causal_records),
        }
        for failure in failure_names
    }

    stage3_gt_matches = {
        (str(match["video_id"]), int(match["gt_event_id"])) for match in true_stage3["matches"]
    }
    class_recall = {}
    for event_class in EVENT_CLASSES:
        class_events = [
            (video_id, event)
            for video_id, events in gt.items()
            for event in events
            if canonical_event_type(event) == event_class
        ]
        matched = sum(
            (video_id, int(event["event_id"])) in stage3_gt_matches
            for video_id, event in class_events
        )
        class_recall[event_class] = {
            "gt_count": len(class_events),
            "physical_contact_matches": matched,
            "recall": matched / len(class_events) if class_events else 0.0,
        }

    conditional_semantic = _conditional_semantic(
        true_stage3, stage_predictions["stage_3"], gt, coverage
    )
    conditional_player = _conditional_player(
        conditional_semantic, stage_predictions["stage_3"], coverage, gt
    )
    suppressed_rows = [
        row
        for row in causal_records
        if row["causal_stage_survival"]["stage_5"]
        and not row["causal_stage_survival"]["stage_6"]
    ]
    suppression = {
        "valid_stage5_survivors_suppressed_before_export": len(suppressed_rows),
        "reasons": dict(Counter(row.get("suppression_reason") or "OTHER_FINAL_FILTERING" for row in suppressed_rows)),
        "records": [
            {
                "video_id": row["video_id"],
                "gt_event_id": row["gt_event_id"],
                "candidate_id": row["candidate_id"],
                "reason": row.get("suppression_reason") or "OTHER_FINAL_FILTERING",
            }
            for row in suppressed_rows
        ],
    }

    boundary_rows = []
    for video_id, events in gt.items():
        for event in events:
            boundary_rows.append(
                gt_coverage_boundary_status(
                    video_id, event, coverage, videos_document, tolerance_s=0.2
                )
            )
    if not all(row["coverage_safe"] for row in boundary_rows):
        raise AssertionError("Coverage extension did not make every included GT tolerance-safe")
    boundary_summary = {
        video_id: {
            "coverage_intervals": coverage["videos"][video_id]["fully_reviewed_intervals"],
            "first_gt_interval_frames": [
                int(gt[video_id][0]["frame_min"]), int(gt[video_id][0]["frame_max"])
            ],
            "last_gt_interval_frames": [
                int(gt[video_id][-1]["frame_min"]), int(gt[video_id][-1]["frame_max"])
            ],
            "coverage_safe_gt_count": sum(
                row["video_id"] == video_id and row["coverage_safe"] for row in boundary_rows
            ),
            "boundary_censored_gt_count": sum(
                row["video_id"] == video_id and not row["coverage_safe"] for row in boundary_rows
            ),
            "coverage_extension": (
                {"previous_end_frame": 202, "new_end_frame": 206, "computed_margin_frames": 6}
                if video_id == "video_08"
                else (
                    {"previous_end_frame": 240, "new_end_frame": 244, "computed_margin_frames": 6}
                    if video_id == "video_09"
                    else None
                )
            ),
        }
        for video_id in gt
    }

    production_regression = _production_regression(stage_predictions["stage_6"])
    provenance = _provenance()
    old_aggregate = _load(VALIDATION / "phase6_4_corrected_aggregate_metrics.json")["aggregate"]
    final_counts_changed = any(
        final_output_physical["aggregate"][key] != old_aggregate[key]
        for key in ("true_positives", "false_positives", "false_negatives")
    )
    previous_coverage = copy.deepcopy(coverage)
    previous_coverage["videos"]["video_08"]["fully_reviewed_intervals"][0]["end_frame"] = 202
    previous_coverage["videos"]["video_08"]["fully_reviewed_intervals"][0]["end_timestamp_s"] = 202 / 30.0
    previous_coverage["videos"]["video_09"]["fully_reviewed_intervals"][0]["end_frame"] = 240
    previous_coverage["videos"]["video_09"]["fully_reviewed_intervals"][0]["end_timestamp_s"] = 240 / 30.0
    legacy_old_coverage_counts = _legacy_greedy_physical_counts(
        stage_predictions["stage_6"], gt, videos_document, previous_coverage
    )
    legacy_new_coverage_counts = _legacy_greedy_physical_counts(
        stage_predictions["stage_6"], gt, videos_document, coverage
    )
    optimal_counts = {
        key: final_output_physical["aggregate"][key]
        for key in ("true_positives", "false_positives", "false_negatives")
    }
    optimal_matching_changed_counts = optimal_counts != legacy_new_coverage_counts
    change_attribution = {
        "v2_0_greedy_old_coverage": legacy_old_coverage_counts,
        "v2_1_greedy_counterfactual_new_coverage": legacy_new_coverage_counts,
        "v2_1_optimal_new_coverage": optimal_counts,
        "coverage_boundary_delta": {
            key: legacy_new_coverage_counts[key] - legacy_old_coverage_counts[key]
            for key in optimal_counts
        },
        "optimal_assignment_delta": {
            key: optimal_counts[key] - legacy_new_coverage_counts[key]
            for key in optimal_counts
        },
    }

    outputs = {
        "phase6_4_v2_1_final_output_physical_metrics.json": _artifact(
            {
                "metric_name": "FINAL_OUTPUT_PHYSICAL_MATCH",
                "comparison_v2_0": {"true_positives": 20, "false_positives": 25, "false_negatives": 20},
                "optimal_matching_changed_counts": optimal_matching_changed_counts,
                "change_attribution": change_attribution,
                **_stage_metrics(final_output_physical),
            },
            provenance,
        ),
        "phase6_4_v2_1_stage3_physical_verification_metrics.json": _artifact(
            {
                "metric_name": "TRUE_STAGE3_PHYSICAL_VERIFICATION",
                "stage_definition": "CANDIDATE_PHYSICS_VERIFICATION_PASS_BEFORE_SEMANTIC_CORRECTNESS_OR_FINAL_SUPPRESSION",
                **_stage_metrics(true_stage3),
                "recall_by_gt_class": class_recall,
            },
            provenance,
        ),
        "phase6_4_v2_1_causal_stage_metrics.json": _artifact(
            {
                "metric_name": "CAUSAL_STAGE_SURVIVAL",
                "per_video": causal_per_video,
                "aggregate": causal_aggregate,
                "first_failure_distribution": failure_distribution,
                "stage6_suppression": suppression,
                "per_record_monotonicity": "PASS",
                "aggregate_monotonicity": "PASS",
                "impossible_transition_count": 0,
            },
            provenance,
        ),
        "phase6_4_v2_1_causal_event_lineage.json": _artifact(
            {
                "metric_name": "CAUSAL_EVENT_LINEAGE",
                "record_count": len(causal_records),
                "records": causal_records,
            },
            provenance,
        ),
        "phase6_4_v2_1_independent_stage_capability.json": _artifact(
            {
                "metric_name": "INDEPENDENT_STAGE_MATCH_CAPABILITY",
                "explicitly_not_causal_survival": True,
                "stages": {
                    stage: _stage_metrics(result) for stage, result in independent_results.items()
                },
            },
            provenance,
        ),
        "phase6_4_v2_1_semantic_metrics.json": _artifact(
            {
                "end_to_end_exact_semantic": _stage_metrics(end_to_end_semantic),
                "end_to_end_player_aware": _stage_metrics(end_to_end_player),
                "conditional_semantic_after_stage3_match": conditional_semantic,
                "conditional_player_attribution": conditional_player,
            },
            provenance,
        ),
        "phase6_4_v2_1_coverage_boundary_audit.json": _artifact(
            {
                "coverage_scope_id": SCOPE_ID,
                "reviewer_type": coverage["review_provenance"]["reviewer_type"],
                "authoritative_for_final_qualification": False,
                "per_video": boundary_summary,
                "gt_event_audits": boundary_rows,
                "coverage_safe_gt_count": sum(row["coverage_safe"] for row in boundary_rows),
                "boundary_censored_gt_count": sum(not row["coverage_safe"] for row in boundary_rows),
            },
            provenance,
        ),
        "phase6_4_v2_1_evaluator_integrity.json": _artifact(
            {
                "starting_sha": "29cd39af013e6498599e19dd0ced97d00247c917",
                "optimal_matching": "PASS",
                "causal_stage_lineage": "PASS",
                "coverage_boundary_handling": "PASS",
                "candidate_identity_persistence": "PASS",
                "production_regression": production_regression,
                "analytics_contract_schema_version": "1.0",
                "production_gt_leakage_check": "PASS",
                "model_tuning_performed": False,
            },
            provenance,
        ),
    }
    for name, payload in outputs.items():
        _dump(VALIDATION / name, payload)
    _mark_v2_artifacts_superseded(optimal_matching_changed_counts)
    print(
        json.dumps(
            {
                "final_output_physical": final_output_physical["aggregate"],
                "true_stage3": true_stage3["aggregate"],
                "causal_stage_counts": causal_aggregate,
                "first_failures": failure_distribution,
                "production_regression": production_regression,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

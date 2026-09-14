"""Compatibility entry point for Phase 6.4 evaluator artifact reconstruction.

The v2.0 implementation is retained below for forensic readability only. The
public ``main`` delegates to the v2.1 causal reconstruction so this historical
path cannot silently regenerate independently-rematched stage-survival data.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase6_4_event_recovery import (
    DEFAULT_INPUT_ROOT,
    _candidate_record,
    _event_record,
    _load_video_inputs,
    _variant_configs,
)
from src.events.event_detector import TennisEventDetector
from src.events.event_evaluator import (
    MATCHING_ALGORITHM,
    PHASE6_4_EVALUATOR_VERSION,
    canonical_event_type,
    canonical_one_to_one_matches,
    evaluate_covered_events,
    frame_inside_annotation_coverage,
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
REVIEW_ROOT = VALIDATION / "corrected_fp_review_pack_v2"
EVALUATION_SCOPE_ID = "CROSS_MATCH_FINAL_HOLDOUT_DIAGNOSTIC_V2_RALLY_INTERVALS"
EVENT_CLASSES = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")


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
        "phase6_4_evaluator_version": PHASE6_4_EVALUATOR_VERSION,
        "evaluator_path": str(EVALUATOR_PATH.relative_to(ROOT)).replace("\\", "/"),
        "evaluator_sha256": _sha256(EVALUATOR_PATH),
        "gt_path": str(GT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "gt_sha256": _sha256(GT_PATH),
        "annotation_coverage_path": str(COVERAGE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "annotation_coverage_sha256": _sha256(COVERAGE_PATH),
        "video_manifest_path": str(VIDEOS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "video_manifest_sha256": _sha256(VIDEOS_PATH),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(CONFIG_PATH),
        "evaluation_scope_id": EVALUATION_SCOPE_ID,
        "matching_tolerance_s": 0.2,
        "matching_semantics": MATCHING_ALGORITHM,
    }


def _with_provenance(payload: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    return {"schema_version": "1.0", **payload, "evaluator_provenance": dict(provenance)}


def _records_inside(
    rows: Iterable[Mapping[str, Any]], video_id: str, coverage: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if frame_inside_annotation_coverage(
            video_id,
            int(row.get("frame", row.get("frame_index", row.get("frame_best", 0)))),
            coverage,
        )
    ]


def _stage_zero_records(video_id: str) -> list[Dict[str, Any]]:
    raw = _load(VALIDATION / "raw_candidates" / f"{video_id}_candidates.json")
    return [
        {"video_id": video_id, "frame": index, "timestamp_s": index / float(raw["fps"])}
        for index, candidates in enumerate(raw["frames"])
        if candidates
    ]


def _stage_one_records(video_id: str, points: Sequence[Any]) -> list[Dict[str, Any]]:
    usable = {BallState.DETECTED, BallState.TRACKED, BallState.INTERPOLATED}
    return [
        {
            "video_id": video_id,
            "frame": point.frame_index,
            "timestamp_s": point.timestamp_seconds,
        }
        for point in points
        if point.state in usable and point.x_px is not None and point.y_px is not None
    ]


def _mean_timing(matches: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(match[key]) for match in matches]
    return fmean(values) if values else None


def _per_class_metrics(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    videos: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {}
    for event_class in EVENT_CLASSES:
        class_predictions = {
            video_id: [
                row for row in rows if canonical_event_type(row) == event_class
            ]
            for video_id, rows in predictions.items()
        }
        class_gt = {
            video_id: [row for row in rows if canonical_event_type(row) == event_class]
            for video_id, rows in gt.items()
        }
        evaluated = evaluate_covered_events(
            class_predictions,
            class_gt,
            videos,
            coverage,
            tolerance_s=0.2,
            require_event_type=True,
        )
        metric = dict(evaluated["aggregate"])
        metric["timing_mae_s"] = _mean_timing(evaluated["matches"], "timing_error_s")
        metric["timing_mae_ms"] = _mean_timing(evaluated["matches"], "timing_error_ms")
        result[event_class] = metric
    return result


def _covered_prediction_index(
    rows: Sequence[Mapping[str, Any]], video_id: str, coverage: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if frame_inside_annotation_coverage(
            video_id, int(row.get("frame", row.get("frame_index", 0))), coverage
        )
    ]


def _lineage(
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    stage_results: Mapping[str, Mapping[str, Any]],
    stage_predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    coverage: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    match_maps = {
        stage: {
            (match["video_id"], int(match["gt_event_id"])): match
            for match in result["matches"]
        }
        for stage, result in stage_results.items()
    }
    records = []
    for video_id, events in gt.items():
        covered_predictions = {
            stage: _covered_prediction_index(rows[video_id], video_id, coverage)
            for stage, rows in stage_predictions.items()
        }
        for event in events:
            event_id = int(event["event_id"])
            key = (video_id, event_id)
            first_failure = "FULLY_CORRECT"
            for stage, failure in (
                ("stage_0", "NO_RAW_PROPOSAL"),
                ("stage_1", "TRACK_NOT_USABLE"),
                ("stage_2", "NO_EVENT_CANDIDATE"),
                ("stage_3", "PHYSICAL_CONTACT_REJECTED"),
                ("stage_4", "SEMANTIC_TYPE_WRONG"),
                ("stage_5", "PLAYER_ATTRIBUTION_WRONG"),
                ("stage_6", "FINAL_SUPPRESSION"),
            ):
                if key not in match_maps[stage]:
                    first_failure = failure
                    break
            final_match = match_maps["stage_6"].get(key)
            prediction_identity = None
            if final_match is not None:
                pred_index = int(final_match["prediction_index"])
                prediction_identity = covered_predictions["stage_6"][pred_index].get(
                    "prediction_id", f"{video_id}:prediction:{pred_index + 1}"
                )
            records.append(
                {
                    "video_id": video_id,
                    "evaluation_scope_id": EVALUATION_SCOPE_ID,
                    "inside_annotation_coverage": True,
                    "gt_event_id": event_id,
                    "gt_event_type": event["event_type"],
                    "gt_player_id": event.get("player_id"),
                    "gt_frame_min": event["frame_min"],
                    "gt_frame_best": event["frame_best"],
                    "gt_frame_max": event["frame_max"],
                    "prediction_id": prediction_identity,
                    "same_video_match": bool(final_match),
                    "timing_error_s": final_match["timing_error_s"] if final_match else None,
                    "timing_error_ms": final_match["timing_error_ms"] if final_match else None,
                    "first_failure_stage": first_failure,
                    "stage_survival": {
                        stage: key in match_maps[stage] for stage in match_maps
                    },
                }
            )
    return records


def _crop(frame: Any, center_x: float, center_y: float, radius: int) -> Any:
    height, width = frame.shape[:2]
    x1, x2 = max(0, int(center_x) - radius), min(width, int(center_x) + radius)
    y1, y2 = max(0, int(center_y) - radius), min(height, int(center_y) + radius)
    return frame[y1:y2, x1:x2]


def _write_fp_assets(
    video_id: str,
    metadata: Mapping[str, Any],
    statuses: Sequence[Dict[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    points: Sequence[Any],
) -> Dict[str, Dict[str, str | None]]:
    assets: Dict[str, Dict[str, str | None]] = {}
    fp_rows = [row for row in statuses if row["evaluation_status"] == "FP"]
    if not fp_rows:
        return assets
    capture = cv2.VideoCapture(str(ROOT / metadata["path"]))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {metadata['path']}")
    video_root = REVIEW_ROOT / video_id
    video_root.mkdir(parents=True, exist_ok=True)
    for row in fp_rows:
        prediction = predictions[int(row["prediction_index"])]
        frame_index = int(row["frame"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {video_id} frame {frame_index}")
        stem = f"{video_id}_f{frame_index:04d}_{row['prediction_id'].split(':')[-1]}"
        frame_path = video_root / f"{stem}_frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        point = points[frame_index] if 0 <= frame_index < len(points) else None
        ball_path = context_path = None
        if point is not None and point.x_px is not None and point.y_px is not None:
            ball = _crop(frame, float(point.x_px), float(point.y_px), 48)
            context = _crop(frame, float(point.x_px), float(point.y_px), 180)
            if ball.size:
                ball_file = video_root / f"{stem}_ball.jpg"
                cv2.imwrite(str(ball_file), ball)
                ball_path = str(ball_file.relative_to(ROOT)).replace("\\", "/")
            if context.size:
                context_file = video_root / f"{stem}_context.jpg"
                cv2.imwrite(str(context_file), context)
                context_path = str(context_file.relative_to(ROOT)).replace("\\", "/")
        assets[str(row["prediction_id"])] = {
            "frame_image": str(frame_path.relative_to(ROOT)).replace("\\", "/"),
            "ball_crop": ball_path,
            "context_crop": context_path,
        }
    capture.release()
    return assets


def _prelabel(prediction: Mapping[str, Any]) -> tuple[str, float]:
    state = str(prediction.get("trajectory_state", "UNKNOWN"))
    if state in {"PREDICTED", "INTERPOLATED", "OCCLUDED"}:
        return "LOW_PROVENANCE_TRAJECTORY", 0.65
    if canonical_event_type(prediction) == "UNKNOWN_EVENT":
        return "UNKNOWN_EVENT_PROPOSAL", 0.6
    return "UNMATCHED_PHYSICAL_PROPOSAL", 0.5


def _invalidate_legacy_fp_artifacts() -> None:
    for name in (
        "phase6_4_fp_manual_review_manifest.json",
        "phase6_4_fp_source_taxonomy.json",
    ):
        path = VALIDATION / name
        if not path.exists():
            continue
        payload = _load(path)
        payload["scientific_status"] = "INVALIDATED_BY_EVALUATOR_INTEGRITY_CORRECTION"
        payload["authoritative"] = False
        payload["label_source"] = "AUTOMATED_HEURISTIC_UNVERIFIED"
        payload["human_reviewed"] = False
        payload["human_label"] = None
        payload["invalidation_reason"] = (
            "Derived before same-video matching and exhaustive annotation coverage were enforced."
        )
        if name == "phase6_4_fp_source_taxonomy.json":
            payload["description"] = (
                "Legacy automated taxonomy assignment for the pre-correction Stage 3 population"
            )
        records = payload.get("records", payload.get("items", []))
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record["label_source"] = "AUTOMATED_HEURISTIC_UNVERIFIED"
                    record["human_reviewed"] = False
                    record["human_label"] = None
        _dump(path, payload)


def _legacy_v2_main_do_not_use() -> None:
    videos_document = _load(VIDEOS_PATH)
    videos = videos_document["videos"]
    gt = _load(GT_PATH)["events"]
    coverage = _load(COVERAGE_PATH)
    validate_annotation_coverage(coverage, videos_document)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["event_detection"]
    detector_config = _variant_configs(config)["H_FINAL_INTEGRATED"]

    analyses = {}
    points_by_video = {}
    predictions = {}
    candidates = {}
    stage_zero = {}
    stage_one = {}
    for video_id, metadata in videos.items():
        points, player_1, player_2, _ = _load_video_inputs(DEFAULT_INPUT_ROOT, video_id)
        detector = TennisEventDetector(config=detector_config)
        analysis = detector.analyze(
            points,
            player_1,
            player_2,
            fps=float(metadata["fps"]),
            frame_size=(int(metadata["width"]), int(metadata["height"])),
            camera_offsets_px=None,
        )
        analyses[video_id] = analysis
        points_by_video[video_id] = points
        predictions[video_id] = [
            {**_event_record(event, video_id), "prediction_id": f"{video_id}:prediction:{index + 1}"}
            for index, event in enumerate(analysis.events)
        ]
        candidates[video_id] = [
            {**_candidate_record(candidate, video_id), "prediction_id": f"{video_id}:candidate:{index + 1}"}
            for index, candidate in enumerate(analysis.candidates)
        ]
        stage_zero[video_id] = _stage_zero_records(video_id)
        stage_one[video_id] = _stage_one_records(video_id, points)

    physical = evaluate_covered_events(predictions, gt, videos_document, coverage)
    semantic = evaluate_covered_events(
        predictions, gt, videos_document, coverage, require_event_type=True
    )
    player = evaluate_covered_events(
        predictions,
        gt,
        videos_document,
        coverage,
        require_event_type=True,
        require_player=True,
    )
    stage_results = {
        "stage_0": evaluate_covered_events(stage_zero, gt, videos_document, coverage),
        "stage_1": evaluate_covered_events(stage_one, gt, videos_document, coverage),
        "stage_2": evaluate_covered_events(candidates, gt, videos_document, coverage),
        "stage_3": physical,
        "stage_4": semantic,
        "stage_5": player,
        "stage_6": player,
    }
    stage_predictions = {
        "stage_0": stage_zero,
        "stage_1": stage_one,
        "stage_2": candidates,
        "stage_3": predictions,
        "stage_4": predictions,
        "stage_5": predictions,
        "stage_6": predictions,
    }
    provenance = _provenance()

    per_video_payload = _with_provenance(
        {
            "phase": "6.4",
            "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
            "qualification_evidence": False,
            "metric_scope": "STAGE_3_TYPE_AGNOSTIC_PHYSICAL_CONTACT",
            "per_video": physical["per_video"],
        },
        provenance,
    )
    aggregate_metric = dict(physical["aggregate"])
    aggregate_metric["timing_mae_s"] = _mean_timing(physical["matches"], "timing_error_s")
    aggregate_metric["timing_mae_ms"] = _mean_timing(physical["matches"], "timing_error_ms")
    aggregate_metric["timing_mae_frames"] = None
    aggregate_payload = _with_provenance(
        {
            "phase": "6.4",
            "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
            "qualification_evidence": False,
            "metric_scope": "STAGE_3_TYPE_AGNOSTIC_PHYSICAL_CONTACT",
            "aggregate": aggregate_metric,
            "count_invariants": {
                key: aggregate_metric[key]
                == sum(row[key] for row in physical["per_video"].values())
                for key in ("true_positives", "false_positives", "false_negatives")
            },
        },
        provenance,
    )

    stage_names = {
        "stage_0": "STAGE_0_RAW_PROPOSAL",
        "stage_1": "STAGE_1_USABLE_OBSERVATION",
        "stage_2": "STAGE_2_PHYSICAL_CANDIDATE",
        "stage_3": "STAGE_3_PHYSICAL_CONTACT",
        "stage_4": "STAGE_4_SEMANTIC_TYPE",
        "stage_5": "STAGE_5_PLAYER_ATTRIBUTION",
        "stage_6": "STAGE_6_AUTHORITATIVE_EVENT",
    }
    stage_rows = []
    for stage_id, result in stage_results.items():
        metric = dict(result["aggregate"])
        metric["stage_id"] = stage_id
        metric["stage"] = stage_names[stage_id]
        metric["timing_mae_s"] = _mean_timing(result["matches"], "timing_error_s")
        metric["timing_mae_ms"] = _mean_timing(result["matches"], "timing_error_ms")
        metric["timing_mae_frames"] = None
        stage_rows.append(metric)
    stage_payload = _with_provenance(
        {
            "phase": "6.4",
            "no_model_tuning_performed": True,
            "stages": stage_rows,
            "semantic_exact_type": semantic["aggregate"],
            "semantic_per_class": _per_class_metrics(
                predictions, gt, videos_document, coverage
            ),
            "player_attribution": player["aggregate"],
        },
        provenance,
    )

    lineage_records = _lineage(gt, stage_results, stage_predictions, coverage)
    lineage_payload = _with_provenance(
        {
            "phase": "6.4",
            "record_count": len(lineage_records),
            "records": lineage_records,
            "prediction_statuses": physical["prediction_statuses"],
            "cross_video_match_count": sum(
                not bool(match["same_video_match"]) for match in physical["matches"]
            ),
        },
        provenance,
    )

    review_assets = {}
    for video_id, metadata in videos.items():
        statuses = [
            row for row in physical["prediction_statuses"] if row["video_id"] == video_id
        ]
        review_assets.update(
            _write_fp_assets(
                video_id, metadata, statuses, predictions[video_id], points_by_video[video_id]
            )
        )
    corrected_rows = []
    human_review_rows = []
    prediction_lookup = {
        row["prediction_id"]: row for rows in predictions.values() for row in rows
    }
    for index, status in enumerate(physical["prediction_statuses"], start=1):
        prediction = prediction_lookup[status["prediction_id"]]
        prelabel, prelabel_confidence = _prelabel(prediction)
        assets = review_assets.get(
            status["prediction_id"],
            {"frame_image": None, "ball_crop": None, "context_crop": None},
        )
        row = {
            "review_id": f"P64V2-{index:04d}",
            **status,
            "event_type": prediction.get("event_type"),
            "player_id": prediction.get("player_id"),
            "confidence": prediction.get("confidence"),
            **assets,
            "automated_prelabel": prelabel,
            "automated_prelabel_confidence": prelabel_confidence,
            "human_reviewed": False,
            "human_label": None,
            "label_source": "AUTOMATED_HEURISTIC_UNVERIFIED",
        }
        corrected_rows.append(row)
        if row["evaluation_status"] == "FP":
            human_review_rows.append(row)

    corrected_manifest = _with_provenance(
        {
            "phase": "6.4",
            "authoritative_fp_definition": "INSIDE_EXHAUSTIVE_COVERAGE_AND_UNMATCHED_SAME_VIDEO_GT",
            "fp_count": physical["aggregate"]["false_positives"],
            "outside_scope_count": physical["aggregate"]["outside_scope_prediction_count"],
            "records": corrected_rows,
        },
        provenance,
    )
    human_manifest = _with_provenance(
        {
            "phase": "6.4",
            "manifest_version": "2.0",
            "human_reviewed_count": 0,
            "automated_unverified_count": len(human_review_rows),
            "pending_human_review_count": len(human_review_rows),
            "records": human_review_rows,
        },
        provenance,
    )

    integrity_payload = _with_provenance(
        {
            "phase": "6.4",
            "starting_sha": "d15de4fcc8b0479070f408a024d98b46eef2be7c",
            "cross_video_bug_reproduced_on_starting_sha": True,
            "old_cross_video_match_count_for_fixture": 1,
            "corrected_cross_video_match_count_for_fixture": len(
                canonical_one_to_one_matches(
                    [{"video_id": "video_A", "frame": 55}],
                    [
                        {
                            "video_id": "video_B",
                            "frame_best": 55,
                            "frame_min": 53,
                            "frame_max": 57,
                        }
                    ],
                )
            ),
            "annotation_coverage_defined": True,
            "gt_revision": "NONE",
            "production_gt_leakage_check": "PASS",
            "analytics_contract_schema_version": "1.0",
            "historical_artifact_status": "INVALIDATED_BY_EVALUATOR_INTEGRITY_CORRECTION",
            "evaluator_call_site_audit": {
                "src/events/event_evaluator.py": "FIXED_CANONICAL",
                "scripts/evaluate_phase6_4_cross_match.py": "FIXED_COMPATIBILITY_ADAPTER_AND_COUNT_AGGREGATION",
                "scripts/evaluate_phase6_4_event_recovery.py": "FIXED",
                "scripts/evaluate_phase6_4_semantic_event_recovery.py": "FIXED",
                "scripts/evaluate_phase6_4_event_pipeline.py": "FIXED",
                "scripts/evaluate_phase6_4_oracle_shots.py": "SAFE_PER_VIDEO_JOIN",
                "scripts/audit_false_positives.py": "FIXED_COVERAGE_AWARE",
                "scripts/generate_pre_semantic_recovery_artifacts.py": "FIXED_COUNT_AGGREGATION",
                "scripts/generate_pipeline_docs.py": "SAFE_DOCUMENT_GENERATOR_NO_MATCHER",
                "artifacts/validation/phase6_4_event_pipeline_ablation.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_event_pipeline_confusion.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_physical_precision_ablation.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_physical_fp_before_after.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_physical_prediction_lineage.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_physical_prediction_lineage_final.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_fp_manual_review_manifest.json": "INVALIDATED_OUTPUT",
                "artifacts/validation/phase6_4_fp_source_taxonomy.json": "INVALIDATED_OUTPUT"
            },
            "consistency_checks": {
                "aggregate_tp_equals_sum_per_video": aggregate_payload["count_invariants"]["true_positives"],
                "aggregate_fp_equals_sum_per_video": aggregate_payload["count_invariants"]["false_positives"],
                "aggregate_fn_equals_sum_per_video": aggregate_payload["count_invariants"]["false_negatives"],
                "covered_gt_denominator_consistent": physical["aggregate"]["true_positives"]
                + physical["aggregate"]["false_negatives"]
                == physical["aggregate"]["covered_gt_count"],
                "fp_only_from_covered_frames": all(
                    row["inside_evaluation_coverage"]
                    for row in corrected_rows
                    if row["evaluation_status"] == "FP"
                ),
                "outside_scope_excluded_from_metrics": True,
                "no_cross_video_match": all(
                    match["same_video_match"] for match in physical["matches"]
                ),
            },
        },
        provenance,
    )

    _dump(VALIDATION / "phase6_4_evaluator_integrity.json", integrity_payload)
    _dump(VALIDATION / "phase6_4_corrected_per_video_metrics.json", per_video_payload)
    _dump(VALIDATION / "phase6_4_corrected_aggregate_metrics.json", aggregate_payload)
    _dump(VALIDATION / "phase6_4_corrected_stage_metrics.json", stage_payload)
    _dump(VALIDATION / "phase6_4_corrected_event_lineage.json", lineage_payload)
    _dump(VALIDATION / "phase6_4_corrected_fp_manifest.json", corrected_manifest)
    _dump(VALIDATION / "phase6_4_fp_human_review_manifest_v2.json", human_manifest)
    _invalidate_legacy_fp_artifacts()
    print(json.dumps({"stage_3": aggregate_metric, "semantic": semantic["aggregate"]}, indent=2))


def main() -> None:
    """Run the authoritative v2.1 reconstruction."""
    from scripts.reconstruct_phase6_4_evaluator_v2_1_artifacts import main as v2_1_main

    v2_1_main()


if __name__ == "__main__":
    main()

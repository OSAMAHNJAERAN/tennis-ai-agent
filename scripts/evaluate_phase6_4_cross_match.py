"""Time-aware Phase 6.4 evaluator for the cross-match diagnostic data.

``video_08`` through ``video_10`` are development diagnostics. Re-scoring their
preserved artifacts is useful for regression analysis but is not qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from src.events.event_evaluator import (  # noqa: E402
    canonical_event_type,
    canonical_one_to_one_matches,
    frame_inside_annotation_coverage,
)


EVENT_CLASSES = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")
SHOT_CLASSES = ("FOREHAND", "BACKHAND", "SERVE")


def compute_sha256(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.fmean(values)) if values else None


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _event_type(record: Dict[str, Any]) -> str:
    return canonical_event_type(record)


def _frame(record: Dict[str, Any]) -> int:
    for key in ("frame_index", "frame", "frame_best", "frame_hit"):
        if key in record:
            return int(record[key])
    raise KeyError(f"No frame field in record: {record}")


def one_to_one_matches(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    tolerance_frames: int,
    *,
    require_event_type: bool = False,
    fps: float = 30.0,
) -> List[Tuple[int, int, int]]:
    """Compatibility adapter to the one canonical timestamp matcher."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    normalized_predictions = [
        {
            **record,
            **(
                {"video_id": record["_video_id"]}
                if "video_id" not in record and "_video_id" in record
                else {}
            ),
        }
        for record in predictions
    ]
    normalized_gt = [
        {
            **record,
            **(
                {"video_id": record["_video_id"]}
                if "video_id" not in record and "_video_id" in record
                else {}
            ),
        }
        for record in ground_truth
    ]
    explicit_ids = {
        str(record["video_id"])
        for record in (*normalized_predictions, *normalized_gt)
        if record.get("video_id") is not None
    }
    single_video_id = next(iter(explicit_ids)) if len(explicit_ids) == 1 else None
    matches = canonical_one_to_one_matches(
        normalized_predictions,
        normalized_gt,
        tolerance_s=tolerance_frames / fps,
        fps=fps,
        require_event_type=require_event_type,
        single_video_id=single_video_id,
    )
    return [
        (pred_index, gt_index, int(round(distance_s * fps)))
        for pred_index, gt_index, distance_s in matches
    ]


def evaluate_split_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth_shots: List[Dict[str, Any]],
    ground_truth_events: List[Dict[str, Any]],
    tolerance_frames: Optional[int] = None,
    *,
    fps: float = 30.0,
    event_predictions: Optional[List[Dict[str, Any]]] = None,
    tolerance_seconds: float = 0.2,
) -> Dict[str, Any]:
    """Evaluate physical events, conditional class, and end-to-end shots."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if tolerance_frames is None:
        tolerance_frames = max(1, round(tolerance_seconds * fps))
    effective_frame_tolerance_ms = 1000.0 * tolerance_frames / fps

    evaluated_event_predictions = event_predictions if event_predictions is not None else predictions
    evaluated_event_gt = ground_truth_events if ground_truth_events else ground_truth_shots
    event_matches = one_to_one_matches(
        evaluated_event_predictions,
        evaluated_event_gt,
        tolerance_frames,
        require_event_type=bool(ground_truth_events and event_predictions is not None),
        fps=fps,
    )
    timing_frames = [difference for _, _, difference in event_matches]
    event_overall = _prf(
        len(event_matches),
        len(evaluated_event_predictions) - len(event_matches),
        len(evaluated_event_gt) - len(event_matches),
    )
    event_overall.update({
        "mean_timing_error_frames": _mean(timing_frames),
        "mean_timing_error_ms": 1000.0 * _mean(timing_frames) / fps if timing_frames else None,
        "matching_tolerance_frames": tolerance_frames,
        "matching_tolerance_ms": 1000.0 * tolerance_seconds,
        "effective_frame_tolerance_ms": effective_frame_tolerance_ms,
        "matching_semantics": "GLOBAL_NEAREST_ONE_TO_ONE_SAME_EVENT_TYPE",
        "scope": "ALL_PHYSICAL_EVENTS" if event_predictions is not None else "HITS_ONLY_COMPATIBILITY",
    })

    per_event_class: Dict[str, Dict[str, Any]] = {}
    if ground_truth_events and event_predictions is not None:
        for event_class in EVENT_CLASSES:
            class_predictions = [p for p in event_predictions if _event_type(p) == event_class]
            class_gt = [g for g in ground_truth_events if _event_type(g) == event_class]
            class_matches = one_to_one_matches(
                class_predictions,
                class_gt,
                tolerance_frames,
                require_event_type=True,
                fps=fps,
            )
            metric = _prf(
                len(class_matches),
                len(class_predictions) - len(class_matches),
                len(class_gt) - len(class_matches),
            )
            errors = [difference for _, _, difference in class_matches]
            metric["timing_mae_frames"] = _mean(errors)
            metric["timing_mae_ms"] = 1000.0 * _mean(errors) / fps if errors else None
            per_event_class[event_class] = metric

    shot_matches = one_to_one_matches(
        predictions, ground_truth_shots, tolerance_frames, fps=fps
    )
    matched_pairs = [(predictions[p], ground_truth_shots[g]) for p, g, _ in shot_matches]
    shot_metrics: Dict[str, Dict[str, Any]] = {}
    for shot_class in SHOT_CLASSES:
        true_positive = sum(
            prediction.get("shot_type") == shot_class and gt.get("shot_type") == shot_class
            for prediction, gt in matched_pairs
        )
        false_positive = sum(
            prediction.get("shot_type") == shot_class and gt.get("shot_type") != shot_class
            for prediction, gt in matched_pairs
        )
        false_negative = sum(
            prediction.get("shot_type") != shot_class and gt.get("shot_type") == shot_class
            for prediction, gt in matched_pairs
        )
        metric = _prf(true_positive, false_positive, false_negative)
        # Conditional classification is defined only on physical hits that the
        # detector matched.  Reporting support from every GT hit would mix
        # detector misses into a conditional classifier metric.
        metric["support"] = sum(
            gt.get("shot_type") == shot_class for _, gt in matched_pairs
        )
        shot_metrics[shot_class] = metric

    classified_matches = sum(
        prediction.get("shot_type") != "UNKNOWN" for prediction, _ in matched_pairs
    )
    unknown_matches = len(matched_pairs) - classified_matches
    e2e_true_positive = sum(
        prediction.get("shot_type") == gt.get("shot_type")
        and prediction.get("player_id") == gt.get("player_id")
        for prediction, gt in matched_pairs
    )
    return {
        "event_detection": event_overall,
        "event_detection_per_class": per_event_class,
        "conditional_shot_classification": {
            "per_class": shot_metrics,
            "macro_f1": float(statistics.fmean(metric["f1"] for metric in shot_metrics.values())),
            "matched_hit_count": len(matched_pairs),
            "unknown_abstentions": unknown_matches,
            "unknown_rate": unknown_matches / len(matched_pairs) if matched_pairs else None,
            "coverage": classified_matches / len(matched_pairs) if matched_pairs else None,
        },
        "end_to_end_shot_recognition": _prf(
            e2e_true_positive,
            len(predictions) - e2e_true_positive,
            len(ground_truth_shots) - e2e_true_positive,
        ),
    }


def aggregate_split_metrics(per_video: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate independent per-video counts without concatenating timelines."""
    event_rows = [row["event_detection"] for row in per_video.values()]
    event_overall = _prf(
        sum(row["true_positives"] for row in event_rows),
        sum(row["false_positives"] for row in event_rows),
        sum(row["false_negatives"] for row in event_rows),
    )
    timing_weight = sum(row["true_positives"] for row in event_rows)
    event_overall.update(
        {
            "mean_timing_error_frames": None,
            "mean_timing_error_ms": (
                sum(
                    row["mean_timing_error_ms"] * row["true_positives"]
                    for row in event_rows
                    if row["mean_timing_error_ms"] is not None
                )
                / timing_weight
                if timing_weight
                else None
            ),
            "matching_tolerance_ms": 200.0,
            "matching_semantics": "PER_VIDEO_CANONICAL_TIMESTAMP_INTERVAL_ONE_TO_ONE",
            "scope": "ALL_PHYSICAL_EVENTS",
        }
    )

    per_event_class = {}
    for event_class in EVENT_CLASSES:
        rows = [
            row["event_detection_per_class"][event_class]
            for row in per_video.values()
            if event_class in row["event_detection_per_class"]
        ]
        if not rows:
            continue
        per_event_class[event_class] = _prf(
            sum(row["true_positives"] for row in rows),
            sum(row["false_positives"] for row in rows),
            sum(row["false_negatives"] for row in rows),
        )

    conditional_rows = [row["conditional_shot_classification"] for row in per_video.values()]
    shot_metrics = {}
    for shot_class in SHOT_CLASSES:
        rows = [row["per_class"][shot_class] for row in conditional_rows]
        metric = _prf(
            sum(row["true_positives"] for row in rows),
            sum(row["false_positives"] for row in rows),
            sum(row["false_negatives"] for row in rows),
        )
        metric["support"] = sum(row["support"] for row in rows)
        shot_metrics[shot_class] = metric
    matched_hits = sum(row["matched_hit_count"] for row in conditional_rows)
    unknown = sum(row["unknown_abstentions"] for row in conditional_rows)

    end_rows = [row["end_to_end_shot_recognition"] for row in per_video.values()]
    return {
        "event_detection": event_overall,
        "event_detection_per_class": per_event_class,
        "conditional_shot_classification": {
            "per_class": shot_metrics,
            "macro_f1": float(statistics.fmean(row["f1"] for row in shot_metrics.values())),
            "matched_hit_count": matched_hits,
            "unknown_abstentions": unknown,
            "unknown_rate": unknown / matched_hits if matched_hits else None,
            "coverage": (matched_hits - unknown) / matched_hits if matched_hits else None,
        },
        "end_to_end_shot_recognition": _prf(
            sum(row["true_positives"] for row in end_rows),
            sum(row["false_positives"] for row in end_rows),
            sum(row["false_negatives"] for row in end_rows),
        ),
    }


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _evaluate_existing_video(
    video_id: str,
    metadata: Dict[str, Any],
    output_root: str,
    gt_events: Dict[str, List[Dict[str, Any]]],
    gt_shots: Dict[str, List[Dict[str, Any]]],
    coverage: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_dir = os.path.join(output_root, video_id)
    shots = _load_json(os.path.join(output_dir, "shot_events.json")).get("shot_events", [])
    events = _load_json(os.path.join(output_dir, "match_events.json")).get("events", [])
    detections = _load_json(os.path.join(output_dir, "detections.json"))
    frames = detections.get("frames", [])
    covered_shots = [
        record
        for record in shots
        if frame_inside_annotation_coverage(video_id, _frame(record), coverage)
    ]
    covered_events = [
        record
        for record in events
        if frame_inside_annotation_coverage(video_id, _frame(record), coverage)
    ]
    metrics = evaluate_split_metrics(
        covered_shots,
        gt_shots.get(video_id, []),
        gt_events.get(video_id, []),
        fps=float(metadata["fps"]),
        event_predictions=covered_events,
    )
    metrics["outside_scope_event_predictions"] = len(events) - len(covered_events)
    metrics["outside_scope_shot_predictions"] = len(shots) - len(covered_shots)
    metrics["player_1_coverage"] = (
        sum(frame.get("player_1") is not None for frame in frames) / len(frames) if frames else None
    )
    metrics["player_2_coverage"] = (
        sum(frame.get("player_2") is not None for frame in frames) / len(frames) if frames else None
    )
    run_metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(run_metrics_path):
        metrics["pipeline_fps"] = _load_json(run_metrics_path).get("processing_fps")
    return metrics, shots, events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument(
        "--output-root",
        default="outputs/phase6_4_qualification/final_cross_match_holdout",
        help="Legacy path containing video_08-10 diagnostic artifacts",
    )
    parser.add_argument(
        "--report",
        default="outputs/phase6_4_qualification/aggregate_cross_match_diagnostic_corrected.json",
    )
    args = parser.parse_args()

    videos = _load_json("data/benchmarks/cross_match_final_holdout/videos.json")["videos"]
    coverage = _load_json(
        "data/benchmarks/cross_match_final_holdout/annotation_coverage.json"
    )
    gt_events = _load_json("data/benchmarks/cross_match_final_holdout/ground_truth_events.json")["events"]
    gt_shots = _load_json("data/benchmarks/cross_match_final_holdout/ground_truth_shots.json")["shots"]

    if args.run_inference:
        from src.pipeline.phase6_pipeline import Phase6Pipeline

        for video_id, metadata in videos.items():
            # Each clip is an independent diagnostic source.  Reusing one
            # stateful scoring/tracking pipeline leaks event IDs and dead-ball
            # state across videos and makes output depend on evaluation order.
            pipeline = Phase6Pipeline(config_path="configs/phase6_analytics/pipeline.yaml")
            if compute_sha256(metadata["path"]) != metadata["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for {video_id}")
            output_dir = os.path.join(args.output_root, video_id)
            os.makedirs(output_dir, exist_ok=True)
            started = time.time()
            pipeline.run(metadata["path"], output_dir)
            print(f"{video_id}: inference completed in {time.time() - started:.2f}s")

    per_video: Dict[str, Any] = {}
    for video_id, metadata in videos.items():
        metrics, shots, events = _evaluate_existing_video(
            video_id, metadata, args.output_root, gt_events, gt_shots, coverage
        )
        per_video[video_id] = metrics
    aggregate = aggregate_split_metrics(per_video)
    report = {
        "schema_version": "1.0",
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC",
        "qualification_evidence": False,
        "per_video": per_video,
        "aggregate": aggregate,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

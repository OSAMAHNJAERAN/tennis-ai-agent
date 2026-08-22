"""Offline Phase 6.4 staged event-recovery experiment.

Only preserved, already-consumed diagnostic trajectories from video_08--10 are
used.  No video inference, GT-assisted production inference, scoring mutation,
or evaluator-semantic change occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_phase6_4_cross_match import evaluate_split_metrics, one_to_one_matches
from src.events.event_detector import EventDetectionAnalysis, TennisEvent, TennisEventDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


EVENT_CLASSES = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")
DEFAULT_INPUT_ROOT = os.path.join(
    REPO_ROOT, "outputs", "phase6_4_qualification", "cross_match_diagnostic_final"
)
LEGACY_BASELINE_ROOT = os.path.join(
    REPO_ROOT, "outputs", "phase6_4_qualification", "final_cross_match_holdout"
)
VALIDATION_ROOT = os.path.join(REPO_ROOT, "artifacts", "validation")
GT_PATH = os.path.join(
    REPO_ROOT, "data", "benchmarks", "cross_match_final_holdout", "ground_truth_events.json"
)
VIDEOS_PATH = os.path.join(
    REPO_ROOT, "data", "benchmarks", "cross_match_final_holdout", "videos.json"
)
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "phase6_analytics", "pipeline.yaml")
PRESERVED_PRODUCTION_REPORT = os.path.join(
    REPO_ROOT,
    "outputs",
    "phase6_4_qualification",
    "aggregate_cross_match_diagnostic_final.json",
)


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(path: str) -> Dict[str, Any]:
    return {
        "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def _dump(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def _canonical_type(record: Mapping[str, Any]) -> str:
    value = str(record.get("event_type", ""))
    return "PLAYER_HIT" if value in {"PLAYER_1_HIT", "PLAYER_2_HIT"} else value


def _frame(record: Mapping[str, Any]) -> int:
    for key in ("frame_index", "frame", "frame_best", "refined_frame"):
        if key in record:
            return int(record[key])
    raise KeyError(record)


def _event_record(event: TennisEvent, video_id: str) -> Dict[str, Any]:
    return {
        "_video_id": video_id,
        "frame": event.frame_index,
        "timestamp_s": event.timestamp_s,
        "event_type": event.event_type.value,
        "player_id": event.player_id,
        "confidence": event.confidence,
        "trajectory_state": event.trajectory_state,
        "evidence": event.evidence,
    }


def _candidate_record(candidate: Any, video_id: str) -> Dict[str, Any]:
    return {
        "_video_id": video_id,
        "frame": int(candidate.frame_index),
        "timestamp_s": float(candidate.timestamp_s),
        "score": float(candidate.score),
        "trajectory_state": candidate.trajectory_state,
        "evidence": candidate.evidence,
    }


def _bbox(record: Optional[Dict[str, Any]]) -> Optional[BBox]:
    if record is None or record.get("bbox") is None:
        return None
    values = [float(value) for value in record["bbox"]]
    return BBox(*values)


def _load_video_inputs(
    input_root: str, video_id: str
) -> Tuple[List[TemporalBallPoint], List[Optional[BBox]], List[Optional[BBox]], List[Dict[str, Any]]]:
    output = os.path.join(input_root, video_id)
    trajectories = _load(os.path.join(output, "trajectories.json"))["ball_trajectory"]
    detections = _load(os.path.join(output, "detections.json"))["frames"]
    if len(trajectories) != len(detections):
        raise RuntimeError(f"Unaligned preserved artifacts for {video_id}")
    points: List[TemporalBallPoint] = []
    p1: List[Optional[BBox]] = []
    p2: List[Optional[BBox]] = []
    for trajectory, detection in zip(trajectories, detections):
        state_name = str(trajectory.get("state", "MISSING"))
        state = BallState[state_name] if state_name in BallState.__members__ else BallState.MISSING
        points.append(TemporalBallPoint(
            frame_index=int(trajectory["frame_index"]),
            timestamp_seconds=float(trajectory["timestamp_seconds"]),
            x_px=float(trajectory["x_px"]) if trajectory.get("x_px") is not None else None,
            y_px=float(trajectory["y_px"]) if trajectory.get("y_px") is not None else None,
            court_x_m=float(trajectory["court_x_m"]) if trajectory.get("court_x_m") is not None else None,
            court_y_m=float(trajectory["court_y_m"]) if trajectory.get("court_y_m") is not None else None,
            confidence=float(trajectory["confidence"]) if trajectory.get("confidence") is not None else None,
            state=state,
            source="PRESERVED_DIAGNOSTIC_ARTIFACT",
        ))
        p1.append(_bbox(detection.get("player_1")))
        p2.append(_bbox(detection.get("player_2")))
    return points, p1, p2, detections


def _variant_configs(base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def variant(**updates: Any) -> Dict[str, Any]:
        value = dict(base)
        value.update(updates)
        return value

    minimal = dict(
        enable_player_temporal_proximity=False,
        enable_event_time_refinement=False,
        enable_bounce_contact_check=False,
        enable_serve_semantics=False,
        enable_dead_ball_gating=False,
        enable_camera_motion_compensation=False,
        enable_provenance_weighting=False,
    )
    return {
        "B_HIGH_RECALL_CANDIDATES": variant(**minimal),
        "C_TEMPORAL_PLAYER_PROXIMITY": variant(**{**minimal, "enable_player_temporal_proximity": True}),
        "D_EVENT_TIME_REFINEMENT": variant(**{
            **minimal,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
        }),
        "E_EVENT_SPECIFIC_VERIFICATION": variant(**{
            **minimal,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_bounce_contact_check": True,
            "enable_serve_semantics": True,
        }),
        "F_VISION_DEAD_BALL_GATING": variant(**{
            **minimal,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_bounce_contact_check": True,
            "enable_serve_semantics": True,
            "enable_dead_ball_gating": True,
        }),
        "G_CAMERA_MOTION_COMPENSATION": variant(**{
            **minimal,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_bounce_contact_check": True,
            "enable_serve_semantics": True,
            "enable_dead_ball_gating": True,
            "enable_camera_motion_compensation": True,
        }),
        "H_FINAL_INTEGRATED": variant(**{
            **minimal,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_bounce_contact_check": True,
            "enable_serve_semantics": True,
            "enable_dead_ball_gating": True,
            "enable_camera_motion_compensation": True,
            "enable_provenance_weighting": True,
        }),
    }


def _matching_by_gt(
    predictions: Sequence[Dict[str, Any]],
    gt: Sequence[Dict[str, Any]],
    tolerance: int,
    require_type: bool,
) -> Dict[int, Tuple[int, int]]:
    return {
        gt_index: (prediction_index, difference)
        for prediction_index, gt_index, difference in one_to_one_matches(
            predictions, gt, tolerance, require_event_type=require_type
        )
    }


def _recall_summary(
    per_video_predictions: Mapping[str, Sequence[Dict[str, Any]]],
    gt_by_video: Mapping[str, Sequence[Dict[str, Any]]],
    videos: Mapping[str, Dict[str, Any]],
    *,
    require_type: bool,
) -> Dict[str, Any]:
    matched: Counter[str] = Counter()
    support: Counter[str] = Counter()
    timing: List[int] = []
    for video_id, gt in gt_by_video.items():
        predictions = list(per_video_predictions.get(video_id, []))
        tolerance = max(1, round(0.2 * float(videos[video_id]["fps"])))
        matches = one_to_one_matches(predictions, gt, tolerance, require_event_type=require_type)
        timing.extend(difference for _, _, difference in matches)
        for _, gt_index, _ in matches:
            matched[_canonical_type(gt[gt_index])] += 1
            matched["OVERALL"] += 1
        for row in gt:
            support[_canonical_type(row)] += 1
            support["OVERALL"] += 1
    recall = {
        name: matched[name] / support[name] if support[name] else 0.0
        for name in (*EVENT_CLASSES, "OVERALL")
    }
    return {
        "matched": dict(matched),
        "support": dict(support),
        "recall": recall,
        "timing_mae_frames": statistics.fmean(timing) if timing else None,
        "timing_mae_ms": 1000.0 * statistics.fmean(timing) / 30.0 if timing else None,
    }


def _aggregate_event_metrics(
    per_video_events: Mapping[str, Sequence[Dict[str, Any]]],
    gt_by_video: Mapping[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    predictions = [
        {**row, "_video_id": video_id}
        for video_id, rows in per_video_events.items() for row in rows
    ]
    gt = [
        {**row, "_video_id": video_id}
        for video_id, rows in gt_by_video.items() for row in rows
    ]
    res = evaluate_split_metrics([], [], gt, fps=30.0, event_predictions=predictions)
    phys = evaluate_split_metrics(predictions, [], gt, fps=30.0, event_predictions=None)
    res["event_detection"] = phys["event_detection"]
    return res


def _metric_row(
    name: str,
    candidate: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    event_count: int,
) -> Dict[str, Any]:
    overall = metrics["event_detection"]
    per_class = metrics["event_detection_per_class"]
    return {
        "variant": name,
        "candidate_recall": candidate["recall"] if candidate else None,
        "candidate_timing_mae_frames": candidate["timing_mae_frames"] if candidate else None,
        "candidate_timing_mae_ms": candidate["timing_mae_ms"] if candidate else None,
        "serve": per_class["SERVE_CONTACT"],
        "player_hit": per_class["PLAYER_HIT"],
        "bounce": per_class["BOUNCE"],
        "overall": overall,
        "event_count": event_count,
    }


def _stage_predictions(
    analyses: Mapping[str, EventDetectionAnalysis],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    names = (
        "raw_candidate_generator", "physics_verification", "player_attribution",
        "event_type_classification", "temporal_suppression", "final_authoritative_events",
    )
    stages: Dict[str, Dict[str, List[Dict[str, Any]]]] = {name: {} for name in names}
    for video_id, analysis in analyses.items():
        stages["raw_candidate_generator"][video_id] = [
            _candidate_record(candidate, video_id) for candidate in analysis.candidates
        ]
        for stage_name in ("physics_verification", "player_attribution", "event_type_classification", "temporal_suppression"):
            rows = []
            for trace in analysis.verification_traces:
                if not trace["stage_pass"].get(stage_name):
                    continue
                row = {"frame": trace["refined_frame"], "_video_id": video_id}
                if stage_name in {"event_type_classification", "temporal_suppression"}:
                    row["event_type"] = trace["candidate_event_type"]
                rows.append(row)
            stages[stage_name][video_id] = rows
        stages["final_authoritative_events"][video_id] = [
            _event_record(event, video_id) for event in analysis.events
        ]
    return stages


def _forensic_rows(
    analyses: Mapping[str, EventDetectionAnalysis],
    gt_by_video: Mapping[str, Sequence[Dict[str, Any]]],
    videos: Mapping[str, Dict[str, Any]],
    points_by_video: Mapping[str, Sequence[TemporalBallPoint]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    fn_taxonomy: Counter[str] = Counter()
    fp_taxonomy: Counter[str] = Counter()
    for video_id, analysis in analyses.items():
        fps = float(videos[video_id]["fps"])
        tolerance = max(1, round(0.2 * fps))
        gt = list(gt_by_video[video_id])
        candidates = [_candidate_record(candidate, video_id) for candidate in analysis.candidates]
        finals = [_event_record(event, video_id) for event in analysis.events]
        candidate_matches = _matching_by_gt(candidates, gt, tolerance, False)
        final_matches = _matching_by_gt(finals, gt, tolerance, False)
        any_final_matches = _matching_by_gt(finals, gt, tolerance, False)
        trace_by_id = {trace["candidate_id"]: trace for trace in analysis.verification_traces}

        for gt_index, ground_truth in enumerate(gt):
            best_frame = _frame(ground_truth)
            candidate_match = candidate_matches.get(gt_index)
            final_match = final_matches.get(gt_index)
            matched_candidate = candidates[candidate_match[0]] if candidate_match else None
            nearest_candidate = min(
                candidates, key=lambda row: (abs(_frame(row) - best_frame), _frame(row))
            ) if candidates else None
            candidate = matched_candidate or nearest_candidate
            candidate_id = candidate_match[0] + 1 if candidate_match else (
                candidates.index(nearest_candidate) + 1 if nearest_candidate else None
            )
            trace = trace_by_id.get(candidate_id) if candidate_id else None
            final_emitted = final_match is not None
            rejection_stage = None
            rejection_reason = None
            if not final_emitted:
                if not candidate_match:
                    low = max(0, int(ground_truth.get("frame_min", best_frame)) - tolerance)
                    high = min(len(points_by_video[video_id]) - 1, int(ground_truth.get("frame_max", best_frame)) + tolerance)
                    if not any(points_by_video[video_id][index].x_px is not None for index in range(low, high + 1)):
                        rejection_stage, rejection_reason = "RAW_BALL_TRAJECTORY", "TRACKING_GAP"
                    else:
                        rejection_stage, rejection_reason = "EVENT_CANDIDATE_GENERATION", "CANDIDATE_OUTSIDE_MATCH_WINDOW"
                elif gt_index in any_final_matches:
                    rejection_stage, rejection_reason = "EVENT_TYPE_CLASSIFICATION", "WRONG_EVENT_TYPE"
                elif trace:
                    rejection_stage = trace.get("rejection_stage") or "FINAL_AUTHORITATIVE_EVENT"
                    rejection_reason = trace.get("rejection_reason") or "UNKNOWN_ROOT_CAUSE"
                else:
                    rejection_stage, rejection_reason = "UNKNOWN", "UNKNOWN_ROOT_CAUSE"
                fn_taxonomy[rejection_reason] += 1

            evidence = candidate.get("evidence", {}) if candidate else {}
            rows.append({
                "video_id": video_id,
                "gt_event_id": int(ground_truth["event_id"]),
                "gt_event_type": _canonical_type(ground_truth),
                "gt_frame_best": best_frame,
                "gt_timestamp": best_frame / fps,
                "gt_timestamp_provenance": "DERIVED_FRAME_BEST_DIVIDED_BY_FPS",
                "nearest_candidate_frame": _frame(candidate) if candidate else None,
                "nearest_candidate_time": candidate.get("timestamp_s") if candidate else None,
                "candidate_time_offset_ms": (
                    1000.0 * (_frame(candidate) - best_frame) / fps if candidate else None
                ),
                "candidate_generated": candidate_match is not None,
                "candidate_score": candidate.get("score") if candidate else None,
                "ball_state": trace.get("ball_state") if trace else candidate.get("trajectory_state") if candidate else None,
                "ball_confidence": trace.get("ball_confidence") if trace else evidence.get("ball_confidence"),
                "pre_velocity": trace.get("pre_velocity_px_s") if trace else evidence.get("pre_velocity_px_s"),
                "post_velocity": trace.get("post_velocity_px_s") if trace else evidence.get("post_velocity_px_s"),
                "normalized_speed": trace.get("normalized_speed_per_s") if trace else evidence.get("normalized_speed_per_s"),
                "direction_change": trace.get("direction_change_degrees") if trace else evidence.get("direction_change_degrees"),
                "acceleration": trace.get("normalized_acceleration_per_s2") if trace else evidence.get("normalized_acceleration_per_s2"),
                "curvature": trace.get("normalized_curvature") if trace else evidence.get("normalized_curvature"),
                "player1_distance_normalized": trace.get("player1_distance_normalized") if trace else None,
                "player2_distance_normalized": trace.get("player2_distance_normalized") if trace else None,
                "selected_player": trace.get("selected_player") if trace else None,
                "player_scale": trace.get("player_scale_px") if trace else None,
                "pose_support": trace.get("pose_support") if trace else None,
                "camera_motion_state": trace.get("camera_motion_state") if trace else evidence.get("camera_motion_state"),
                "candidate_event_type": trace.get("candidate_event_type") if trace else None,
                "physical_event_type": trace.get("physical_event_type") if trace else None,
                "verification_score": trace.get("verification_score") if trace else None,
                "verification_pass": trace.get("verification_pass") if trace else False,
                "rejection_stage": rejection_stage,
                "rejection_reason": rejection_reason,
                "final_event_emitted": final_emitted,
            })

        matched_prediction_indexes = {
            prediction_index
            for prediction_index, _, _ in one_to_one_matches(finals, gt, tolerance, require_event_type=True)
        }
        any_matches = {
            prediction_index
            for prediction_index, _, _ in one_to_one_matches(finals, gt, tolerance, require_event_type=False)
        }
        last_gt = max(_frame(row) for row in gt)
        for index, prediction in enumerate(finals):
            if index in matched_prediction_indexes:
                continue
            if index in any_matches:
                cause = "WRONG_EVENT_TYPE"
            elif _frame(prediction) > last_gt + tolerance:
                cause = "POST_RALLY_BALL_MOTION"
            elif min(abs(_frame(prediction) - _frame(row)) for row in gt) <= 2 * tolerance:
                cause = "TIMING_DRIFT_OR_DUPLICATE"
            elif prediction.get("trajectory_state") in {"PREDICTED", "INTERPOLATED", "OCCLUDED"}:
                cause = "LOW_PROVENANCE_MOTION"
            else:
                cause = "UNMATCHED_KINEMATIC_MOTION"
            fp_taxonomy[cause] += 1
    return rows, dict(sorted(fn_taxonomy.items())), dict(sorted(fp_taxonomy.items()))


def _state_distributions(
    analyses: Mapping[str, EventDetectionAnalysis],
    gt_by_video: Mapping[str, Sequence[Dict[str, Any]]],
    videos: Mapping[str, Dict[str, Any]],
    points_by_video: Mapping[str, Sequence[TemporalBallPoint]],
) -> Dict[str, Dict[str, int]]:
    result = {"true_positive": Counter(), "false_positive": Counter(), "false_negative": Counter()}
    for video_id, analysis in analyses.items():
        gt = list(gt_by_video[video_id])
        predictions = [_event_record(event, video_id) for event in analysis.events]
        tolerance = max(1, round(0.2 * float(videos[video_id]["fps"])))
        matches = one_to_one_matches(predictions, gt, tolerance, require_event_type=True)
        matched_predictions = {prediction for prediction, _, _ in matches}
        matched_gt = {ground_truth for _, ground_truth, _ in matches}
        for index, prediction in enumerate(predictions):
            bucket = "true_positive" if index in matched_predictions else "false_positive"
            result[bucket][str(prediction.get("trajectory_state", "UNKNOWN"))] += 1
        for index, ground_truth in enumerate(gt):
            if index not in matched_gt:
                result["false_negative"][points_by_video[video_id][_frame(ground_truth)].state.value] += 1
    return {name: dict(sorted(counter.items())) for name, counter in result.items()}


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "minimum": None, "median": None, "p90": None, "maximum": None}
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "maximum": ordered[-1],
    }


def _reach_distributions(
    analyses: Mapping[str, EventDetectionAnalysis],
    videos: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float]]] = {
        "720p": {"near_player_distance_over_height": [], "far_player_distance_over_height": [], "player_scale_px": []},
        "1080p": {"near_player_distance_over_height": [], "far_player_distance_over_height": [], "player_scale_px": []},
    }
    for video_id, analysis in analyses.items():
        resolution = "720p" if int(videos[video_id]["height"]) == 720 else "1080p"
        for trace in analysis.verification_traces:
            distances = [
                float(value)
                for value in (
                    trace.get("player1_distance_normalized"),
                    trace.get("player2_distance_normalized"),
                )
                if value is not None and math.isfinite(float(value))
            ]
            if distances:
                grouped[resolution]["near_player_distance_over_height"].append(min(distances))
                grouped[resolution]["far_player_distance_over_height"].append(max(distances))
            if trace.get("player_scale_px") is not None:
                grouped[resolution]["player_scale_px"].append(float(trace["player_scale_px"]))
    return {
        resolution: {name: _distribution(values) for name, values in rows.items()}
        for resolution, rows in grouped.items()
    }


def _resolution_metrics(
    events: Mapping[str, Sequence[Dict[str, Any]]],
    gt: Mapping[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    groups = {"720p": ("video_08", "video_09"), "1080p": ("video_10",)}
    return {
        name: _aggregate_event_metrics(
            {video_id: events[video_id] for video_id in ids},
            {video_id: gt[video_id] for video_id in ids},
        )
        for name, ids in groups.items()
    }


def _legacy_metrics(root: str, gt: Mapping[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    events = {
        video_id: _load(os.path.join(root, video_id, "match_events.json")).get("events", [])
        for video_id in gt
    }
    return _aggregate_event_metrics(events, gt)


def _pre_recovery_stages(
    input_root: str,
    gt: Mapping[str, Sequence[Dict[str, Any]]],
    videos: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    verifier: Dict[str, List[Dict[str, Any]]] = {}
    finals: Dict[str, List[Dict[str, Any]]] = {}
    for video_id in gt:
        candidates[video_id] = _load(
            os.path.join(input_root, video_id, "event_candidates.json")
        ).get("candidates", [])
        verifier[video_id] = _load(
            os.path.join(input_root, video_id, "scoring_events.json")
        ).get("scoring_events", [])
        finals[video_id] = _load(
            os.path.join(input_root, video_id, "match_events.json")
        ).get("events", [])
    return {
        "candidate_output_already_thresholded_and_nms": _recall_summary(
            candidates, gt, videos, require_type=False
        ),
        "any_integrated_verifier_emission_type_agnostic": _recall_summary(
            verifier, gt, videos, require_type=False
        ),
        "correctly_typed_integrated_verifier_emission": _recall_summary(
            verifier, gt, videos, require_type=True
        ),
        "scoring_filtered_final_authoritative": _recall_summary(
            finals, gt, videos, require_type=True
        ),
        "limitation": (
            "Physics-only, player-attribution-only, and pre-NMS stages were not persisted; "
            "scoring rows are the only artifact-observable verifier proxy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--variant", default="all", help="all or one B_* through H_*")
    args = parser.parse_args()

    gt_by_video = _load(GT_PATH)["events"]
    videos = _load(VIDEOS_PATH)["videos"]
    with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)["event_detection"]
    variant_configs = _variant_configs(base_config)
    requested = list(variant_configs) if args.variant == "all" else [args.variant]
    if any(name not in variant_configs for name in requested):
        raise ValueError(f"Unknown variant; choose from {sorted(variant_configs)}")

    inputs: Dict[str, Tuple[List[TemporalBallPoint], List[Optional[BBox]], List[Optional[BBox]], List[Dict[str, Any]]]] = {}
    points_by_video: Dict[str, List[TemporalBallPoint]] = {}
    for video_id in gt_by_video:
        inputs[video_id] = _load_video_inputs(args.input_root, video_id)
        points_by_video[video_id] = inputs[video_id][0]

    ablations: List[Dict[str, Any]] = []
    all_variant_analyses: Dict[str, Dict[str, EventDetectionAnalysis]] = {}
    for variant_name in requested:
        variant_started = time.perf_counter()
        analyses: Dict[str, EventDetectionAnalysis] = {}
        candidate_predictions: Dict[str, List[Dict[str, Any]]] = {}
        event_predictions: Dict[str, List[Dict[str, Any]]] = {}
        for video_id, metadata in videos.items():
            points, p1, p2, _ = inputs[video_id]
            detector = TennisEventDetector(config=variant_configs[variant_name])
            analysis = detector.analyze(
                points, p1, p2, fps=float(metadata["fps"]),
                frame_size=(int(metadata["width"]), int(metadata["height"])),
                camera_offsets_px=None,
            )
            analyses[video_id] = analysis
            candidate_predictions[video_id] = [_candidate_record(row, video_id) for row in analysis.candidates]
            event_predictions[video_id] = [_event_record(row, video_id) for row in analysis.events]
        candidate_metrics = _recall_summary(candidate_predictions, gt_by_video, videos, require_type=False)
        metrics = _aggregate_event_metrics(event_predictions, gt_by_video)
        ablations.append(_metric_row(
            variant_name, candidate_metrics, metrics,
            sum(len(rows) for rows in event_predictions.values()),
        ))
        elapsed = time.perf_counter() - variant_started
        ablations[-1]["offline_event_analysis_seconds"] = elapsed
        ablations[-1]["offline_event_analysis_fps"] = (
            sum(int(video["frame_count"]) for video in videos.values()) / elapsed
            if elapsed > 0 else None
        )
        all_variant_analyses[variant_name] = analyses

    if args.variant != "all":
        print(json.dumps(ablations[0], indent=2))
        return

    final_analyses = all_variant_analyses["H_FINAL_INTEGRATED"]
    final_events = {
        video_id: [_event_record(event, video_id) for event in analysis.events]
        for video_id, analysis in final_analyses.items()
    }
    stage_predictions = _stage_predictions(final_analyses)
    stage_rows = []
    for stage_name, predictions in stage_predictions.items():
        stage_rows.append({
            "stage": stage_name,
            **_recall_summary(predictions, gt_by_video, videos, require_type=False),
            "type_matching_required": False,
        })
    forensic_rows, fn_taxonomy, fp_taxonomy = _forensic_rows(
        final_analyses, gt_by_video, videos, points_by_video
    )
    final_metrics = _aggregate_event_metrics(final_events, gt_by_video)
    state_distribution = _state_distributions(final_analyses, gt_by_video, videos, points_by_video)

    grouped = []
    video_ids = list(gt_by_video)
    for validation_video in video_ids:
        development = [video_id for video_id in video_ids if video_id != validation_video]
        validation_metrics = _aggregate_event_metrics(
            {validation_video: final_events[validation_video]},
            {validation_video: gt_by_video[validation_video]},
        )["event_detection"]
        grouped.append({
            "development_groups": development,
            "validation_group": validation_video,
            "source_match_grouping": videos[validation_video]["match_event"],
            "resolution": videos[validation_video]["resolution"],
            "threshold_refit_on_development_groups": False,
            "validation_metrics": validation_metrics,
        })

    pre_audit_path = os.path.join(
        VALIDATION_ROOT, "phase6_4_false_negative_audit_pre_recovery.json"
    )
    pre_audit = _load(pre_audit_path) if os.path.exists(pre_audit_path) else None
    pre_stages = _pre_recovery_stages(args.input_root, gt_by_video, videos)
    integrated_pre_metrics = _legacy_metrics(args.input_root, gt_by_video)
    integrated_pre_events = integrated_pre_metrics["event_detection"]
    ablations.insert(0, {
        "variant": "A_INTEGRATED_PRE_RECOVERY",
        "candidate_recall": pre_stages[
            "candidate_output_already_thresholded_and_nms"
        ]["recall"],
        "candidate_timing_mae_frames": pre_stages[
            "candidate_output_already_thresholded_and_nms"
        ]["timing_mae_frames"],
        "candidate_timing_mae_ms": pre_stages[
            "candidate_output_already_thresholded_and_nms"
        ]["timing_mae_ms"],
        "serve": integrated_pre_metrics["event_detection_per_class"]["SERVE_CONTACT"],
        "player_hit": integrated_pre_metrics["event_detection_per_class"]["PLAYER_HIT"],
        "bounce": integrated_pre_metrics["event_detection_per_class"]["BOUNCE"],
        "overall": integrated_pre_events,
        "event_count": integrated_pre_events["true_positives"] + integrated_pre_events["false_positives"],
        "offline_event_analysis_seconds": None,
        "offline_event_analysis_fps": None,
    })
    common = {
        "schema_version": "1.0",
        "phase": "6.4",
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
        "qualification_evidence": False,
        "matching_semantics": "GLOBAL_NEAREST_ONE_TO_ONE; GT interval expanded by 0.2 seconds",
        "fps_semantics": "Per-video native timestamps; all available diagnostic clips happen to be 30 FPS",
        "camera_motion_input": "UNAVAILABLE; G records the enabled-but-unavailable no-op honestly",
        "input_provenance": {
            "ground_truth_events": _provenance(GT_PATH),
            "video_manifest": _provenance(VIDEOS_PATH),
            "active_event_config": _provenance(CONFIG_PATH),
            "event_detector_implementation": _provenance(
                os.path.join(REPO_ROOT, "src", "events", "event_detector.py")
            ),
            "recovery_evaluator": _provenance(os.path.abspath(__file__)),
            "preserved_video_inputs": {
                video_id: {
                    "trajectories": _provenance(
                        os.path.join(args.input_root, video_id, "trajectories.json")
                    ),
                    "detections": _provenance(
                        os.path.join(args.input_root, video_id, "detections.json")
                    ),
                }
                for video_id in gt_by_video
            },
        },
    }
    stage_artifact = {
        **common,
        "pre_recovery_observable_stage_limit": "Old artifacts combine physics/player/type stages and retain stale refined timestamps.",
        "pre_recovery_observable_stage_metrics": pre_stages,
        "final_stage_metrics": stage_rows,
        "ball_state_distribution_around_outcomes": state_distribution,
        "normalized_player_reach_distributions": _reach_distributions(final_analyses, videos),
        "resolution_metrics": _resolution_metrics(final_events, gt_by_video),
        "grouped_leave_one_video_out": grouped,
    }
    forensic_artifact = {
        **common,
        "record_count": len(forensic_rows),
        "required_unavailable_features_are_null": True,
        "records": forensic_rows,
    }
    false_negative_artifact = {
        **common,
        "pre_recovery_audit": pre_audit,
        "post_recovery_false_negative_count": sum(fn_taxonomy.values()),
        "post_recovery_taxonomy": fn_taxonomy,
        "post_recovery_false_positive_taxonomy": fp_taxonomy,
        "post_recovery_records": [row for row in forensic_rows if not row["final_event_emitted"]],
    }
    recovery = {
        **common,
        "reported_corrected_baseline_from_previous_run": {
            "true_positives": 11, "false_positives": 113, "false_negatives": 29,
            "precision": 0.08870967741935484, "recall": 0.275,
            "f1": 0.13414634146341464,
            "warning": "Published artifact predates the current interval-aware evaluator fields; retained as the requested historical before-state.",
        },
        "legacy_baseline_fixed_evaluator_rescore": _legacy_metrics(LEGACY_BASELINE_ROOT, gt_by_video),
        "integrated_pre_recovery_fixed_evaluator": integrated_pre_metrics,
        "ablations": ablations,
        "final_metrics": final_metrics,
        "production_conditional_shot_metrics": {
            "source": os.path.relpath(PRESERVED_PRODUCTION_REPORT, REPO_ROOT).replace("\\", "/"),
            "scope": "PRESERVED_PRE_RECOVERY_PRODUCTION_PATH; NOT RECOMPUTED BY OFFLINE EVENT ABLATION",
            "metrics": _load(PRESERVED_PRODUCTION_REPORT)["aggregate"]["conditional_shot_classification"],
        },
        "oracle_shot_diagnostic": {
            "source": "artifacts/validation/phase6_4_oracle_shot_diagnostic.json",
            "scope": "GT_ASSISTED_DIAGNOSTIC_ONLY; NOT_PRODUCTION; NOT_END_TO_END",
        },
        "analytics_contract_regression": {
            "contract_version": "1.0",
            "existing_event_schema_changed": False,
            "event_verification_trace_is_additive_diagnostic_output": True,
            "scoring_remains_downstream_and_no_longer_deletes_physical_match_events": True,
        },
        "performance_context": {
            "full_pipeline_inference_rerun": False,
            "reason": "Preserved trajectories are sufficient for the single event-verifier blocker; GPU proposal inference was not repeated.",
            "previous_pipeline_fps": {
                "video_08_720p": 26.662918324959257,
                "video_09_720p": 28.010673242269625,
                "video_10_1080p": 8.956220195256629,
            },
            "final_offline_event_analysis_fps": ablations[-1]["offline_event_analysis_fps"],
            "detector_runs_candidate_generation_once_per_pipeline_run": True,
        },
        "readiness_gate": {
            "targets": {
                "overall_precision": 0.70,
                "overall_recall": 0.75,
                "overall_f1": 0.70,
                "player_hit_recall": 0.80,
                "serve_not_completely_broken": True,
                "bounce_not_nearly_broken": True,
            },
            "observed": {
                "overall_precision": final_metrics["event_detection"]["precision"],
                "overall_recall": final_metrics["event_detection"]["recall"],
                "overall_f1": final_metrics["event_detection"]["f1"],
                "player_hit_recall": final_metrics["event_detection_per_class"]["PLAYER_HIT"]["recall"],
                "serve_recall": final_metrics["event_detection_per_class"]["SERVE_CONTACT"]["recall"],
                "bounce_recall": final_metrics["event_detection_per_class"]["BOUNCE"]["recall"],
            },
            "passed": False,
            "ready_for_production_evaluator_freeze": False,
            "primary_remaining_blocker": "Discontinuous and missing ball association, especially the 1080p video_10 rally, prevents source-independent physical-event verification.",
        },
        "final_events_by_video": final_events,
        "final_event_counts_by_video": {
            video_id: len(rows) for video_id, rows in final_events.items()
        },
        "candidate_recall": ablations[-1]["candidate_recall"],
        "false_negative_taxonomy": fn_taxonomy,
        "false_positive_taxonomy": fp_taxonomy,
        "stage_artifact": "artifacts/validation/phase6_4_event_stage_metrics.json",
        "forensic_artifact": "artifacts/validation/phase6_4_event_forensic_table.json",
    }
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_event_stage_metrics.json"), stage_artifact)
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_event_forensic_table.json"), forensic_artifact)
    _dump(
        os.path.join(VALIDATION_ROOT, "phase6_4_false_negative_audit.json"),
        false_negative_artifact,
    )
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_event_recovery_results.json"), recovery)
    print(json.dumps({
        "candidate_recall": recovery["candidate_recall"],
        "final_event_detection": final_metrics["event_detection"],
        "per_class": final_metrics["event_detection_per_class"],
        "false_negative_taxonomy": fn_taxonomy,
        "false_positive_taxonomy": fp_taxonomy,
    }, indent=2))


if __name__ == "__main__":
    main()

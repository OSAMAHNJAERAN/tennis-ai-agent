"""Canonical Phase 6.4 Event Evaluator & Stage Lineage Architecture.

Defines the authoritative 7-stage event processing hierarchy (Stages 0 to 6),
centralizes 1-to-1 bipartite matching with temporal tolerance,
and generates complete lineage traces for every ground-truth event.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.events.event_detector import (
    EventCandidate,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
    _Feature,
)
from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


class CanonicalStage(str, Enum):
    STAGE_0_RAW_PROPOSAL = "STAGE_0_RAW_PROPOSAL"
    STAGE_1_USABLE_OBSERVATION = "STAGE_1_USABLE_OBSERVATION"
    STAGE_2_PHYSICAL_CANDIDATE = "STAGE_2_PHYSICAL_CANDIDATE"
    STAGE_3_PHYSICAL_CONTACT = "STAGE_3_PHYSICAL_CONTACT"
    STAGE_4_SEMANTIC_TYPE = "STAGE_4_SEMANTIC_TYPE"
    STAGE_5_PLAYER_ATTRIBUTION = "STAGE_5_PLAYER_ATTRIBUTION"
    STAGE_6_AUTHORITATIVE_EVENT = "STAGE_6_AUTHORITATIVE_EVENT"


class FirstFailureStage(str, Enum):
    NO_RAW_PROPOSAL = "NO_RAW_PROPOSAL"
    TRACK_NOT_USABLE = "TRACK_NOT_USABLE"
    NO_EVENT_CANDIDATE = "NO_EVENT_CANDIDATE"
    PHYSICAL_CONTACT_REJECTED = "PHYSICAL_CONTACT_REJECTED"
    CONTACT_FAMILY_WRONG = "CONTACT_FAMILY_WRONG"
    SEMANTIC_TYPE_WRONG = "SEMANTIC_TYPE_WRONG"
    PLAYER_ATTRIBUTION_WRONG = "PLAYER_ATTRIBUTION_WRONG"
    TEMPORAL_MATCH_FAILURE = "TEMPORAL_MATCH_FAILURE"
    FINAL_SUPPRESSION = "FINAL_SUPPRESSION"
    FULLY_CORRECT = "FULLY_CORRECT"


@dataclass
class EventLineageRecord:
    video_id: str
    gt_event_id: int
    gt_event_type: str
    gt_player_id: Optional[int]
    gt_frame: int
    gt_frame_min: int
    gt_frame_max: int
    gt_timestamp_s: float
    raw_proposal_present: bool
    usable_ball_observation: bool
    candidate_generated: bool
    candidate_frame: Optional[int]
    candidate_time_error_ms: Optional[float]
    physical_contact_verified: bool
    contact_family: str  # PLAYER_CONTACT, COURT_CONTACT, UNKNOWN_CONTACT, NONE
    semantic_type: Optional[str]
    predicted_player: Optional[int]
    authoritative_event_emitted: bool
    final_match_status: str  # MATCHED_EXACT, MATCHED_WRONG_TYPE, MATCHED_WRONG_PLAYER, UNMATCHED
    first_failure_stage: str
    first_failure_reason: str
    evaluation_scope_id: str = "UNSPECIFIED"
    inside_annotation_coverage: bool = True
    prediction_id: Optional[str] = None
    matched_gt_event_id: Optional[int] = None
    same_video_match: bool = False
    timing_error_s: Optional[float] = None


PHASE6_4_EVALUATOR_VERSION = "2.1"
PHYSICAL_EVENT_ANNOTATION_SCOPE = "PHYSICAL_EVENTS_EXHAUSTIVE"
MATCHING_ALGORITHM = (
    "PER_VIDEO_NATIVE_TIMESTAMP_GT_INTERVAL_MAX_CARDINALITY_MIN_COST_ONE_TO_ONE"
)
FPSValue = Union[float, Mapping[str, float]]


def canonical_event_type(record: Mapping[str, Any]) -> str:
    """Return the benchmark event family used by exact semantic evaluation."""
    value = str(record.get("event_type", "")).upper()
    if value in ("PLAYER_1_HIT", "PLAYER_2_HIT"):
        return "PLAYER_HIT"
    return value


def normalize_video_id(
    record: Mapping[str, Any], *, single_video_id: Optional[str] = None
) -> str:
    """Normalize legacy ``_video_id`` once and fail on ambiguous identity."""
    canonical = record.get("video_id")
    legacy = record.get("_video_id")
    if canonical is not None and legacy is not None and str(canonical) != str(legacy):
        raise ValueError(
            f"Conflicting media identity: video_id={canonical!r}, _video_id={legacy!r}"
        )
    value = canonical if canonical is not None else legacy
    if value is None:
        value = single_video_id
    if value is None or not str(value).strip():
        raise ValueError(
            "Missing video_id. Multi-video evaluation must provide explicit media identity; "
            "single-video callers must pass single_video_id."
        )
    return str(value)


def _record_frame(record: Mapping[str, Any]) -> int:
    for key in ("frame_index", "frame", "frame_best", "frame_hit"):
        if key in record:
            return int(record[key])
    raise KeyError(f"No frame field in evaluation record: {record}")


def _native_fps(fps: FPSValue, video_id: str) -> float:
    value = fps[video_id] if isinstance(fps, Mapping) else fps
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid native FPS for {video_id}: {value!r}")
    return value


def _prediction_timestamp(record: Mapping[str, Any], fps: float) -> float:
    timestamp = record.get("timestamp_s")
    if timestamp is not None:
        timestamp = float(timestamp)
        if math.isfinite(timestamp) and timestamp >= 0:
            return timestamp
    return _record_frame(record) / fps


def _gt_timestamp_interval(record: Mapping[str, Any], fps: float) -> Tuple[float, float]:
    if record.get("timestamp_min_s") is not None and record.get("timestamp_max_s") is not None:
        start = float(record["timestamp_min_s"])
        end = float(record["timestamp_max_s"])
    else:
        best = _record_frame(record)
        start = int(record.get("frame_min", best)) / fps
        end = int(record.get("frame_max", best)) / fps
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        raise ValueError(f"Invalid GT timestamp interval: {record}")
    return start, end


@dataclass
class _FlowEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int
    pair: Optional[Tuple[int, int]] = None
    temporal_error_s: Optional[float] = None


def _valid_pair_distance(
    prediction: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    fps: FPSValue,
    tolerance_s: float,
    require_event_type: bool,
    require_player: bool,
    single_video_id: Optional[str],
) -> Optional[float]:
    """Return the temporal edge cost, or ``None`` when the pair is invalid."""
    prediction_video_id = normalize_video_id(
        prediction, single_video_id=single_video_id
    )
    gt_video_id = normalize_video_id(gt, single_video_id=single_video_id)
    if prediction_video_id != gt_video_id:
        return None

    prediction_fps = _native_fps(fps, prediction_video_id)
    gt_fps = _native_fps(fps, gt_video_id)
    prediction_time = _prediction_timestamp(prediction, prediction_fps)
    gt_min, gt_max = _gt_timestamp_interval(gt, gt_fps)
    if prediction_time < gt_min:
        distance = gt_min - prediction_time
    elif prediction_time > gt_max:
        distance = prediction_time - gt_max
    else:
        distance = 0.0
    if distance > tolerance_s + 1e-12:
        return None

    prediction_type = canonical_event_type(prediction)
    gt_type = canonical_event_type(gt)
    if require_event_type and prediction_type != gt_type:
        return None
    if require_player and gt_type != "BOUNCE" and gt.get("player_id") is not None:
        if prediction.get("player_id") != gt.get("player_id"):
            return None
    return distance


def _add_flow_edge(
    graph: List[List[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: int,
    *,
    pair: Optional[Tuple[int, int]] = None,
    temporal_error_s: Optional[float] = None,
) -> None:
    forward = _FlowEdge(
        target,
        len(graph[target]),
        capacity,
        cost,
        pair=pair,
        temporal_error_s=temporal_error_s,
    )
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)


def _minimum_cost_maximum_matching(
    prediction_count: int,
    gt_count: int,
    valid_pairs: Sequence[Tuple[float, int, int]],
) -> List[Tuple[int, int, float]]:
    """Solve maximum-cardinality matching, then minimum total temporal error.

    Successive shortest augmenting paths run until no augmenting path remains,
    which fixes maximum cardinality first. Edge cost uses picosecond temporal
    units followed by a bounded stable index cost, so timing error is the second
    objective and deterministic input-index ordering is the third.
    """
    if not valid_pairs or prediction_count == 0 or gt_count == 0:
        return []

    source = 0
    prediction_offset = 1
    gt_offset = prediction_offset + prediction_count
    sink = gt_offset + gt_count
    node_count = sink + 1
    graph: List[List[_FlowEdge]] = [[] for _ in range(node_count)]

    for prediction_index in range(prediction_count):
        _add_flow_edge(graph, source, prediction_offset + prediction_index, 1, 0)
    for gt_index in range(gt_count):
        _add_flow_edge(graph, gt_offset + gt_index, sink, 1, 0)

    maximum_matches = min(prediction_count, gt_count)
    maximum_tie_sum = maximum_matches * prediction_count * (gt_count + 1)
    timing_scale = maximum_tie_sum + 1
    for distance, prediction_index, gt_index in sorted(
        valid_pairs, key=lambda item: (item[1], item[2], item[0])
    ):
        temporal_units = int(round(distance * 1_000_000_000_000))
        stable_tie_cost = prediction_index * (gt_count + 1) + gt_index
        composite_cost = temporal_units * timing_scale + stable_tie_cost
        _add_flow_edge(
            graph,
            prediction_offset + prediction_index,
            gt_offset + gt_index,
            1,
            composite_cost,
            pair=(prediction_index, gt_index),
            temporal_error_s=distance,
        )

    while True:
        distances: List[Optional[int]] = [None] * node_count
        predecessors: List[Optional[Tuple[int, int]]] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node in range(node_count):
                if distances[node] is None:
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distances[node] + edge.cost
                    if distances[edge.to] is None or candidate_distance < distances[edge.to]:
                        distances[edge.to] = candidate_distance
                        predecessors[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if predecessors[sink] is None:
            break
        node = sink
        while node != source:
            previous, edge_index = predecessors[node]  # type: ignore[misc]
            edge = graph[previous][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous

    matches: List[Tuple[int, int, float]] = []
    for prediction_index in range(prediction_count):
        node = prediction_offset + prediction_index
        for edge in graph[node]:
            if edge.pair is not None and edge.capacity == 0:
                assert edge.temporal_error_s is not None
                matches.append((*edge.pair, edge.temporal_error_s))
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches


def canonical_one_to_one_matches(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    tolerance_s: float = 0.200,
    fps: FPSValue = 30.0,
    require_event_type: bool = False,
    require_player: bool = False,
    *,
    single_video_id: Optional[str] = None,
) -> List[Tuple[int, int, float]]:
    """Canonical same-video, timestamp-based optimal one-to-one matcher.

    The GT event is an uncertainty interval and distance is measured from the
    prediction timestamp to that interval.  The returned distance is seconds.
    Explicit media identity is mandatory unless a single-video caller supplies
    ``single_video_id``.  Legacy ``_video_id`` is accepted only at this boundary.

    The lexicographic objective is: (1) maximum match cardinality, (2) minimum
    total temporal error, and (3) deterministic input-index tie-breaking.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and non-negative")
    explicit_video_ids = {
        str(record.get("video_id", record.get("_video_id")))
        for record in (*predictions, *ground_truth)
        if record.get("video_id", record.get("_video_id")) is not None
    }
    if single_video_id is None and not explicit_video_ids:
        # Backwards-compatible single-video mode.  As soon as any item carries
        # media identity, every item must carry it and normalization fails closed.
        single_video_id = "__single_video__"
    pairs: List[Tuple[float, int, int]] = []

    for p_idx, pred in enumerate(predictions):
        for g_idx, gt in enumerate(ground_truth):
            distance = _valid_pair_distance(
                pred,
                gt,
                fps=fps,
                tolerance_s=tolerance_s,
                require_event_type=require_event_type,
                require_player=require_player,
                single_video_id=single_video_id,
            )
            if distance is not None:
                pairs.append((distance, p_idx, g_idx))

    return _minimum_cost_maximum_matching(len(predictions), len(ground_truth), pairs)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validate_annotation_coverage(
    document: Mapping[str, Any], video_manifest: Mapping[str, Any]
) -> None:
    """Validate exhaustive raw-video coverage metadata against media metadata."""
    coverage_videos = document.get("videos")
    if not isinstance(coverage_videos, Mapping):
        raise ValueError("annotation coverage must contain a videos mapping")
    if str(document.get("annotation_scope")) != PHYSICAL_EVENT_ANNOTATION_SCOPE:
        raise ValueError("annotation coverage must declare PHYSICAL_EVENTS_EXHAUSTIVE")
    if str(document.get("coverage_basis")) != "RAW_VIDEO":
        raise ValueError("annotation coverage must be defined from RAW_VIDEO, not predictions")

    manifest_videos = video_manifest.get("videos", video_manifest)
    for video_id, metadata in manifest_videos.items():
        if video_id not in coverage_videos:
            raise ValueError(f"Missing annotation coverage for {video_id}")
        coverage = coverage_videos[video_id]
        if str(coverage.get("video_id")) != str(video_id):
            raise ValueError(f"Coverage video_id mismatch for {video_id}")
        fps = float(metadata["fps"])
        frame_count = int(metadata["frame_count"])
        if float(coverage.get("fps", 0)) != fps:
            raise ValueError(f"Coverage FPS disagrees with video manifest for {video_id}")
        if int(coverage.get("frame_count", -1)) != frame_count:
            raise ValueError(f"Coverage frame_count disagrees with video manifest for {video_id}")
        intervals = coverage.get("fully_reviewed_intervals")
        if not isinstance(intervals, list):
            raise ValueError(f"Coverage intervals must be a list for {video_id}")
        previous_end = -1
        for interval in intervals:
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
            if start < 0 or end < start or end >= frame_count:
                raise ValueError(f"Coverage interval outside media bounds for {video_id}: {interval}")
            if start <= previous_end:
                raise ValueError(f"Coverage intervals overlap or are unsorted for {video_id}")
            previous_end = end
            if not bool(interval.get("physical_event_annotation_complete")):
                raise ValueError(f"Non-exhaustive interval listed as reviewed for {video_id}")
            expected_start = start / fps
            expected_end = end / fps
            max_error = 0.5 / fps + 1e-9
            if abs(float(interval["start_timestamp_s"]) - expected_start) > max_error:
                raise ValueError(f"Coverage start timestamp disagrees with FPS for {video_id}")
            if abs(float(interval["end_timestamp_s"]) - expected_end) > max_error:
                raise ValueError(f"Coverage end timestamp disagrees with FPS for {video_id}")


def frame_inside_annotation_coverage(
    video_id: str, frame: int, coverage_document: Mapping[str, Any]
) -> bool:
    coverage = coverage_document["videos"][video_id]
    return any(
        int(interval["start_frame"]) <= frame <= int(interval["end_frame"])
        and bool(interval.get("physical_event_annotation_complete"))
        for interval in coverage["fully_reviewed_intervals"]
    )


def gt_coverage_boundary_status(
    video_id: str,
    gt: Mapping[str, Any],
    coverage_document: Mapping[str, Any],
    video_manifest: Mapping[str, Any],
    *,
    tolerance_s: float = 0.200,
) -> Dict[str, Any]:
    """Audit whether a GT event's complete match neighborhood is observable."""
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and non-negative")
    videos = video_manifest.get("videos", video_manifest)
    metadata = videos[video_id]
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    gt_start_s, gt_end_s = _gt_timestamp_interval(gt, fps)
    video_start_s = 0.0
    video_end_s = (frame_count - 1) / fps
    required_start_s = max(video_start_s, gt_start_s - tolerance_s)
    required_end_s = min(video_end_s, gt_end_s + tolerance_s)

    containing_interval: Optional[Mapping[str, Any]] = None
    for interval in coverage_document["videos"][video_id]["fully_reviewed_intervals"]:
        interval_start_s = int(interval["start_frame"]) / fps
        interval_end_s = int(interval["end_frame"]) / fps
        if interval_start_s <= gt_start_s + 1e-12 and interval_end_s >= gt_end_s - 1e-12:
            containing_interval = interval
            break

    if containing_interval is None:
        left_margin_s = None
        right_margin_s = None
        coverage_safe = False
        interval_id = None
    else:
        interval_start_s = int(containing_interval["start_frame"]) / fps
        interval_end_s = int(containing_interval["end_frame"]) / fps
        left_margin_s = gt_start_s - interval_start_s
        right_margin_s = interval_end_s - gt_end_s
        coverage_safe = (
            interval_start_s <= required_start_s + 1e-12
            and interval_end_s >= required_end_s - 1e-12
        )
        interval_id = containing_interval.get("interval_id")

    return {
        "video_id": video_id,
        "gt_event_id": gt.get("event_id"),
        "gt_interval_start_s": gt_start_s,
        "gt_interval_end_s": gt_end_s,
        "required_start_s": required_start_s,
        "required_end_s": required_end_s,
        "left_margin_s": left_margin_s,
        "right_margin_s": right_margin_s,
        "tolerance_s": tolerance_s,
        "physical_video_boundary_truncation": {
            "left": required_start_s == video_start_s and gt_start_s - tolerance_s < video_start_s,
            "right": required_end_s == video_end_s and gt_end_s + tolerance_s > video_end_s,
        },
        "coverage_interval_id": interval_id,
        "coverage_safe": coverage_safe,
        "evaluation_status": "EVALUABLE" if coverage_safe else "BOUNDARY_CENSORED",
    }


def evaluate_covered_events(
    predictions_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    video_manifest: Mapping[str, Any],
    coverage_document: Mapping[str, Any],
    *,
    tolerance_s: float = 0.200,
    require_event_type: bool = False,
    require_player: bool = False,
) -> Dict[str, Any]:
    """Evaluate each video independently, then aggregate counts.

    Predictions outside exhaustive coverage are labelled ``OUTSIDE_SCOPE`` and
    never contribute to precision, recall, or F1.
    """
    validate_annotation_coverage(coverage_document, video_manifest)
    videos = video_manifest.get("videos", video_manifest)
    per_video: Dict[str, Any] = {}
    prediction_statuses: List[Dict[str, Any]] = []
    match_records: List[Dict[str, Any]] = []

    for video_id, metadata in videos.items():
        fps = float(metadata["fps"])
        normalized_predictions = []
        for index, record in enumerate(predictions_by_video.get(video_id, [])):
            normalized_id = normalize_video_id(record, single_video_id=video_id)
            if normalized_id != video_id:
                raise ValueError(f"Prediction placed under wrong video bucket: {normalized_id}")
            normalized = {**record, "video_id": video_id}
            normalized_predictions.append(normalized)
        normalized_gt = []
        for record in ground_truth_by_video.get(video_id, []):
            normalized_id = normalize_video_id(record, single_video_id=video_id)
            if normalized_id != video_id:
                raise ValueError(f"GT placed under wrong video bucket: {normalized_id}")
            normalized_gt.append({**record, "video_id": video_id})

        covered_predictions = [
            record
            for record in normalized_predictions
            if frame_inside_annotation_coverage(video_id, _record_frame(record), coverage_document)
        ]
        inside_coverage_gt = [
            record
            for record in normalized_gt
            if frame_inside_annotation_coverage(video_id, _record_frame(record), coverage_document)
        ]
        gt_boundary_audit = [
            gt_coverage_boundary_status(
                video_id,
                record,
                coverage_document,
                video_manifest,
                tolerance_s=tolerance_s,
            )
            for record in inside_coverage_gt
        ]
        covered_gt = [
            record
            for record, audit in zip(inside_coverage_gt, gt_boundary_audit)
            if audit["coverage_safe"]
        ]
        boundary_censored_gt = [
            record
            for record, audit in zip(inside_coverage_gt, gt_boundary_audit)
            if not audit["coverage_safe"]
        ]
        boundary_censored_prediction_ids = {
            id(prediction)
            for prediction in covered_predictions
            if any(
                _valid_pair_distance(
                    prediction,
                    censored_gt,
                    fps=fps,
                    tolerance_s=tolerance_s,
                    require_event_type=require_event_type,
                    require_player=require_player,
                    single_video_id=video_id,
                )
                is not None
                for censored_gt in boundary_censored_gt
            )
        }
        evaluable_predictions = [
            record
            for record in covered_predictions
            if id(record) not in boundary_censored_prediction_ids
        ]
        matches = canonical_one_to_one_matches(
            evaluable_predictions,
            covered_gt,
            tolerance_s=tolerance_s,
            fps=fps,
            require_event_type=require_event_type,
            require_player=require_player,
            single_video_id=video_id,
        )
        matched_predictions = {pred_index for pred_index, _, _ in matches}
        matched_gt = {gt_index for _, gt_index, _ in matches}

        metric = precision_recall_f1(
            len(matches), len(evaluable_predictions) - len(matches), len(covered_gt) - len(matches)
        )
        metric.update(
            {
                "video_id": video_id,
                "native_fps": fps,
                "covered_prediction_count": len(evaluable_predictions),
                "inside_coverage_prediction_count": len(covered_predictions),
                "boundary_censored_prediction_count": len(boundary_censored_prediction_ids),
                "outside_scope_prediction_count": len(normalized_predictions) - len(covered_predictions),
                "covered_gt_count": len(covered_gt),
                "inside_coverage_gt_count": len(inside_coverage_gt),
                "boundary_censored_gt_count": len(boundary_censored_gt),
            }
        )
        per_video[video_id] = metric

        covered_index_by_identity = {
            id(record): index for index, record in enumerate(evaluable_predictions)
        }
        match_by_prediction = {pred_index: (gt_index, error) for pred_index, gt_index, error in matches}
        for source_index, record in enumerate(normalized_predictions):
            inside = frame_inside_annotation_coverage(
                video_id, _record_frame(record), coverage_document
            )
            boundary_censored = id(record) in boundary_censored_prediction_ids
            covered_index = (
                covered_index_by_identity.get(id(record))
                if inside and not boundary_censored
                else None
            )
            match = match_by_prediction.get(covered_index) if covered_index is not None else None
            gt_record = covered_gt[match[0]] if match else None
            status = (
                "OUTSIDE_SCOPE"
                if not inside
                else (
                    "BOUNDARY_CENSORED"
                    if boundary_censored
                    else ("TP" if match else "FP")
                )
            )
            prediction_statuses.append(
                {
                    "video_id": video_id,
                    "prediction_index": source_index,
                    "prediction_id": record.get("prediction_id", f"{video_id}:prediction:{source_index + 1}"),
                    "frame": _record_frame(record),
                    "timestamp_s": _prediction_timestamp(record, fps),
                    "inside_evaluation_coverage": inside,
                    "boundary_censored": boundary_censored,
                    "matched_gt_event_id": gt_record.get("event_id") if gt_record else None,
                    "evaluation_status": status,
                }
            )
        for pred_index, gt_index, error_s in matches:
            match_records.append(
                {
                    "video_id": video_id,
                    "prediction_index": pred_index,
                    "gt_index": gt_index,
                    "gt_event_id": covered_gt[gt_index].get("event_id"),
                    "same_video_match": True,
                    "timing_error_s": error_s,
                    "timing_error_ms": error_s * 1000.0,
                }
            )

    aggregate = precision_recall_f1(
        sum(metric["true_positives"] for metric in per_video.values()),
        sum(metric["false_positives"] for metric in per_video.values()),
        sum(metric["false_negatives"] for metric in per_video.values()),
    )
    aggregate["outside_scope_prediction_count"] = sum(
        metric["outside_scope_prediction_count"] for metric in per_video.values()
    )
    aggregate["covered_prediction_count"] = sum(
        metric["covered_prediction_count"] for metric in per_video.values()
    )
    aggregate["inside_coverage_prediction_count"] = sum(
        metric["inside_coverage_prediction_count"] for metric in per_video.values()
    )
    aggregate["covered_gt_count"] = sum(metric["covered_gt_count"] for metric in per_video.values())
    aggregate["boundary_censored_gt_count"] = sum(
        metric["boundary_censored_gt_count"] for metric in per_video.values()
    )
    aggregate["boundary_censored_prediction_count"] = sum(
        metric["boundary_censored_prediction_count"] for metric in per_video.values()
    )
    assert aggregate["true_positives"] == sum(m["true_positives"] for m in per_video.values())
    assert aggregate["false_positives"] == sum(
        m["false_positives"] for m in per_video.values()
    )
    assert aggregate["false_negatives"] == sum(m["false_negatives"] for m in per_video.values())
    assert aggregate["true_positives"] + aggregate["false_negatives"] == aggregate["covered_gt_count"]
    return {
        "evaluator_version": PHASE6_4_EVALUATOR_VERSION,
        "matching_algorithm": MATCHING_ALGORITHM,
        "evaluation_scope_id": coverage_document.get("evaluation_scope_id"),
        "per_video": per_video,
        "aggregate": aggregate,
        "prediction_statuses": prediction_statuses,
        "matches": match_records,
    }


def evaluate_event_lineage(
    video_id: str,
    raw_proposals: Sequence[Sequence[Dict[str, Any]]],
    trajectory: Sequence[TemporalBallPoint],
    player1_boxes: Sequence[Optional[BBox]],
    player2_boxes: Sequence[Optional[BBox]],
    gt_events: Sequence[Dict[str, Any]],
    detector: TennisEventDetector,
    fps: float = 30.0,
    frame_size: Tuple[int, int] = (1280, 720),
) -> Tuple[List[EventLineageRecord], Dict[str, Any]]:
    """Evaluate causal GT lineage from one detector run.

    Stage 2 is temporally associated once. Stages 3-6 then follow that exact
    candidate through its verification trace; later stages are never rematched
    independently and therefore cannot resurrect a failed GT event.
    """
    tol_frames = max(1, round(0.200 * fps))
    analysis = detector.analyze(
        trajectory, player1_boxes, player2_boxes, fps=fps, frame_size=frame_size
    )

    candidates = analysis.candidates
    cand_records = [
        {
            "video_id": video_id,
            "frame": c.frame_index,
            "timestamp_s": c.timestamp_s,
            "score": c.score,
        }
        for c in candidates
    ]

    # Map candidate matches
    cand_matches = canonical_one_to_one_matches(
        cand_records,
        gt_events,
        tolerance_s=0.200,
        fps=fps,
        require_event_type=False,
        single_video_id=video_id,
    )
    cand_by_gt = {g_idx: (p_idx, dist) for p_idx, g_idx, dist in cand_matches}

    traces_by_candidate = {
        int(trace["candidate_id"]): trace for trace in analysis.verification_traces
    }

    lineage_records: List[EventLineageRecord] = []

    for g_idx, gt in enumerate(gt_events):
        g_id = int(gt.get("event_id", g_idx + 1))
        g_type = str(gt.get("event_type", "UNKNOWN"))
        g_player = gt.get("player_id")
        g_frame = int(gt.get("frame_best", gt.get("frame", 0)))
        g_min = int(gt.get("frame_min", g_frame))
        g_max = int(gt.get("frame_max", g_frame))
        g_time = float(g_frame / fps)

        # Stage 0: Raw ball proposal presence
        raw_count_in_win = sum(
            len(raw_proposals[f])
            for f in range(max(0, g_min - tol_frames), min(len(raw_proposals), g_max + tol_frames + 1))
        )
        raw_present = raw_count_in_win > 0

        # Stage 1: Usable ball observation
        tracked_count_in_win = sum(
            1
            for f in range(max(0, g_min - tol_frames), min(len(trajectory), g_max + tol_frames + 1))
            if trajectory[f].state in (BallState.DETECTED, BallState.TRACKED, BallState.INTERPOLATED)
            and trajectory[f].x_px is not None
        )
        usable_obs = raw_present and tracked_count_in_win >= 3

        # Stage 2: Physical event candidate
        cand_info = cand_by_gt.get(g_idx)
        cand_gen = usable_obs and cand_info is not None
        cand_frame = candidates[cand_info[0]].frame_index if cand_gen else None
        cand_err_ms = (cand_info[1] * 1000.0) if cand_gen else None

        # Stages 3-6 follow the matched Stage-2 candidate identity directly.
        candidate_id = cand_info[0] + 1 if cand_gen else None
        trace = traces_by_candidate.get(candidate_id) if candidate_id is not None else None
        stage_pass = trace.get("stage_pass", {}) if trace else {}
        phys_verified = cand_gen and bool(stage_pass.get("physics_verification", False))
        sem_type = trace.get("candidate_event_type") if phys_verified and trace else None
        semantic_correct = bool(
            phys_verified
            and stage_pass.get("event_type_classification", False)
            and canonical_event_type({"event_type": sem_type}) == canonical_event_type(gt)
        )
        pred_player = trace.get("selected_player") if semantic_correct and trace else None
        player_correct = bool(
            semantic_correct
            and (
                canonical_event_type(gt) == "BOUNCE"
                or g_player is None
                or pred_player == g_player
            )
        )

        # Contact family
        if sem_type in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT"):
            contact_family = "PLAYER_CONTACT"
        elif sem_type == "BOUNCE":
            contact_family = "COURT_CONTACT"
        elif sem_type == "UNKNOWN_EVENT":
            contact_family = "UNKNOWN_CONTACT"
        else:
            contact_family = "NONE"

        auth_emitted = player_correct and bool(trace and trace.get("final_event_emitted", False))

        # Match status
        if auth_emitted:
            final_status = "MATCHED_EXACT"
        elif semantic_correct and not player_correct:
            final_status = "MATCHED_WRONG_PLAYER"
        elif phys_verified:
            final_status = "MATCHED_WRONG_TYPE"
        else:
            final_status = "UNMATCHED"

        # Determine FIRST failure stage
        if not raw_present:
            first_fail_stage = FirstFailureStage.NO_RAW_PROPOSAL.value
            first_fail_reason = "No raw detector-level ball proposals in temporal match window."
        elif not usable_obs:
            first_fail_stage = FirstFailureStage.TRACK_NOT_USABLE.value
            first_fail_reason = f"Tracking gap: only {tracked_count_in_win} usable points in temporal window."
        elif not cand_gen:
            first_fail_stage = FirstFailureStage.NO_EVENT_CANDIDATE.value
            first_fail_reason = "Ball tracked but kinematic score / speed below candidate threshold."
        elif not phys_verified:
            first_fail_stage = FirstFailureStage.PHYSICAL_CONTACT_REJECTED.value
            first_fail_reason = "Candidate generated but failed physical multi-cue verification or debounce."
        elif not semantic_correct:
            # Check if contact family was wrong
            gt_is_player = g_type in ("PLAYER_HIT", "SERVE_CONTACT")
            pred_is_player = sem_type in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")
            if gt_is_player != pred_is_player:
                first_fail_stage = FirstFailureStage.CONTACT_FAMILY_WRONG.value
                first_fail_reason = f"Contact family mismatch: GT is {g_type} but predicted as {sem_type}."
            else:
                first_fail_stage = FirstFailureStage.SEMANTIC_TYPE_WRONG.value
                first_fail_reason = f"Semantic subtype mismatch: GT is {g_type} but predicted as {sem_type}."
        elif not player_correct:
            first_fail_stage = FirstFailureStage.PLAYER_ATTRIBUTION_WRONG.value
            first_fail_reason = f"Player attribution error: GT player {g_player} vs predicted {pred_player}."
        elif not auth_emitted:
            first_fail_stage = FirstFailureStage.FINAL_SUPPRESSION.value
            first_fail_reason = str(
                (trace or {}).get("suppression_reason")
                or (trace or {}).get("rejection_reason")
                or "Candidate did not survive final export filtering."
            )
        else:
            first_fail_stage = FirstFailureStage.FULLY_CORRECT.value
            first_fail_reason = "Fully correct event detection, exact type, and player attribution."

        rec = EventLineageRecord(
            video_id=video_id,
            gt_event_id=g_id,
            gt_event_type=g_type,
            gt_player_id=g_player,
            gt_frame=g_frame,
            gt_frame_min=g_min,
            gt_frame_max=g_max,
            gt_timestamp_s=g_time,
            raw_proposal_present=raw_present,
            usable_ball_observation=usable_obs,
            candidate_generated=cand_gen,
            candidate_frame=cand_frame,
            candidate_time_error_ms=cand_err_ms,
            physical_contact_verified=phys_verified,
            contact_family=contact_family,
            semantic_type=sem_type,
            predicted_player=pred_player,
            authoritative_event_emitted=auth_emitted,
            final_match_status=final_status,
            first_failure_stage=first_fail_stage,
            first_failure_reason=first_fail_reason,
            evaluation_scope_id="CROSS_MATCH_FINAL_HOLDOUT_DIAGNOSTIC_V2",
            inside_annotation_coverage=True,
            prediction_id=(
                f"{video_id}:candidate:{candidate_id}" if candidate_id is not None else None
            ),
            matched_gt_event_id=g_id if cand_gen else None,
            same_video_match=cand_gen,
            timing_error_s=cand_info[1] if cand_gen else None,
        )
        lineage_records.append(rec)

    metrics = {
        "video_id": video_id,
        "gt_count": len(gt_events),
        "raw_proposal_count": sum(1 for r in lineage_records if r.raw_proposal_present),
        "usable_obs_count": sum(1 for r in lineage_records if r.usable_ball_observation),
        "candidate_count": sum(1 for r in lineage_records if r.candidate_generated),
        "physical_verified_count": sum(1 for r in lineage_records if r.physical_contact_verified),
        "authoritative_count": len(analysis.events),
        "exact_matched_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_EXACT"),
        "wrong_type_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_WRONG_TYPE"),
        "wrong_player_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_WRONG_PLAYER"),
        "unmatched_count": sum(1 for r in lineage_records if r.final_match_status == "UNMATCHED"),
    }
    monotonic_counts = [
        metrics["raw_proposal_count"],
        metrics["usable_obs_count"],
        metrics["candidate_count"],
        metrics["physical_verified_count"],
        sum(
            record.first_failure_stage
            not in {
                FirstFailureStage.NO_RAW_PROPOSAL.value,
                FirstFailureStage.TRACK_NOT_USABLE.value,
                FirstFailureStage.NO_EVENT_CANDIDATE.value,
                FirstFailureStage.PHYSICAL_CONTACT_REJECTED.value,
                FirstFailureStage.CONTACT_FAMILY_WRONG.value,
                FirstFailureStage.SEMANTIC_TYPE_WRONG.value,
            }
            for record in lineage_records
        ),
        sum(
            record.first_failure_stage
            in {FirstFailureStage.FINAL_SUPPRESSION.value, FirstFailureStage.FULLY_CORRECT.value}
            for record in lineage_records
        ),
        sum(record.first_failure_stage == FirstFailureStage.FULLY_CORRECT.value for record in lineage_records),
    ]
    assert all(
        earlier >= later for earlier, later in zip(monotonic_counts, monotonic_counts[1:])
    ), f"Non-monotonic causal lineage counts: {monotonic_counts}"
    return lineage_records, metrics

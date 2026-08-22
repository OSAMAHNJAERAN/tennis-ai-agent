"""GT-assisted oracle-event diagnostic for Phase 6.4 shot classification.

This evaluator is intentionally isolated from production inference.  It replaces
only the upstream hit metadata with the manually annotated diagnostic hit frame,
player, and physical event type, then invokes the existing shot classifier with
exact-frame features exported by an already completed diagnostic run.

It is NOT production, NOT end-to-end, and NOT qualification evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_types import (
    PlayerHandedness,
    ShotClassificationSource,
    ShotType,
)
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


ARTIFACT_TYPE = "ORACLE_EVENT_CONDITIONAL_SHOT_DIAGNOSTIC"
DIAGNOSTIC_LABELS = (
    "NOT PRODUCTION",
    "NOT END-TO-END",
    "GT-ASSISTED DIAGNOSTIC ONLY",
)
ALLOWED_GT_ASSISTED_FIELDS = ("frame_index", "player_id", "event_type")
EVALUATION_ONLY_GT_FIELDS = (
    "shot_type",
    "direction",
    "court_side",
    "annotation_confidence",
)
SHOT_CLASSES = ("FOREHAND", "BACKHAND", "SERVE")

DEFAULT_DIAGNOSTIC_OUTPUT_ROOT = Path(
    "outputs/phase6_4_qualification/cross_match_diagnostic_final"
)
OUTPUT_PATH = Path("artifacts/validation/phase6_4_oracle_shot_diagnostic.json")
VIDEOS_PATH = Path("data/benchmarks/cross_match_final_holdout/videos.json")
GT_EVENTS_PATH = Path("data/benchmarks/cross_match_final_holdout/ground_truth_events.json")
GT_SHOTS_PATH = Path("data/benchmarks/cross_match_final_holdout/ground_truth_shots.json")
PIPELINE_CONFIG_PATH = Path("configs/phase6_analytics/pipeline.yaml")
CLASSIFIER_PATH = Path("src/shot_analysis/shot_classifier.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _relative_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hashed_file(path: Path, repository_root: Path) -> Dict[str, Any]:
    return {
        "path": _relative_path(path, repository_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, Any]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_oracle_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute direct-row conditional metrics; no temporal detector matching occurs."""
    per_class: Dict[str, Dict[str, Any]] = {}
    for shot_class in SHOT_CLASSES:
        true_positives = sum(
            row["classifier_output"]["shot_type"] == shot_class
            and row["evaluation_only"]["shot_type"] == shot_class
            for row in records
        )
        false_positives = sum(
            row["classifier_output"]["shot_type"] == shot_class
            and row["evaluation_only"]["shot_type"] != shot_class
            for row in records
        )
        false_negatives = sum(
            row["classifier_output"]["shot_type"] != shot_class
            and row["evaluation_only"]["shot_type"] == shot_class
            for row in records
        )
        metric = _prf(true_positives, false_positives, false_negatives)
        metric["support"] = sum(
            row["evaluation_only"]["shot_type"] == shot_class for row in records
        )
        per_class[shot_class] = metric

    unknown_count = sum(
        row["classifier_output"]["shot_type"] == "UNKNOWN" for row in records
    )
    classified_count = len(records) - unknown_count
    correct_count = sum(
        row["classifier_output"]["shot_type"] == row["evaluation_only"]["shot_type"]
        for row in records
    )
    confusion: Dict[str, Dict[str, int]] = {}
    for actual in SHOT_CLASSES:
        confusion[actual] = {
            predicted: sum(
                row["evaluation_only"]["shot_type"] == actual
                and row["classifier_output"]["shot_type"] == predicted
                for row in records
            )
            for predicted in (*SHOT_CLASSES, "UNKNOWN")
        }

    return {
        "evaluation_semantics": "EXACT_GT_HIT_ROW_CONDITIONAL_CLASSIFICATION",
        "per_class": per_class,
        "macro_f1": sum(per_class[name]["f1"] for name in SHOT_CLASSES) / len(SHOT_CLASSES),
        "record_count": len(records),
        "correct_count": correct_count,
        "accuracy": correct_count / len(records) if records else None,
        "unknown_abstentions": unknown_count,
        "unknown_rate": unknown_count / len(records) if records else None,
        "coverage": classified_count / len(records) if records else None,
        "confusion_matrix": confusion,
    }


def join_oracle_annotations(
    shots_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    events_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Join GT shot labels to the one allowed GT physical-hit metadata row."""
    joined: List[Dict[str, Any]] = []
    for video_id, shots in shots_by_video.items():
        hit_events = [
            event
            for event in events_by_video.get(video_id, [])
            if event.get("event_type") in ("SERVE_CONTACT", "PLAYER_HIT")
        ]
        for shot in shots:
            matches = [
                event
                for event in hit_events
                if int(event["frame_best"]) == int(shot["frame_index"])
                and int(event["player_id"]) == int(shot["player_id"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one GT hit event for {video_id} shot {shot['shot_id']}; "
                    f"found {len(matches)}"
                )
            event = matches[0]
            expected_event_type = "SERVE_CONTACT" if shot["shot_type"] == "SERVE" else "PLAYER_HIT"
            if event["event_type"] != expected_event_type:
                raise ValueError(
                    f"GT semantic mismatch for {video_id} shot {shot['shot_id']}: "
                    f"{shot['shot_type']} vs {event['event_type']}"
                )
            joined.append(
                {
                    "video_id": video_id,
                    "shot": dict(shot),
                    "event": dict(event),
                }
            )
    return joined


def _unique_frame_map(records: Iterable[Mapping[str, Any]], label: str) -> Dict[int, Mapping[str, Any]]:
    result: Dict[int, Mapping[str, Any]] = {}
    for record in records:
        frame_index = int(record["frame_index"])
        if frame_index in result:
            raise ValueError(f"Duplicate frame {frame_index} in {label}")
        result[frame_index] = record
    return result


def _player_box(frame_record: Mapping[str, Any], player_id: int) -> Tuple[Optional[BBox], Dict[str, Any]]:
    player = frame_record.get(f"player_{player_id}")
    if player is None or player.get("bbox") is None:
        return None, {
            "available": False,
            "bbox_px": None,
            "confidence": None,
            "source": "detections.json exact frame",
        }
    coords = [float(value) for value in player["bbox"]]
    if len(coords) != 4:
        raise ValueError(f"Invalid player bbox: {coords}")
    # The export contains coordinates but no player-box confidence.  Preserve
    # that absence in both the adapter and the auditable artifact; the current
    # classifier consumes only the box geometry.
    box = BBox(*coords, confidence=None, class_id=0)  # type: ignore[arg-type]
    return box, {
        "available": True,
        "bbox_px": coords,
        "confidence": None,
        "source": "detections.json exact frame",
    }


def _ball_point(record: Mapping[str, Any]) -> Tuple[TemporalBallPoint, Dict[str, Any]]:
    state = BallState(record.get("state", "MISSING"))
    point = TemporalBallPoint(
        frame_index=int(record["frame_index"]),
        timestamp_seconds=float(record["timestamp_seconds"]),
        x_px=float(record["x_px"]) if record.get("x_px") is not None else None,
        y_px=float(record["y_px"]) if record.get("y_px") is not None else None,
        court_x_m=float(record["court_x_m"]) if record.get("court_x_m") is not None else None,
        court_y_m=float(record["court_y_m"]) if record.get("court_y_m") is not None else None,
        confidence=float(record["confidence"]) if record.get("confidence") is not None else None,
        state=state,
        source="EXPORTED_DIAGNOSTIC_TRAJECTORY",
        speed_kmh=float(record["speed_kmh"]) if record.get("speed_kmh") is not None else None,
    )
    provenance = {
        "available": point.x_px is not None and point.y_px is not None,
        "frame_index": point.frame_index,
        "timestamp_s": point.timestamp_seconds,
        "position_px": [point.x_px, point.y_px] if point.x_px is not None and point.y_px is not None else None,
        "court_position_m": (
            [point.court_x_m, point.court_y_m]
            if point.court_x_m is not None and point.court_y_m is not None
            else None
        ),
        "speed_kmh": point.speed_kmh,
        "confidence": point.confidence,
        "state": point.state.value,
        "source": "trajectories.json exact frame",
    }
    return point, provenance


def classify_oracle_row(
    joined: Mapping[str, Any],
    detection_frame: Mapping[str, Any],
    trajectory_frame: Mapping[str, Any],
    fps: float,
    classifier: Any,
) -> Dict[str, Any]:
    """Classify one exact GT hit while keeping its label outside classifier inputs."""
    shot = joined["shot"]
    event = joined["event"]
    frame_index = int(event["frame_best"])
    player_id = int(event["player_id"])
    if int(detection_frame["frame_index"]) != frame_index:
        raise ValueError("Detection feature frame does not equal oracle hit frame")
    if int(trajectory_frame["frame_index"]) != frame_index:
        raise ValueError("Trajectory feature frame does not equal oracle hit frame")

    player_box, player_provenance = _player_box(detection_frame, player_id)
    ball_point, ball_provenance = _ball_point(trajectory_frame)
    gt_event_type = str(event["event_type"])
    classifier_event_type = (
        gt_event_type if gt_event_type == "SERVE_CONTACT" else f"PLAYER_{player_id}_HIT"
    )

    # Only the permitted oracle metadata enter semantic classification.  The
    # player box and ball point are prediction artifacts from the exact frame,
    # not GT features.  GT shot type remains evaluation-only below.
    shot_type, confidence, source, components, reason = classifier.classify_shot(
        event_type=classifier_event_type,
        hit_frame=frame_index,
        player_id=player_id,
        player_box=player_box,
        ball_point=ball_point,
    )

    predicted_type = shot_type.value if isinstance(shot_type, ShotType) else str(shot_type)
    source_value = source.value if isinstance(source, ShotClassificationSource) else str(source)
    return {
        "video_id": joined["video_id"],
        "gt_shot_id": int(shot["shot_id"]),
        "gt_event_id": int(event["event_id"]),
        "oracle_input": {
            "frame_index": frame_index,
            "timestamp_s": frame_index / fps,
            "player_id": player_id,
            "event_type": gt_event_type,
            "classifier_event_type": classifier_event_type,
        },
        "predicted_feature_inputs": {
            "player_box": player_provenance,
            "ball_point": ball_provenance,
            "pose_feature_extraction": {
                "invoked": False,
                "available": False,
                "reason": "Required handedness/orientation metadata are absent in active production config.",
            },
            "raw_frames_supplied": False,
        },
        "classifier_output": {
            "shot_type": predicted_type,
            "confidence": float(confidence),
            "classification_source": source_value,
            "confidence_components": components,
            "reason": reason,
        },
        "evaluation_only": {
            "shot_type": str(shot["shot_type"]),
            "direction": shot.get("direction"),
            "court_side": shot.get("court_side"),
            "annotation_confidence": shot.get("annotation_confidence"),
            "passed_to_classifier": False,
        },
    }


def _classifier_from_config(config: Mapping[str, Any]) -> Tuple[TennisShotClassifier, Dict[str, Any]]:
    shot_config = config.get("shot_classification", {})
    pose_config = config.get("pose_estimation", {})
    configured_handedness = shot_config.get("players_handedness", {})
    handedness_map = {
        int(player_id): PlayerHandedness(value)
        for player_id, value in configured_handedness.items()
    }
    orientation_map = {
        int(player_id): float(sign)
        for player_id, sign in shot_config.get("court_orientation_sign", {}).items()
    }
    if handedness_map or orientation_map:
        raise RuntimeError(
            "Oracle runner requires review before enabling pose/geometry metadata; "
            "the current diagnostic is defined for the active empty metadata maps."
        )
    classifier = TennisShotClassifier(
        pose_extractor=None,
        min_pose_confidence=float(pose_config.get("min_pose_confidence", 0.35)),
        ambiguity_threshold=float(shot_config.get("ambiguity_threshold", 0.15)),
        handedness_map=handedness_map,
        court_orientation_map=orientation_map,
    )
    return classifier, {
        "handedness_map": {},
        "court_orientation_map": {},
        "pose_extractor_instantiated": False,
        "pose_extractor_behavior_equivalence": (
            "With both required metadata maps empty, production classify_shot gates pose "
            "before invocation; omitting model initialization does not change predictions."
        ),
    }


def _feature_availability(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    state_counts = {state.value: 0 for state in BallState}
    for row in records:
        state = row["predicted_feature_inputs"]["ball_point"]["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "player_box_available": sum(
            row["predicted_feature_inputs"]["player_box"]["available"] for row in records
        ),
        "ball_position_available": sum(
            row["predicted_feature_inputs"]["ball_point"]["available"] for row in records
        ),
        "record_count": len(records),
        "ball_state_counts": state_counts,
    }


def build_oracle_artifact(
    repository_root: Path = REPOSITORY_ROOT,
    diagnostic_output_root: Path = DEFAULT_DIAGNOSTIC_OUTPUT_ROOT,
    classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    repository_root = repository_root.resolve()
    output_root = (
        diagnostic_output_root
        if diagnostic_output_root.is_absolute()
        else repository_root / diagnostic_output_root
    ).resolve()
    videos_path = repository_root / VIDEOS_PATH
    gt_events_path = repository_root / GT_EVENTS_PATH
    gt_shots_path = repository_root / GT_SHOTS_PATH
    config_path = repository_root / PIPELINE_CONFIG_PATH

    videos_document = load_json(videos_path)
    if videos_document.get("scientific_split") != "CROSS_MATCH_DIAGNOSTIC":
        raise ValueError("Oracle evaluator may only consume the declared diagnostic split")
    if videos_document.get("qualification_evidence") is not False:
        raise ValueError("Diagnostic manifest must explicitly reject qualification use")
    events_document = load_json(gt_events_path)
    shots_document = load_json(gt_shots_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    if classifier is None:
        classifier, classifier_runtime = _classifier_from_config(config)
    else:
        classifier_runtime = {
            "handedness_map": {},
            "court_orientation_map": {},
            "pose_extractor_instantiated": False,
            "pose_extractor_behavior_equivalence": "Injected classifier used by diagnostic test harness.",
        }

    joined = join_oracle_annotations(shots_document["shots"], events_document["events"])
    records: List[Dict[str, Any]] = []
    per_video_inputs: Dict[str, Any] = {}
    for video_id, metadata in videos_document["videos"].items():
        video_output = output_root / video_id
        detections_path = video_output / "detections.json"
        trajectories_path = video_output / "trajectories.json"
        detections_document = load_json(detections_path)
        trajectories_document = load_json(trajectories_path)
        detection_frames = _unique_frame_map(detections_document["frames"], f"{video_id} detections")
        trajectory_frames = _unique_frame_map(
            trajectories_document["ball_trajectory"], f"{video_id} trajectory"
        )
        fps = float(metadata["fps"])
        if fps <= 0:
            raise ValueError(f"Invalid FPS for {video_id}")

        media_path = repository_root / metadata["path"]
        actual_media_hash = sha256_file(media_path)
        if actual_media_hash != metadata["sha256"]:
            raise ValueError(f"Media SHA256 mismatch for {video_id}")
        per_video_inputs[video_id] = {
            "media": {
                "path": metadata["path"],
                "sha256": actual_media_hash,
                "manifest_hash_verified": True,
                "fps": fps,
                "resolution": metadata["resolution"],
            },
            "detections": _hashed_file(detections_path, repository_root),
            "trajectories": _hashed_file(trajectories_path, repository_root),
        }

        for annotation in joined:
            if annotation["video_id"] != video_id:
                continue
            frame_index = int(annotation["event"]["frame_best"])
            if frame_index not in detection_frames or frame_index not in trajectory_frames:
                raise ValueError(f"Missing exact-frame features for {video_id} frame {frame_index}")
            records.append(
                classify_oracle_row(
                    annotation,
                    detection_frames[frame_index],
                    trajectory_frames[frame_index],
                    fps,
                    classifier,
                )
            )

    expected_count = sum(len(rows) for rows in shots_document["shots"].values())
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} oracle rows, produced {len(records)}")

    per_video: Dict[str, Any] = {}
    for video_id in videos_document["videos"]:
        video_records = [record for record in records if record["video_id"] == video_id]
        per_video[video_id] = {
            "metrics": compute_oracle_metrics(video_records),
            "feature_availability": _feature_availability(video_records),
        }

    return {
        "schema_version": "1.0",
        "artifact_type": ARTIFACT_TYPE,
        "diagnostic_labels": list(DIAGNOSTIC_LABELS),
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC",
        "qualification_evidence": False,
        "production_inference": False,
        "end_to_end_evidence": False,
        "purpose": (
            "Measure current stroke classification when upstream physical-hit frame, "
            "player, and event type are supplied from diagnostic GT."
        ),
        "oracle_boundary": {
            "gt_assisted_fields": list(ALLOWED_GT_ASSISTED_FIELDS),
            "evaluation_only_gt_fields": list(EVALUATION_ONLY_GT_FIELDS),
            "predicted_feature_policy": "EXACT_FRAME_ONLY_NO_NEAREST_FILL",
            "production_pipeline_modified": False,
            "production_conditional_metrics_included": False,
            "warning": (
                "Oracle metrics cannot be used as production, end-to-end, holdout, "
                "qualification, or generalization evidence."
            ),
        },
        "input_provenance": {
            "diagnostic_output_root": _relative_path(output_root, repository_root),
            "video_manifest": _hashed_file(videos_path, repository_root),
            "ground_truth_events": _hashed_file(gt_events_path, repository_root),
            "ground_truth_shots": _hashed_file(gt_shots_path, repository_root),
            "active_pipeline_config": _hashed_file(config_path, repository_root),
            "classifier_implementation": _hashed_file(repository_root / CLASSIFIER_PATH, repository_root),
            "evaluator_implementation": _hashed_file(Path(__file__).resolve(), repository_root),
            "per_video": per_video_inputs,
        },
        "classifier_runtime": classifier_runtime,
        "dataset_integrity": {
            "gt_shot_count": expected_count,
            "oracle_record_count": len(records),
            "one_to_one_gt_hit_join": True,
            "class_support": {
                shot_class: sum(
                    row["evaluation_only"]["shot_type"] == shot_class for row in records
                )
                for shot_class in SHOT_CLASSES
            },
        },
        "aggregate_metrics": compute_oracle_metrics(records),
        "aggregate_feature_availability": _feature_availability(records),
        "per_video": per_video,
        "records": records,
        "interpretation_constraint": (
            "SERVE is an event-type passthrough because the permitted GT event type "
            "already identifies SERVE_CONTACT; it is not an independent serve-mechanics result."
        ),
    }


def write_oracle_artifact(
    artifact: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    output_path = (repository_root / OUTPUT_PATH).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-output-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_OUTPUT_ROOT,
        help="Existing Phase 6.4 diagnostic artifacts; no inference is run.",
    )
    args = parser.parse_args()
    artifact = build_oracle_artifact(
        repository_root=REPOSITORY_ROOT,
        diagnostic_output_root=args.diagnostic_output_root,
    )
    output_path = write_oracle_artifact(artifact, repository_root=REPOSITORY_ROOT)
    print(json.dumps({
        "artifact": _relative_path(output_path, REPOSITORY_ROOT),
        "diagnostic_labels": list(DIAGNOSTIC_LABELS),
        "aggregate_metrics": artifact["aggregate_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()

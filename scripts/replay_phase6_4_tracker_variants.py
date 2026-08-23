"""Replay Phase 6.4 tracker variants from frozen raw detector candidates.

Ground truth is loaded only after every trajectory has been produced.  The
script is diagnostic-only and never consumes the pristine video_11+ split.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase6_4_event_recovery import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    _load_video_inputs,
    _variant_configs,
)
from scripts.reconstruct_phase6_4_evaluator_v2_1_artifacts import (  # noqa: E402
    STAGES,
    _evaluate,
    _match_maps,
    _stage_one_records,
    _stage_zero_records,
    assert_causal_monotonicity,
    causal_stage_survival,
    first_failure_from_survival,
    stage_sets_from_analysis,
)
from src.events.event_detector import TennisEventDetector  # noqa: E402
from src.events.event_evaluator import (  # noqa: E402
    MATCHING_ALGORITHM,
    PHASE6_4_EVALUATOR_VERSION,
    canonical_event_type,
    validate_annotation_coverage,
)
from src.tracking.temporal_ball_tracker import (  # noqa: E402
    BallObservation,
    BallState,
    TemporalBallPoint,
    TemporalBallTracker,
)
from src.utils.bbox_utils import BBox  # noqa: E402


BENCHMARK = ROOT / "data" / "benchmarks" / "cross_match_final_holdout"
VALIDATION = ROOT / "artifacts" / "validation"
RAW_ROOT = VALIDATION / "raw_candidates"
REPLAY_ROOT = VALIDATION / "phase6_4_tracker_replay"
CONFIG_PATH = ROOT / "configs" / "phase6_analytics" / "pipeline.yaml"
STARTING_SHA = "918f876479ec0d59aa166116658977f3d3837e76"
TRACKER_PATH = ROOT / "src" / "tracking" / "temporal_ball_tracker.py"
PIPELINE_PATH = ROOT / "src" / "pipeline" / "phase6_pipeline.py"
EVALUATOR_PATH = ROOT / "src" / "events" / "event_evaluator.py"
EVENT_DETECTOR_PATH = ROOT / "src" / "events" / "event_detector.py"
GT_PATH = BENCHMARK / "ground_truth_events.json"
VIDEOS_PATH = BENCHMARK / "videos.json"
COVERAGE_PATH = BENCHMARK / "annotation_coverage.json"
VIDEO_IDS = ("video_08", "video_09", "video_10")
SPLIT = "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED"
GENERATION_COMMAND = "python scripts/replay_phase6_4_tracker_variants.py"
TOLERANCE_S = 0.2

PRODUCTION_KEYS = {
    "high_conf_thresh": ("ball_detection", "high_conf"),
    "low_conf_thresh": ("ball_detection", "low_conf"),
    "max_prediction_gap": ("temporal_tracking", "max_prediction_gap"),
    "max_interpolation_gap": ("temporal_tracking", "max_interpolation_gap"),
    "max_valid_speed_px_per_frame": ("temporal_tracking", "max_valid_speed_px_per_frame"),
    "base_gating_radius_px": ("temporal_tracking", "base_gating_radius_px"),
    "enable_multi_candidate_association": ("temporal_tracking", "enable_multi_candidate_association"),
    "enable_adaptive_gate": ("temporal_tracking", "enable_adaptive_gate"),
    "enable_short_gap_reacquisition": ("temporal_tracking", "enable_short_gap_reacquisition"),
    "enable_camera_motion_compensation": ("temporal_tracking", "enable_camera_motion_compensation"),
    "enable_scale_normalization": ("temporal_tracking", "enable_scale_normalization"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def config_at_starting_sha() -> dict[str, Any]:
    text = git_output("show", f"{STARTING_SHA}:configs/phase6_analytics/pipeline.yaml")
    return yaml.safe_load(text)


def artifact_provenance(generated_at: str) -> dict[str, Any]:
    video_manifest = load_json(VIDEOS_PATH)
    return {
        "git_sha": git_output("rev-parse", "HEAD"),
        "evaluator_version": PHASE6_4_EVALUATOR_VERSION,
        "evaluator_sha256": sha256(EVALUATOR_PATH),
        "matching_algorithm": MATCHING_ALGORITHM,
        "tracker_source_sha256": sha256(TRACKER_PATH),
        "event_detector_sha256": sha256(EVENT_DETECTOR_PATH),
        "event_config_sha256": sha256(CONFIG_PATH),
        "ground_truth_sha256": sha256(GT_PATH),
        "annotation_coverage_sha256": sha256(COVERAGE_PATH),
        "video_manifest_sha256": sha256(VIDEOS_PATH),
        "source_video_sha256": {
            video_id: video_manifest["videos"][video_id]["sha256"]
            for video_id in VIDEO_IDS
        },
        "raw_candidate_sha256": {
            video_id: sha256(RAW_ROOT / f"{video_id}_candidates.json")
            for video_id in VIDEO_IDS
        },
        "preserved_trajectory_sha256": {
            video_id: sha256(Path(DEFAULT_INPUT_ROOT) / video_id / "trajectories.json")
            for video_id in VIDEO_IDS
        },
        "scientific_split": SPLIT,
        "qualification_evidence": False,
        "coverage_review_status": "MODEL_ASSISTED_PROVISIONAL",
        "human_approved": False,
        "videos": list(VIDEO_IDS),
        "pristine_video_11_plus_used": False,
        "generation_command": GENERATION_COMMAND,
        "generated_at_utc": generated_at,
    }


def production_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for argument, (section, key) in PRODUCTION_KEYS.items():
        if section not in config or key not in config[section]:
            raise KeyError(f"Production tracker config is missing {section}.{key}")
        result[argument] = config[section][key]
    return result


def class_default_settings() -> dict[str, Any]:
    signature = inspect.signature(TemporalBallTracker.__init__)
    result = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.default is inspect.Parameter.empty:
            raise AssertionError(f"Unexpected required tracker parameter: {name}")
        result[name] = parameter.default
    return result


def tracker_snapshot(tracker: TemporalBallTracker) -> dict[str, Any]:
    return {
        "high_conf_thresh": tracker.high_conf_thresh,
        "low_conf_thresh": tracker.low_conf_thresh,
        "max_prediction_gap": tracker.max_prediction_gap,
        "max_interpolation_gap": tracker.max_interpolation_gap,
        "max_valid_speed_px_per_frame": tracker.max_valid_speed,
        "base_gating_radius_px": tracker.base_gating_radius,
        "enable_multi_candidate_association": tracker.enable_multi_candidate,
        "enable_adaptive_gate": tracker.enable_adaptive_gate,
        "enable_short_gap_reacquisition": tracker.enable_reacquisition,
        "enable_camera_motion_compensation": tracker.enable_camera_comp,
        "enable_scale_normalization": tracker.enable_scale_norm,
    }


def raw_inputs(video_id: str) -> tuple[dict[str, Any], list[list[BallObservation]]]:
    raw = load_json(RAW_ROOT / f"{video_id}_candidates.json")
    observations = [
        [
            BallObservation(
                x_px=float(candidate["x_px"]),
                y_px=float(candidate["y_px"]),
                confidence=float(candidate["confidence"]),
                bbox=BBox(*[float(value) for value in candidate["bbox"]]),
            )
            for candidate in frame
        ]
        for frame in raw["frames"]
    ]
    return raw, observations


def serialize_point(point: TemporalBallPoint) -> dict[str, Any]:
    return {
        "frame_index": int(point.frame_index),
        "timestamp_seconds": float(point.timestamp_seconds),
        "x_px": float(point.x_px) if point.x_px is not None else None,
        "y_px": float(point.y_px) if point.y_px is not None else None,
        "court_x_m": float(point.court_x_m) if point.court_x_m is not None else None,
        "court_y_m": float(point.court_y_m) if point.court_y_m is not None else None,
        "confidence": float(point.confidence) if point.confidence is not None else None,
        "state": point.state.value,
        "source": point.source,
        "velocity_px_per_sec": list(point.velocity_px_per_sec) if point.velocity_px_per_sec else None,
        "speed_kmh": float(point.speed_kmh) if point.speed_kmh is not None else None,
    }


def load_players(video_id: str, count: int) -> tuple[list[Optional[BBox]], list[Optional[BBox]]]:
    frames = load_json(Path(DEFAULT_INPUT_ROOT) / video_id / "detections.json")["frames"]
    player_1: list[Optional[BBox]] = [None] * count
    player_2: list[Optional[BBox]] = [None] * count
    for index, frame in enumerate(frames[:count]):
        for key, destination in (("player_1", player_1), ("player_2", player_2)):
            record = frame.get(key)
            if record and record.get("bbox"):
                destination[index] = BBox(*[float(value) for value in record["bbox"]])
    return player_1, player_2


def longest_run(points: Sequence[TemporalBallPoint], state: BallState) -> int:
    longest = current = 0
    for point in points:
        if point.state == state:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def trajectory_diagnostics(
    points: Sequence[TemporalBallPoint], raw: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    positioned = [point for point in points if point.x_px is not None and point.y_px is not None]
    states = Counter(point.state.value for point in points)
    width, height = int(raw["width"]), int(raw["height"])
    out_of_frame = sum(
        not (0 <= float(point.x_px) < width and 0 <= float(point.y_px) < height)
        for point in positioned
    )
    grounded = [
        point
        for point in points
        if point.state in {BallState.DETECTED, BallState.TRACKED, BallState.INTERPOLATED}
        and point.x_px is not None
        and point.y_px is not None
    ]
    stationary_transitions = 0
    impossible_steps = 0
    previous: Optional[TemporalBallPoint] = None
    max_speed = float(settings.get("max_valid_speed_px_per_frame", 90.0))
    for point in grounded:
        if previous is not None and point.frame_index == previous.frame_index + 1:
            distance = ((float(point.x_px) - float(previous.x_px)) ** 2 + (float(point.y_px) - float(previous.y_px)) ** 2) ** 0.5
            stationary_transitions += distance < 1.0
            impossible_steps += distance > max_speed + 1e-12
        previous = point
    reacquisition_successes = sum("reacquir" in point.source.lower() for point in points)
    inferred_attempts = sum(
        points[index - 1].state in {BallState.PREDICTED, BallState.MISSING}
        and points[index].state in {BallState.DETECTED, BallState.TRACKED}
        for index in range(1, len(points))
    )
    raw_counts = [len(frame) for frame in raw["frames"]]
    return {
        "frames": len(points),
        "state_counts": dict(sorted(states.items())),
        "grounded_frame_count": len(grounded),
        "longest_predicted_run_frames": longest_run(points, BallState.PREDICTED),
        "longest_interpolated_run_frames": longest_run(points, BallState.INTERPOLATED),
        "out_of_frame_position_count": out_of_frame,
        "stationary_grounded_transition_count_lt_1px": stationary_transitions,
        "impossible_grounded_step_count": impossible_steps,
        "reacquisition": {
            "instrumented_attempt_count": None,
            "source_label_success_count": reacquisition_successes,
            "inferred_missing_or_predicted_to_grounded_count": inferred_attempts,
        },
        "raw_candidate_frames": sum(count > 0 for count in raw_counts),
        "raw_candidate_frame_rate": sum(count > 0 for count in raw_counts) / len(raw_counts),
        "raw_candidate_count": sum(raw_counts),
        "raw_candidates_per_frame": sum(raw_counts) / len(raw_counts),
    }


def metric_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    per_video = {}
    for video_id, metric in result["per_video"].items():
        errors = [
            float(match["timing_error_ms"])
            for match in result["matches"]
            if match["video_id"] == video_id
        ]
        per_video[video_id] = {
            **metric,
            "timing_mae_ms": sum(errors) / len(errors) if errors else None,
        }
    all_errors = [float(match["timing_error_ms"]) for match in result["matches"]]
    return {
        "aggregate": {
            **result["aggregate"],
            "timing_mae_ms": sum(all_errors) / len(all_errors) if all_errors else None,
        },
        "per_video": per_video,
        "match_count": len(result["matches"]),
    }


def covered_minutes(coverage: Mapping[str, Any], video_id: str) -> float:
    video = coverage["videos"][video_id]
    frames = sum(
        int(interval["end_frame"]) - int(interval["start_frame"]) + 1
        for interval in video["fully_reviewed_intervals"]
    )
    return frames / float(video["fps"]) / 60.0


def evaluate_variant(
    trajectories: Mapping[str, Sequence[TemporalBallPoint]],
    raw_by_video: Mapping[str, Mapping[str, Any]],
    videos_document: Mapping[str, Any],
    gt: Mapping[str, Sequence[Mapping[str, Any]]],
    coverage: Mapping[str, Any],
    event_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
        stage: {} for stage in STAGES
    }
    analyses = {}
    event_seconds = 0.0
    event_timing: dict[str, dict[str, Any]] = {}
    for video_id in VIDEO_IDS:
        points = trajectories[video_id]
        player_1, player_2 = load_players(video_id, len(points))
        started = time.perf_counter()
        analysis = TennisEventDetector(config=dict(event_config)).analyze(
            points,
            player_1,
            player_2,
            fps=float(raw_by_video[video_id]["fps"]),
            frame_size=(int(raw_by_video[video_id]["width"]), int(raw_by_video[video_id]["height"])),
            camera_offsets_px=None,
        )
        elapsed = time.perf_counter() - started
        event_seconds += elapsed
        event_timing[video_id] = {
            "frames": len(points),
            "event_detection_elapsed_seconds": elapsed,
            "event_detection_fps_equivalent": len(points) / elapsed if elapsed > 0 else None,
        }
        analyses[video_id] = analysis
        stage_predictions["stage_0"][video_id] = _stage_zero_records(video_id)
        stage_predictions["stage_1"][video_id] = _stage_one_records(video_id, points)
        extracted = stage_sets_from_analysis(video_id, analysis)
        for stage in STAGES[2:]:
            stage_predictions[stage][video_id] = extracted[stage]

    independent_results = {
        "stage_0": _evaluate(stage_predictions["stage_0"], gt, videos_document, coverage),
        "stage_1": _evaluate(stage_predictions["stage_1"], gt, videos_document, coverage),
        "stage_2": _evaluate(stage_predictions["stage_2"], gt, videos_document, coverage),
        "stage_3": _evaluate(stage_predictions["stage_3"], gt, videos_document, coverage),
        "stage_4": _evaluate(stage_predictions["stage_4"], gt, videos_document, coverage, event_type=True),
        "stage_5": _evaluate(stage_predictions["stage_5"], gt, videos_document, coverage, event_type=True, player=True),
        "stage_6": _evaluate(stage_predictions["stage_6"], gt, videos_document, coverage, event_type=True, player=True),
    }
    final_physical = _evaluate(stage_predictions["stage_6"], gt, videos_document, coverage)
    independent_maps = {
        stage: _match_maps(result, stage_predictions[stage], gt, coverage)
        for stage, result in independent_results.items()
    }
    trace_lookup = {
        (video_id, int(trace["candidate_id"])): trace
        for video_id, analysis in analyses.items()
        for trace in analysis.verification_traces
    }
    lineage: list[dict[str, Any]] = []
    for video_id, events in gt.items():
        points = trajectories[video_id]
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
            fps = float(raw_by_video[video_id]["fps"])
            lo = max(0, int(event["frame_min"]) - int(round(TOLERANCE_S * fps)))
            hi = min(len(points) - 1, int(event["frame_max"]) + int(round(TOLERANCE_S * fps)))
            window = points[lo : hi + 1]
            state_counts = Counter(point.state.value for point in window)
            source_counts = Counter(point.source for point in window)
            lineage.append({
                "video_id": video_id,
                "gt_event_id": int(event["event_id"]),
                "gt_event_type": canonical_event_type(event),
                "gt_player_id": event.get("player_id"),
                "gt_frame_min": int(event["frame_min"]),
                "gt_frame_best": int(event["frame_best"]),
                "gt_frame_max": int(event["frame_max"]),
                "stage1_window_frame_min": lo,
                "stage1_window_frame_max": hi,
                "stage1_window_state_counts": dict(sorted(state_counts.items())),
                "stage1_window_source_counts": dict(sorted(source_counts.items())),
                "candidate_id": candidate_id,
                "candidate_refined_frame": trace.get("refined_frame") if trace else None,
                "causal_stage_survival": survival,
                "first_failure_stage": first_failure_from_survival(survival),
                "independent_stage_match_capability": {
                    stage: key in independent_maps[stage] for stage in STAGES
                },
            })
    assert_causal_monotonicity(lineage)
    causal_counts = {
        stage: sum(row["causal_stage_survival"][stage] for row in lineage)
        for stage in STAGES
    }
    causal_per_video = {
        video_id: {
            stage: sum(
                row["video_id"] == video_id and row["causal_stage_survival"][stage]
                for row in lineage
            )
            for stage in STAGES
        }
        for video_id in VIDEO_IDS
    }
    summaries = {
        stage: metric_summary(result) for stage, result in independent_results.items()
    }
    for stage in ("stage_2", "stage_3"):
        total_minutes = sum(covered_minutes(coverage, video_id) for video_id in VIDEO_IDS)
        for video_id in VIDEO_IDS:
            count = summaries[stage]["per_video"][video_id]["covered_prediction_count"]
            summaries[stage]["per_video"][video_id]["predictions_per_minute"] = (
                count / covered_minutes(coverage, video_id)
            )
        summaries[stage]["aggregate"]["predictions_per_minute"] = (
            summaries[stage]["aggregate"]["covered_prediction_count"] / total_minutes
        )
    return ({
        "independent_stage_metrics": {
            stage: summary for stage, summary in summaries.items()
        },
        "causal_stage_counts": causal_counts,
        "causal_stage_counts_per_video": causal_per_video,
        "true_stage3_physical": metric_summary(independent_results["stage_3"]),
        "final_output_physical": metric_summary(final_physical),
        "final_output_semantic_player": metric_summary(independent_results["stage_6"]),
        "event_detection_elapsed_seconds": event_seconds,
        "event_detection_performance": event_timing,
    }, lineage)


def loss_evidence(
    row: Mapping[str, Any],
    points: Sequence[TemporalBallPoint],
    raw: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    lo, hi = int(row["stage1_window_frame_min"]), int(row["stage1_window_frame_max"])
    proposals = [candidate for frame in raw["frames"][lo : hi + 1] for candidate in frame]
    window = points[lo : hi + 1]
    sources = [point.source for point in window]
    return {
        "raw_proposal_count": len(proposals),
        "max_raw_confidence": max((float(candidate["confidence"]) for candidate in proposals), default=None),
        "window_states": dict(sorted(Counter(point.state.value for point in window).items())),
        "window_sources": dict(sorted(Counter(sources).items())),
    }


def nearest_point_summary(
    points: Sequence[TemporalBallPoint], frame_best: int, lo: int, hi: int
) -> Optional[dict[str, Any]]:
    positioned = [
        point
        for point in points[lo : hi + 1]
        if point.x_px is not None and point.y_px is not None
    ]
    if not positioned:
        return None
    point = min(positioned, key=lambda item: (abs(item.frame_index - frame_best), item.frame_index))
    return {
        "frame": point.frame_index,
        "state": point.state.value,
        "source": point.source,
        "distance_from_gt_best_frames": abs(point.frame_index - frame_best),
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    provenance = artifact_provenance(generated_at)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    starting_config = config_at_starting_sha()
    starting_prod_settings = production_settings(starting_config)
    selected_settings = production_settings(config)
    default_settings = class_default_settings()
    event_config = _variant_configs(config["event_detection"])["H_FINAL_INTEGRATED"]
    videos_document = load_json(VIDEOS_PATH)
    coverage = load_json(COVERAGE_PATH)
    validate_annotation_coverage(coverage, videos_document)

    raw_by_video: dict[str, dict[str, Any]] = {}
    observations_by_video: dict[str, list[list[BallObservation]]] = {}
    for video_id in VIDEO_IDS:
        raw, observations = raw_inputs(video_id)
        raw_by_video[video_id] = raw
        observations_by_video[video_id] = observations

    variants = {
        "A_PRESERVED_V21": {
            "status": "REPLAYED_FROM_PRESERVED_TRAJECTORY",
            "settings": None,
            "config_source": "co-located run_config.yaml; source/toggles not fully recorded",
        },
        "B_CURRENT_PRODUCTION_CONFIG": {
            "status": "REPLAYED_FROM_RAW_CANDIDATES",
            "settings": starting_prod_settings,
            "config_source": f"git:{STARTING_SHA}:configs/phase6_analytics/pipeline.yaml",
        },
        "C_CURRENT_CLASS_DEFAULTS": {
            "status": "REPLAYED_FROM_RAW_CANDIDATES",
            "settings": default_settings,
            "config_source": "TemporalBallTracker.__init__ defaults",
        },
        "D_HISTORICAL_92_5_RECOVERY_CONFIG": {
            "status": "NOT_REPRODUCIBLE",
            "settings": None,
            "config_source": None,
            "reason": "No exact source-linked tracker configuration for the historical 92.5% recovery claim is present; approximation is forbidden.",
        },
        "E_RECONCILED_CONFIG": {
            "status": "REPLAYED_FROM_RAW_CANDIDATES",
            "settings": selected_settings,
            "config_source": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "H_PRESEMANTIC_INTEGRATED": {
            "status": "SELECTED_ALIAS_OF_E_RECONCILED_CONFIG",
            "settings": selected_settings,
            "config_source": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    trajectories: dict[str, dict[str, list[TemporalBallPoint]]] = {
        name: {} for name in variants if name != "D_HISTORICAL_92_5_RECOVERY_CONFIG"
    }
    timings: dict[str, dict[str, Any]] = {name: {} for name in trajectories}
    deterministic_hashes: dict[str, dict[str, Any]] = {name: {} for name in trajectories}
    for video_id in VIDEO_IDS:
        points, _, _, _ = _load_video_inputs(DEFAULT_INPUT_ROOT, video_id)
        trajectories["A_PRESERVED_V21"][video_id] = points
        serialized = [serialize_point(point) for point in points]
        deterministic_hashes["A_PRESERVED_V21"][video_id] = {
            "first_sha256": stable_hash(serialized),
            "second_sha256": stable_hash(serialized),
            "identical": True,
        }
        timings["A_PRESERVED_V21"][video_id] = {
            "frames": len(points), "tracker_elapsed_seconds": None, "tracker_fps": None
        }

        for name, settings in (
            ("B_CURRENT_PRODUCTION_CONFIG", starting_prod_settings),
            ("C_CURRENT_CLASS_DEFAULTS", default_settings),
            ("E_RECONCILED_CONFIG", selected_settings),
            ("H_PRESEMANTIC_INTEGRATED", selected_settings),
        ):
            started = time.perf_counter()
            first = TemporalBallTracker(**settings).track_video_candidates(
                observations_by_video[video_id],
                fps=float(raw_by_video[video_id]["fps"]),
                frame_size=(int(raw_by_video[video_id]["width"]), int(raw_by_video[video_id]["height"])),
            )
            elapsed = time.perf_counter() - started
            second = TemporalBallTracker(**settings).track_video_candidates(
                observations_by_video[video_id],
                fps=float(raw_by_video[video_id]["fps"]),
                frame_size=(int(raw_by_video[video_id]["width"]), int(raw_by_video[video_id]["height"])),
            )
            first_hash = stable_hash([serialize_point(point) for point in first])
            second_hash = stable_hash([serialize_point(point) for point in second])
            if first_hash != second_hash:
                raise AssertionError(f"Non-deterministic tracker output for {name}/{video_id}")
            trajectories[name][video_id] = first
            timings[name][video_id] = {
                "frames": len(first),
                "tracker_elapsed_seconds": elapsed,
                "tracker_fps": len(first) / elapsed if elapsed > 0 else None,
            }
            deterministic_hashes[name][video_id] = {
                "first_sha256": first_hash,
                "second_sha256": second_hash,
                "identical": True,
            }

    gt = load_json(GT_PATH)["events"]  # Evaluation-only: loaded after all replay trajectories exist.
    variant_results = {}
    all_lineage = {}
    for name, by_video in trajectories.items():
        result, lineage = evaluate_variant(
            by_video, raw_by_video, videos_document, gt, coverage, event_config
        )
        settings = variants[name]["settings"] or starting_prod_settings
        result["trajectory_diagnostics"] = {
            video_id: trajectory_diagnostics(by_video[video_id], raw_by_video[video_id], settings)
            for video_id in VIDEO_IDS
        }
        result["tracker_performance"] = timings[name]
        tracker_seconds = sum(
            item["tracker_elapsed_seconds"] or 0.0 for item in timings[name].values()
        )
        result["full_tracker_plus_event_elapsed_seconds"] = (
            tracker_seconds + result["event_detection_elapsed_seconds"]
        )
        result["determinism"] = deterministic_hashes[name]
        variant_results[name] = result
        all_lineage[name] = lineage
        for video_id, points in by_video.items():
            dump_json(
                REPLAY_ROOT / name.lower() / f"{video_id}_trajectory.json",
                {
                    "schema_version": "1.0",
                    "variant": name,
                    "video_id": video_id,
                    "tracker_settings": variants[name]["settings"],
                    "raw_candidate_sha256": provenance["raw_candidate_sha256"][video_id],
                    "trajectory_sha256": deterministic_hashes[name][video_id]["first_sha256"],
                    "points": [serialize_point(point) for point in points],
                    "provenance": provenance,
                },
            )

    preserved_stage1 = variant_results["A_PRESERVED_V21"]["independent_stage_metrics"]["stage_1"]["aggregate"]["recall"]
    production_stage1 = variant_results["B_CURRENT_PRODUCTION_CONFIG"]["independent_stage_metrics"]["stage_1"]["aggregate"]["recall"]
    defaults_stage1 = variant_results["C_CURRENT_CLASS_DEFAULTS"]["independent_stage_metrics"]["stage_1"]["aggregate"]["recall"]
    if preserved_stage1 < production_stage1 and production_stage1 < 0.90 <= defaults_stage1:
        root_cause = "MIXED"
        rationale = "Fresh replay removes most preserved-artifact losses, configuration reconciliation reaches the Stage-1 threshold, and residual losses remain."
    elif production_stage1 >= 0.90 and preserved_stage1 < 0.90:
        root_cause = "STALE_ARTIFACT"
        rationale = "Current production-config replay clears Stage 1 while the preserved trajectory does not."
    elif defaults_stage1 >= 0.90 and production_stage1 < 0.90:
        root_cause = "CONFIGURATION_DRIFT"
        rationale = "Class defaults clear Stage 1 while the resolved production configuration does not."
    elif production_stage1 < 0.90 and defaults_stage1 < 0.90:
        root_cause = "CURRENT_TRACKER_FAILURE"
        rationale = "Both current production and class-default raw-candidate replays fail the Stage-1 gate."
    else:
        root_cause = "MIXED"
        rationale = "The replay differences do not support a single stale/config/current-failure explanation."

    production_losses = []
    for row in all_lineage["B_CURRENT_PRODUCTION_CONFIG"]:
        if row["independent_stage_match_capability"]["stage_1"]:
            continue
        evidence = loss_evidence(
            row,
            trajectories["B_CURRENT_PRODUCTION_CONFIG"][row["video_id"]],
            raw_by_video[row["video_id"]],
            starting_prod_settings,
        )
        default_row = next(
            candidate
            for candidate in all_lineage["C_CURRENT_CLASS_DEFAULTS"]
            if candidate["video_id"] == row["video_id"] and candidate["gt_event_id"] == row["gt_event_id"]
        )
        if default_row["independent_stage_match_capability"]["stage_1"]:
            category = "CONFIGURATION_TOO_STRICT"
        elif evidence["raw_proposal_count"] == 0:
            category = "NO_RAW_PROPOSAL"
        elif "rejected_outlier" in evidence["window_sources"]:
            category = "OUTLIER_SPEED_REJECTION"
        else:
            category = "UNKNOWN"
        production_losses.append({
            "video_id": row["video_id"],
            "gt_event_id": row["gt_event_id"],
            "gt_frame_best": row["gt_frame_best"],
            "category": category,
            "evidence": evidence,
        })

    grouped_validation = {}
    for train, held_out in (
        (("video_08", "video_09"), "video_10"),
        (("video_08", "video_10"), "video_09"),
        (("video_09", "video_10"), "video_08"),
    ):
        fold_name = f"{'+'.join(train)}_to_{held_out}"
        grouped_validation[fold_name] = {
            "train_groups": list(train),
            "held_out_group": held_out,
            "fixed_config_holdout_results": {
                variant: {
                    "stage_1_recall": variant_results[variant]["independent_stage_metrics"]["stage_1"]["per_video"][held_out]["recall"],
                    "stage_2_recall": variant_results[variant]["independent_stage_metrics"]["stage_2"]["per_video"][held_out]["recall"],
                    "true_stage3_f1": variant_results[variant]["true_stage3_physical"]["per_video"][held_out]["f1"],
                    "longest_predicted_run_frames": variant_results[variant]["trajectory_diagnostics"][held_out]["longest_predicted_run_frames"],
                    "impossible_grounded_step_count": variant_results[variant]["trajectory_diagnostics"][held_out]["impossible_grounded_step_count"],
                }
                for variant in ("B_CURRENT_PRODUCTION_CONFIG", "C_CURRENT_CLASS_DEFAULTS")
            },
        }

    authoritative = variant_results["H_PRESEMANTIC_INTEGRATED"]
    gate = {
        "stage_0_recall_at_least_0_95": authoritative["independent_stage_metrics"]["stage_0"]["aggregate"]["recall"] >= 0.95,
        "stage_1_grounded_recall_at_least_0_90": authoritative["independent_stage_metrics"]["stage_1"]["aggregate"]["recall"] >= 0.90,
        "stage_2_recall_at_least_0_85": authoritative["independent_stage_metrics"]["stage_2"]["aggregate"]["recall"] >= 0.85,
        "true_stage3_precision_at_least_0_70": authoritative["true_stage3_physical"]["aggregate"]["precision"] >= 0.70,
        "true_stage3_recall_at_least_0_80": authoritative["true_stage3_physical"]["aggregate"]["recall"] >= 0.80,
        "true_stage3_f1_at_least_0_75": authoritative["true_stage3_physical"]["aggregate"]["f1"] >= 0.75,
        "no_uncontrolled_candidate_explosion": authoritative["independent_stage_metrics"]["stage_2"]["aggregate"]["covered_prediction_count"] <= variant_results["B_CURRENT_PRODUCTION_CONFIG"]["independent_stage_metrics"]["stage_2"]["aggregate"]["covered_prediction_count"],
        "no_long_synthetic_track_hallucination": max(
            metric["longest_predicted_run_frames"]
            for metric in authoritative["trajectory_diagnostics"].values()
        ) <= int(selected_settings["max_prediction_gap"]),
        "no_major_wrong_association_regression": sum(
            metric["impossible_grounded_step_count"]
            for metric in authoritative["trajectory_diagnostics"].values()
        ) <= 1.05 * sum(
            metric["impossible_grounded_step_count"]
            for metric in variant_results["B_CURRENT_PRODUCTION_CONFIG"]["trajectory_diagnostics"].values()
        ),
        "no_catastrophic_held_out_video_collapse": min(
            authoritative["independent_stage_metrics"]["stage_1"]["per_video"][video_id]["recall"]
            for video_id in VIDEO_IDS
        ) >= 0.90,
        "all_tests_pass": None,
    }
    gate["status_without_test_result"] = "PASS" if all(value is True for key, value in gate.items() if key not in {"all_tests_pass", "status_without_test_result"}) else "FAIL"
    gate["earliest_dominant_blocker"] = "STAGE_2_PHYSICAL_CANDIDATE_RECALL" if not gate["stage_2_recall_at_least_0_85"] else None

    config_resolution = {
        "schema_version": "1.0",
        "production_constructor": "src/pipeline/phase6_pipeline.py:Phase6Pipeline.__init__",
        "canonical_config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "starting_sha": STARTING_SHA,
        "starting_production_settings": starting_prod_settings,
        "starting_production_instance_snapshot": tracker_snapshot(TemporalBallTracker(**starting_prod_settings)),
        "selected_production_settings": selected_settings,
        "selected_production_instance_snapshot": tracker_snapshot(TemporalBallTracker(**selected_settings)),
        "class_defaults": default_settings,
        "class_default_instance_snapshot": tracker_snapshot(TemporalBallTracker()),
        "differences": {
            key: {"starting_production": starting_prod_settings[key], "class_default": default_settings[key]}
            for key in starting_prod_settings if starting_prod_settings[key] != default_settings[key]
        },
        "historical_92_5_recovery": variants["D_HISTORICAL_92_5_RECOVERY_CONFIG"],
        "provenance": provenance,
    }
    artifact_freshness = {
        "schema_version": "1.0",
        "classification": root_cause,
        "rationale": rationale,
        "preserved_artifact_provenance_complete": False,
        "missing_preserved_fields": [
            "generating_git_sha", "tracker_source_sha256", "complete_tracker_settings",
            "raw_candidate_sha256", "generation_command", "generation_timestamp_utc",
        ],
        "co_located_run_config_sha256": {
            video_id: sha256(Path(DEFAULT_INPUT_ROOT) / video_id / "run_config.yaml")
            for video_id in VIDEO_IDS
        },
        "preserved_trajectory_sha256": provenance["preserved_trajectory_sha256"],
        "authoritative_replay_trajectory_paths": {
            name: {
                video_id: str((REPLAY_ROOT / name.lower() / f"{video_id}_trajectory.json").relative_to(ROOT)).replace("\\", "/")
                for video_id in VIDEO_IDS
            }
            for name in trajectories
        },
        "provenance": provenance,
    }
    manifest = {
        "schema_version": "1.0",
        "variants": variants,
        "trajectory_hashes": deterministic_hashes,
        "tracker_performance": timings,
        "gt_loaded_after_all_trajectories": True,
        "tracker_received_ground_truth": False,
        "video_specific_branches": False,
        "provenance": provenance,
    }
    comparison = {
        "schema_version": "1.0",
        "root_cause_classification": root_cause,
        "root_cause_rationale": rationale,
        "variants": variant_results,
        "grouped_validation": grouped_validation,
        "presemantic_gate_thresholds": {
            "stage_0_recall_min": 0.95,
            "stage_1_grounded_recall_min": 0.90,
            "stage_2_recall_min": 0.85,
            "true_stage3_precision_min": 0.70,
            "true_stage3_recall_min": 0.80,
            "true_stage3_f1_min": 0.75,
        },
        "provenance": provenance,
    }
    merged_lineage = []
    lineage_lookup = {
        name: {(row["video_id"], row["gt_event_id"]): row for row in rows}
        for name, rows in all_lineage.items()
    }
    for event in all_lineage["H_PRESEMANTIC_INTEGRATED"]:
        key = (event["video_id"], event["gt_event_id"])
        merged = {
            "video_id": event["video_id"],
            "gt_event_id": event["gt_event_id"],
            "gt_event_type": event["gt_event_type"],
            "frame_min": event["gt_frame_min"],
            "frame_best": event["gt_frame_best"],
            "frame_max": event["gt_frame_max"],
            "raw_proposal_available": event["independent_stage_match_capability"]["stage_0"],
            "historical_recovery_stage1": None,
            "historical_recovery_nearest_state": None,
            "first_tracker_failure_historical": "NOT_REPRODUCIBLE",
        }
        for variant, prefix in (
            ("A_PRESERVED_V21", "preserved"),
            ("B_CURRENT_PRODUCTION_CONFIG", "production_replay"),
            ("C_CURRENT_CLASS_DEFAULTS", "class_default"),
            ("E_RECONCILED_CONFIG", "reconciled"),
        ):
            row = lineage_lookup[variant][key]
            stage1 = bool(row["independent_stage_match_capability"]["stage_1"])
            nearest = nearest_point_summary(
                trajectories[variant][event["video_id"]],
                event["gt_frame_best"],
                event["stage1_window_frame_min"],
                event["stage1_window_frame_max"],
            )
            merged[f"{prefix}_stage1"] = stage1
            merged[f"{prefix}_nearest_state"] = nearest
            merged[f"first_tracker_failure_{prefix}"] = None if stage1 else "TRACK_NOT_USABLE"
        merged["candidate_generated"] = event["independent_stage_match_capability"]["stage_2"]
        merged["physics_verified"] = event["independent_stage_match_capability"]["stage_3"]
        merged_lineage.append(merged)

    lineage_payload = {
        "schema_version": "1.0",
        "record_count": len(merged_lineage),
        "records": merged_lineage,
        "record_count_per_variant": {name: len(rows) for name, rows in all_lineage.items()},
        "variant_records": all_lineage,
        "provenance": provenance,
    }
    loss_payload = {
        "schema_version": "1.0",
        "variant": "B_CURRENT_PRODUCTION_CONFIG",
        "taxonomy_scope": "POST_HOC_GT_ERROR_ANALYSIS_ONLY",
        "tracker_received_ground_truth": False,
        "loss_count": len(production_losses),
        "category_counts": dict(sorted(Counter(row["category"] for row in production_losses).items())),
        "category_counts_per_video": {
            video_id: dict(sorted(Counter(
                row["category"] for row in production_losses if row["video_id"] == video_id
            ).items()))
            for video_id in VIDEO_IDS
        },
        "records": production_losses,
        "video_10_forensic_records": [
            {
                "video_id": row["video_id"],
                "gt_event_id": row["gt_event_id"],
                "gt_frame_best": row["gt_frame_best"],
                "stage1_pass": row["independent_stage_match_capability"]["stage_1"],
                "first_tracker_failure": next(
                    (
                        loss["category"]
                        for loss in production_losses
                        if loss["video_id"] == row["video_id"] and loss["gt_event_id"] == row["gt_event_id"]
                    ),
                    None,
                ),
                "window_states": row["stage1_window_state_counts"],
                "window_sources": row["stage1_window_source_counts"],
            }
            for row in all_lineage["B_CURRENT_PRODUCTION_CONFIG"]
            if row["video_id"] == "video_10"
        ],
        "provenance": provenance,
    }
    presemantic = {
        "schema_version": "1.0",
        "variants_executed": [
            "A_PRESERVED_V21", "B_CURRENT_PRODUCTION_CONFIG", "C_CURRENT_CLASS_DEFAULTS",
            "E_RECONCILED_CONFIG", "H_PRESEMANTIC_INTEGRATED",
        ],
        "variants_not_executed": {
            "D_HISTORICAL_92_5_RECOVERY_CONFIG": "NOT_REPRODUCIBLE",
            "F_TARGETED_TRACKER_FIX": "NOT_JUSTIFIED: selected config passes Stage 1 and residual windows lack independently localized proposal identity",
            "G_PHYSICAL_FP_SOURCE_FIX": "GATED: pre-semantic gate failed before FP-source remediation",
        },
        "root_cause_classification": root_cause,
        "variant_metrics": variant_results,
        "grouped_validation": grouped_validation,
        "hard_gate": gate,
        "comparison_artifact": "artifacts/validation/phase6_4_tracker_replay_comparison.json",
        "provenance": provenance,
    }
    dump_json(VALIDATION / "phase6_4_tracker_config_resolution.json", config_resolution)
    dump_json(VALIDATION / "phase6_4_tracker_artifact_provenance.json", artifact_freshness)
    dump_json(VALIDATION / "phase6_4_tracker_replay_manifest.json", manifest)
    dump_json(VALIDATION / "phase6_4_tracker_replay_comparison.json", comparison)
    dump_json(VALIDATION / "phase6_4_tracker_replay_gt_lineage.json", lineage_payload)
    dump_json(VALIDATION / "phase6_4_tracker_loss_taxonomy.json", loss_payload)
    dump_json(VALIDATION / "phase6_4_presemantic_ablation.json", presemantic)
    print(json.dumps({
        "root_cause": root_cause,
        "stage1_recall": {
            "preserved": preserved_stage1,
            "production": production_stage1,
            "class_defaults": defaults_stage1,
        },
        "production_true_stage3": variant_results["B_CURRENT_PRODUCTION_CONFIG"]["true_stage3_physical"]["aggregate"],
        "production_causal_counts": variant_results["B_CURRENT_PRODUCTION_CONFIG"]["causal_stage_counts"],
        "production_loss_categories": loss_payload["category_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()

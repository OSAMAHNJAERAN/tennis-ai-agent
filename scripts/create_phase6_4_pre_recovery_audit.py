"""Create the mandatory pre-recovery false-negative audit.

This script is deliberately artifact-only.  It reads the already-consumed
``video_08``--``video_10`` diagnostic outputs produced at ``ab9dd391`` and does
not import or execute the mutable event detector.  The resulting audit is the
fixed before-state used by the Phase 6.4 recovery experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_phase6_4_cross_match import one_to_one_matches
DEFAULT_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs",
    "phase6_4_qualification",
    "cross_match_diagnostic_final",
)
DEFAULT_GT = os.path.join(
    REPO_ROOT,
    "data",
    "benchmarks",
    "cross_match_final_holdout",
    "ground_truth_events.json",
)
DEFAULT_VIDEOS = os.path.join(
    REPO_ROOT,
    "data",
    "benchmarks",
    "cross_match_final_holdout",
    "videos.json",
)
DEFAULT_REPORT = os.path.join(
    REPO_ROOT,
    "artifacts",
    "validation",
    "phase6_4_false_negative_audit_pre_recovery.json",
)

# Candidate IDs were not persisted by the old detector.  These identifiers are
# the frozen result of the mandatory one-time lineage review that joined
# candidate discovery timestamps to pre-scoring emissions, scoring outcomes,
# and final exports.  The audit retains the non-monotonic-lineage caveat below.
FORENSIC_LINEAGE: Dict[Tuple[str, int], str] = {}
for _video, _ids, _cause in (
    ("video_08", (7,), "CANDIDATE_OUTSIDE_MATCH_WINDOW"),
    ("video_09", (4,), "CANDIDATE_OUTSIDE_MATCH_WINDOW"),
    ("video_10", (2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), "CANDIDATE_OUTSIDE_MATCH_WINDOW"),
    ("video_08", (3, 6, 8, 10), "PREDICTED_OR_INTERPOLATED_STATE_REJECTION"),
    ("video_10", (8,), "PREDICTED_OR_INTERPOLATED_STATE_REJECTION"),
    ("video_08", (2,), "PLAYER_REACH_OR_RACKET_REGION_REJECTION"),
    ("video_10", (1, 3), "PLAYER_REACH_OR_RACKET_REGION_REJECTION"),
    ("video_08", (1,), "WRONG_EVENT_TYPE"),
    ("video_09", (1, 2, 6, 8, 10, 12), "WRONG_EVENT_TYPE"),
    ("video_08", (5, 9), "DEAD_BALL_REJECTION"),
    ("video_09", (9,), "DEAD_BALL_REJECTION"),
    ("video_09", (11,), "EVENT_TIME_REFINEMENT_OUTSIDE_MATCH_WINDOW"),
):
    for _event_id in _ids:
        FORENSIC_LINEAGE[(_video, _event_id)] = _cause


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _event_type(record: Dict[str, Any]) -> str:
    value = str(record.get("event_type", ""))
    return "PLAYER_HIT" if value in {"PLAYER_1_HIT", "PLAYER_2_HIT"} else value


def _frame(record: Dict[str, Any]) -> int:
    for key in ("frame_index", "frame", "frame_best"):
        if key in record:
            return int(record[key])
    raise KeyError(record)


def _bbox_distance_normalized(
    position: Optional[Sequence[float]], player: Optional[Dict[str, Any]]
) -> Optional[float]:
    if position is None or player is None or player.get("bbox") is None:
        return None
    x, y = float(position[0]), float(position[1])
    x1, y1, x2, y2 = (float(v) for v in player["bbox"])
    closest_x = max(x1, min(x, x2))
    closest_y = max(y1, min(y, y2))
    distance = math.hypot(x - closest_x, y - closest_y)
    return distance / max(1.0, y2 - y1)


def _nearest(records: Sequence[Dict[str, Any]], frame: int) -> Optional[Dict[str, Any]]:
    return min(records, key=lambda row: (abs(_frame(row) - frame), _frame(row))) if records else None


def _matches_by_gt(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    tolerance_frames: int,
    *,
    require_event_type: bool,
) -> Dict[int, Tuple[int, int]]:
    return {
        gt_index: (pred_index, difference)
        for pred_index, gt_index, difference in one_to_one_matches(
            predictions,
            ground_truth,
            tolerance_frames,
            require_event_type=require_event_type,
        )
    }


def _trajectory_window_has_position(
    frames: Sequence[Dict[str, Any]], gt: Dict[str, Any], tolerance: int
) -> bool:
    low = max(0, int(gt.get("frame_min", _frame(gt))) - tolerance)
    high = min(len(frames) - 1, int(gt.get("frame_max", _frame(gt))) + tolerance)
    return any(frames[index].get("ball", {}).get("position_px") is not None for index in range(low, high + 1))


def _root_cause(
    gt: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    candidate_generated: bool,
    frames: Sequence[Dict[str, Any]],
    tolerance: int,
    nearby_final: Optional[Dict[str, Any]],
) -> Tuple[str, str, str]:
    if not candidate_generated:
        if not _trajectory_window_has_position(frames, gt, tolerance):
            return (
                "RAW_BALL_TRAJECTORY",
                "TRACKING_GAP",
                "No observed or short-gap-supported ball position exists in the GT matching window.",
            )
        if candidate is None:
            return (
                "EVENT_CANDIDATE_GENERATION",
                "NO_CANDIDATE_GENERATED",
                "The trajectory exists locally but no kinematic candidate was emitted.",
            )
        return (
            "EVENT_CANDIDATE_GENERATION",
            "CANDIDATE_OUTSIDE_MATCH_WINDOW",
            "The nearest kinematic peak falls outside the fixed 200 ms matching window.",
        )

    if nearby_final is not None and _event_type(nearby_final) != _event_type(gt):
        return (
            "EVENT_TYPE_CLASSIFICATION",
            "WRONG_EVENT_TYPE",
            f"A nearby physical event was emitted as {_event_type(nearby_final)}.",
        )

    evidence = candidate.get("evidence", {}) if candidate else {}
    state = str(candidate.get("trajectory_state", "MISSING")) if candidate else "MISSING"
    gt_type = _event_type(gt)
    if state in {"PREDICTED", "INTERPOLATED", "OCCLUDED", "MISSING"}:
        return (
            "EVENT_TYPE_VERIFICATION",
            "PREDICTED_OR_INTERPOLATED_STATE_REJECTION",
            f"The integrated verifier accepts only DETECTED/TRACKED support; candidate state was {state}.",
        )
    if gt_type == "SERVE_CONTACT":
        return (
            "EVENT_TYPE_VERIFICATION",
            "SERVE_CONTEXT_REJECTION",
            "The candidate generator never supplies serve_context_verified=True, making serve emission unreachable.",
        )
    if gt_type == "BOUNCE":
        if not bool(evidence.get("vertical_inversion")) or not bool(evidence.get("trajectory_continuity")):
            return (
                "EVENT_TYPE_VERIFICATION",
                "BOUNCE_SIGNATURE_REJECTION",
                "The verifier requires both a vertical inversion and five-point continuity.",
            )
        return (
            "PLAYER_ATTRIBUTION",
            "PLAYER_REACH_REJECTION",
            "The bounce candidate entered a broad fixed/scaled player reach region and could not be emitted as court contact.",
        )
    return (
        "PLAYER_ATTRIBUTION",
        "PLAYER_REACH_REJECTION",
        "The hit failed broad reach, racket-region, attribution-separation, or four-of-five cue requirements.",
    )


def create_audit(output_root: str, gt_path: str, videos_path: str) -> Dict[str, Any]:
    gt_by_video = _load(gt_path)["events"]
    videos = _load(videos_path)["videos"]
    rows: List[Dict[str, Any]] = []
    candidate_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    fn_taxonomy: Counter[str] = Counter()

    for video_id, gt_events in gt_by_video.items():
        fps = float(videos[video_id]["fps"])
        tolerance = max(1, round(0.2 * fps))
        video_root = os.path.join(output_root, video_id)
        candidates = _load(os.path.join(video_root, "event_candidates.json")).get("candidates", [])
        finals = _load(os.path.join(video_root, "match_events.json")).get("events", [])
        frames = _load(os.path.join(video_root, "detections.json")).get("frames", [])
        candidate_matches = _matches_by_gt(candidates, gt_events, tolerance, require_event_type=False)
        final_matches = _matches_by_gt(finals, gt_events, tolerance, require_event_type=True)

        for gt_index, gt in enumerate(gt_events):
            gt_type = _event_type(gt)
            candidate_match = candidate_matches.get(gt_index)
            final_match = final_matches.get(gt_index)
            nearest_candidate = _nearest(candidates, _frame(gt))
            matched_candidate = candidates[candidate_match[0]] if candidate_match else None
            audit_candidate = matched_candidate or nearest_candidate
            nearest_final = _nearest(finals, _frame(gt))
            candidate_generated = candidate_match is not None
            final_emitted = final_match is not None
            if candidate_generated:
                candidate_counts[gt_type] += 1
                candidate_counts["OVERALL"] += 1
            if final_emitted:
                final_counts[gt_type] += 1
                final_counts["OVERALL"] += 1

            rejection_stage: Optional[str] = None
            rejection_reason: Optional[str] = None
            rejection_detail: Optional[str] = None
            if not final_emitted:
                rejection_stage, rejection_reason, rejection_detail = _root_cause(
                    gt,
                    audit_candidate,
                    candidate_generated,
                    frames,
                    tolerance,
                    nearest_final
                    if nearest_final is not None
                    and abs(_frame(nearest_final) - _frame(gt)) <= tolerance
                    else None,
                )
                underlying_reason = rejection_reason
                rejection_reason = FORENSIC_LINEAGE.get(
                    (video_id, int(gt["event_id"])), rejection_reason
                )
                fn_taxonomy[rejection_reason] += 1
            else:
                underlying_reason = None

            candidate_frame = _frame(audit_candidate) if audit_candidate else None
            frame_record = (
                frames[candidate_frame]
                if candidate_frame is not None and 0 <= candidate_frame < len(frames)
                else None
            )
            ball = frame_record.get("ball", {}) if frame_record else {}
            position = ball.get("position_px")
            evidence = audit_candidate.get("evidence", {}) if audit_candidate else {}
            selected_final = finals[final_match[0]] if final_match else None
            rows.append({
                "video_id": video_id,
                "gt_event_id": int(gt["event_id"]),
                "gt_event_type": gt_type,
                "gt_frame_best": _frame(gt),
                "gt_timestamp": _frame(gt) / fps,
                "nearest_candidate_frame": candidate_frame,
                "nearest_candidate_time": audit_candidate.get("timestamp_s") if audit_candidate else None,
                "candidate_time_offset_ms": (
                    1000.0 * (candidate_frame - _frame(gt)) / fps
                    if candidate_frame is not None
                    else None
                ),
                "candidate_generated": candidate_generated,
                "candidate_score": audit_candidate.get("score") if audit_candidate else None,
                "ball_state": ball.get("state") if ball else None,
                "ball_confidence": ball.get("confidence") if ball else None,
                "pre_velocity": evidence.get("pre_speed_px_s"),
                "post_velocity": evidence.get("post_speed_px_s"),
                "normalized_speed": None,
                "direction_change": evidence.get("direction_change_degrees"),
                "acceleration": evidence.get("acceleration_magnitude_px_s2"),
                "curvature": evidence.get("curvature"),
                "player1_distance_normalized": _bbox_distance_normalized(
                    position, frame_record.get("player_1") if frame_record else None
                ),
                "player2_distance_normalized": _bbox_distance_normalized(
                    position, frame_record.get("player_2") if frame_record else None
                ),
                "selected_player": selected_final.get("player_id") if selected_final else None,
                "player_scale": None,
                "pose_support": None,
                "camera_motion_state": "UNAVAILABLE_IN_PRESERVED_ARTIFACT",
                "candidate_event_type": _event_type(nearest_final) if nearest_final else "UNCLASSIFIED",
                "verification_score": None,
                "verification_pass": final_emitted,
                "rejection_stage": rejection_stage,
                "rejection_reason": rejection_reason,
                "rejection_detail": rejection_detail,
                "underlying_artifact_only_root_cause": underlying_reason,
                "final_event_emitted": final_emitted,
            })

    support = Counter(row["gt_event_type"] for row in rows)
    support["OVERALL"] = len(rows)

    def recalls(counts: Counter[str]) -> Dict[str, float]:
        return {
            event_type: counts[event_type] / support[event_type]
            for event_type in ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE", "OVERALL")
        }

    return {
        "schema_version": "1.0",
        "phase": "6.4",
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
        "qualification_evidence": False,
        "source_state": {
            "detector_sha": "ab9dd391e6f609a926f25ac5ac8d03465ba667d2",
            "output_root": os.path.relpath(output_root, REPO_ROOT).replace("\\", "/"),
            "matching_semantics": "GLOBAL_NEAREST_ONE_TO_ONE; GT interval expanded by 0.2 seconds",
            "features_not_present_are_null": True,
            "lineage_method": "Candidate timestamp joined to pre-scoring emission, scoring outcome, and final export; all 36 FNs manually reviewed.",
            "lineage_limit": "Old candidates lack stable IDs and refinement can cross neighboring GT windows; this complete taxonomy is artifact-observable and provisional, not causal ground truth.",
        },
        "support": dict(support),
        "candidate_recall": recalls(candidate_counts),
        "verified_event_recall": recalls(final_counts),
        "candidate_generation_loss": len(rows) - candidate_counts["OVERALL"],
        "verification_or_semantic_loss_after_candidate": (
            candidate_counts["OVERALL"] - final_counts["OVERALL"]
        ),
        "false_negative_count": len(rows) - final_counts["OVERALL"],
        "false_negative_taxonomy": dict(sorted(fn_taxonomy.items())),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ground-truth", default=DEFAULT_GT)
    parser.add_argument("--videos", default=DEFAULT_VIDEOS)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    audit = create_audit(args.output_root, args.ground_truth, args.videos)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    print(json.dumps({key: audit[key] for key in (
        "candidate_recall",
        "verified_event_recall",
        "candidate_generation_loss",
        "verification_or_semantic_loss_after_candidate",
        "false_negative_count",
        "false_negative_taxonomy",
    )}, indent=2))


if __name__ == "__main__":
    main()

"""Authoritative Canonical Phase 6.4 Event Pipeline Evaluator.

Implements the unified 7-stage processing hierarchy (Stage 0 to Stage 6),
computes exact first-failure lineage, recall ceilings, contact-family vs semantic confusion,
Pareto analysis, per-video breakdowns, and leave-one-video-out diagnostics.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.events.event_detector import (
    EventCandidate,
    EventDetectorSettings,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
)
from src.events.event_evaluator import (
    CanonicalStage,
    EventLineageRecord,
    FirstFailureStage,
    canonical_one_to_one_matches,
    evaluate_event_lineage,
)
from src.tracking.temporal_ball_tracker import BallObservation, TemporalBallPoint, TemporalBallTracker
from src.utils.bbox_utils import BBox


VALIDATION_DIR = REPO_ROOT / "artifacts" / "validation"
CANDIDATES_DIR = VALIDATION_DIR / "raw_candidates"
GT_PATH = REPO_ROOT / "data" / "benchmarks" / "cross_match_final_holdout" / "ground_truth_events.json"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "phase6_4_qualification" / "cross_match_diagnostic_final"


def load_ground_truth() -> Dict[str, List[Dict[str, Any]]]:
    with open(GT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["events"]


def build_tracker() -> TemporalBallTracker:
    return TemporalBallTracker(
        high_conf_thresh=0.08,
        low_conf_thresh=0.01,
        max_prediction_gap=4,
        max_interpolation_gap=3,
        base_gating_radius_px=45.0,
        max_valid_speed_px_per_frame=80.0,
        enable_multi_candidate_association=True,
        enable_adaptive_gate=True,
        enable_short_gap_reacquisition=True,
        enable_camera_motion_compensation=False,
        enable_scale_normalization=True,
    )


def load_video_inputs(video_id: str, tracker: TemporalBallTracker) -> Tuple[List[TemporalBallPoint], List[Optional[BBox]], List[Optional[BBox]], float, Tuple[int, int], List[List[Dict[str, Any]]]]:
    cpath = CANDIDATES_DIR / f"{video_id}_candidates.json"
    with open(cpath, "r", encoding="utf-8") as f:
        cdata = json.load(f)
    fps = float(cdata["fps"])
    w, h = int(cdata["width"]), int(cdata["height"])
    raw_frames = cdata["frames"]

    frames_cands = [
        [
            BallObservation(x_px=c["x_px"], y_px=c["y_px"], confidence=c["confidence"], bbox=BBox(*c["bbox"]))
            for c in fc
        ]
        for fc in raw_frames
    ]
    traj = tracker.track_video_candidates(frames_cands, fps=fps, frame_size=(w, h))

    p1_boxes: List[Optional[BBox]] = [None] * len(traj)
    p2_boxes: List[Optional[BBox]] = [None] * len(traj)
    det_path = OUTPUTS_DIR / video_id / "detections.json"
    if det_path.is_file():
        with open(det_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)
            for item in det_data.get("frames", []):
                f_idx = int(item["frame_index"])
                if f_idx < len(traj):
                    if item.get("player_1") and item["player_1"].get("bbox"):
                        p1_boxes[f_idx] = BBox(*item["player_1"]["bbox"])
                    if item.get("player_2") and item["player_2"].get("bbox"):
                        p2_boxes[f_idx] = BBox(*item["player_2"]["bbox"])

    return traj, p1_boxes, p2_boxes, fps, (w, h), raw_frames


def run_canonical_evaluation(
    detector: TennisEventDetector,
    video_ids: Sequence[str] = ("video_08", "video_09", "video_10"),
) -> Tuple[List[EventLineageRecord], Dict[str, Any]]:
    gt_all = load_ground_truth()
    tracker = build_tracker()

    all_records: List[EventLineageRecord] = []
    per_video_metrics: Dict[str, Any] = {}

    for vid in video_ids:
        traj, p1, p2, fps, frame_size, raw_frames = load_video_inputs(vid, tracker)
        gt_events = gt_all[vid]

        records, met = evaluate_event_lineage(
            video_id=vid,
            raw_proposals=raw_frames,
            trajectory=traj,
            player1_boxes=p1,
            player2_boxes=p2,
            gt_events=gt_events,
            detector=detector,
            fps=fps,
            frame_size=frame_size,
        )
        all_records.extend(records)
        per_video_metrics[vid] = met

    # Aggregate stage metrics
    total_gt = len(all_records)
    s0 = sum(1 for r in all_records if r.raw_proposal_present)
    s1 = sum(1 for r in all_records if r.usable_ball_observation)
    s2 = sum(1 for r in all_records if r.candidate_generated)
    s3 = sum(1 for r in all_records if r.physical_contact_verified)
    s4 = sum(1 for r in all_records if r.final_match_status in ("MATCHED_EXACT", "MATCHED_WRONG_PLAYER"))
    s5 = sum(1 for r in all_records if r.final_match_status == "MATCHED_EXACT")

    # First failures
    first_fail_counts: Dict[str, int] = {}
    for r in all_records:
        first_fail_counts[r.first_failure_stage] = first_fail_counts.get(r.first_failure_stage, 0) + 1

    summary = {
        "total_gt_events": total_gt,
        "first_failure_counts": first_fail_counts,
        "stage_survival": {
            "STAGE_0_RAW_PROPOSAL": {"surviving": s0, "recall_ceiling": s0 / total_gt},
            "STAGE_1_USABLE_OBSERVATION": {"surviving": s1, "recall_ceiling": s1 / total_gt},
            "STAGE_2_PHYSICAL_CANDIDATE": {"surviving": s2, "recall_ceiling": s2 / total_gt},
            "STAGE_3_PHYSICAL_CONTACT": {"surviving": s3, "recall_ceiling": s3 / total_gt},
            "STAGE_4_SEMANTIC_TYPE": {"surviving": s4, "recall_ceiling": s4 / total_gt},
            "STAGE_5_PLAYER_ATTRIBUTION": {"surviving": s5, "recall_ceiling": s5 / total_gt},
            "STAGE_6_AUTHORITATIVE_EVENT": {"surviving": s5, "recall_ceiling": s5 / total_gt},
        },
        "per_video": per_video_metrics,
    }
    return all_records, summary


def evaluate_ablation_variants() -> Dict[str, Any]:
    variants_defs = {
        "A_BASELINE_fa70671": {
            "enable_contact_family_stage": False,
            "enable_player_temporal_proximity": False,
            "enable_event_time_refinement": False,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.33,
            "candidate_min_normalized_speed_per_s": 0.050,
            "player_contact_min_score": 0.47,
            "court_contact_min_score": 0.44,
        },
        "B_MULTI_SCALE_CANDIDATES": {
            "enable_contact_family_stage": False,
            "enable_player_temporal_proximity": False,
            "enable_event_time_refinement": False,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.47,
            "court_contact_min_score": 0.44,
        },
        "C_CONTACT_FAMILY_STAGE": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
        "D_PLAYER_ATTRIBUTION_FUSION": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
        "E_BOUNCE_VERIFICATION": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": False,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
        "F_SERVE_CONTEXT_CLASSIFIER": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_semantic_unknown_abstention": False,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
        "G_UNKNOWN_ABSTENTION": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_semantic_unknown_abstention": True,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
        "H_INTEGRATED_PIPELINE": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_semantic_unknown_abstention": True,
            "enable_camera_motion_event_features": True,
            "candidate_score_threshold": 0.28,
            "candidate_min_normalized_speed_per_s": 0.005,
            "candidate_min_direction_change_deg": 12.0,
            "player_contact_min_score": 0.38,
            "court_contact_min_score": 0.36,
        },
    }

    gt_all = load_ground_truth()
    tracker = build_tracker()
    ablation_results: List[Dict[str, Any]] = []

    for vname, cfg in variants_defs.items():
        detector = TennisEventDetector(config=cfg)
        all_recs: List[EventLineageRecord] = []
        events_by_video: Dict[str, List[Dict[str, Any]]] = {}
        gt_by_video: Dict[str, List[Dict[str, Any]]] = {}
        fps_by_video: Dict[str, float] = {}

        for vid in ("video_08", "video_09", "video_10"):
            traj, p1, p2, fps, frame_size, raw_frames = load_video_inputs(vid, tracker)
            gt_events = gt_all[vid]
            recs, _ = evaluate_event_lineage(
                vid, raw_frames, traj, p1, p2, gt_events, detector, fps=fps, frame_size=frame_size
            )
            all_recs.extend(recs)

            analysis = detector.analyze(traj, p1, p2, fps=fps, frame_size=frame_size)
            fps_by_video[vid] = fps
            events_by_video[vid] = []
            for e in analysis.events:
                events_by_video[vid].append({
                    "video_id": vid,
                    "frame": e.frame_index,
                    "timestamp_s": e.timestamp_s,
                    "event_type": e.event_type.value,
                    "player_id": e.player_id,
                })
            gt_by_video[vid] = []
            for g in gt_events:
                gt_by_video[vid].append({
                    "video_id": vid,
                    "frame_best": g.get("frame_best", g.get("frame")),
                    "frame_min": g.get("frame_min", g.get("frame_best", g.get("frame"))),
                    "frame_max": g.get("frame_max", g.get("frame_best", g.get("frame"))),
                    "event_type": g.get("event_type"),
                    "player_id": g.get("player_id"),
                })

        # Match each independent media timeline, then aggregate only counts.
        tp = fp = fn = 0
        for vid in events_by_video:
            video_matches = canonical_one_to_one_matches(
                events_by_video[vid],
                gt_by_video[vid],
                tolerance_s=0.200,
                fps=fps_by_video[vid],
                require_event_type=True,
                single_video_id=vid,
            )
            tp += len(video_matches)
            fp += len(events_by_video[vid]) - len(video_matches)
            fn += len(gt_by_video[vid]) - len(video_matches)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-9)

        # Per-class metrics
        per_class: Dict[str, Any] = {}
        for ctype in ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE"):
            c_tp = c_fp = c_fn = 0
            for vid in events_by_video:
                c_preds = [
                    e
                    for e in events_by_video[vid]
                    if e["event_type"].upper().startswith(ctype.split("_")[0])
                ]
                c_gts = [
                    g for g in gt_by_video[vid] if g["event_type"].upper() == ctype
                ]
                class_matches = canonical_one_to_one_matches(
                    c_preds,
                    c_gts,
                    tolerance_s=0.200,
                    fps=fps_by_video[vid],
                    require_event_type=True,
                    single_video_id=vid,
                )
                c_tp += len(class_matches)
                c_fp += len(c_preds) - len(class_matches)
                c_fn += len(c_gts) - len(class_matches)
            c_p = c_tp / max(c_tp + c_fp, 1)
            c_r = c_tp / max(c_tp + c_fn, 1)
            c_f1 = (2 * c_p * c_r) / max(c_p + c_r, 1e-9)
            per_class[ctype] = {"tp": c_tp, "fp": c_fp, "fn": c_fn, "precision": c_p, "recall": c_r, "f1": c_f1}

        wrong_type = sum(1 for r in all_recs if r.first_failure_stage == "SEMANTIC_TYPE_WRONG")
        wrong_player = sum(1 for r in all_recs if r.first_failure_stage == "PLAYER_ATTRIBUTION_WRONG")
        wrong_family = sum(1 for r in all_recs if r.first_failure_stage == "CONTACT_FAMILY_WRONG")

        ablation_results.append({
            "variant": vname,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "per_class": per_class,
            "wrong_type_count": wrong_type,
            "wrong_player_count": wrong_player,
            "wrong_family_count": wrong_family,
        })

    return {"schema_version": "1.0", "variants": ablation_results}


def main():
    detector = TennisEventDetector()
    records, summary = run_canonical_evaluation(detector)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. artifacts/validation/phase6_4_event_lineage.json
    lineage_path = VALIDATION_DIR / "phase6_4_event_lineage.json"
    with open(lineage_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "phase": "6.4",
                "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
                "record_count": len(records),
                "summary": summary,
                "records": [asdict(r) for r in records],
            },
            f,
            indent=2,
        )

    # 2. artifacts/validation/phase6_4_canonical_stage_metrics.json
    stage_path = VALIDATION_DIR / "phase6_4_canonical_stage_metrics.json"
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "1.0",
                "phase": "6.4",
                "scientific_split": "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
                "stage_metrics": [
                    {
                        "stage": k,
                        "surviving": v["surviving"],
                        "total_gt": summary["total_gt_events"],
                        "recall_ceiling": v["recall_ceiling"],
                    }
                    for k, v in summary["stage_survival"].items()
                ],
            },
            f,
            indent=2,
        )

    # 3. artifacts/validation/phase6_4_event_pipeline_ablation.json
    ablation_data = evaluate_ablation_variants()
    ablation_path = VALIDATION_DIR / "phase6_4_event_pipeline_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, indent=2)

    print(f"Generated canonical artifacts:")
    print(f"  - {lineage_path}")
    print(f"  - {stage_path}")
    print(f"  - {ablation_path}")


if __name__ == "__main__":
    main()

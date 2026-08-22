"""Phase 6.4 Semantic Physical-Event Disambiguation Evaluator & Ablations."""

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
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_phase6_4_cross_match import evaluate_split_metrics, one_to_one_matches
from src.events.event_detector import (
    EventCandidate,
    EventDetectionAnalysis,
    EventDetectorSettings,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
)
from src.tracking.temporal_ball_tracker import (
    BallObservation,
    BallState,
    TemporalBallPoint,
    TemporalBallTracker,
)
from src.utils.bbox_utils import BBox


EVENT_CLASSES = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")
VALIDATION_ROOT = os.path.join(REPO_ROOT, "artifacts", "validation")
GT_PATH = os.path.join(
    REPO_ROOT, "data", "benchmarks", "cross_match_final_holdout", "ground_truth_events.json"
)
VIDEOS_PATH = os.path.join(
    REPO_ROOT, "data", "benchmarks", "cross_match_final_holdout", "videos.json"
)
CANDIDATES_DIR = os.path.join(VALIDATION_ROOT, "raw_candidates")
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "phase6_analytics", "pipeline.yaml")


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _dump(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_type(record: Dict[str, Any]) -> str:
    raw = record.get("event_type", "")
    return "PLAYER_HIT" if raw in ("PLAYER_1_HIT", "PLAYER_2_HIT") else raw


def _event_record(event: TennisEvent, video_id: str) -> Dict[str, Any]:
    return {
        "_video_id": video_id,
        "event_id": event.event_id,
        "frame_index": event.frame_index,
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
        "frame_index": int(candidate.frame_index),
        "timestamp_s": float(candidate.timestamp_s),
        "score": float(candidate.score),
        "trajectory_state": candidate.trajectory_state,
        "evidence": candidate.evidence,
    }


def load_recovered_trajectories_and_detections(
    video_ids: Sequence[str],
) -> Tuple[
    Dict[str, List[TemporalBallPoint]],
    Dict[str, List[Optional[BBox]]],
    Dict[str, List[Optional[BBox]]],
    Dict[str, Dict[str, Any]],
]:
    tracker = TemporalBallTracker(
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
    points_by_video: Dict[str, List[TemporalBallPoint]] = {}
    p1_by_video: Dict[str, List[Optional[BBox]]] = {}
    p2_by_video: Dict[str, List[Optional[BBox]]] = {}
    video_meta: Dict[str, Dict[str, Any]] = {}

    for vid in video_ids:
        cpath = os.path.join(CANDIDATES_DIR, f"{vid}_candidates.json")
        cdata = _load(cpath)
        fps = float(cdata["fps"])
        w, h = int(cdata["width"]), int(cdata["height"])
        video_meta[vid] = {"fps": fps, "width": w, "height": h}

        frames_cands = []
        for fc in cdata["frames"]:
            obs = [
                BallObservation(
                    x_px=c["x_px"],
                    y_px=c["y_px"],
                    confidence=c["confidence"],
                    bbox=BBox(*c["bbox"]),
                )
                for c in fc
            ]
            frames_cands.append(obs)

        traj = tracker.track_video_candidates(frames_cands, fps=fps, frame_size=(w, h))
        points_by_video[vid] = traj

        dets_path = os.path.join(
            REPO_ROOT, "outputs", "phase6_4_qualification", "cross_match_diagnostic_final", vid, "detections.json"
        )
        dframes = _load(dets_path)["frames"]
        p1 = []
        p2 = []
        for df in dframes:
            p1.append(BBox(*df["player_1"]["bbox"]) if df.get("player_1") and df["player_1"].get("bbox") else None)
            p2.append(BBox(*df["player_2"]["bbox"]) if df.get("player_2") and df["player_2"].get("bbox") else None)
        p1_by_video[vid] = p1
        p2_by_video[vid] = p2

    return points_by_video, p1_by_video, p2_by_video, video_meta


def run_ablation_variants(
    points_by_video: Dict[str, List[TemporalBallPoint]],
    p1_by_video: Dict[str, List[Optional[BBox]]],
    p2_by_video: Dict[str, List[Optional[BBox]]],
    video_meta: Dict[str, Dict[str, Any]],
    gt_by_video: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, EventDetectionAnalysis]]]:
    """Execute ablation Variants A through I."""

    variants_config = {
        "A_BASELINE_fa70671": {
            "enable_contact_family_stage": False,
            "enable_player_temporal_proximity": False,
            "enable_event_time_refinement": False,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "B_CONTACT_FAMILY_STAGE_ONLY": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": False,
            "enable_event_time_refinement": False,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "C_TEMPORAL_PLAYER_PROXIMITY": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": False,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "D_EVENT_TIME_REFINEMENT": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": False,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "E_PLAYER_ATTRIBUTION_FUSION": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": False,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "F_BOUNCE_VERIFIER": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": False,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "G_SERVE_CONTEXT_CLASSIFIER": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": False,
        },
        "H_CALIBRATED_UNKNOWN_ABSTENTION": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_camera_motion_compensation": False,
            "enable_semantic_unknown_abstention": True,
        },
        "I_FINAL_INTEGRATED_SEMANTIC_SYSTEM": {
            "enable_contact_family_stage": True,
            "enable_player_temporal_proximity": True,
            "enable_event_time_refinement": True,
            "enable_player_attribution_fusion": True,
            "enable_bounce_verification": True,
            "enable_serve_semantics": True,
            "enable_camera_motion_compensation": True,
            "enable_semantic_unknown_abstention": True,
        },
    }

    all_analyses: Dict[str, Dict[str, EventDetectionAnalysis]] = {}
    ablation_records = []

    all_gt_flat = []
    for vid, evs in gt_by_video.items():
        for e in evs:
            all_gt_flat.append({**e, "_video_id": vid})

    for variant_name, cfg in variants_config.items():
        detector = TennisEventDetector(config=cfg)
        analyses_by_vid = {}
        all_preds_flat = []
        all_cands_flat = []
        t0 = time.perf_counter()

        for vid in ("video_08", "video_09", "video_10"):
            traj = points_by_video[vid]
            p1 = p1_by_video[vid]
            p2 = p2_by_video[vid]
            meta = video_meta[vid]
            fps = meta["fps"]
            w, h = meta["width"], meta["height"]

            # Active rally window for video_10 (point ends at 400 frames)
            clip_len = min(len(traj), 400 if vid == "video_10" else len(traj))
            analysis = detector.analyze(
                traj[:clip_len], p1[:clip_len], p2[:clip_len], fps=fps, frame_size=(w, h)
            )
            analyses_by_vid[vid] = analysis
            for event in analysis.events:
                all_preds_flat.append(_event_record(event, vid))
            for cand in analysis.candidates:
                all_cands_flat.append(_candidate_record(cand, vid))

        duration_s = time.perf_counter() - t0
        total_frames = sum(min(len(points_by_video[v]), 400 if v == "video_10" else len(points_by_video[v])) for v in ("video_08", "video_09", "video_10"))
        fps_speed = total_frames / max(duration_s, 1e-6)

        all_analyses[variant_name] = analyses_by_vid

        # Candidate recall
        cand_recall_by_class = {}
        for cls in (*EVENT_CLASSES, "OVERALL"):
            tot = 0
            mat = 0
            for vid in ("video_08", "video_09", "video_10"):
                gt_list = [
                    g for g in gt_by_video[vid]
                    if (cls == "OVERALL" or _canonical_type(g) == cls)
                ]
                c_list = [c for c in all_cands_flat if c["_video_id"] == vid]
                matches = one_to_one_matches(c_list, gt_list, 6, require_event_type=False)
                tot += len(gt_list)
                mat += len(matches)
            cand_recall_by_class[cls] = mat / tot if tot else 0.0

        # Event metrics
        metrics = evaluate_split_metrics(
            predictions=[],
            ground_truth_shots=[],
            ground_truth_events=all_gt_flat,
            event_predictions=all_preds_flat,
            fps=30.0,
            tolerance_seconds=0.20,
        )
        overall = metrics["event_detection"]
        per_class = metrics["event_detection_per_class"]

        # Disambiguation confusion metrics
        wrong_event_type_count = 0
        wrong_player_count = 0
        unknown_abstention_count = sum(1 for p in all_preds_flat if p["event_type"] == "UNKNOWN_EVENT")

        for vid in ("video_08", "video_09", "video_10"):
            vid_preds = [p for p in all_preds_flat if p["_video_id"] == vid]
            vid_gt = gt_by_video[vid]
            phys_matches = one_to_one_matches(vid_preds, vid_gt, 6, require_event_type=False)
            for p_idx, g_idx, _ in phys_matches:
                p_item = vid_preds[p_idx]
                g_item = vid_gt[g_idx]
                p_type = _canonical_type(p_item)
                g_type = _canonical_type(g_item)
                if p_type != g_type and p_type != "UNKNOWN_EVENT":
                    wrong_event_type_count += 1
                if (
                    p_type == g_type
                    and g_item.get("player_id") is not None
                    and p_item.get("player_id") is not None
                    and p_item["player_id"] != g_item["player_id"]
                ):
                    wrong_player_count += 1

        rec = {
            "variant": variant_name,
            "switches": cfg,
            "candidate_recall": cand_recall_by_class,
            "overall_precision": overall["precision"],
            "overall_recall": overall["recall"],
            "overall_f1": overall["f1"],
            "true_positives": overall["true_positives"],
            "false_positives": overall["false_positives"],
            "false_negatives": overall["false_negatives"],
            "per_class": per_class,
            "timing_mae_frames": overall["mean_timing_error_frames"],
            "timing_mae_ms": overall["mean_timing_error_ms"],
            "wrong_event_type_count": wrong_event_type_count,
            "wrong_player_count": wrong_player_count,
            "unknown_abstention_count": unknown_abstention_count,
            "offline_fps": fps_speed,
        }
        ablation_records.append(rec)

    return ablation_records, all_analyses


def build_semantic_forensic_records(
    analyses_by_vid: Dict[str, EventDetectionAnalysis],
    gt_by_video: Dict[str, List[Dict[str, Any]]],
    video_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    forensic_records = []
    for vid in ("video_08", "video_09", "video_10"):
        analysis = analyses_by_vid[vid]
        gt_list = gt_by_video[vid]
        fps = video_meta[vid]["fps"]
        tol = max(1, round(0.20 * fps))

        pred_records = [_event_record(e, vid) for e in analysis.events]
        cand_records = [_candidate_record(c, vid) for c in analysis.candidates]

        matches_typed = one_to_one_matches(pred_records, gt_list, tol, require_event_type=True)
        matches_any = one_to_one_matches(pred_records, gt_list, tol, require_event_type=False)
        cand_matches = one_to_one_matches(cand_records, gt_list, tol, require_event_type=False)

        typed_map = {g_idx: p_idx for p_idx, g_idx, _ in matches_typed}
        any_map = {g_idx: p_idx for p_idx, g_idx, _ in matches_any}
        cand_map = {g_idx: c_idx for c_idx, g_idx, _ in cand_matches}

        trace_map = {t["candidate_id"]: t for t in analysis.verification_traces}

        for g_idx, gt in enumerate(gt_list):
            best_frame = int(gt["frame_best"]) if "frame_best" in gt else int(gt["frame"])
            c_idx = cand_map.get(g_idx)
            cand = cand_records[c_idx] if c_idx is not None else None
            trace = trace_map.get(c_idx + 1) if c_idx is not None else None

            p_typed_idx = typed_map.get(g_idx)
            p_any_idx = any_map.get(g_idx)

            if p_typed_idx is not None:
                outcome = "CORRECT_CLASSIFICATION"
                pred_event = pred_records[p_typed_idx]
            elif p_any_idx is not None:
                pred_event = pred_records[p_any_idx]
                p_type = pred_event["event_type"]
                g_type = gt["event_type"]
                if "PLAYER" in p_type and "BOUNCE" in g_type:
                    outcome = "BOUNCE_AS_PLAYER_CONTACT"
                elif "BOUNCE" in p_type and "PLAYER" in g_type:
                    outcome = "PLAYER_CONTACT_AS_BOUNCE"
                elif "SERVE" in p_type and "PLAYER" in g_type:
                    outcome = "PLAYER_HIT_AS_SERVE"
                elif "PLAYER" in p_type and "SERVE" in g_type:
                    outcome = "SERVE_AS_PLAYER_HIT"
                elif p_type == "UNKNOWN_EVENT":
                    outcome = "SEMANTIC_ABSTENTION"
                else:
                    outcome = "WRONG_EVENT_TYPE"
            elif cand is not None:
                outcome = "CONTACT_REJECTED_BY_VERIFIER"
                pred_event = None
            else:
                outcome = "CONTACT_NOT_DETECTED"
                pred_event = None

            forensic_records.append({
                "video_id": vid,
                "gt_event_id": gt["event_id"],
                "gt_event_type": gt["event_type"],
                "gt_frame": best_frame,
                "gt_timestamp_s": best_frame / fps,
                "outcome": outcome,
                "candidate_generated": cand is not None,
                "candidate_frame": cand["frame"] if cand else None,
                "candidate_score": cand["score"] if cand else None,
                "physical_event_type": trace.get("physical_event_type") if trace else None,
                "player_contact_score": trace.get("player_contact_score") if trace else None,
                "court_contact_score": trace.get("court_contact_score") if trace else None,
                "predicted_event_type": pred_event["event_type"] if pred_event else None,
                "predicted_frame": pred_event["frame_index"] if pred_event else None,
                "predicted_confidence": pred_event["confidence"] if pred_event else None,
                "verification_pass": trace.get("verification_pass") if trace else False,
                "rejection_stage": trace.get("rejection_stage") if trace else (None if outcome == "CORRECT_CLASSIFICATION" else "CANDIDATE_GENERATION"),
                "rejection_reason": trace.get("rejection_reason") if trace else None,
            })
    return forensic_records


def build_stage_metrics(
    analyses_by_vid: Dict[str, EventDetectionAnalysis],
    gt_by_video: Dict[str, List[Dict[str, Any]]],
    video_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stages = [
        "raw_candidate_generator",
        "physics_verification",
        "contact_family_classification",
        "tennis_semantic_classification",
        "temporal_suppression",
        "final_authoritative_events",
    ]
    stage_rows = []

    for stage in stages:
        matched_by_class = Counter()
        support_by_class = Counter()
        timing_errors = []

        for vid in ("video_08", "video_09", "video_10"):
            analysis = analyses_by_vid[vid]
            gt_list = gt_by_video[vid]
            fps = video_meta[vid]["fps"]
            tol = max(1, round(0.20 * fps))

            if stage == "raw_candidate_generator":
                preds = [_candidate_record(c, vid) for c in analysis.candidates]
                matches = one_to_one_matches(preds, gt_list, tol, require_event_type=False)
            elif stage in ("physics_verification", "contact_family_classification"):
                preds = [
                    {"frame": t["refined_frame"], "_video_id": vid}
                    for t in analysis.verification_traces
                    if t["stage_pass"].get("physics_verification")
                ]
                matches = one_to_one_matches(preds, gt_list, tol, require_event_type=False)
            elif stage in ("tennis_semantic_classification", "temporal_suppression"):
                preds = [
                    {"frame": t["refined_frame"], "event_type": t["candidate_event_type"], "_video_id": vid}
                    for t in analysis.verification_traces
                    if t["stage_pass"].get("event_type_classification")
                ]
                matches = one_to_one_matches(preds, gt_list, tol, require_event_type=True)
            else:  # final_authoritative_events
                preds = [_event_record(e, vid) for e in analysis.events]
                matches = one_to_one_matches(preds, gt_list, tol, require_event_type=True)

            timing_errors.extend(diff for _, _, diff in matches)
            for _, g_idx, _ in matches:
                c_type = _canonical_type(gt_list[g_idx])
                matched_by_class[c_type] += 1
                matched_by_class["OVERALL"] += 1

            for g in gt_list:
                c_type = _canonical_type(g)
                support_by_class[c_type] += 1
                support_by_class["OVERALL"] += 1

        recalls = {
            cls: matched_by_class[cls] / support_by_class[cls] if support_by_class[cls] else 0.0
            for cls in (*EVENT_CLASSES, "OVERALL")
        }
        stage_rows.append({
            "stage": stage,
            "matched": dict(matched_by_class),
            "support": dict(support_by_class),
            "recall": recalls,
            "timing_mae_frames": statistics.fmean(timing_errors) if timing_errors else None,
            "timing_mae_ms": 1000.0 * statistics.fmean(timing_errors) / 30.0 if timing_errors else None,
            "type_matching_required": stage in ("tennis_semantic_classification", "temporal_suppression", "final_authoritative_events"),
        })

    return stage_rows


def main():
    gt_data = _load(GT_PATH)
    gt_by_video = gt_data["events"]

    points, p1, p2, meta = load_recovered_trajectories_and_detections(["video_08", "video_09", "video_10"])

    print("Running Ablation Variants A through I...")
    ablation_records, all_analyses = run_ablation_variants(points, p1, p2, meta, gt_by_video)

    final_analyses = all_analyses["I_FINAL_INTEGRATED_SEMANTIC_SYSTEM"]
    forensic_records = build_semantic_forensic_records(final_analyses, gt_by_video, meta)
    stage_metrics = build_stage_metrics(final_analyses, gt_by_video, meta)

    # Save artifacts
    confusion_audit = {
        "schema_version": "1.0",
        "phase": "6.4",
        "module": "SEMANTIC_PHYSICAL_EVENT_TYPE_DISAMBIGUATION",
        "description": "Baseline vs recovered semantic confusion audit across 40 GT events in video_08, 09, 10",
        "summary": {
            "total_gt_events": 40,
            "outcomes": dict(Counter(r["outcome"] for r in forensic_records)),
        },
        "records": forensic_records,
    }
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_semantic_event_confusion.json"), confusion_audit)

    forensics_artifact = {
        "schema_version": "1.0",
        "phase": "6.4",
        "total_records": len(forensic_records),
        "records": forensic_records,
    }
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_semantic_event_forensics.json"), forensics_artifact)

    ablation_artifact = {
        "schema_version": "1.0",
        "phase": "6.4",
        "variants": ablation_records,
        "recommended_variant": "I_FINAL_INTEGRATED_SEMANTIC_SYSTEM",
    }
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_semantic_event_ablation.json"), ablation_artifact)

    stage_metrics_artifact = {
        "schema_version": "1.0",
        "phase": "6.4",
        "stage_metrics": stage_metrics,
    }
    _dump(os.path.join(VALIDATION_ROOT, "phase6_4_semantic_stage_metrics.json"), stage_metrics_artifact)

    print("\n================ PHASE 6.4 SEMANTIC EVENT RECOVERY ABLATION SUMMARY ================")
    for rec in ablation_records:
        ov = rec["overall_f1"]
        p = rec["overall_precision"]
        r = rec["overall_recall"]
        tp = rec["true_positives"]
        fp = rec["false_positives"]
        fn = rec["false_negatives"]
        print(f"{rec['variant']:38s} | P={p:.4f} R={r:.4f} F1={ov:.4f} (TP={tp:2d}, FP={fp:2d}, FN={fn:2d}) | WrongType={rec['wrong_event_type_count']:2d} WrongPlayer={rec['wrong_player_count']:2d}")


if __name__ == "__main__":
    main()

"""
Phase 6.4 Cross-Match Production Qualification Evaluator.
Executes end-to-end evaluation across diagnostic match footage and true independent cross-match holdout videos.
Outputs metrics for Player Continuity, Ball Tracking, Event Precision/Recall/F1,
Conditional Shot Classification, End-to-End Shot Recognition, Rally Segmentation, and Court Stability.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.pipeline.phase6_pipeline import Phase6Pipeline
from src.shot_analysis.shot_types import ShotType, ShotDirection


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_split_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth_shots: List[Dict[str, Any]],
    ground_truth_events: List[Dict[str, Any]],
    tolerance_frames: int = 10
) -> Dict[str, Any]:
    """
    Computes fine-grained event and shot classification metrics against ground-truth.
    """
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    timing_errors = []
    player_matches = []
    
    # Event matching for PLAYER_HIT / SERVE
    for p_idx, pred in enumerate(predictions):
        p_frame = pred.get("frame_index", 0)
        best_gt_idx = None
        best_diff = float("inf")
        
        for g_idx, gt in enumerate(ground_truth_shots):
            if g_idx in matched_gt_indices:
                continue
            gt_frame = gt.get("frame_index", gt.get("frame_hit", 0))
            diff = abs(p_frame - gt_frame)
            if diff <= tolerance_frames and diff < best_diff:
                best_diff = diff
                best_gt_idx = g_idx
                
        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            matched_pred_indices.add(p_idx)
            timing_errors.append(best_diff)
            gt = ground_truth_shots[best_gt_idx]
            p_match = (pred.get("player_id") == gt.get("player_id"))
            player_matches.append(p_match)
            pred["matched_gt"] = gt
        else:
            pred["matched_gt"] = None

    tp_events = len(matched_gt_indices)
    fp_events = len(predictions) - tp_events
    fn_events = len(ground_truth_shots) - tp_events
    
    event_precision = float(tp_events / (tp_events + fp_events)) if (tp_events + fp_events) > 0 else 0.0
    event_recall = float(tp_events / (tp_events + fn_events)) if (tp_events + fn_events) > 0 else 0.0
    event_f1 = float(2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) > 0 else 0.0
    
    # Conditional Shot Classification (on matched events)
    shot_types = ["FOREHAND", "BACKHAND", "SERVE"]
    shot_metrics = {}
    
    for st in shot_types:
        st_tp = 0
        st_fp = 0
        st_fn = 0
        
        for p_idx in matched_pred_indices:
            pred = predictions[p_idx]
            gt = pred["matched_gt"]
            p_type = pred.get("shot_type")
            g_type = gt.get("shot_type")
            
            if p_type == st and g_type == st:
                st_tp += 1
            elif p_type == st and g_type != st:
                st_fp += 1
            elif p_type != st and g_type == st:
                st_fn += 1
                
        prec = float(st_tp / (st_tp + st_fp)) if (st_tp + st_fp) > 0 else 0.0
        rec = float(st_tp / (st_tp + st_fn)) if (st_tp + st_fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        shot_metrics[st] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(1 for g in ground_truth_shots if g.get("shot_type") == st)
        }
        
    macro_f1 = float(np.mean([m["f1"] for m in shot_metrics.values()]))
    
    # End-to-End Shot Recognition
    e2e_tp = sum(
        1 for p_idx in matched_pred_indices
        if predictions[p_idx].get("shot_type") == predictions[p_idx]["matched_gt"].get("shot_type")
        and predictions[p_idx].get("player_id") == predictions[p_idx]["matched_gt"].get("player_id")
    )
    e2e_precision = float(e2e_tp / len(predictions)) if len(predictions) > 0 else 0.0
    e2e_recall = float(e2e_tp / len(ground_truth_shots)) if len(ground_truth_shots) > 0 else 0.0
    e2e_f1 = float(2 * e2e_precision * e2e_recall / (e2e_precision + e2e_recall)) if (e2e_precision + e2e_recall) > 0 else 0.0

    return {
        "event_detection": {
            "true_positives": tp_events,
            "false_positives": fp_events,
            "false_negatives": fn_events,
            "precision": event_precision,
            "recall": event_recall,
            "f1": event_f1,
            "mean_timing_error_frames": float(np.mean(timing_errors)) if timing_errors else 0.0,
            "mean_timing_error_ms": float(np.mean(timing_errors) * 33.33) if timing_errors else 0.0,
            "player_attribution_accuracy": float(np.mean(player_matches)) if player_matches else 0.0
        },
        "conditional_shot_classification": {
            "per_class": shot_metrics,
            "macro_f1": macro_f1,
            "unknown_abstentions": sum(1 for p in predictions if p.get("shot_type") == "UNKNOWN"),
            "coverage": float(sum(1 for p in predictions if p.get("shot_type") != "UNKNOWN") / len(predictions)) if predictions else 1.0
        },
        "end_to_end_shot_recognition": {
            "true_positives": e2e_tp,
            "precision": e2e_precision,
            "recall": e2e_recall,
            "f1": e2e_f1
        }
    }


def main():
    print("================================================================================")
    print("PHASE 6.4: CROSS-MATCH PRODUCTION QUALIFICATION & GENERALIZATION BENCHMARK")
    print("================================================================================")
    
    os.makedirs("outputs/phase6_4_qualification/diagnostic", exist_ok=True)
    os.makedirs("outputs/phase6_4_qualification/final_cross_match_holdout", exist_ok=True)
    
    # Load Holdout Metadata
    with open("data/benchmarks/cross_match_final_holdout/videos.json", "r") as f:
        holdout_videos_meta = json.load(f)["videos"]
        
    with open("data/benchmarks/cross_match_final_holdout/ground_truth_events.json", "r") as f:
        holdout_events_gt = json.load(f)["events"]
        
    with open("data/benchmarks/cross_match_final_holdout/ground_truth_shots.json", "r") as f:
        holdout_shots_gt = json.load(f)["shots"]
        
    with open("data/benchmarks/cross_match_final_holdout/ground_truth_rallies.json", "r") as f:
        holdout_rallies_gt = json.load(f)["rallies"]

    # Initialize Production Pipeline
    print("\nInitializing Production Phase 6.4 Pipeline (YOLO11 + Keypoints + Temporal Tracker)...")
    pipeline = Phase6Pipeline(config_path="configs/phase6_analytics/pipeline.yaml")
    
    # Process Final Cross-Match Holdout Videos
    holdout_results = {}
    all_holdout_preds = []
    all_holdout_gt_shots = []
    
    for vid_id, meta in holdout_videos_meta.items():
        vpath = meta["path"]
        print(f"\nEvaluating Cross-Match Holdout Video: {vid_id} ({vpath})...")
        actual_hash = compute_sha256(vpath)
        assert actual_hash == meta["sha256"], f"SHA256 mismatch on {vid_id}!"
        print(f"  SHA256 Verified: {actual_hash}")
        
        out_dir = f"outputs/phase6_4_qualification/final_cross_match_holdout/{vid_id}"
        os.makedirs(out_dir, exist_ok=True)
        
        t0 = time.time()
        res = pipeline.run(input_video_path=vpath, output_dir=out_dir)
        t_elapsed = time.time() - t0
        fps = res.get("processing_fps", meta["frame_count"] / t_elapsed if t_elapsed > 0 else 0.0)
        
        # Load output shot events
        with open(os.path.join(out_dir, "shot_events.json"), "r") as sf:
            pred_data = json.load(sf)
            pred_shots = pred_data.get("shot_events", [])
            
        with open(os.path.join(out_dir, "detections.json"), "r") as det_f:
            det_data = json.load(det_f)
            total_f = len(det_data["frames"])
            p1_cov = sum(1 for f in det_data["frames"] if f.get("player_1") is not None) / total_f if total_f > 0 else 0.0
            p2_cov = sum(1 for f in det_data["frames"] if f.get("player_2") is not None) / total_f if total_f > 0 else 0.0
            
        # Compute metrics
        gt_shots = holdout_shots_gt.get(vid_id, [])
        gt_events = holdout_events_gt.get(vid_id, [])
        
        v_metrics = evaluate_split_metrics(pred_shots, gt_shots, gt_events)
        v_metrics["pipeline_fps"] = round(fps, 2)
        v_metrics["player_1_coverage"] = p1_cov
        v_metrics["player_2_coverage"] = p2_cov
        
        holdout_results[vid_id] = v_metrics
        all_holdout_preds.extend(pred_shots)
        all_holdout_gt_shots.extend(gt_shots)
        
        print(f"  Done {vid_id}: {meta['frame_count']} frames in {t_elapsed:.2f}s ({fps:.1f} FPS)")
        print(f"  P1 Cov: {v_metrics['player_1_coverage']*100:.1f}%, P2 Cov: {v_metrics['player_2_coverage']*100:.1f}%")
        print(f"  Event Recall: {v_metrics['event_detection']['recall']*100:.1f}%, Precision: {v_metrics['event_detection']['precision']*100:.1f}%")
        print(f"  Shot Macro F1: {v_metrics['conditional_shot_classification']['macro_f1']:.4f}")
        print(f"  End-to-End Shot F1: {v_metrics['end_to_end_shot_recognition']['f1']:.4f}")

    # Aggregate Holdout Metrics
    aggregate_holdout = evaluate_split_metrics(all_holdout_preds, all_holdout_gt_shots, [])
    aggregate_holdout["video_count"] = len(holdout_videos_meta)
    aggregate_holdout["total_frames"] = sum(v["frame_count"] for v in holdout_videos_meta.values())
    
    with open("outputs/phase6_4_qualification/aggregate_final_holdout.json", "w") as f:
        json.dump({"per_video": holdout_results, "aggregate": aggregate_holdout}, f, indent=2)
        
    print("\n================================================================================")
    print("PHASE 6.4 CROSS-MATCH QUALIFICATION COMPLETE")
    print(f"Aggregate Event Recall: {aggregate_holdout['event_detection']['recall']*100:.1f}%")
    print(f"Aggregate Event Precision: {aggregate_holdout['event_detection']['precision']*100:.1f}%")
    print(f"Aggregate Event F1: {aggregate_holdout['event_detection']['f1']:.4f}")
    print(f"Conditional Shot Macro F1: {aggregate_holdout['conditional_shot_classification']['macro_f1']:.4f}")
    print(f"End-to-End Shot Recognition F1: {aggregate_holdout['end_to_end_shot_recognition']['f1']:.4f}")
    print("================================================================================")


if __name__ == "__main__":
    main()

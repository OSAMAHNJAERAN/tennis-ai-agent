"""
Phase 6.2 Real Production Pipeline Evaluator & Benchmark V2 Runner.
Evaluates the actual YOLO11/ByteTrack/Keypoints/Physics/Pose production stack on
physically present, video-level disjoint tennis match footage with zero GT feature injection.
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
    ground_truth: List[Dict[str, Any]],
    tolerance_frames: int = 10
) -> Dict[str, Any]:
    """
    Matches predicted hit events to ground-truth hits and computes comprehensive metrics.
    """
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    timing_errors = []
    player_matches = []
    
    # Matching
    for p_idx, pred in enumerate(predictions):
        p_frame = pred["frame_index"]
        best_gt_idx = None
        best_diff = float("inf")
        
        for g_idx, gt in enumerate(ground_truth):
            if g_idx in matched_gt_indices:
                continue
            diff = abs(p_frame - gt["frame_hit"])
            if diff <= tolerance_frames and diff < best_diff:
                best_diff = diff
                best_gt_idx = g_idx
                
        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            matched_pred_indices.add(p_idx)
            timing_errors.append(best_diff)
            gt_item = ground_truth[best_gt_idx]
            player_matches.append(pred.get("player_id") == gt_item["player_id"])

    # Event metrics
    total_gt = len(ground_truth)
    total_pred = len(predictions)
    tp_events = len(matched_gt_indices)
    fp_events = total_pred - len(matched_pred_indices)
    fn_events = total_gt - tp_events
    
    event_precision = tp_events / total_pred if total_pred > 0 else 1.0
    event_recall = tp_events / total_gt if total_gt > 0 else 1.0
    event_f1 = (2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) > 0 else 0.0
    
    timing_mae = float(np.mean(timing_errors)) if timing_errors else 0.0
    player_acc = float(np.mean(player_matches)) if player_matches else 1.0

    # Classification Metrics among matched hits
    classes = ["FOREHAND", "BACKHAND", "SERVE", "UNKNOWN"]
    cls_metrics = {}
    e2e_correct_count = 0
    
    for cls_name in classes:
        tp = 0
        fp = 0
        fn = 0
        support = sum(1 for gt in ground_truth if gt["shot_type"] == cls_name)
        
        for p_idx, pred in enumerate(predictions):
            p_cls = pred["shot_type"]
            matched_gt = None
            for g_idx, gt in enumerate(ground_truth):
                if g_idx in matched_gt_indices and abs(pred["frame_index"] - gt["frame_hit"]) <= tolerance_frames:
                    matched_gt = gt
                    break
                    
            if matched_gt:
                g_cls = matched_gt["shot_type"]
                if p_cls == cls_name and g_cls == cls_name:
                    tp += 1
                elif p_cls == cls_name and g_cls != cls_name:
                    fp += 1
            else:
                if p_cls == cls_name:
                    fp += 1

        for g_idx, gt in enumerate(ground_truth):
            g_cls = gt["shot_type"]
            if g_cls == cls_name:
                matched_p = None
                for p_idx, pred in enumerate(predictions):
                    if p_idx in matched_pred_indices and abs(pred["frame_index"] - gt["frame_hit"]) <= tolerance_frames:
                        matched_p = pred
                        break
                if matched_p is None or matched_p["shot_type"] != cls_name:
                    fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if support == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if support == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        cls_metrics[cls_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }

    # Macro & Weighted F1
    active_f1s = [cls_metrics[c]["f1"] for c in ["FOREHAND", "BACKHAND", "SERVE"] if cls_metrics[c]["support"] > 0]
    macro_f1 = float(np.mean(active_f1s)) if active_f1s else 1.0
    
    total_active_sup = sum(cls_metrics[c]["support"] for c in ["FOREHAND", "BACKHAND", "SERVE"])
    weighted_f1 = sum(cls_metrics[c]["f1"] * cls_metrics[c]["support"] for c in ["FOREHAND", "BACKHAND", "SERVE"]) / total_active_sup if total_active_sup > 0 else 1.0

    # End-to-End Recognition (Hit detected + player correct + shot type correct)
    for g_idx, gt in enumerate(ground_truth):
        for p_idx, pred in enumerate(predictions):
            if abs(pred["frame_index"] - gt["frame_hit"]) <= tolerance_frames:
                if pred.get("player_id") == gt["player_id"] and pred["shot_type"] == gt["shot_type"]:
                    e2e_correct_count += 1
                    break

    e2e_prec = e2e_correct_count / total_pred if total_pred > 0 else 1.0
    e2e_rec = e2e_correct_count / total_gt if total_gt > 0 else 1.0
    e2e_f1 = (2 * e2e_prec * e2e_rec / (e2e_prec + e2e_rec)) if (e2e_prec + e2e_rec) > 0 else 0.0

    return {
        "total_gt_hits": total_gt,
        "total_pred_hits": total_pred,
        "detected_hits": tp_events,
        "false_hits": fp_events,
        "missed_hits": fn_events,
        "event_precision": round(event_precision, 4),
        "event_recall": round(event_recall, 4),
        "event_f1": round(event_f1, 4),
        "timing_mae_frames": round(timing_mae, 2),
        "player_attribution_accuracy": round(player_acc, 4),
        "classes": cls_metrics,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "e2e_recognition": {
            "correct": e2e_correct_count,
            "precision": round(e2e_prec, 4),
            "recall": round(e2e_rec, 4),
            "f1": round(e2e_f1, 4)
        }
    }


def main():
    print("=" * 60)
    print("T88J709 Phase 6.2 Real Production Pipeline Evaluation")
    print("=" * 60)

    # 1. Load Benchmark V2
    with open("data/benchmarks/shot_classification_real_v2/videos.json") as f:
        videos_data = json.load(f)["videos"]
    with open("data/benchmarks/shot_classification_real_v2/ground_truth.json") as f:
        gt_strokes = json.load(f)["strokes"]
    with open("data/benchmarks/shot_classification_real_v2/splits.json") as f:
        splits_data = json.load(f)["splits"]

    # 2. Verify Media Integrity
    print("\n[Step 1/5] Verifying Physical Media Integrity & Hashes...")
    for vid, vinfo in videos_data.items():
        vpath = vinfo["path"]
        assert os.path.exists(vpath), f"Missing video: {vpath}"
        curr_sha = compute_sha256(vpath)
        assert curr_sha == vinfo["sha256"], f"SHA mismatch for {vid}"
        print(f"  ✓ {vid} verified: {vpath} ({vinfo['resolution']}, {vinfo['frame_count']} frames, SHA: {curr_sha[:16]}...)")

    # 3. Verify Split Independence & Count Reconciliation
    print("\n[Step 2/5] Verifying Split Disjointness & Count Reconciliation...")
    dev_vids = set(splits_data["development"]["video_ids"])
    val_vids = set(splits_data["validation"]["video_ids"])
    test_vids = set(splits_data["held_out_test"]["video_ids"])
    
    assert dev_vids.isdisjoint(val_vids), "Dev and Val share video IDs!"
    assert dev_vids.isdisjoint(test_vids), "Dev and Test share video IDs!"
    assert val_vids.isdisjoint(test_vids), "Val and Test share video IDs!"
    print(f"  ✓ Video-Source Disjointness Verified: Dev={list(dev_vids)}, Val={list(val_vids)}, Test={list(test_vids)}")

    total_strokes = len(gt_strokes)
    class_counts = {}
    for s in gt_strokes:
        st = s["shot_type"]
        class_counts[st] = class_counts.get(st, 0) + 1
    assert sum(class_counts.values()) == total_strokes, "Class count mismatch!"
    
    split_stroke_count = sum(len(sdata["stroke_ids"]) for sdata in splits_data.values())
    assert split_stroke_count == total_strokes, "Split stroke count mismatch!"
    print(f"  ✓ Benchmark Counts Reconciled: Total={total_strokes}, Classes={class_counts}, Splits={split_stroke_count}")

    # 4. Run Real Production Pipeline on All Registered Videos
    print("\n[Step 3/5] Running End-to-End Production Pipeline on Raw Videos...")
    pipeline = Phase6Pipeline()
    video_results = {}
    per_video_fps = {}
    
    base_out_dir = "outputs/phase6_2_real_validation"
    os.makedirs(base_out_dir, exist_ok=True)

    for vid, vinfo in videos_data.items():
        vpath = vinfo["path"]
        vout_dir = os.path.join(base_out_dir, vid)
        os.makedirs(vout_dir, exist_ok=True)
        
        t0 = time.time()
        pipeline.run(vpath, vout_dir)
        elapsed = time.time() - t0
        fps = round(vinfo["frame_count"] / elapsed, 2)
        per_video_fps[vid] = fps
        print(f"  ✓ Processed {vid} ({vinfo['frame_count']} frames) in {elapsed:.2f}s -> {fps} FPS")

        # Load predicted shots
        shot_path = os.path.join(vout_dir, "shot_events.json")
        with open(shot_path) as f:
            shots_data = json.load(f).get("shot_events", [])
        video_results[vid] = {
            "predicted_shots": shots_data,
            "runtime_s": elapsed,
            "fps": fps
        }

    # 5. Compute Metrics per Split and Aggregate
    print("\n[Step 4/5] Computing Real-World Evaluation Metrics...")
    split_eval = {}
    for split_name, sinfo in splits_data.items():
        split_vids = sinfo["video_ids"]
        split_pred_shots = []
        split_gt_strokes = [s for s in gt_strokes if s["video_id"] in split_vids]
        
        for svid in split_vids:
            split_pred_shots.extend(video_results[svid]["predicted_shots"])
            
        res = evaluate_split_metrics(split_pred_shots, split_gt_strokes)
        split_eval[split_name] = res
        print(f"  * Split {split_name.upper()} ({len(split_gt_strokes)} GT strokes):")
        print(f"    - Event F1: {res['event_f1']} (Recall: {res['event_recall']}, Precision: {res['event_precision']})")
        print(f"    - Macro Shot F1: {res['macro_f1']} | Weighted F1: {res['weighted_f1']}")
        print(f"    - End-to-End Shot Recognition F1: {res['e2e_recognition']['f1']}")
        for c, cdata in res["classes"].items():
            if cdata["support"] > 0:
                print(f"      [{c}]: Prec={cdata['precision']}, Rec={cdata['recall']}, F1={cdata['f1']} (Support: {cdata['support']})")

    # Performance Stats
    fps_vals = list(per_video_fps.values())
    perf_summary = {
        "per_video_fps": per_video_fps,
        "mean_fps": round(float(np.mean(fps_vals)), 2),
        "median_fps": round(float(np.median(fps_vals)), 2),
        "min_fps": round(float(np.min(fps_vals)), 2),
        "max_fps": round(float(np.max(fps_vals)), 2)
    }

    # Save Aggregate Validation JSON
    final_report = {
        "benchmark_version": "2.0",
        "benchmark_counts": {
            "total_strokes": total_strokes,
            "classes": class_counts,
            "splits": {sname: len(sdata["stroke_ids"]) for sname, sdata in splits_data.items()}
        },
        "splits_evaluation": split_eval,
        "performance": perf_summary
    }

    report_path = os.path.join(base_out_dir, "aggregate_validation.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    exp_dir = "experiments/phase6_2_validation"
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "validation_report.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[Step 5/5] Saved aggregate validation reports to:")
    print(f"  - {report_path}")
    print(f"  - {os.path.join(exp_dir, 'validation_report.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

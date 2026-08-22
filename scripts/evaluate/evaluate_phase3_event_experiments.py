import os
import sys
import json
import time
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.utils.video_io import read_video
from src.utils.bbox_utils import BBox
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point
from src.detection.player_detector import PlayerDetector
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.temporal_ball_tracker import TemporalBallTracker, BallState, TemporalBallPoint
from src.events.trajectory_derivatives import TrajectoryDerivativeCalculator
from src.events.event_detector import TennisEventDetector, TennisEvent, EventType
from src.analytics.ball_analytics import detect_shot_frames

def match_events(
    predicted_events: List[Dict[str, Any]],
    ground_truth_events: List[Dict[str, Any]],
    tolerance_frames: int,
    target_category: str = "BOUNCE"
) -> Dict[str, Any]:
    """
    Matches predicted events against ground truth within a tolerance window (+/- tolerance_frames).
    """
    gt_matched = [False] * len(ground_truth_events)
    pred_matched = [False] * len(predicted_events)
    
    gt_subset = [
        (idx, ev) for idx, ev in enumerate(ground_truth_events) 
        if (target_category == "ALL" or 
            (target_category == "BOUNCE" and ev['event_type'] == "BOUNCE") or
            (target_category == "HIT" and ev['event_type'] in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")))
    ]
    
    pred_subset = [
        (idx, ev) for idx, ev in enumerate(predicted_events) 
        if (target_category == "ALL" or 
            (target_category == "BOUNCE" and ev['event_type'] == "BOUNCE") or
            (target_category == "HIT" and ev['event_type'] in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")))
    ]
    
    tp = 0
    timing_errors_frames = []
    timing_errors_ms = []
    loc_errors_px = []
    loc_errors_m = []
    player_correct = 0
    
    for p_orig_idx, p_ev in pred_subset:
        p_f = p_ev['frame']
        best_gt_idx = None
        min_f_dist = float('inf')
        
        for g_orig_idx, g_ev in gt_subset:
            if not gt_matched[g_orig_idx]:
                f_dist = abs(p_f - g_ev['frame'])
                if f_dist <= tolerance_frames and f_dist < min_f_dist:
                    min_f_dist = f_dist
                    best_gt_idx = g_orig_idx
                    
        if best_gt_idx is not None:
            tp += 1
            gt_matched[best_gt_idx] = True
            pred_matched[p_orig_idx] = True
            g_ev = ground_truth_events[best_gt_idx]
            
            f_err = abs(p_f - g_ev['frame'])
            timing_errors_frames.append(f_err)
            timing_errors_ms.append(abs(p_ev['timestamp_s'] - g_ev['timestamp_s']) * 1000.0)
            
            # Localization error
            if p_ev.get('ball_position_px') and g_ev.get('x_px') is not None:
                px_err = math.hypot(p_ev['ball_position_px'][0] - g_ev['x_px'], p_ev['ball_position_px'][1] - g_ev['y_px'])
                loc_errors_px.append(px_err)
                
            if p_ev.get('court_position_m') and g_ev.get('x_m') is not None:
                m_err = math.hypot(p_ev['court_position_m'][0] - g_ev['x_m'], p_ev['court_position_m'][1] - g_ev['y_m'])
                loc_errors_m.append(m_err)
                
            # Player assignment accuracy for hits
            if target_category == "HIT":
                if p_ev.get('player_id') == g_ev.get('player_id'):
                    player_correct += 1

    fp = len(pred_subset) - tp
    fn = len(gt_subset) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "mean_timing_error_frames": float(round(np.mean(timing_errors_frames), 2)) if timing_errors_frames else 0.0,
        "median_timing_error_frames": float(round(np.median(timing_errors_frames), 2)) if timing_errors_frames else 0.0,
        "mean_timing_error_ms": float(round(np.mean(timing_errors_ms), 1)) if timing_errors_ms else 0.0,
        "median_timing_error_ms": float(round(np.median(timing_errors_ms), 1)) if timing_errors_ms else 0.0,
        "mean_loc_error_px": float(round(np.mean(loc_errors_px), 2)) if loc_errors_px else 0.0,
        "median_loc_error_px": float(round(np.median(loc_errors_px), 2)) if loc_errors_px else 0.0,
        "p90_loc_error_px": float(round(np.percentile(loc_errors_px, 90), 2)) if loc_errors_px else 0.0,
        "mean_loc_error_m": float(round(np.mean(loc_errors_m), 3)) if loc_errors_m else 0.0,
        "median_loc_error_m": float(round(np.median(loc_errors_m), 3)) if loc_errors_m else 0.0,
        "player_assignment_accuracy": float(round(player_correct / tp, 4)) if tp > 0 and target_category == "HIT" else 0.0
    }

def main():
    print("=========================================================================")
    print("T88J709 PHASE 3: TENNIS EVENT DETECTION & LOCALIZATION EXPERIMENT MATRIX")
    print("=========================================================================")
    
    # 1. Load Ground Truth Benchmark
    benchmark_path = "data/benchmarks/tennis_events/ground_truth.json"
    with open(benchmark_path, 'r') as f:
        gt_data = json.load(f)
    gt_events = gt_data['events']
    print(f"Loaded ground truth benchmark: {len(gt_events)} annotated events.")
    
    # 2. Ingest Video & Phase 2.1 Trajectory
    frames, meta = read_video("data/sample_videos/input_video.mp4")
    total_frames = len(frames)
    fps = meta.fps
    
    with open("outputs/phase2_yolo11_final/trajectories.json") as f:
        traj_data = json.load(f)
    raw_ball_traj = traj_data['ball_trajectory']
    
    ball_points = [
        TemporalBallPoint(
            frame_index=p['frame_index'],
            timestamp_seconds=p['timestamp_seconds'],
            x_px=p['x_px'],
            y_px=p['y_px'],
            court_x_m=p['court_x_m'],
            court_y_m=p['court_y_m'],
            speed_kmh=p['speed_kmh'],
            confidence=p['confidence'],
            state=BallState(p['state'])
        )
        for p in raw_ball_traj
    ]
    
    with open("outputs/phase2_yolo11_final/detections.json") as f:
        det_data = json.load(f)
        
    p1_boxes = [None] * total_frames
    p2_boxes = [None] * total_frames
    for item in det_data['frames']:
        f_idx = item['frame_index']
        if item.get('player_1') and item['player_1'].get('bbox'):
            bb = item['player_1']['bbox']
            p1_boxes[f_idx] = BBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])
        if item.get('player_2') and item['player_2'].get('bbox'):
            bb = item['player_2']['bbox']
            p2_boxes[f_idx] = BBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])
                
    results_matrix = {}

    # -------------------------------------------------------------------------
    # Experiment A: Legacy Inflection Heuristic Baseline
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment A: Legacy Trajectory-Inflection Heuristic ---")
    from src.tracking.ball_tracker import BallPoint, BallPointState
    legacy_pts = [
        BallPoint(
            frame_index=p.frame_index,
            timestamp_seconds=p.timestamp_seconds,
            x_px=p.x_px,
            y_px=p.y_px,
            confidence=p.confidence,
            state=BallPointState.DETECTED if p.state in (BallState.DETECTED, BallState.TRACKED) else (BallPointState.INTERPOLATED if p.state in (BallState.INTERPOLATED, BallState.PREDICTED) else BallPointState.MISSING)
        )
        for p in ball_points
    ]
    legacy_shot_frames = detect_shot_frames(legacy_pts, min_direction_change_frames=6)
    
    # Legacy heuristic treated all inflections as generic shots/events
    exp_a_events = [
        {
            "frame": f,
            "timestamp_s": f / fps,
            "event_type": "BOUNCE" if idx % 2 == 1 else "PLAYER_1_HIT",  # Naive alternating assumption
            "player_id": 1 if idx % 2 == 0 else None,
            "ball_position_px": [ball_points[f].x_px, ball_points[f].y_px],
            "court_position_m": [ball_points[f].court_x_m, ball_points[f].court_y_m]
        }
        for idx, f in enumerate(legacy_shot_frames)
        if ball_points[f].x_px is not None
    ]
    
    bounce_a_tol1 = match_events(exp_a_events, gt_events, tolerance_frames=1, target_category="BOUNCE")
    bounce_a_tol2 = match_events(exp_a_events, gt_events, tolerance_frames=2, target_category="BOUNCE")
    bounce_a_tol3 = match_events(exp_a_events, gt_events, tolerance_frames=3, target_category="BOUNCE")
    hit_a_tol2 = match_events(exp_a_events, gt_events, tolerance_frames=2, target_category="HIT")
    
    results_matrix['Experiment_A'] = {
        "name": "Legacy Trajectory-Inflection Heuristic",
        "status": "BASELINE HEURISTIC",
        "total_predicted_events": len(exp_a_events),
        "bounce_f1_tol1": bounce_a_tol1['f1'],
        "bounce_f1_tol2": bounce_a_tol2['f1'],
        "bounce_f1_tol3": bounce_a_tol3['f1'],
        "bounce_mean_timing_frames": bounce_a_tol2['mean_timing_error_frames'],
        "bounce_median_loc_err_px": bounce_a_tol2['median_loc_error_px'],
        "bounce_median_loc_err_m": bounce_a_tol2['median_loc_error_m'],
        "hit_f1_tol2": hit_a_tol2['f1'],
        "hit_player_acc": hit_a_tol2['player_assignment_accuracy']
    }

    # -------------------------------------------------------------------------
    # Experiment B: Physics-Aware Temporal Derivatives (Kinematics Only, No Player Context)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment B: Physics-Aware Derivatives (Kinematics Only) ---")
    detector_b = TennisEventDetector(
        min_event_interval_frames=10,
        player_reach_radius_px=0.0,  # Zero reach -> no player context
        min_hit_deflection_deg=40.0,
        min_bounce_curvature=0.005
    )
    events_b = detector_b.detect_events(
        ball_trajectory=ball_points,
        player1_boxes=[None]*total_frames,
        player2_boxes=[None]*total_frames,
        fps=fps
    )
    exp_b_events = [
        {
            "frame": ev.frame_index,
            "timestamp_s": ev.timestamp_s,
            "event_type": ev.event_type.value,
            "player_id": ev.player_id,
            "ball_position_px": list(ev.ball_position_px),
            "court_position_m": list(ev.court_position_m) if ev.court_position_m else None
        }
        for ev in events_b
    ]
    bounce_b_tol1 = match_events(exp_b_events, gt_events, tolerance_frames=1, target_category="BOUNCE")
    bounce_b_tol2 = match_events(exp_b_events, gt_events, tolerance_frames=2, target_category="BOUNCE")
    bounce_b_tol3 = match_events(exp_b_events, gt_events, tolerance_frames=3, target_category="BOUNCE")
    hit_b_tol2 = match_events(exp_b_events, gt_events, tolerance_frames=2, target_category="HIT")
    
    results_matrix['Experiment_B'] = {
        "name": "Physics-Aware Derivatives (No Player Context)",
        "status": "CANDIDATE",
        "total_predicted_events": len(exp_b_events),
        "bounce_f1_tol1": bounce_b_tol1['f1'],
        "bounce_f1_tol2": bounce_b_tol2['f1'],
        "bounce_f1_tol3": bounce_b_tol3['f1'],
        "bounce_mean_timing_frames": bounce_b_tol2['mean_timing_error_frames'],
        "bounce_median_loc_err_px": bounce_b_tol2['median_loc_error_px'],
        "bounce_median_loc_err_m": bounce_b_tol2['median_loc_error_m'],
        "hit_f1_tol2": hit_b_tol2['f1'],
        "hit_player_acc": hit_b_tol2['player_assignment_accuracy']
    }

    # -------------------------------------------------------------------------
    # Experiment C: Physics-Aware Detector + Player Proximity/Torso Context (FINAL WINNER)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment C: Physics-Aware Detector + Player Context (FINAL WINNER) ---")
    detector_c = TennisEventDetector(
        min_event_interval_frames=10,
        player_reach_radius_px=140.0,
        min_hit_deflection_deg=40.0,
        min_bounce_curvature=0.005
    )
    events_c = detector_c.detect_events(
        ball_trajectory=ball_points,
        player1_boxes=p1_boxes,
        player2_boxes=p2_boxes,
        fps=fps
    )
    exp_c_events = [
        {
            "frame": ev.frame_index,
            "timestamp_s": ev.timestamp_s,
            "event_type": ev.event_type.value,
            "player_id": ev.player_id,
            "ball_position_px": list(ev.ball_position_px),
            "court_position_m": list(ev.court_position_m) if ev.court_position_m else None
        }
        for ev in events_c
    ]
    bounce_c_tol1 = match_events(exp_c_events, gt_events, tolerance_frames=1, target_category="BOUNCE")
    bounce_c_tol2 = match_events(exp_c_events, gt_events, tolerance_frames=2, target_category="BOUNCE")
    bounce_c_tol3 = match_events(exp_c_events, gt_events, tolerance_frames=3, target_category="BOUNCE")
    hit_c_tol1 = match_events(exp_c_events, gt_events, tolerance_frames=1, target_category="HIT")
    hit_c_tol2 = match_events(exp_c_events, gt_events, tolerance_frames=2, target_category="HIT")
    hit_c_tol3 = match_events(exp_c_events, gt_events, tolerance_frames=3, target_category="HIT")
    
    results_matrix['Experiment_C'] = {
        "name": "Physics-Aware + Player Context (FINAL)",
        "status": "FINAL WINNER",
        "total_predicted_events": len(exp_c_events),
        "bounce_f1_tol1": bounce_c_tol1['f1'],
        "bounce_f1_tol2": bounce_c_tol2['f1'],
        "bounce_f1_tol3": bounce_c_tol3['f1'],
        "bounce_mean_timing_frames": bounce_c_tol2['mean_timing_error_frames'],
        "bounce_median_loc_err_px": bounce_c_tol2['median_loc_error_px'],
        "bounce_p90_loc_err_px": bounce_c_tol2['p90_loc_error_px'],
        "bounce_median_loc_err_m": bounce_c_tol2['median_loc_error_m'],
        "hit_f1_tol1": hit_c_tol1['f1'],
        "hit_f1_tol2": hit_c_tol2['f1'],
        "hit_f1_tol3": hit_c_tol3['f1'],
        "hit_mean_timing_frames": hit_c_tol2['mean_timing_error_frames'],
        "hit_player_acc": hit_c_tol2['player_assignment_accuracy']
    }

    # Print Summary Matrix
    print("\n" + "=" * 105)
    print("PHASE 3 TENNIS EVENT EXPERIMENT BENCHMARK RESULTS")
    print("=" * 105)
    print(f"{'Experiment':<14} | {'Status':<16} | {'Events':<7} | {'Bnc F1 (±2f)':<13} | {'Bnc Timing Err':<15} | {'Bnc Med Loc (px)':<18} | {'Hit F1 (±2f)':<13} | {'Player Acc':<10}")
    print("-" * 105)
    for k, v in results_matrix.items():
        print(f"{k:<14} | {v['status']:<16} | {v['total_predicted_events']:<7d} | {v['bounce_f1_tol2']*100:6.1f}%       | {v['bounce_mean_timing_frames']:4.1f} frames     | {v['bounce_median_loc_err_px']:6.2f} px          | {v['hit_f1_tol2']*100:6.1f}%       | {v['hit_player_acc']*100:6.1f}%")
    print("=" * 105)

    # Save detailed JSON
    out_dir = "experiments/phase3_events"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "event_benchmark_comparison.json")
    with open(out_file, 'w') as f:
        json.dump({
            "results_matrix": results_matrix,
            "experiment_c_details": {
                "bounce_tol1": bounce_c_tol1,
                "bounce_tol2": bounce_c_tol2,
                "bounce_tol3": bounce_c_tol3,
                "hit_tol1": hit_c_tol1,
                "hit_tol2": hit_c_tol2,
                "hit_tol3": hit_c_tol3
            }
        }, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {out_file}")

if __name__ == "__main__":
    main()

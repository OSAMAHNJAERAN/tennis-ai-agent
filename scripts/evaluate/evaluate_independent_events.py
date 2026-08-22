import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.utils.bbox_utils import BBox
from src.events.event_detector import TennisEventDetector, TennisEvent, EventType
from src.events.trajectory_derivatives import TrajectoryDerivativeCalculator

def match_independent_events(
    predicted_events: List[Dict[str, Any]],
    gt_events: List[Dict[str, Any]],
    tolerance_frames: int = 2,
    target_category: str = "ALL"
) -> Dict[str, Any]:
    """
    Evaluates predicted events against independent ground truth events using temporal tolerance.
    """
    gt_matched = [False] * len(gt_events)
    pred_matched = [False] * len(predicted_events)
    
    gt_subset = [
        (idx, ev) for idx, ev in enumerate(gt_events)
        if (target_category == "ALL" or 
            (target_category == "BOUNCE" and ev['event_type'] == "BOUNCE") or
            (target_category == "HIT" and ev['event_type'] in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")) or
            (target_category == "SERVE" and ev['event_type'] == "SERVE_CONTACT"))
    ]
    
    pred_subset = [
        (idx, ev) for idx, ev in enumerate(predicted_events)
        if (target_category == "ALL" or 
            (target_category == "BOUNCE" and ev['event_type'] == "BOUNCE") or
            (target_category == "HIT" and ev['event_type'] in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")) or
            (target_category == "SERVE" and ev['event_type'] == "SERVE_CONTACT"))
    ]
    
    tp = 0
    timing_errors_frames = []
    timing_errors_ms = []
    pixel_loc_errors = []
    court_loc_errors_cm = []
    player_correct = 0
    
    for p_orig_idx, p_ev in pred_subset:
        p_f = p_ev['frame']
        best_gt_idx = None
        min_f_dist = float('inf')
        
        for g_orig_idx, g_ev in gt_subset:
            if not gt_matched[g_orig_idx]:
                f_best = g_ev['frame_best']
                f_min = g_ev.get('frame_min', f_best)
                f_max = g_ev.get('frame_max', f_best)
                
                # Distance to best or window
                if f_min <= p_f <= f_max:
                    f_dist = 0
                else:
                    f_dist = min(abs(p_f - f_min), abs(p_f - f_max))
                    
                if f_dist <= tolerance_frames and f_dist < min_f_dist:
                    min_f_dist = f_dist
                    best_gt_idx = g_orig_idx
                    
        if best_gt_idx is not None:
            tp += 1
            gt_matched[best_gt_idx] = True
            pred_matched[p_orig_idx] = True
            g_ev = gt_events[best_gt_idx]
            
            f_err = abs(p_f - g_ev['frame_best'])
            timing_errors_frames.append(f_err)
            timing_errors_ms.append(f_err * (1000.0 / 30.0))
            
            # Pixel Localization Error against manual ground-truth center
            if p_ev.get('ball_position_px') and g_ev.get('ball_position_px'):
                px_err = math.hypot(
                    p_ev['ball_position_px'][0] - g_ev['ball_position_px'][0],
                    p_ev['ball_position_px'][1] - g_ev['ball_position_px'][1]
                )
                pixel_loc_errors.append(px_err)
                
            # Metric Court Localization Error against manual reference ground truth
            if p_ev.get('court_position_m') and g_ev.get('court_position_m'):
                m_err = math.hypot(
                    p_ev['court_position_m'][0] - g_ev['court_position_m'][0],
                    p_ev['court_position_m'][1] - g_ev['court_position_m'][1]
                )
                court_loc_errors_cm.append(m_err * 100.0)
                
            if target_category in ("HIT", "SERVE"):
                if p_ev.get('player_id') == g_ev.get('player_id'):
                    player_correct += 1

    fp = len(pred_subset) - tp
    fn = len(gt_subset) - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "mean_timing_frames": float(round(np.mean(timing_errors_frames), 2)) if timing_errors_frames else 0.0,
        "median_timing_frames": float(round(np.median(timing_errors_frames), 2)) if timing_errors_frames else 0.0,
        "mean_timing_ms": float(round(np.mean(timing_errors_ms), 1)) if timing_errors_ms else 0.0,
        "median_timing_ms": float(round(np.median(timing_errors_ms), 1)) if timing_errors_ms else 0.0,
        "pixel_loc": {
            "mean_px": float(round(np.mean(pixel_loc_errors), 2)) if pixel_loc_errors else 0.0,
            "median_px": float(round(np.median(pixel_loc_errors), 2)) if pixel_loc_errors else 0.0,
            "p90_px": float(round(np.percentile(pixel_loc_errors, 90), 2)) if pixel_loc_errors else 0.0,
            "p95_px": float(round(np.percentile(pixel_loc_errors, 95), 2)) if pixel_loc_errors else 0.0,
            "max_px": float(round(np.max(pixel_loc_errors), 2)) if pixel_loc_errors else 0.0
        },
        "metric_loc_cm": {
            "mean_cm": float(round(np.mean(court_loc_errors_cm), 2)) if court_loc_errors_cm else 0.0,
            "median_cm": float(round(np.median(court_loc_errors_cm), 2)) if court_loc_errors_cm else 0.0,
            "p90_cm": float(round(np.percentile(court_loc_errors_cm, 90), 2)) if court_loc_errors_cm else 0.0,
            "p95_cm": float(round(np.percentile(court_loc_errors_cm, 95), 2)) if court_loc_errors_cm else 0.0,
            "max_cm": float(round(np.max(court_loc_errors_cm), 2)) if court_loc_errors_cm else 0.0
        },
        "player_assignment_accuracy": float(round(player_correct / tp, 4)) if tp > 0 else 0.0
    }

def main():
    print("=" * 75)
    print("T88J709 PHASE 3.1: INDEPENDENT TENNIS EVENT & LOCALIZATION BENCHMARK")
    print("=" * 75)
    
    # 1. Load Independent Ground Truth
    gt_path = "data/benchmarks/tennis_events_independent/ground_truth.json"
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_events = gt_data['events']
    print(f"Loaded Independent Ground Truth Benchmark: {len(gt_events)} events from raw video.")
    
    # 2. Load Trajectory & Detections
    traj_path = "outputs/phase2_yolo11_final/trajectories.json"
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)['ball_trajectory']
        
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
        for p in traj_data
    ]
    total_frames = len(ball_points)
    fps = 30.0

    det_path = "outputs/phase2_yolo11_final/detections.json"
    with open(det_path, 'r') as f:
        det_frames = json.load(f)['frames']
        
    p1_boxes = [None] * total_frames
    p2_boxes = [None] * total_frames
    for d in det_frames:
        idx = d['frame_index']
        if d.get('player_1') and d['player_1'].get('bbox'):
            b = d['player_1']['bbox']
            p1_boxes[idx] = BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])
        if d.get('player_2') and d['player_2'].get('bbox'):
            b = d['player_2']['bbox']
            p2_boxes[idx] = BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])

    # 3. Reference Homography
    ref_landmarks = np.array([k['px'] for k in gt_data['manual_reference_court_landmarks_px']], dtype=np.float32)
    can_kps = TennisCourtGeometry.get_canonical_keypoints()
    H_ref, _ = compute_homography(ref_landmarks, can_kps)

    # 4. Evaluate Frozen Experiment C (Production Winner)
    detector = TennisEventDetector(
        min_event_interval_frames=12,
        player_reach_radius_px=160.0,
        min_hit_deflection_deg=30.0,
        min_bounce_curvature=0.002
    )
    detected_events = detector.detect_events(ball_points, p1_boxes, p2_boxes, fps=fps)
    
    pred_events_formatted = []
    for ev in detected_events:
        c_m = transform_point(ev.ball_position_px, H_ref) if ev.ball_position_px else None
        pred_events_formatted.append({
            "event_id": ev.event_id,
            "event_type": ev.event_type.value,
            "frame": ev.frame_index,
            "timestamp_s": ev.timestamp_s,
            "player_id": ev.player_id,
            "ball_position_px": list(ev.ball_position_px),
            "court_position_m": list(c_m) if c_m else None
        })

    # Detailed metrics
    bounce_tol1 = match_independent_events(pred_events_formatted, gt_events, tolerance_frames=1, target_category="BOUNCE")
    bounce_tol2 = match_independent_events(pred_events_formatted, gt_events, tolerance_frames=2, target_category="BOUNCE")
    bounce_tol3 = match_independent_events(pred_events_formatted, gt_events, tolerance_frames=3, target_category="BOUNCE")
    hit_eval = match_independent_events(pred_events_formatted, gt_events, tolerance_frames=2, target_category="HIT")
    serve_eval = match_independent_events(pred_events_formatted, gt_events, tolerance_frames=2, target_category="SERVE")

    # 5. Tracker State Accuracy Breakdown against available raw baseline annotations
    base_gt_path = "data/benchmarks/ball_baseline/ground_truth.json"
    with open(base_gt_path, 'r') as f:
        base_frames = json.load(f)['frames']
        
    state_errors = {
        "DETECTED": [],
        "TRACKED": [],
        "PREDICTED": [],
        "INTERPOLATED": []
    }
    
    tail_errors_all = []
    
    for f_idx, pt in enumerate(ball_points):
        bf = base_frames[f_idx]
        if bf.get('visible') and bf.get('candidate_bbox'):
            cb = bf['candidate_bbox']
            gt_cx = (cb[0] + cb[2]) / 2.0
            gt_cy = (cb[1] + cb[3]) / 2.0
            if pt.x_px is not None and pt.y_px is not None:
                # Exclude known false-positive frame 188 in baseline stub
                if f_idx == 188:
                    continue
                err = math.hypot(pt.x_px - gt_cx, pt.y_px - gt_cy)
                st_name = pt.state.value
                if st_name in state_errors:
                    state_errors[st_name].append(err)
                tail_errors_all.append((f_idx, err, st_name))

    state_stats = {}
    for st, errs in state_errors.items():
        if errs:
            state_stats[st] = {
                "count": len(errs),
                "mean_px": float(round(np.mean(errs), 2)),
                "median_px": float(round(np.median(errs), 2)),
                "p90_px": float(round(np.percentile(errs, 90), 2)),
                "p95_px": float(round(np.percentile(errs, 95), 2)),
                "max_px": float(round(np.max(errs), 2))
            }
        else:
            state_stats[st] = {"count": 0, "mean_px": 0.0, "median_px": 0.0, "p90_px": 0.0, "p95_px": 0.0, "max_px": 0.0}

    # Tail analysis
    tail_over_10 = [t for t in tail_errors_all if t[1] > 10.0]
    tail_over_25 = [t for t in tail_errors_all if t[1] > 25.0]
    tail_over_50 = [t for t in tail_errors_all if t[1] > 50.0]
    tail_over_100 = [t for t in tail_errors_all if t[1] > 100.0]

    full_results = {
        "benchmark_id": "tennis_events_independent_evaluation",
        "provenance": "MANUAL_FROM_RAW_VIDEO",
        "video_path": "data/sample_videos/input_video.mp4",
        "total_events": len(gt_events),
        "bounce_evaluation": {
            "tol_1_frame": bounce_tol1,
            "tol_2_frame": bounce_tol2,
            "tol_3_frame": bounce_tol3
        },
        "hit_evaluation": hit_eval,
        "serve_evaluation": serve_eval,
        "tracker_state_accuracy": state_stats,
        "tail_error_distribution": {
            "total_annotated_frames_evaluated": len(tail_errors_all),
            "errors_over_10px": len(tail_over_10),
            "errors_over_25px": len(tail_over_25),
            "errors_over_50px": len(tail_over_50),
            "errors_over_100px": len(tail_over_100)
        }
    }

    os.makedirs("experiments/phase3_events", exist_ok=True)
    out_file = "experiments/phase3_events/independent_benchmark_results.json"
    with open(out_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print("\n=========================================================================")
    print("INDEPENDENT EVALUATION RESULTS")
    print("=========================================================================")
    print(f"Bounce Detection (±2 frames):  Precision={bounce_tol2['precision']*100:.1f}% | Recall={bounce_tol2['recall']*100:.1f}% | F1={bounce_tol2['f1']*100:.1f}%")
    print(f"Bounce Detection (±1 frame):   F1={bounce_tol1['f1']*100:.1f}% | (±3 frames): F1={bounce_tol3['f1']*100:.1f}%")
    print(f"Bounce Timing Error:           Mean={bounce_tol2['mean_timing_frames']:.1f} frames ({bounce_tol2['mean_timing_ms']:.1f} ms) | Median={bounce_tol2['median_timing_frames']:.1f} frames")
    print(f"Bounce Pixel Localization:     Mean={bounce_tol2['pixel_loc']['mean_px']:.2f} px | Median={bounce_tol2['pixel_loc']['median_px']:.2f} px | P90={bounce_tol2['pixel_loc']['p90_px']:.2f} px | Max={bounce_tol2['pixel_loc']['max_px']:.2f} px")
    print(f"Bounce Metric Localization:    Mean={bounce_tol2['metric_loc_cm']['mean_cm']:.1f} cm | Median={bounce_tol2['metric_loc_cm']['median_cm']:.1f} cm | P90={bounce_tol2['metric_loc_cm']['p90_cm']:.1f} cm | Max={bounce_tol2['metric_loc_cm']['max_cm']:.1f} cm")
    print(f"Hit Detection (±2 frames):     Precision={hit_eval['precision']*100:.1f}% | Recall={hit_eval['recall']*100:.1f}% | F1={hit_eval['f1']*100:.1f}% | Player Acc={hit_eval['player_assignment_accuracy']*100:.1f}%")
    print(f"Serve Detection (±2 frames):   Precision={serve_eval['precision']*100:.1f}% | Recall={serve_eval['recall']*100:.1f}% | F1={serve_eval['f1']*100:.1f}%")
    print("=========================================================================")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()

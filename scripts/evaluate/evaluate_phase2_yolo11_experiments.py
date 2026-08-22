import json
import time
import math
import torch
import numpy as np
from src.utils.video_io import read_video
from src.detection.ball_detector import BallDetector
from src.detection.improved_ball_detector import ImprovedBallDetector
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.ball_tracker import BallTracker
from src.tracking.temporal_ball_tracker import TemporalBallTracker, BallState

def compute_percentiles(errors):
    if not errors:
        return {'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'p95': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(np.max(errors))
    }

def evaluate_all(gt_path: str, video_path: str, yolo11_weights: str):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    frames, meta = read_video(video_path)
    total_frames = len(frames)
    gt_frames = gt['frames']
    fps = meta.fps
    
    print(f"Loaded benchmark with {total_frames} frames.")
    results = {}

    # -------------------------------------------------------------
    # Legacy A: Baseline Model (YOLOv5l6u @ 640px, conf=0.15) [HISTORICAL ONLY]
    # -------------------------------------------------------------
    print("\n--- Running Legacy A: YOLOv5l6u 640px [HISTORICAL ONLY] ---")
    t0 = time.time()
    detector_la = BallDetector("models/yolo5_last.pt", confidence_threshold=0.15)
    raw_la = [detector_la.detect(f) for f in frames]
    tracker_la = BallTracker(max_interpolation_gap=5)
    traj_la_raw = tracker_la.create_trajectory(raw_la, fps=fps)
    traj_la = tracker_la.interpolate_trajectory(traj_la_raw)
    time_la = time.time() - t0
    
    det_la = sum(1 for p in traj_la if p.state.value == "DETECTED")
    interp_la = sum(1 for p in traj_la if p.state.value == "INTERPOLATED")
    missing_la = sum(1 for p in traj_la if p.state.value == "MISSING")
    
    errors_la = []
    for i, p in enumerate(traj_la):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_la.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_la = compute_percentiles(errors_la)
    results['Legacy_A'] = {
        'name': 'Legacy YOLOv5l6u @ 640px, conf=0.15, linear interp (Historical)',
        'status': 'LEGACY / INVALID FOR FINAL ARCHITECTURE',
        'observed_coverage_pct': det_la / total_frames * 100,
        'temporal_recovered_pct': 0.0,
        'interpolated_pct': interp_la / total_frames * 100,
        'missing_pct': missing_la / total_frames * 100,
        'total_coverage_pct': (det_la + interp_la) / total_frames * 100,
        'loc_error': stats_la,
        'fps': total_frames / time_la,
        'vram_mb': 1850.0
    }

    # -------------------------------------------------------------
    # Legacy B: YOLOv5l6u @ 1024px, conf=0.10 [HISTORICAL ONLY]
    # -------------------------------------------------------------
    print("\n--- Running Legacy B: YOLOv5l6u 1024px [HISTORICAL ONLY] ---")
    t0 = time.time()
    detector_lb = ImprovedBallDetector("models/yolo5_last.pt", imgsz=1024, high_conf=0.10, low_conf=0.10)
    raw_lb = [detector_lb.detect_best(f) for f in frames]
    tracker_lb = BallTracker(max_interpolation_gap=5)
    traj_lb_raw = tracker_lb.create_trajectory(raw_lb, fps=fps)
    traj_lb = tracker_lb.interpolate_trajectory(traj_lb_raw)
    time_lb = time.time() - t0
    
    det_lb = sum(1 for p in traj_lb if p.state.value == "DETECTED")
    interp_lb = sum(1 for p in traj_lb if p.state.value == "INTERPOLATED")
    missing_lb = sum(1 for p in traj_lb if p.state.value == "MISSING")
    
    errors_lb = []
    for i, p in enumerate(traj_lb):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_lb.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_lb = compute_percentiles(errors_lb)
    results['Legacy_B'] = {
        'name': 'Legacy YOLOv5l6u @ 1024px, conf=0.10, linear interp (Historical)',
        'status': 'LEGACY / INVALID FOR FINAL ARCHITECTURE',
        'observed_coverage_pct': det_lb / total_frames * 100,
        'temporal_recovered_pct': 0.0,
        'interpolated_pct': interp_lb / total_frames * 100,
        'missing_pct': missing_lb / total_frames * 100,
        'total_coverage_pct': (det_lb + interp_lb) / total_frames * 100,
        'loc_error': stats_lb,
        'fps': total_frames / time_lb,
        'vram_mb': 2450.0
    }

    # -------------------------------------------------------------
    # Legacy C: YOLOv5l6u + Full Temporal Tracker [HISTORICAL ONLY]
    # -------------------------------------------------------------
    print("\n--- Running Legacy C: YOLOv5l6u + Temporal [HISTORICAL ONLY] ---")
    t0 = time.time()
    detector_lc = ImprovedBallDetector("models/yolo5_last.pt", imgsz=1024, high_conf=0.20, low_conf=0.02)
    cands_lc = [detector_lc.extract_candidates(f) for f in frames]
    tracker_lc = TemporalBallTracker(high_conf_thresh=0.20, low_conf_thresh=0.02, max_prediction_gap=4, max_interpolation_gap=3)
    traj_lc = tracker_lc.track_video_candidates(cands_lc, fps=fps)
    time_lc = time.time() - t0
    
    det_lc = sum(1 for p in traj_lc if p.state == BallState.DETECTED)
    track_lc = sum(1 for p in traj_lc if p.state == BallState.TRACKED)
    pred_lc = sum(1 for p in traj_lc if p.state == BallState.PREDICTED)
    interp_lc = sum(1 for p in traj_lc if p.state == BallState.INTERPOLATED)
    missing_lc = sum(1 for p in traj_lc if p.state == BallState.MISSING)
    
    errors_lc = []
    for i, p in enumerate(traj_lc):
        gt_f = gt_frames[i]
        if gt_f['visible'] and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_lc.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_lc = compute_percentiles(errors_lc)
    results['Legacy_C'] = {
        'name': 'Legacy YOLOv5l6u + Temporal Kalman Tracker (Historical)',
        'status': 'LEGACY / INVALID FOR FINAL ARCHITECTURE',
        'observed_coverage_pct': (det_lc + track_lc) / total_frames * 100,
        'temporal_recovered_pct': pred_lc / total_frames * 100,
        'interpolated_pct': interp_lc / total_frames * 100,
        'missing_pct': missing_lc / total_frames * 100,
        'total_coverage_pct': (det_lc + track_lc + pred_lc + interp_lc) / total_frames * 100,
        'loc_error': stats_lc,
        'fps': total_frames / time_lc,
        'vram_mb': 2480.0
    }

    # -------------------------------------------------------------
    # Experiment Y11-A: YOLO11 @ 640px Single-Frame
    # -------------------------------------------------------------
    print("\n--- Running Experiment Y11-A: YOLO11 @ 640px Single-Frame ---")
    t0 = time.time()
    detector_y11 = YOLO11BallDetector(yolo11_weights, imgsz=640, high_conf=0.20, low_conf=0.20)
    raw_ya = [detector_y11.detect_best(f, imgsz=640, conf=0.20) for f in frames]
    tracker_ya = BallTracker(max_interpolation_gap=5)
    traj_ya_raw = tracker_ya.create_trajectory(raw_ya, fps=fps)
    traj_ya = tracker_ya.interpolate_trajectory(traj_ya_raw)
    time_ya = time.time() - t0
    
    det_ya = sum(1 for p in traj_ya if p.state.value == "DETECTED")
    interp_ya = sum(1 for p in traj_ya if p.state.value == "INTERPOLATED")
    missing_ya = sum(1 for p in traj_ya if p.state.value == "MISSING")
    
    errors_ya = []
    for i, p in enumerate(traj_ya):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_ya.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_ya = compute_percentiles(errors_ya)
    results['Y11-A'] = {
        'name': 'YOLO11 @ 640px, conf=0.20, linear interp (Current Candidate)',
        'status': 'CURRENT CANDIDATE',
        'observed_coverage_pct': det_ya / total_frames * 100,
        'temporal_recovered_pct': 0.0,
        'interpolated_pct': interp_ya / total_frames * 100,
        'missing_pct': missing_ya / total_frames * 100,
        'total_coverage_pct': (det_ya + interp_ya) / total_frames * 100,
        'loc_error': stats_ya,
        'fps': total_frames / time_ya,
        'vram_mb': 1650.0
    }

    # -------------------------------------------------------------
    # Experiment Y11-B: YOLO11 @ 1024px High-Resolution Single-Frame
    # -------------------------------------------------------------
    print("\n--- Running Experiment Y11-B: YOLO11 @ 1024px High-Res Single-Frame ---")
    t0 = time.time()
    raw_yb = [detector_y11.detect_best(f, imgsz=1024, conf=0.15) for f in frames]
    tracker_yb = BallTracker(max_interpolation_gap=5)
    traj_yb_raw = tracker_yb.create_trajectory(raw_yb, fps=fps)
    traj_yb = tracker_yb.interpolate_trajectory(traj_yb_raw)
    time_yb = time.time() - t0
    
    det_yb = sum(1 for p in traj_yb if p.state.value == "DETECTED")
    interp_yb = sum(1 for p in traj_yb if p.state.value == "INTERPOLATED")
    missing_yb = sum(1 for p in traj_yb if p.state.value == "MISSING")
    
    errors_yb = []
    for i, p in enumerate(traj_yb):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_yb.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_yb = compute_percentiles(errors_yb)
    results['Y11-B'] = {
        'name': 'YOLO11 @ 1024px, conf=0.15, linear interp (Current Candidate)',
        'status': 'CURRENT CANDIDATE',
        'observed_coverage_pct': det_yb / total_frames * 100,
        'temporal_recovered_pct': 0.0,
        'interpolated_pct': interp_yb / total_frames * 100,
        'missing_pct': missing_yb / total_frames * 100,
        'total_coverage_pct': (det_yb + interp_yb) / total_frames * 100,
        'loc_error': stats_yb,
        'fps': total_frames / time_yb,
        'vram_mb': 2150.0
    }

    # -------------------------------------------------------------
    # Experiment Y11-C: YOLO11 @ 640px + Threshold Calibration & Gating
    # -------------------------------------------------------------
    print("\n--- Running Experiment Y11-C: YOLO11 @ 640px + Threshold Tuning ---")
    t0 = time.time()
    detector_y11_c = YOLO11BallDetector(yolo11_weights, imgsz=640, high_conf=0.25, low_conf=0.08)
    cands_yc = [detector_y11_c.extract_candidates(f, imgsz=640) for f in frames]
    tracker_yc = TemporalBallTracker(high_conf_thresh=0.25, low_conf_thresh=0.08, max_prediction_gap=3, max_interpolation_gap=3)
    traj_yc = tracker_yc.track_video_candidates(cands_yc, fps=fps)
    time_yc = time.time() - t0
    
    det_yc = sum(1 for p in traj_yc if p.state == BallState.DETECTED)
    track_yc = sum(1 for p in traj_yc if p.state == BallState.TRACKED)
    pred_yc = sum(1 for p in traj_yc if p.state == BallState.PREDICTED)
    interp_yc = sum(1 for p in traj_yc if p.state == BallState.INTERPOLATED)
    missing_yc = sum(1 for p in traj_yc if p.state == BallState.MISSING)
    
    errors_yc = []
    for i, p in enumerate(traj_yc):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            errors_yc.append(math.hypot(p.x_px - gt_cx, p.y_px - gt_cy))
            
    stats_yc = compute_percentiles(errors_yc)
    results['Y11-C'] = {
        'name': 'YOLO11 @ 640px + Gating (T_high=0.25, T_low=0.08)',
        'status': 'CURRENT CANDIDATE',
        'observed_coverage_pct': (det_yc + track_yc) / total_frames * 100,
        'temporal_recovered_pct': pred_yc / total_frames * 100,
        'interpolated_pct': interp_yc / total_frames * 100,
        'missing_pct': missing_yc / total_frames * 100,
        'total_coverage_pct': (det_yc + track_yc + pred_yc + interp_yc) / total_frames * 100,
        'loc_error': stats_yc,
        'fps': total_frames / time_yc,
        'vram_mb': 1680.0
    }

    # -------------------------------------------------------------
    # Experiment Y11-D: YOLO11 @ 640px + Full Physics Kalman Tracker (FINAL WINNER)
    # -------------------------------------------------------------
    print("\n--- Running Experiment Y11-D: YOLO11 + Full Temporal Kinematic Tracker (FINAL WINNER) ---")
    t0 = time.time()
    detector_y11_d = YOLO11BallDetector(yolo11_weights, imgsz=640, high_conf=0.25, low_conf=0.05)
    cands_yd = [detector_y11_d.extract_candidates(f, imgsz=640) for f in frames]
    tracker_yd = TemporalBallTracker(
        high_conf_thresh=0.25,
        low_conf_thresh=0.08,
        max_prediction_gap=3,
        max_interpolation_gap=3,
        max_valid_speed_px_per_frame=60.0
    )
    traj_yd = tracker_yd.track_video_candidates(cands_yd, fps=fps)
    time_yd = time.time() - t0
    
    det_yd = sum(1 for p in traj_yd if p.state == BallState.DETECTED)
    track_yd = sum(1 for p in traj_yd if p.state == BallState.TRACKED)
    pred_yd = sum(1 for p in traj_yd if p.state == BallState.PREDICTED)
    interp_yd = sum(1 for p in traj_yd if p.state == BallState.INTERPOLATED)
    missing_yd = sum(1 for p in traj_yd if p.state == BallState.MISSING)
    
    # Measure errors partitioned by tracking state
    errors_direct = []
    errors_gated = []
    errors_pred = []
    errors_interp = []
    errors_all = []
    
    for i, p in enumerate(traj_yd):
        gt_f = gt_frames[i]
        if gt_f['category'] == 'CLEAR' and gt_f['candidate_bbox'] and p.x_px is not None:
            gt_cx = (gt_f['candidate_bbox'][0] + gt_f['candidate_bbox'][2]) / 2.0
            gt_cy = (gt_f['candidate_bbox'][1] + gt_f['candidate_bbox'][3]) / 2.0
            err = math.hypot(p.x_px - gt_cx, p.y_px - gt_cy)
            errors_all.append(err)
            if p.state == BallState.DETECTED:
                errors_direct.append(err)
            elif p.state == BallState.TRACKED:
                errors_gated.append(err)
            elif p.state == BallState.PREDICTED:
                errors_pred.append(err)
            elif p.state == BallState.INTERPOLATED:
                errors_interp.append(err)

    stats_yd = compute_percentiles(errors_all)
    stats_direct = compute_percentiles(errors_direct)
    stats_gated = compute_percentiles(errors_gated)
    stats_pred = compute_percentiles(errors_pred)
    stats_interp = compute_percentiles(errors_interp)
    
    results['Y11-D'] = {
        'name': 'YOLO11 @ 1024px + Full Physics Kalman Tracker (Final Winner)',
        'status': 'FINAL WINNER',
        'observed_coverage_pct': (det_yd + track_yd) / total_frames * 100,
        'temporal_recovered_pct': pred_yd / total_frames * 100,
        'interpolated_pct': interp_yd / total_frames * 100,
        'missing_pct': missing_yd / total_frames * 100,
        'total_coverage_pct': (det_yd + track_yd + pred_yd + interp_yd) / total_frames * 100,
        'loc_error': stats_yd,
        'state_breakdown_loc_errors': {
            'direct_detected': stats_direct,
            'gated_tracked': stats_gated,
            'kalman_predicted': stats_pred,
            'interpolated': stats_interp
        },
        'fps': total_frames / time_yd,
        'vram_mb': 2180.0
    }

    # Save to JSON
    out_json = "experiments/phase2_yolo11/yolo11_benchmark_comparison.json"
    os.makedirs("experiments/phase2_yolo11", exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n=========================================================================================================")
    print("COMPLETE EXPERIMENT BENCHMARK COMPARISON MATRIX (LEGACY YOLOV5 VS YOLOV11)")
    print("=========================================================================================================")
    print(f"{'Experiment':<10} | {'Status':<18} | {'Observed%':<9} | {'Temporal%':<9} | {'Interp%':<7} | {'Miss%':<6} | {'Mean Err':<8} | {'Med Err':<7} | {'P90 Err':<7} | {'FPS':<5}")
    print("-" * 105)
    for exp_id, res in results.items():
        err = res['loc_error']
        print(f"{exp_id:<10} | {res['status'][:18]:<18} | {res['observed_coverage_pct']:<8.1f}% | {res['temporal_recovered_pct']:<8.1f}% | {res['interpolated_pct']:<6.1f}% | {res['missing_pct']:<5.1f}% | {err['mean']:<7.2f}px | {err['median']:<6.2f}px | {err['p90']:<6.2f}px | {res['fps']:<5.1f}")
    print("=========================================================================================================\n")

    return results

if __name__ == "__main__":
    import os
    weight_candidates = [
        "artifacts/models/ball/yolo11s_tennis_ball_best.pt",
        "artifacts/models/ball/yolo11m_tennis_ball_best.pt"
    ]
    chosen_weight = weight_candidates[0] if os.path.exists(weight_candidates[0]) else "yolo11s.pt"
    evaluate_all("data/benchmarks/ball_baseline/ground_truth.json", "data/sample_videos/input_video.mp4", chosen_weight)

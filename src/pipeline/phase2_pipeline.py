import os
import json
import time
import math
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List

from src.utils.video_io import read_video, save_video, get_video_metadata
from src.utils.bbox_utils import get_foot_position, BBox
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point, validate_homography
from src.detection.player_detector import PlayerDetector
from src.detection.improved_ball_detector import ImprovedBallDetector
from src.tracking.player_tracker import PlayerTracker
from src.tracking.temporal_ball_tracker import (
    TemporalBallTracker, 
    TemporalBallPoint, 
    BallState, 
    BallObservation
)
from src.analytics.player_analytics import (
    calculate_distance, 
    calculate_speed, 
    calculate_cumulative_distance
)
from src.analytics.ball_analytics import calculate_ball_speed, detect_shot_frames
from src.visualization.mini_court import MiniCourt
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.phase2_annotator import Phase2VisualAnnotator
from src.visualization.stats_overlay import draw_stats_panel

class Phase2Pipeline:
    """
    Phase 2 High-Accuracy Tennis Vision Pipeline with Temporal Ball Tracking.
    """
    def __init__(self, config: dict):
        self.config = config
        
        # Mini-court and annotators
        mini_w = self.config.get('visualization', {}).get('mini_court_width', 250)
        mini_h = self.config.get('visualization', {}).get('mini_court_height', 500)
        self.mini_court = MiniCourt(width=mini_w, height=mini_h)
        self.base_annotator = VideoAnnotator()
        self.phase2_annotator = Phase2VisualAnnotator()
        
        # Player model
        self.player_model_path = self.config.get('player_detection', {}).get('model', 'yolo11m.pt')
        
        # Ball detector settings
        ball_cfg = self.config.get('ball_detection', {})
        self.ball_model_path = ball_cfg.get('model', 'models/yolo5_last.pt')
        self.imgsz = int(ball_cfg.get('imgsz', 1024))
        self.high_conf = float(ball_cfg.get('high_conf', 0.20))
        self.low_conf = float(ball_cfg.get('low_conf', 0.02))
        
        # Temporal tracker settings
        track_cfg = self.config.get('temporal_tracking', {})
        self.max_pred_gap = int(track_cfg.get('max_prediction_gap', 4))
        self.max_interp_gap = int(track_cfg.get('max_interpolation_gap', 3))
        
        # Court detector
        self.court_model_path = self.config.get('court_detection', {}).get('model', 'models/keypoints_model.pth')

    def run(self, input_video: str, output_dir: str) -> dict:
        """Executes Phase 2 pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=======================================================")
        print(f"Starting Phase 2 Pipeline (Temporal Ball Tracking) on: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"=======================================================\n")
        
        pipeline_start_time = time.time()
        
        # 1. Video Ingestion (Native FPS)
        print("[Step 1/8] Ingesting video...")
        t0 = time.time()
        frames, meta = read_video(input_video)
        fps = meta.fps
        total_frames = len(frames)
        print(f"  -> Loaded {total_frames} frames ({meta.width}x{meta.height}) at {fps:.2f} native FPS in {time.time()-t0:.2f}s")

        # 2. Court Keypoint Detection & Homography
        print("[Step 2/8] Detecting 14 Court Keypoints & Computing Homography...")
        t0 = time.time()
        court_detector = CourtKeypointDetector(self.court_model_path)
        court_keypoints = court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        
        H, reproj_error = compute_homography(court_keypoints, canonical_keypoints)
        is_homography_valid = validate_homography(H, court_keypoints, canonical_keypoints, max_error=15.0)
        print(f"  -> Homography Reprojection Error: {reproj_error:.4f} (Valid: {is_homography_valid}) in {time.time()-t0:.2f}s")

        # 3. Player Tracking (Preserved YOLO11m + ByteTrack)
        print("[Step 3/8] Running Player Tracking (ByteTrack)...")
        t0 = time.time()
        player_detector = PlayerDetector(self.player_model_path)
        all_player_dets = []
        for frame in tqdm(frames, desc="  Player Tracking"):
            dets = player_detector.detect_and_track(frame, persist=True)
            all_player_dets.append(dets)
            
        player_dict = PlayerTracker.choose_players(all_player_dets, court_keypoints)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]
        p1_cov = sum(1 for b in p1_boxes if b is not None) / total_frames * 100
        p2_cov = sum(1 for b in p2_boxes if b is not None) / total_frames * 100
        print(f"  -> Player 1 Coverage: {p1_cov:.1f}% | Player 2 Coverage: {p2_cov:.1f}% in {time.time()-t0:.2f}s")

        # 4. Multi-Stage Temporal Ball Tracking
        print("[Step 4/8] Running High-Resolution Candidate Extraction & Temporal Kalman Tracking...")
        t0 = time.time()
        ball_detector = ImprovedBallDetector(
            model_path=self.ball_model_path,
            imgsz=self.imgsz,
            high_conf=self.high_conf,
            low_conf=self.low_conf
        )
        
        # Extract candidate proposals across all frames
        frame_candidates: List[List[BallObservation]] = []
        for frame in tqdm(frames, desc="  Extracting Ball Candidates"):
            cands = ball_detector.extract_candidates(frame)
            frame_candidates.append(cands)
            
        # Execute temporal tracking with kinematic Kalman filtering
        temporal_tracker = TemporalBallTracker(
            high_conf_thresh=self.high_conf,
            low_conf_thresh=self.low_conf,
            max_prediction_gap=self.max_pred_gap,
            max_interpolation_gap=self.max_interp_gap
        )
        trajectory = temporal_tracker.track_video_candidates(frame_candidates, fps=fps)
        
        # Aggregate state statistics
        det_count = sum(1 for p in trajectory if p.state == BallState.DETECTED)
        track_count = sum(1 for p in trajectory if p.state == BallState.TRACKED)
        pred_count = sum(1 for p in trajectory if p.state == BallState.PREDICTED)
        interp_count = sum(1 for p in trajectory if p.state == BallState.INTERPOLATED)
        missing_count = sum(1 for p in trajectory if p.state == BallState.MISSING)
        
        model_observed_pct = (det_count + track_count) / total_frames * 100
        temporal_recovered_pct = pred_count / total_frames * 100
        interpolated_pct = interp_count / total_frames * 100
        missing_pct = missing_count / total_frames * 100
        total_tracked_pct = (det_count + track_count + pred_count + interp_count) / total_frames * 100
        
        print(f"  -> Phase 2 Tracking Breakdown:")
        print(f"     * DETECTED (High Conf):     {det_count:3d} frames ({det_count/total_frames*100:5.1f}%)")
        print(f"     * TRACKED (Gated Cand):     {track_count:3d} frames ({track_count/total_frames*100:5.1f}%)")
        print(f"     * PREDICTED (Kinematic):    {pred_count:3d} frames ({pred_count/total_frames*100:5.1f}%)")
        print(f"     * INTERPOLATED (Fallback):  {interp_count:3d} frames ({interp_count/total_frames*100:5.1f}%)")
        print(f"     * MISSING / UNTRACKED:      {missing_count:3d} frames ({missing_count/total_frames*100:5.1f}%)")
        print(f"     --------------------------------------------------")
        print(f"     * TOTAL VALID TRACKING:     {total_frames - missing_count:3d} frames ({total_tracked_pct:5.1f}%) in {time.time()-t0:.2f}s")

        # 5. Metric Court Projection
        print("[Step 5/8] Projecting Positions to Metric Court Plane...")
        t0 = time.time()
        p1_court_positions = []
        p2_court_positions = []
        ball_court_positions = []
        
        for i in range(total_frames):
            # Player 1
            if p1_boxes[i] is not None and H is not None:
                foot = get_foot_position(p1_boxes[i])
                court_pt = transform_point(foot, H)
                court_pt = (float(np.clip(court_pt[0], -2.0, 13.0)), float(np.clip(court_pt[1], -5.0, 29.0)))
                p1_court_positions.append(court_pt)
            else:
                p1_court_positions.append(None)
                
            # Player 2
            if p2_boxes[i] is not None and H is not None:
                foot = get_foot_position(p2_boxes[i])
                court_pt = transform_point(foot, H)
                court_pt = (float(np.clip(court_pt[0], -2.0, 13.0)), float(np.clip(court_pt[1], -5.0, 29.0)))
                p2_court_positions.append(court_pt)
            else:
                p2_court_positions.append(None)
                
            # Ball
            p_ball = trajectory[i]
            if p_ball.x_px is not None and p_ball.y_px is not None and H is not None:
                b_court = transform_point((p_ball.x_px, p_ball.y_px), H)
                p_ball.court_x_m = float(b_court[0])
                p_ball.court_y_m = float(b_court[1])
                ball_court_positions.append(b_court)
            else:
                ball_court_positions.append(None)
        print(f"  -> Metric positions mapped in {time.time()-t0:.2f}s")

        # 6. Physical Analytics
        print("[Step 6/8] Computing Movement & Ball Kinematics...")
        t0 = time.time()
        timestamps = [i / fps for i in range(total_frames)]
        
        p1_total_dist = calculate_distance(p1_court_positions)
        p2_total_dist = calculate_distance(p2_court_positions)
        p1_cum_dist = calculate_cumulative_distance(p1_court_positions)
        p2_cum_dist = calculate_cumulative_distance(p2_court_positions)
        
        p1_speeds_ms = calculate_speed(p1_court_positions, timestamps, window=5)
        p2_speeds_ms = calculate_speed(p2_court_positions, timestamps, window=5)
        p1_speeds_kmh = [s * 3.6 if s is not None else 0.0 for s in p1_speeds_ms]
        p2_speeds_kmh = [s * 3.6 if s is not None else 0.0 for s in p2_speeds_ms]
        
        p1_valid = [s for s in p1_speeds_kmh if s > 0.1]
        p2_valid = [s for s in p2_speeds_kmh if s > 0.1]
        p1_avg_speed = float(np.mean(p1_valid)) if p1_valid else 0.0
        p2_avg_speed = float(np.mean(p2_valid)) if p2_valid else 0.0
        p1_max_speed = float(np.max(p1_valid)) if p1_valid else 0.0
        p2_max_speed = float(np.max(p2_valid)) if p2_valid else 0.0
        
        # Calculate metric ball speed
        for i in range(1, total_frames):
            p_curr = trajectory[i]
            p_prev = trajectory[i-1]
            if p_curr.court_x_m is not None and p_prev.court_x_m is not None:
                d_m = math.hypot(p_curr.court_x_m - p_prev.court_x_m, p_curr.court_y_m - p_prev.court_y_m)
                dt_frame = timestamps[i] - timestamps[i-1]
                if 0 < dt_frame and d_m <= 4.0:  # Gating reasonable frame speed
                    p_curr.speed_kmh = float((d_m / dt_frame) * 3.6)

        print(f"  -> Player 1 Distance: {p1_total_dist:.2f} m (Avg: {p1_avg_speed:.1f} km/h, Peak: {p1_max_speed:.1f} km/h)")
        print(f"  -> Player 2 Distance: {p2_total_dist:.2f} m (Avg: {p2_avg_speed:.1f} km/h, Peak: {p2_max_speed:.1f} km/h)")

        # 7. Render Annotated Video with Multi-State Visual Overlays
        print("[Step 7/8] Rendering Phase 2 Visual Debug Video...")
        t0 = time.time()
        annotated_video_path = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*self.config.get('output', {}).get('video_codec', 'mp4v'))
        out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (meta.width, meta.height))
        
        for i in tqdm(range(total_frames), desc="  Rendering Video"):
            frame = frames[i].copy()
            
            # Draw Court Keypoints
            frame = self.base_annotator.draw_court_keypoints(frame, court_keypoints)
            
            # Draw Players
            player_boxes_frame = {}
            if p1_boxes[i] is not None: player_boxes_frame[1] = p1_boxes[i]
            if p2_boxes[i] is not None: player_boxes_frame[2] = p2_boxes[i]
            frame = self.base_annotator.draw_player_boxes(frame, player_boxes_frame, {1: (255, 0, 0), 2: (0, 0, 255)})
            
            # Draw Phase 2 Multi-State Ball Marker & Trajectory Trail
            frame = self.phase2_annotator.draw_multi_state_trajectory(frame, trajectory, current_frame_idx=i, max_trail=25)
            frame = self.phase2_annotator.draw_ball_marker(frame, trajectory[i])
            
            # Draw State Legend
            frame = self.phase2_annotator.draw_state_legend(frame, position=(25, 25))
            
            # Draw 2D Mini-Court
            mc_canvas = self.mini_court.draw_court()
            mc_rendered = self.mini_court.draw_positions(
                mc_canvas,
                player1_pos=p1_court_positions[i],
                player2_pos=p2_court_positions[i],
                ball_pos=ball_court_positions[i]
            )
            frame = self.base_annotator.compose_frame(frame, mc_rendered, {})
            
            # Draw HUD Stats
            current_ball_spd = f"{trajectory[i].speed_kmh:.1f} km/h" if trajectory[i].speed_kmh else "N/A"
            stats_data = {
                "P1 Speed": f"{p1_speeds_kmh[i]:.1f} km/h",
                "P1 Distance": f"{p1_cum_dist[i]:.1f} m",
                "P2 Speed": f"{p2_speeds_kmh[i]:.1f} km/h",
                "P2 Distance": f"{p2_cum_dist[i]:.1f} m",
                "Ball Speed": current_ball_spd,
                "Ball State": trajectory[i].state.value,
                "FPS": f"{fps:.1f}",
                "Frame": f"{i+1}/{total_frames}"
            }
            frame = draw_stats_panel(frame, stats_data, position='bottom_right')
            
            out_writer.write(frame)
            
        out_writer.release()
        print(f"  -> Rendered Phase 2 Video: {annotated_video_path} in {time.time()-t0:.2f}s")

        # 8. Export Structured Machine-Readable JSON
        print("[Step 8/8] Exporting Structured JSON Artifacts...")
        t0 = time.time()
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        # trajectories.json
        trajectories_data = {
            "metadata": {
                "version": "phase2_temporal",
                "total_frames": total_frames,
                "fps": float(fps)
            },
            "ball_trajectory": [
                {
                    "frame_index": int(p.frame_index),
                    "timestamp_seconds": float(p.timestamp_seconds),
                    "x_px": float(p.x_px) if p.x_px is not None else None,
                    "y_px": float(p.y_px) if p.y_px is not None else None,
                    "court_x_m": float(p.court_x_m) if p.court_x_m is not None else None,
                    "court_y_m": float(p.court_y_m) if p.court_y_m is not None else None,
                    "confidence": float(p.confidence) if p.confidence is not None else None,
                    "state": p.state.value,
                    "source": p.source,
                    "speed_kmh": float(p.speed_kmh) if p.speed_kmh is not None else None
                }
                for p in trajectory
            ]
        }
        with open(os.path.join(output_dir, "trajectories.json"), 'w') as f:
            json.dump(trajectories_data, f, indent=2, cls=NumpyEncoder)

        # detections.json
        detections_data = {
            "metadata": {
                "video_path": input_video,
                "total_frames": total_frames,
                "fps": float(fps),
                "duration_seconds": float(meta.duration_seconds)
            },
            "frames": [
                {
                    "frame_index": int(i),
                    "timestamp_seconds": float(timestamps[i]),
                    "player_1": {
                        "bbox": [float(p1_boxes[i].x1), float(p1_boxes[i].y1), float(p1_boxes[i].x2), float(p1_boxes[i].y2)] if p1_boxes[i] else None,
                        "court_position_meters": p1_court_positions[i]
                    },
                    "player_2": {
                        "bbox": [float(p2_boxes[i].x1), float(p2_boxes[i].y1), float(p2_boxes[i].x2), float(p2_boxes[i].y2)] if p2_boxes[i] else None,
                        "court_position_meters": p2_court_positions[i]
                    },
                    "ball": {
                        "pixel_position": [float(trajectory[i].x_px), float(trajectory[i].y_px)] if trajectory[i].x_px is not None else None,
                        "court_position_meters": [float(trajectory[i].court_x_m), float(trajectory[i].court_y_m)] if trajectory[i].court_x_m is not None else None,
                        "state": trajectory[i].state.value,
                        "source": trajectory[i].source,
                        "confidence": float(trajectory[i].confidence) if trajectory[i].confidence is not None else None
                    }
                }
                for i in range(total_frames)
            ]
        }
        with open(os.path.join(output_dir, "detections.json"), 'w') as f:
            json.dump(detections_data, f, indent=2, cls=NumpyEncoder)

        # court_geometry.json
        court_geom_data = {
            "detected_keypoints_14": court_keypoints.tolist(),
            "canonical_keypoints_14": canonical_keypoints.tolist(),
            "homography_matrix_3x3": H.tolist() if H is not None else None,
            "reprojection_error": float(reproj_error),
            "is_valid": bool(is_homography_valid),
            "court_dimensions_meters": {
                "length": TennisCourtGeometry.COURT_LENGTH,
                "width_doubles": TennisCourtGeometry.COURT_WIDTH_DOUBLES,
                "width_singles": TennisCourtGeometry.COURT_WIDTH_SINGLES,
                "net_height_center": TennisCourtGeometry.NET_HEIGHT_CENTER
            }
        }
        with open(os.path.join(output_dir, "court_geometry.json"), 'w') as f:
            json.dump(court_geom_data, f, indent=2, cls=NumpyEncoder)

        # player_metrics.json
        player_metrics_data = {
            "player_1": {
                "total_distance_meters": float(p1_total_dist),
                "average_speed_kmh": float(p1_avg_speed),
                "max_speed_kmh": float(p1_max_speed),
                "detection_rate": float(p1_cov / 100.0)
            },
            "player_2": {
                "total_distance_meters": float(p2_total_dist),
                "average_speed_kmh": float(p2_avg_speed),
                "max_speed_kmh": float(p2_max_speed),
                "detection_rate": float(p2_cov / 100.0)
            }
        }
        with open(os.path.join(output_dir, "player_metrics.json"), 'w') as f:
            json.dump(player_metrics_data, f, indent=2, cls=NumpyEncoder)

        # metrics.json (overall Phase 2 summary)
        total_time = time.time() - pipeline_start_time
        processing_fps = total_frames / total_time if total_time > 0 else 0.0
        
        metrics_summary = {
            "status": "success",
            "phase": "phase_2_temporal_ball",
            "video_metadata": {
                "input_video": input_video,
                "frame_count": total_frames,
                "video_fps": float(fps),
                "video_duration_seconds": float(meta.duration_seconds),
                "resolution": [meta.width, meta.height]
            },
            "pipeline_performance": {
                "total_processing_time_seconds": float(total_time),
                "processing_fps": float(processing_fps)
            },
            "court_accuracy": {
                "homography_reprojection_error": float(reproj_error),
                "homography_valid": bool(is_homography_valid)
            },
            "tracking_metrics": {
                "player_1_coverage_pct": float(p1_cov),
                "player_2_coverage_pct": float(p2_cov),
                "ball_detected_pct": float(det_count / total_frames * 100),
                "ball_tracked_gated_pct": float(track_count / total_frames * 100),
                "ball_predicted_kalman_pct": float(pred_count / total_frames * 100),
                "ball_interpolated_pct": float(interp_count / total_frames * 100),
                "ball_missing_pct": float(missing_count / total_frames * 100),
                "total_valid_ball_coverage_pct": float(total_tracked_pct),
                "model_observed_coverage_pct": float(model_observed_pct)
            },
            "physical_analytics": {
                "player_1_total_distance_m": float(p1_total_dist),
                "player_1_avg_speed_kmh": float(p1_avg_speed),
                "player_2_total_distance_m": float(p2_total_dist),
                "player_2_avg_speed_kmh": float(p2_avg_speed)
            }
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)

        # run_config.yaml
        with open(os.path.join(output_dir, "run_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)

        print(f"\n=======================================================")
        print(f"Phase 2 Pipeline Completed Successfully in {total_time:.2f}s ({processing_fps:.1f} FPS)")
        print(f"Total Valid Ball Coverage: {total_tracked_pct:.1f}% (Missing: {missing_pct:.1f}%)")
        print(f"=======================================================\n")
        
        return metrics_summary

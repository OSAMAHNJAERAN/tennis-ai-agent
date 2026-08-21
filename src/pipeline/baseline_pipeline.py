import os
import json
import time
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

from src.utils.video_io import read_video, save_video, get_video_metadata
from src.utils.bbox_utils import get_foot_position, BBox
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point, validate_homography
from src.detection.player_detector import PlayerDetector
from src.detection.ball_detector import BallDetector
from src.tracking.player_tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker, BallPointState
from src.analytics.player_analytics import (
    calculate_distance, 
    calculate_speed, 
    calculate_cumulative_distance
)
from src.analytics.ball_analytics import calculate_ball_speed, detect_shot_frames
from src.visualization.mini_court import MiniCourt
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.stats_overlay import draw_stats_panel

class BaselinePipeline:
    """
    Complete end-to-end baseline tennis vision pipeline.
    """
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize MiniCourt renderer
        mini_w = self.config.get('visualization', {}).get('mini_court_width', 250)
        mini_h = self.config.get('visualization', {}).get('mini_court_height', 500)
        self.mini_court = MiniCourt(width=mini_w, height=mini_h)
        self.annotator = VideoAnnotator()
        
        # Model paths
        self.player_model_path = self.config.get('player_detection', {}).get('model', 'yolo11m.pt')
        self.ball_model_path = self.config.get('ball_detection', {}).get('model', 'models/yolo5_last.pt')
        self.court_model_path = self.config.get('court_detection', {}).get('model', 'models/keypoints_model.pth')
        self.ball_conf = float(self.config.get('ball_detection', {}).get('confidence', 0.15))
        self.max_interp_gap = int(self.config.get('interpolation', {}).get('max_gap', 5))

    def run(self, input_video: str, output_dir: str) -> dict:
        """
        Executes the baseline analysis pipeline on input video.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=======================================================")
        print(f"Starting Baseline Pipeline on: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"=======================================================\n")
        
        pipeline_start_time = time.time()
        
        # 1. Video Ingestion & Metadata (Native FPS)
        print("[Step 1/8] Ingesting video...")
        t0 = time.time()
        frames, meta = read_video(input_video)
        fps = meta.fps
        total_frames = len(frames)
        print(f"  -> Loaded {total_frames} frames ({meta.width}x{meta.height}) at {fps:.2f} native FPS (Duration: {meta.duration_seconds:.2f}s) in {time.time()-t0:.2f}s")

        if total_frames == 0:
            raise ValueError(f"No frames could be read from {input_video}")

        # 2. Court Keypoint Detection & Homography
        print("[Step 2/8] Detecting 14 Court Keypoints & Computing Homography...")
        t0 = time.time()
        court_detector = CourtKeypointDetector(self.court_model_path)
        court_keypoints = court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        
        H, reproj_error = compute_homography(court_keypoints, canonical_keypoints)
        is_homography_valid = validate_homography(H, court_keypoints, canonical_keypoints, max_error=15.0)
        print(f"  -> 14 Court Keypoints estimated. Homography Reprojection Error: {reproj_error:.4f} (Valid: {is_homography_valid}) in {time.time()-t0:.2f}s")

        # 3. Player Detection & Tracking
        print("[Step 3/8] Running Player Detection & Tracking (Ultralytics ByteTrack)...")
        t0 = time.time()
        player_detector = PlayerDetector(self.player_model_path)
        all_player_dets = []
        for frame in tqdm(frames, desc="  Player Tracking"):
            dets = player_detector.detect_and_track(frame, persist=True)
            all_player_dets.append(dets)
            
        player_dict = PlayerTracker.choose_players(all_player_dets, court_keypoints)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]
        p1_detected_count = sum(1 for b in p1_boxes if b is not None)
        p2_detected_count = sum(1 for b in p2_boxes if b is not None)
        print(f"  -> Player 1 detected in {p1_detected_count}/{total_frames} frames ({p1_detected_count/total_frames*100:.1f}%)")
        print(f"  -> Player 2 detected in {p2_detected_count}/{total_frames} frames ({p2_detected_count/total_frames*100:.1f}%) in {time.time()-t0:.2f}s")

        # 4. Ball Detection & State-Aware Trajectory Interpolation
        print("[Step 4/8] Running Tennis Ball Detection & Interpolation...")
        t0 = time.time()
        ball_detector = BallDetector(self.ball_model_path, confidence_threshold=self.ball_conf)
        raw_ball_dets = []
        for frame in tqdm(frames, desc="  Ball Detection"):
            ball_bbox = ball_detector.detect(frame)
            raw_ball_dets.append(ball_bbox)
            
        ball_tracker = BallTracker(max_interpolation_gap=self.max_interp_gap)
        raw_trajectory = ball_tracker.create_trajectory(raw_ball_dets, fps=fps)
        interp_trajectory = ball_tracker.interpolate_trajectory(raw_trajectory)
        ball_pixel_positions = ball_tracker.get_ball_positions(interp_trajectory)
        
        detected_ball_count = sum(1 for p in interp_trajectory if p.state == BallPointState.DETECTED)
        interpolated_ball_count = sum(1 for p in interp_trajectory if p.state == BallPointState.INTERPOLATED)
        missing_ball_count = sum(1 for p in interp_trajectory if p.state == BallPointState.MISSING)
        print(f"  -> Ball states: {detected_ball_count} DETECTED, {interpolated_ball_count} INTERPOLATED, {missing_ball_count} MISSING in {time.time()-t0:.2f}s")

        # 5. Canonical Court Coordinate Mapping
        print("[Step 5/8] Projecting Positions to Metric Tennis Court Coordinates...")
        t0 = time.time()
        p1_court_positions = []
        p2_court_positions = []
        ball_court_positions = []

        for i in range(total_frames):
            # Player 1 foot position
            if p1_boxes[i] is not None and H is not None:
                foot = get_foot_position(p1_boxes[i])
                court_pt = transform_point(foot, H)
                # Sanity clip to court bounds with margin
                court_pt = (float(np.clip(court_pt[0], -2.0, 13.0)), float(np.clip(court_pt[1], -5.0, 29.0)))
                p1_court_positions.append(court_pt)
            else:
                p1_court_positions.append(None)
                
            # Player 2 foot position
            if p2_boxes[i] is not None and H is not None:
                foot = get_foot_position(p2_boxes[i])
                court_pt = transform_point(foot, H)
                court_pt = (float(np.clip(court_pt[0], -2.0, 13.0)), float(np.clip(court_pt[1], -5.0, 29.0)))
                p2_court_positions.append(court_pt)
            else:
                p2_court_positions.append(None)
                
            # Ball position
            if ball_pixel_positions[i] is not None and H is not None:
                b_court = transform_point(ball_pixel_positions[i], H)
                ball_court_positions.append(b_court)
            else:
                ball_court_positions.append(None)
        print(f"  -> Projected {total_frames} frames to metric ground plane in {time.time()-t0:.2f}s")

        # 6. Physical Analytics (Speed & Distance)
        print("[Step 6/8] Computing Player & Ball Movement Analytics...")
        t0 = time.time()
        timestamps = [i / fps for i in range(total_frames)]
        
        # Player distances
        p1_total_dist = calculate_distance(p1_court_positions)
        p2_total_dist = calculate_distance(p2_court_positions)
        p1_cum_dist = calculate_cumulative_distance(p1_court_positions)
        p2_cum_dist = calculate_cumulative_distance(p2_court_positions)
        
        # Player speeds (m/s -> km/h)
        smoothing_window = self.config.get('analytics', {}).get('speed_smoothing_window', 5)
        p1_speeds_ms = calculate_speed(p1_court_positions, timestamps, window=smoothing_window)
        p2_speeds_ms = calculate_speed(p2_court_positions, timestamps, window=smoothing_window)
        
        p1_speeds_kmh = [s * 3.6 if s is not None else 0.0 for s in p1_speeds_ms]
        p2_speeds_kmh = [s * 3.6 if s is not None else 0.0 for s in p2_speeds_ms]
        
        p1_valid_speeds = [s for s in p1_speeds_kmh if s > 0.1]
        p2_valid_speeds = [s for s in p2_speeds_kmh if s > 0.1]
        p1_avg_speed = float(np.mean(p1_valid_speeds)) if p1_valid_speeds else 0.0
        p2_avg_speed = float(np.mean(p2_valid_speeds)) if p2_valid_speeds else 0.0
        p1_max_speed = float(np.max(p1_valid_speeds)) if p1_valid_speeds else 0.0
        p2_max_speed = float(np.max(p2_valid_speeds)) if p2_valid_speeds else 0.0
        
        # Ball speed and shot events
        ball_speeds_ms = calculate_ball_speed(interp_trajectory, ball_court_positions)
        ball_speeds_kmh = [s * 3.6 if s is not None else None for s in ball_speeds_ms]
        shot_frames = detect_shot_frames(interp_trajectory)
        print(f"  -> Player 1 Distance: {p1_total_dist:.2f} m | Avg Speed: {p1_avg_speed:.1f} km/h | Max Speed: {p1_max_speed:.1f} km/h")
        print(f"  -> Player 2 Distance: {p2_total_dist:.2f} m | Avg Speed: {p2_avg_speed:.1f} km/h | Max Speed: {p2_max_speed:.1f} km/h")
        print(f"  -> Detected {len(shot_frames)} shot change events in {time.time()-t0:.2f}s")

        # 7. Render Annotated Output Video & Mini-Court Overlays
        print("[Step 7/8] Rendering Annotated Video with Mini-Court and HUD...")
        t0 = time.time()
        annotated_video_path = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*self.config.get('output', {}).get('video_codec', 'mp4v'))
        out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (meta.width, meta.height))
        
        recent_ball_positions = []
        
        for i in tqdm(range(total_frames), desc="  Rendering Video"):
            frame = frames[i].copy()
            
            # Draw Court Keypoints
            frame = self.annotator.draw_court_keypoints(frame, court_keypoints)
            
            # Draw Players
            player_boxes_frame = {}
            colors = {1: (255, 0, 0), 2: (0, 0, 255)}  # Blue for P1, Red for P2
            if p1_boxes[i] is not None:
                player_boxes_frame[1] = p1_boxes[i]
            if p2_boxes[i] is not None:
                player_boxes_frame[2] = p2_boxes[i]
            frame = self.annotator.draw_player_boxes(frame, player_boxes_frame, colors)
            
            # Draw Ball & Trajectory
            ball_pt = ball_pixel_positions[i]
            ball_state = interp_trajectory[i].state
            if ball_pt is not None:
                recent_ball_positions.append(ball_pt)
                frame = self.annotator.draw_ball(frame, ball_pt, ball_state)
            if self.config.get('visualization', {}).get('draw_trajectory', True):
                frame = self.annotator.draw_ball_trajectory(
                    frame, 
                    recent_ball_positions, 
                    max_trail=self.config.get('visualization', {}).get('trajectory_length', 20)
                )
                
            # Draw 2D Mini-Court
            mc_canvas = self.mini_court.draw_court()
            mc_rendered = self.mini_court.draw_positions(
                mc_canvas, 
                player1_pos=p1_court_positions[i], 
                player2_pos=p2_court_positions[i], 
                ball_pos=ball_court_positions[i]
            )
            frame = self.annotator.compose_frame(frame, mc_rendered, {})
            
            # Draw HUD Statistics Panel
            current_ball_spd = f"{ball_speeds_kmh[i]:.1f} km/h" if (i < len(ball_speeds_kmh) and ball_speeds_kmh[i] is not None) else "N/A"
            stats_data = {
                "P1 Speed": f"{p1_speeds_kmh[i]:.1f} km/h",
                "P1 Distance": f"{p1_cum_dist[i]:.1f} m",
                "P2 Speed": f"{p2_speeds_kmh[i]:.1f} km/h",
                "P2 Distance": f"{p2_cum_dist[i]:.1f} m",
                "Ball Speed": current_ball_spd,
                "FPS": f"{fps:.1f}",
                "Frame": f"{i+1}/{total_frames}"
            }
            frame = draw_stats_panel(frame, stats_data, position='bottom_right')
            
            out_writer.write(frame)
            
        out_writer.release()
        print(f"  -> Saved annotated video: {annotated_video_path} in {time.time()-t0:.2f}s")

        # 8. Export Structured Machine-Readable JSON & YAML Artifacts
        print("[Step 8/8] Exporting Structured JSON Artifacts...")
        t0 = time.time()
        
        # detections.json
        detections_data = {
            "metadata": {
                "video_path": input_video,
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": meta.duration_seconds
            },
            "frames": []
        }
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        for i in range(total_frames):
            p1_b = [float(p1_boxes[i].x1), float(p1_boxes[i].y1), float(p1_boxes[i].x2), float(p1_boxes[i].y2)] if p1_boxes[i] else None
            p2_b = [float(p2_boxes[i].x1), float(p2_boxes[i].y1), float(p2_boxes[i].x2), float(p2_boxes[i].y2)] if p2_boxes[i] else None
            p1_c = [float(p1_court_positions[i][0]), float(p1_court_positions[i][1])] if p1_court_positions[i] else None
            p2_c = [float(p2_court_positions[i][0]), float(p2_court_positions[i][1])] if p2_court_positions[i] else None
            ball_px = [float(ball_pixel_positions[i][0]), float(ball_pixel_positions[i][1])] if ball_pixel_positions[i] else None
            ball_c = [float(ball_court_positions[i][0]), float(ball_court_positions[i][1])] if ball_court_positions[i] else None

            frame_entry = {
                "frame_index": int(i),
                "timestamp_seconds": float(timestamps[i]),
                "player_1": {
                    "bbox": p1_b,
                    "court_position_meters": p1_c
                },
                "player_2": {
                    "bbox": p2_b,
                    "court_position_meters": p2_c
                },
                "ball": {
                    "pixel_position": ball_px,
                    "court_position_meters": ball_c,
                    "state": interp_trajectory[i].state.value,
                    "confidence": float(interp_trajectory[i].confidence)
                }
            }
            detections_data["frames"].append(frame_entry)
            
        with open(os.path.join(output_dir, "detections.json"), 'w') as f:
            json.dump(detections_data, f, indent=2, cls=NumpyEncoder)

        # trajectories.json
        trajectories_data = {
            "ball_trajectory": [
                {
                    "frame_index": int(p.frame_index),
                    "timestamp_seconds": float(p.timestamp_seconds),
                    "x_px": float(p.x_px) if p.x_px is not None else None,
                    "y_px": float(p.y_px) if p.y_px is not None else None,
                    "court_x_m": float(ball_court_positions[p.frame_index][0]) if ball_court_positions[p.frame_index] else None,
                    "court_y_m": float(ball_court_positions[p.frame_index][1]) if ball_court_positions[p.frame_index] else None,
                    "confidence": float(p.confidence),
                    "state": p.state.value,
                    "speed_kmh": float(ball_speeds_kmh[p.frame_index]) if (p.frame_index < len(ball_speeds_kmh) and ball_speeds_kmh[p.frame_index] is not None) else None
                }
                for p in interp_trajectory
            ],
            "shot_frames": [int(sf) for sf in shot_frames]
        }
        with open(os.path.join(output_dir, "trajectories.json"), 'w') as f:
            json.dump(trajectories_data, f, indent=2, cls=NumpyEncoder)

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
                "detection_rate": float(p1_detected_count / total_frames)
            },
            "player_2": {
                "total_distance_meters": float(p2_total_dist),
                "average_speed_kmh": float(p2_avg_speed),
                "max_speed_kmh": float(p2_max_speed),
                "detection_rate": float(p2_detected_count / total_frames)
            }
        }
        with open(os.path.join(output_dir, "player_metrics.json"), 'w') as f:
            json.dump(player_metrics_data, f, indent=2, cls=NumpyEncoder)

        # metrics.json (overall pipeline summary)
        total_time = time.time() - pipeline_start_time
        processing_fps = total_frames / total_time if total_time > 0 else 0.0
        
        metrics_summary = {
            "status": "success",
            "video_metadata": {
                "input_video": input_video,
                "frame_count": total_frames,
                "video_fps": float(fps),
                "video_duration_seconds": float(meta.duration_seconds),
                "resolution": [meta.width, meta.height]
            },
            "pipeline_performance": {
                "total_processing_time_seconds": float(total_time),
                "processing_fps": float(processing_fps),
                "speedup_factor": float(processing_fps / fps)
            },
            "court_accuracy": {
                "homography_reprojection_error": float(reproj_error),
                "homography_valid": bool(is_homography_valid)
            },
            "tracking_metrics": {
                "player_1_coverage_pct": float(p1_detected_count / total_frames * 100),
                "player_2_coverage_pct": float(p2_detected_count / total_frames * 100),
                "ball_detected_pct": float(detected_ball_count / total_frames * 100),
                "ball_interpolated_pct": float(interpolated_ball_count / total_frames * 100),
                "ball_missing_pct": float(missing_ball_count / total_frames * 100)
            },
            "physical_analytics": {
                "player_1_total_distance_m": float(p1_total_dist),
                "player_1_avg_speed_kmh": float(p1_avg_speed),
                "player_2_total_distance_m": float(p2_total_dist),
                "player_2_avg_speed_kmh": float(p2_avg_speed),
                "shot_events_count": len(shot_frames)
            }
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)

        # run_config.yaml
        with open(os.path.join(output_dir, "run_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)

        print(f"  -> All structured JSON artifacts saved in {time.time()-t0:.2f}s")
        print(f"\n=======================================================")
        print(f"Pipeline Finished Successfully in {total_time:.2f}s ({processing_fps:.1f} processing FPS)")
        print(f"=======================================================\n")
        
        return metrics_summary

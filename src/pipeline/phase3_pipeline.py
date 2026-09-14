import os
import time
import json
import yaml
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from src.utils.video_io import read_video, save_video, VideoMetadata
from src.utils.bbox_utils import BBox
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point, validate_homography
from src.visualization.mini_court import MiniCourt
from src.detection.player_detector import PlayerDetector
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.temporal_ball_tracker import TemporalBallTracker, BallState, TemporalBallPoint
from src.events.event_detector import TennisEventDetector, TennisEvent, EventType
from src.analytics.ball_speed_estimator import BallSpeedEstimator
from src.analytics.player_analytics import calculate_distance, calculate_speed
from src.visualization.video_annotator import VideoAnnotator

class Phase3Pipeline:
    """
    Phase 3 Complete Orchestrator:
    - Ultralytics YOLO11m Player Tracking (ByteTrack)
    - Ultralytics YOLO11s High-Accuracy Ball Detection
    - 2D Kinematic Kalman Filter Ball Tracking
    - 14-Point ResNet-50 Court Keypoint Detection & RANSAC Homography
    - Physics-Informed Tennis Event Detection (Bounces, Hits, Serves)
    - Segment-Level 2D Court-Projected Ball Speed Estimation
    - Visual Debug Video Rendering & Structured JSON Export
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.court_detector = CourtKeypointDetector(
            model_path=self.config['court_detection']['model']
        )
        self.player_detector = PlayerDetector(
            model_path=self.config['player_detection']['model']
        )
        self.ball_detector = YOLO11BallDetector(
            model_path=self.config['ball_detection']['model'],
            imgsz=self.config['ball_detection'].get('imgsz', 640),
            high_conf=self.config['ball_detection'].get('high_conf', 0.25),
            low_conf=self.config['ball_detection'].get('low_conf', 0.05)
        )
        self.temporal_tracker = TemporalBallTracker(
            high_conf_thresh=self.config['ball_detection'].get('high_conf', 0.25),
            low_conf_thresh=self.config['ball_detection'].get('low_conf', 0.05),
            max_prediction_gap=self.config['temporal_tracking'].get('max_prediction_gap', 3),
            max_interpolation_gap=self.config['temporal_tracking'].get('max_interpolation_gap', 3),
            max_valid_speed_px_per_frame=self.config['temporal_tracking'].get('max_valid_speed_px_per_frame', 60.0)
        )
        self.event_detector = TennisEventDetector(
            min_event_interval_frames=self.config.get('event_detection', {}).get('min_event_interval', 12),
            player_reach_radius_px=self.config.get('event_detection', {}).get('player_reach_radius_px', 160.0),
            min_hit_deflection_deg=self.config.get('event_detection', {}).get('min_hit_deflection_deg', 30.0),
            min_bounce_curvature=self.config.get('event_detection', {}).get('min_bounce_curvature', 0.002)
        )

    def run(self, input_video_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        print("=" * 55)
        print(f"Starting Phase 3 Pipeline (Events & Physics) on: {input_video_path}")
        print(f"Output directory: {output_dir}")
        print("=" * 55)
        
        # [Step 1] Ingest Video
        print("\n[Step 1/9] Ingesting video...")
        t0 = time.time()
        frames, meta = read_video(input_video_path)
        total_frames = len(frames)
        fps = meta.fps
        timestamps = [f_idx / fps for f_idx in range(total_frames)]
        print(f"  -> Loaded {total_frames} frames ({meta.width}x{meta.height}) at {fps:.2f} native FPS in {time.time()-t0:.2f}s")
        
        # [Step 2] Court Detection & Homography
        print("\n[Step 2/9] Detecting 14 Court Keypoints & Computing Homography...")
        t0 = time.time()
        court_keypoints = self.court_detector.predict(frames[0])
        canonical_kps = TennisCourtGeometry.get_canonical_keypoints()
        H, reproj_err = compute_homography(court_keypoints, canonical_kps)
        homography_valid = validate_homography(H, court_keypoints, canonical_kps, max_error=15.0)
        print(f"  -> Homography Reprojection Error: {reproj_err:.4f} (Valid: {homography_valid}) in {time.time()-t0:.2f}s")
        
        # [Step 3] Player Tracking (ByteTrack)
        print("\n[Step 3/9] Running Player Tracking (ByteTrack)...")
        t0 = time.time()
        from src.tracking.player_tracker import PlayerTracker
        from tqdm import tqdm
        all_player_dets = []
        for frame in tqdm(frames, desc="  Player Tracking"):
            dets = self.player_detector.detect_and_track(frame, persist=True)
            all_player_dets.append(dets)
            
        player_dict = PlayerTracker.choose_players(all_player_dets, court_keypoints)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]
            
        p1_cov = sum(1 for b in p1_boxes if b is not None) / total_frames * 100
        p2_cov = sum(1 for b in p2_boxes if b is not None) / total_frames * 100
        print(f"  -> Player 1 Coverage: {p1_cov:.1f}% | Player 2 Coverage: {p2_cov:.1f}% in {time.time()-t0:.2f}s")
        
        # [Step 4] YOLO11 Ball Proposal Extraction & Temporal Kalman Tracking
        print("\n[Step 4/9] Running High-Resolution Candidate Extraction & Temporal Kalman Tracking (YOLO11)...")
        t0 = time.time()
        imgsz = self.config['ball_detection'].get('imgsz', 640)
        from tqdm import tqdm
        all_candidates = []
        for f in tqdm(frames, desc="  Extracting Ball Candidates"):
            cands = self.ball_detector.extract_candidates(f, imgsz=imgsz)
            all_candidates.append(cands)
            
        ball_trajectory = self.temporal_tracker.track_video_candidates(all_candidates, fps=fps, frame_size=(meta.width, meta.height))
        print(f"  -> Ball Tracking Complete in {time.time()-t0:.2f}s")
        
        # [Step 5] Project Coordinates to Metric Court Plane
        print("\n[Step 5/9] Projecting Positions to Metric Court Plane...")
        t0 = time.time()
        for pt in ball_trajectory:
            if pt.x_px is not None and pt.y_px is not None:
                cx, cy = transform_point((pt.x_px, pt.y_px), H)
                pt.court_x_m = float(cx)
                pt.court_y_m = float(cy)
        print(f"  -> Metric positions mapped in {time.time()-t0:.2f}s")
        
        # [Step 6] Tennis Event Detection (Bounces, Hits, Serves)
        print("\n[Step 6/9] Detecting Tennis Match Events (Bounces, Hits, Serves)...")
        t0 = time.time()
        events = self.event_detector.detect_events(
            ball_trajectory=ball_trajectory,
            player1_boxes=p1_boxes,
            player2_boxes=p2_boxes,
            homography_matrix=H,
            fps=fps
        )
        bounces_count = sum(1 for e in events if e.event_type == EventType.BOUNCE)
        hits_count = sum(1 for e in events if e.event_type in (EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT, EventType.SERVE_CONTACT))
        print(f"  -> Detected {len(events)} Total Match Events: {bounces_count} Bounces, {hits_count} Racket Hits in {time.time()-t0:.2f}s")
        
        # [Step 7] Segment-Level Ball Speed Estimation
        print("\n[Step 7/9] Computing Segment-Level 2D Court-Projected Ball Speed...")
        t0 = time.time()
        per_frame_speeds, flight_segments, speed_summary = BallSpeedEstimator.estimate_speeds(
            trajectory=ball_trajectory,
            events=events,
            fps=fps
        )
        print(f"  -> Speed Estimation Complete: {len(flight_segments)} Flight Segments (Avg: {speed_summary.get('average_speed_kmh', 0.0)} km/h, Max: {speed_summary.get('maximum_speed_kmh', 0.0)} km/h) in {time.time()-t0:.2f}s")
        
        # [Step 8] Player Physical Analytics
        print("\n[Step 8/9] Computing Player Movement Analytics...")
        p1_positions = []
        p2_positions = []
        for i in range(total_frames):
            p1_b = p1_boxes[i]
            p2_b = p2_boxes[i]
            if p1_b is not None:
                foot_px = ((p1_b.x1 + p1_b.x2)/2.0, p1_b.y2)
                p1_positions.append(transform_point(foot_px, H))
            else:
                p1_positions.append(None)
                
            if p2_b is not None:
                foot_px = ((p2_b.x1 + p2_b.x2)/2.0, p2_b.y2)
                p2_positions.append(transform_point(foot_px, H))
            else:
                p2_positions.append(None)
                
        p1_dist = calculate_distance(p1_positions)
        p2_dist = calculate_distance(p2_positions)
        p1_spd_raw = calculate_speed(p1_positions, timestamps)
        p2_spd_raw = calculate_speed(p2_positions, timestamps)
        p1_spd = [s * 3.6 if s is not None else 0.0 for s in p1_spd_raw]
        p2_spd = [s * 3.6 if s is not None else 0.0 for s in p2_spd_raw]
        
        # [Step 9] Render Visual Debug Video
        print("\n[Step 9/9] Rendering Visual Debug Video with Match Events...")
        t0 = time.time()
        annotated_frames = []
        mini_court = MiniCourt(width=250, height=500)
        
        recent_ball_positions = []
        event_dict_by_frame = {ev.frame_index: ev for ev in events}
        active_event_banner: Optional[Tuple[str, tuple, int]] = None  # text, color, expire_frame
        
        # Collect persistent bounces on mini court
        bounces_court_points = [
            ev.court_position_m for ev in events 
            if ev.event_type == EventType.BOUNCE and ev.court_position_m is not None
        ]
        
        for i in tqdm(range(total_frames), desc="  Rendering Video"):
            frame = frames[i].copy()
            
            # 1. Draw Player Boxes
            player_bboxes = {}
            if p1_boxes[i] is not None:
                player_bboxes[1] = p1_boxes[i]
            if p2_boxes[i] is not None:
                player_bboxes[2] = p2_boxes[i]
            frame = VideoAnnotator.draw_player_boxes(frame, player_bboxes, {1: (255, 0, 0), 2: (0, 0, 255)})
            
            # 2. Draw Court Keypoints
            frame = VideoAnnotator.draw_court_keypoints(frame, court_keypoints)
            
            # 3. Draw Ball & Trajectory Trail
            b_pt = ball_trajectory[i]
            if b_pt.x_px is not None and b_pt.y_px is not None:
                recent_ball_positions.append((b_pt.x_px, b_pt.y_px))
                frame = VideoAnnotator.draw_ball(frame, (b_pt.x_px, b_pt.y_px), b_pt.state)
                
            frame = VideoAnnotator.draw_ball_trajectory(frame, recent_ball_positions, max_trail=25)
            
            # 4. Check for Match Events & Display Banner
            if i in event_dict_by_frame:
                ev = event_dict_by_frame[i]
                if ev.event_type == EventType.SERVE_CONTACT:
                    active_event_banner = (f"SERVE CONTACT (Player {ev.player_id})", (0, 255, 255), i + 12)
                elif ev.event_type == EventType.BOUNCE:
                    active_event_banner = ("COURT BOUNCE", (0, 165, 255), i + 12)
                elif ev.event_type in (EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT):
                    active_event_banner = (f"PLAYER {ev.player_id} HIT", (255, 100, 0), i + 12)
                    
            if active_event_banner is not None and i <= active_event_banner[2]:
                frame = VideoAnnotator.draw_event_badge(frame, active_event_banner[0], active_event_banner[1], i)
                
            # 5. Draw Mini-Court Overlay with Players, Ball, and Bounces
            ball_court_pos = (b_pt.court_x_m, b_pt.court_y_m) if b_pt.court_x_m is not None else None
            p1_c_pos = p1_positions[i]
            p2_c_pos = p2_positions[i]
            
            mc_canvas = mini_court.draw_court()
            mc_img = mini_court.draw_positions(
                court_image=mc_canvas,
                player1_pos=p1_c_pos,
                player2_pos=p2_c_pos,
                ball_pos=ball_court_pos
            )
            
            # Draw persistent bounce circles on mini court
            for b_pos in bounces_court_points:
                bx, by = mini_court.court_to_mini_court(b_pos[0], b_pos[1])
                cv2.circle(mc_img, (bx, by), 5, (0, 165, 255), -1)
                cv2.circle(mc_img, (bx, by), 7, (255, 255, 255), 1)
                
            # 6. Compose Top-Right Mini Court & HUD
            frame = VideoAnnotator.compose_frame(frame, mc_img)
            
            # Telemetry HUD on Left
            cv2.putText(frame, f"Frame: {i:03d} / {total_frames} | Time: {timestamps[i]:.2f}s", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"P1: Dist {p1_dist:.1f}m | Speed {p1_spd[i]:.1f} km/h", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2)
            cv2.putText(frame, f"P2: Dist {p2_dist:.1f}m | Speed {p2_spd[i]:.1f} km/h", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2)
            
            spd_val = per_frame_speeds[i] if i < len(per_frame_speeds) and per_frame_speeds[i] is not None else 0.0
            cv2.putText(frame, f"Ball 2D Speed: {spd_val:.1f} km/h | State: {b_pt.state.value}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 120), 2)
                        
            annotated_frames.append(frame)
            
        video_out_path = os.path.join(output_dir, "annotated.mp4")
        save_video(annotated_frames, video_out_path, fps=fps)
        print(f"  -> Rendered Phase 3 Video: {video_out_path} in {time.time()-t0:.2f}s")
        
        # [Step 10] Export Structured JSON Artifacts
        print("\nExporting Structured JSON Artifacts...")
        
        # match_events.json
        events_json_data = {
            "events_count": len(events),
            "summary": {
                "serves": sum(1 for e in events if e.event_type == EventType.SERVE_CONTACT),
                "bounces": bounces_count,
                "player_hits": hits_count
            },
            "events": [
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type.value,
                    "frame": ev.frame_index,
                    "timestamp_s": round(ev.timestamp_s, 4),
                    "player_id": ev.player_id,
                    "ball_position_px": [round(ev.ball_position_px[0], 2), round(ev.ball_position_px[1], 2)],
                    "court_position_m": [round(ev.court_position_m[0], 2), round(ev.court_position_m[1], 2)] if ev.court_position_m else None,
                    "confidence": round(ev.confidence, 3),
                    "trajectory_state": ev.trajectory_state,
                    "evidence": ev.evidence
                }
                for ev in events
            ]
        }
        with open(os.path.join(output_dir, "match_events.json"), 'w') as f:
            json.dump(events_json_data, f, indent=2)
            
        # ball_metrics.json
        ball_metrics_data = {
            "speed_overview": speed_summary,
            "flight_segments": [
                {
                    "segment_id": s.segment_id,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "start_event": s.start_event,
                    "end_event": s.end_event,
                    "duration_s": s.duration_s,
                    "total_2d_distance_m": s.total_2d_distance_m,
                    "mean_speed_kmh": s.mean_speed_kmh,
                    "peak_speed_kmh": s.peak_speed_kmh,
                    "launch_speed_kmh": s.launch_speed_kmh,
                    "confidence_tier": s.confidence_tier,
                    "confidence_score": s.confidence_score
                }
                for s in flight_segments
            ]
        }
        with open(os.path.join(output_dir, "ball_metrics.json"), 'w') as f:
            json.dump(ball_metrics_data, f, indent=2)
            
        # trajectories.json
        traj_export = {
            "total_frames": total_frames,
            "fps": fps,
            "ball_trajectory": [
                {
                    "frame_index": int(p.frame_index),
                    "timestamp_seconds": float(round(p.timestamp_seconds, 4)),
                    "x_px": float(round(p.x_px, 2)) if p.x_px is not None else None,
                    "y_px": float(round(p.y_px, 2)) if p.y_px is not None else None,
                    "court_x_m": float(round(p.court_x_m, 2)) if p.court_x_m is not None else None,
                    "court_y_m": float(round(p.court_y_m, 2)) if p.court_y_m is not None else None,
                    "speed_kmh": float(round(p.speed_kmh, 1)) if p.speed_kmh is not None else None,
                    "confidence": float(round(p.confidence, 3)) if p.confidence is not None else None,
                    "state": p.state.value
                }
                for p in ball_trajectory
            ]
        }
        with open(os.path.join(output_dir, "trajectories.json"), 'w') as f:
            json.dump(traj_export, f, indent=2)
            
        # detections.json
        det_export = {
            "metadata": {
                "video_path": input_video_path,
                "total_frames": int(total_frames),
                "fps": float(fps),
                "duration_seconds": float(total_frames / fps)
            },
            "frames": [
                {
                    "frame_index": int(i),
                    "timestamp_seconds": float(round(timestamps[i], 4)),
                    "player_1": {
                        "bbox": [float(round(p1_boxes[i].x1, 1)), float(round(p1_boxes[i].y1, 1)), float(round(p1_boxes[i].x2, 1)), float(round(p1_boxes[i].y2, 1))],
                        "court_position_meters": [float(round(p1_positions[i][0], 2)), float(round(p1_positions[i][1], 2))] if p1_positions[i] else None
                    } if p1_boxes[i] else None,
                    "player_2": {
                        "bbox": [float(round(p2_boxes[i].x1, 1)), float(round(p2_boxes[i].y1, 1)), float(round(p2_boxes[i].x2, 1)), float(round(p2_boxes[i].y2, 1))],
                        "court_position_meters": [float(round(p2_positions[i][0], 2)), float(round(p2_positions[i][1], 2))] if p2_positions[i] else None
                    } if p2_boxes[i] else None
                }
                for i in range(total_frames)
            ]
        }
        with open(os.path.join(output_dir, "detections.json"), 'w') as f:
            json.dump(det_export, f, indent=2)
            
        # court_geometry.json
        court_export = {
            "keypoints": court_keypoints.tolist(),
            "canonical_keypoints": canonical_kps.tolist(),
            "homography_matrix": H.tolist(),
            "reprojection_error": float(reproj_err),
            "valid": bool(homography_valid)
        }
        with open(os.path.join(output_dir, "court_geometry.json"), 'w') as f:
            json.dump(court_export, f, indent=2)
            
        # player_metrics.json
        p_metrics_export = {
            "player_1": {
                "total_distance_meters": float(p1_dist),
                "average_speed_kmh": float(np.mean([s for s in p1_spd if s > 0])) if any(s > 0 for s in p1_spd) else 0.0,
                "peak_speed_kmh": float(np.max(p1_spd)) if p1_spd else 0.0
            },
            "player_2": {
                "total_distance_meters": float(p2_dist),
                "average_speed_kmh": float(np.mean([s for s in p2_spd if s > 0])) if any(s > 0 for s in p2_spd) else 0.0,
                "peak_speed_kmh": float(np.max(p2_spd)) if p2_spd else 0.0
            }
        }
        with open(os.path.join(output_dir, "player_metrics.json"), 'w') as f:
            json.dump(p_metrics_export, f, indent=2)
            
        # metrics.json & run_config.yaml
        total_time = time.time() - start_time
        metrics_export = {
            "status": "success",
            "phase": "phase_3_tennis_events_physics",
            "pipeline_performance": {
                "total_processing_time_seconds": total_time,
                "processing_fps": total_frames / total_time
            },
            "events_metrics": {
                "events_count": len(events),
                "bounces": bounces_count,
                "player_hits": hits_count
            },
            "tracking_metrics": {
                "player_1_coverage_pct": p1_cov,
                "player_2_coverage_pct": p2_cov,
                "total_valid_ball_coverage_pct": sum(1 for p in ball_trajectory if p.state != BallState.MISSING) / total_frames * 100
            }
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_export, f, indent=2)
            
        with open(os.path.join(output_dir, "run_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
            
        print("\n" + "=" * 55)
        print(f"Phase 3 Pipeline Completed Successfully in {total_time:.2f}s ({total_frames/total_time:.1f} FPS)")
        print(f"Match Events Reconstructed: {len(events)} events ({bounces_count} bounces, {hits_count} hits)")
        print("=" * 55)
        
        return metrics_export

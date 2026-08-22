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
from src.line_calling.line_geometry import CourtLineGeometry, CourtLineType, ServiceBoxType
from src.line_calling.contact_refinement import BounceContactRefiner
from src.line_calling.line_call_engine import TennisLineCallEngine, LineCallDecision, LineCallContext, LineCallEvidence

class Phase4Pipeline:
    """
    Phase 4 Assisted Tennis IN/OUT Line-Calling Orchestrator:
    - Ultralytics YOLO11m Player Tracking (ByteTrack)
    - Ultralytics YOLO11s High-Accuracy Ball Detection
    - 2D Kinematic Kalman Filter Ball Tracking
    - 14-Point ResNet-50 Court Keypoint Detection & Corrected Canonical Homography
    - Physics-Informed Tennis Event Detection (Serves, Bounces, Hits)
    - Uncertainty-Aware Assisted IN/OUT & Serve Line-Calling Engine
    - 2D Court-Projected Ball Speed Estimation
    - Visual Debug Video Rendering & Structured Evidence JSON Export
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
            min_event_interval_frames=self.config.get('event_detection', {}).get('min_event_interval', 10),
            player_reach_radius_px=self.config.get('event_detection', {}).get('player_reach_radius_px', 140.0),
            min_hit_deflection_deg=self.config.get('event_detection', {}).get('min_hit_deflection_deg', 40.0),
            min_bounce_curvature=self.config.get('event_detection', {}).get('min_bounce_curvature', 0.005)
        )
        self.line_call_engine = TennisLineCallEngine(
            refiner=BounceContactRefiner(search_window_radius=self.config.get('line_calling', {}).get('contact_refinement_window', 2)),
            effective_ball_radius_cm=self.config.get('line_calling', {}).get('effective_ball_radius_cm', 3.35),
            uncertainty_safety_factor=self.config.get('line_calling', {}).get('uncertainty_safety_factor', 1.5)
        )

    def run(self, input_video_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        print("=" * 60)
        print(f"Starting Phase 4 Pipeline (Line Calling Engine) on: {input_video_path}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        # 1. Ingest Video
        print("\n[Step 1/9] Ingesting video...")
        t0 = time.time()
        frames, metadata = read_video(input_video_path)
        total_frames = len(frames)
        fps = metadata.fps
        h, w = frames[0].shape[:2]
        print(f"  -> Loaded {total_frames} frames ({w}x{h}) at {fps:.2f} native FPS in {time.time()-t0:.2f}s")

        # 2. Court Keypoint Detection & Homography
        print("\n[Step 2/9] Detecting 14 Court Keypoints & Computing Canonical Homography...")
        t0 = time.time()
        court_keypoints = self.court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        homography_matrix, reproj_err = compute_homography(court_keypoints, canonical_keypoints)
        is_h_valid = validate_homography(homography_matrix, court_keypoints, canonical_keypoints)
        print(f"  -> Homography Reprojection Error: {reproj_err:.4f} px (Valid: {is_h_valid}) in {time.time()-t0:.2f}s")

        # 3. Player Detection & Tracking (ByteTrack)
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

        # 4. Ball Detection & Temporal Kalman Tracking (YOLO11)
        print("\n[Step 4/9] Running High-Resolution Candidate Extraction & Temporal Kalman Tracking (YOLO11)...")
        t0 = time.time()
        imgsz = self.config['ball_detection'].get('imgsz', 640)
        all_candidates = []
        for f in tqdm(frames, desc="  Extracting Ball Candidates"):
            cands = self.ball_detector.extract_candidates(f, imgsz=imgsz)
            all_candidates.append(cands)
            
        ball_points = self.temporal_tracker.track_video_candidates(all_candidates, fps=fps)
        print(f"  -> Ball Tracking Complete in {time.time()-t0:.2f}s")

        # 5. Metric Court Projection
        print("\n[Step 5/9] Projecting Ball Positions to Metric Court Plane...")
        t0 = time.time()
        for pt in ball_points:
            if pt.x_px is not None and pt.y_px is not None and is_h_valid:
                court_pt = transform_point((pt.x_px, pt.y_px), homography_matrix)
                pt.court_x_m = float(court_pt[0])
                pt.court_y_m = float(court_pt[1])
        print(f"  -> Metric positions mapped in {time.time()-t0:.2f}s")

        # 6. Tennis Match Event Detection
        print("\n[Step 6/9] Detecting Tennis Match Events (Bounces, Hits, Serves)...")
        t0 = time.time()
        detected_events = self.event_detector.detect_events(
            ball_trajectory=ball_points,
            player1_boxes=p1_boxes,
            player2_boxes=p2_boxes,
            homography_matrix=homography_matrix,
            fps=fps
        )
        bounces = [e for e in detected_events if e.event_type == EventType.BOUNCE]
        hits = [e for e in detected_events if e.event_type in (EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT, EventType.SERVE_CONTACT)]
        print(f"  -> Detected {len(detected_events)} Total Match Events: {len(bounces)} Bounces, {len(hits)} Racket Hits in {time.time()-t0:.2f}s")

        # 7. Assisted Line-Calling Decision Engine
        print("\n[Step 7/9] Running Assisted IN/OUT & Serve Line-Calling Engine...")
        t0 = time.time()
        line_call_evidences: List[LineCallEvidence] = []
        is_first_bounce = True
        
        for ev in detected_events:
            if ev.event_type == EventType.BOUNCE:
                ctx = LineCallContext.SERVE if is_first_bounce else LineCallContext.RALLY
                box = ServiceBoxType.NEAR_DEUCE if is_first_bounce else None
                is_first_bounce = False

                evidence = self.line_call_engine.evaluate_bounce(
                    event_id=ev.event_id,
                    bounce_frame=ev.frame_index,
                    ball_trajectory=ball_points,
                    homography_matrix=homography_matrix,
                    context=ctx,
                    target_service_box=box,
                    raw_frames=frames
                )
                line_call_evidences.append(evidence)
                print(f"  -> Bounce Frame {ev.frame_index:3d} ({ctx.value:5s}): Decision={evidence.decision.value:15s} | Margin={evidence.ball_edge_margin_cm:+6.1f}cm | Unc=±{evidence.position_uncertainty_cm:4.1f}cm | Conf={evidence.confidence:.2f} | Reason: {evidence.reason}")
        print(f"  -> Completed Line Calling for {len(line_call_evidences)} Bounces in {time.time()-t0:.2f}s")

        # 8. Ball Speed & Player Analytics
        print("\n[Step 8/9] Computing Ball Speed & Player Analytics...")
        t0 = time.time()
        per_frame_speeds, speed_segments, speed_summary = BallSpeedEstimator.estimate_speeds(
            trajectory=ball_points,
            events=detected_events,
            fps=fps
        )
        for i in range(total_frames):
            ball_points[i].speed_kmh = per_frame_speeds[i]
        
        # Player 1 & 2 movement metrics
        p1_positions = []
        p2_positions = []
        timestamps = [i / fps for i in range(total_frames)]
        for i in range(total_frames):
            p1_b = p1_boxes[i]
            p2_b = p2_boxes[i]
            if p1_b is not None and is_h_valid:
                foot_px = ((p1_b.x1 + p1_b.x2)/2.0, p1_b.y2)
                p1_positions.append(transform_point(foot_px, homography_matrix))
            else:
                p1_positions.append(None)
                
            if p2_b is not None and is_h_valid:
                foot_px = ((p2_b.x1 + p2_b.x2)/2.0, p2_b.y2)
                p2_positions.append(transform_point(foot_px, homography_matrix))
            else:
                p2_positions.append(None)

        p1_dist = calculate_distance(p1_positions)
        p2_dist = calculate_distance(p2_positions)
        p1_spd_raw = calculate_speed(p1_positions, timestamps)
        p2_spd_raw = calculate_speed(p2_positions, timestamps)
        p1_speed = float(np.mean([s * 3.6 for s in p1_spd_raw if s is not None])) if p1_spd_raw else 0.0
        p2_speed = float(np.mean([s * 3.6 for s in p2_spd_raw if s is not None])) if p2_spd_raw else 0.0
        print(f"  -> Speed Estimation Complete ({len(speed_segments)} segments) in {time.time()-t0:.2f}s")

        # 9. Rendering Visual Debug Video with Line Call Overlays
        print("\n[Step 9/9] Rendering Visual Debug Video with Line-Calling Overlays...")
        t0 = time.time()
        mini_court = MiniCourt(width=250, height=500)
        output_video_path = os.path.join(output_dir, "annotated.mp4")
        annotated_frames = []
        recent_ball_positions = []
        
        # Map bounce frame to line call evidence for banner rendering
        line_call_map = {e.bounce_frame: e for e in line_call_evidences}
        bounces_court_points = [(e.bounce_position_m[0], e.bounce_position_m[1]) for e in line_call_evidences]

        for i, frame in enumerate(frames):
            ann_frame = frame.copy()
            
            # 1. Draw Player Boxes
            player_bboxes = {}
            if p1_boxes[i] is not None:
                player_bboxes[1] = p1_boxes[i]
            if p2_boxes[i] is not None:
                player_bboxes[2] = p2_boxes[i]
            ann_frame = VideoAnnotator.draw_player_boxes(ann_frame, player_bboxes, {1: (255, 0, 0), 2: (0, 0, 255)})
            
            # 2. Draw Court Keypoints
            ann_frame = VideoAnnotator.draw_court_keypoints(ann_frame, court_keypoints)
            
            # 3. Draw Ball & Trajectory Trail
            b_pt = ball_points[i]
            if b_pt.x_px is not None and b_pt.y_px is not None:
                recent_ball_positions.append((b_pt.x_px, b_pt.y_px))
                ann_frame = VideoAnnotator.draw_ball(ann_frame, (b_pt.x_px, b_pt.y_px), b_pt.state)
                
            ann_frame = VideoAnnotator.draw_ball_trajectory(ann_frame, recent_ball_positions, max_trail=25)
            
            # 4. Check for Match Events & Display Banner
            for ev in detected_events:
                if abs(ev.frame_index - i) <= 6:
                    ann_frame = VideoAnnotator.draw_event_badge(ann_frame, ev.event_type.value, (0, 255, 255), i)

            # 5. Phase 4 Line Call Banner (Active for 18 frames after bounce)
            for b_frame, evidence in line_call_map.items():
                if 0 <= (i - b_frame) <= 18:
                    call_text = f"LINE CALL: [{evidence.decision.value}] ({evidence.nearest_line.value} {evidence.ball_edge_margin_cm:+.1f}cm, Unc: +- {evidence.position_uncertainty_cm:.1f}cm)"
                    bg_color = (0, 180, 0) if "IN" in evidence.decision.value else ((0, 0, 200) if "OUT" in evidence.decision.value or "FAULT" in evidence.decision.value else (0, 165, 255))
                    
                    cv2.rectangle(ann_frame, (40, 110), (1200, 160), bg_color, -1)
                    cv2.rectangle(ann_frame, (40, 110), (1200, 160), (255, 255, 255), 2)
                    cv2.putText(ann_frame, call_text, (55, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # 6. Mini-court overlay
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
                
            ann_frame = VideoAnnotator.compose_frame(ann_frame, mc_img)

            # Telemetry HUD on Left
            cv2.putText(ann_frame, f"Frame: {i:03d} / {total_frames} | Time: {timestamps[i]:.2f}s", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(ann_frame, f"P1: Dist {p1_dist:.1f}m | Speed {p1_speed:.1f} km/h", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2)
            cv2.putText(ann_frame, f"P2: Dist {p2_dist:.1f}m | Speed {p2_speed:.1f} km/h", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2)
            
            spd_val = per_frame_speeds[i] if i < len(per_frame_speeds) and per_frame_speeds[i] is not None else 0.0
            cv2.putText(ann_frame, f"Ball 2D Speed: {spd_val:.1f} km/h | State: {b_pt.state.value}", (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 120), 2)

            annotated_frames.append(ann_frame)

        save_video(annotated_frames, output_video_path, fps=fps)
        print(f"  -> Rendered Phase 4 Video: {output_video_path} in {time.time()-t0:.2f}s")
        print(f"  -> Rendered Phase 4 Video: {output_video_path} in {time.time()-t0:.2f}s")

        # 10. Export Structured JSON Artifacts
        print("\nExporting Structured JSON Artifacts...")
        self._export_artifacts(
            output_dir=output_dir,
            input_video_path=input_video_path,
            total_frames=total_frames,
            fps=fps,
            court_keypoints=court_keypoints,
            homography_matrix=homography_matrix,
            reproj_err=reproj_err,
            p1_boxes=p1_boxes,
            p2_boxes=p2_boxes,
            p1_positions=p1_positions,
            p2_positions=p2_positions,
            p1_cov=p1_cov,
            p2_cov=p2_cov,
            ball_points=ball_points,
            detected_events=detected_events,
            line_call_evidences=line_call_evidences,
            speed_segments=speed_segments,
            speed_summary=speed_summary,
            p1_dist=p1_dist,
            p2_dist=p2_dist,
            p1_speed=p1_speed,
            p2_speed=p2_speed,
            start_time=start_time
        )

        total_pipeline_time = time.time() - start_time
        fps_pipeline = total_frames / total_pipeline_time
        print(f"\n=======================================================")
        print(f"Phase 4 Pipeline Completed Successfully in {total_pipeline_time:.2f}s ({fps_pipeline:.1f} FPS)")
        print(f"Line Calls Issued: {len(line_call_evidences)} Bounces Evaluated with Uncertainty Gating")
        print(f"=======================================================")

        return {
            "status": "success",
            "phase": "phase_4_assisted_line_calling",
            "pipeline_performance": {
                "total_processing_time_seconds": total_pipeline_time,
                "processing_fps": fps_pipeline
            },
            "line_calling_metrics": {
                "bounces_evaluated": len(line_call_evidences),
                "decisions": [e.decision.value for e in line_call_evidences]
            }
        }

    def _export_artifacts(
        self,
        output_dir: str,
        input_video_path: str,
        total_frames: int,
        fps: float,
        court_keypoints: np.ndarray,
        homography_matrix: np.ndarray,
        reproj_err: float,
        p1_boxes: List[Any],
        p2_boxes: List[Any],
        p1_positions: List[Any],
        p2_positions: List[Any],
        p1_cov: float,
        p2_cov: float,
        ball_points: List[TemporalBallPoint],
        detected_events: List[TennisEvent],
        line_call_evidences: List[LineCallEvidence],
        speed_segments: List[Any],
        speed_summary: Dict[str, Any],
        p1_dist: float,
        p2_dist: float,
        p1_speed: float,
        p2_speed: float,
        start_time: float
    ):
        # 1. detections.json
        det_data = {
            "metadata": {"video": input_video_path, "frames": total_frames, "fps": fps},
            "frames": []
        }
        for i in range(total_frames):
            p1_b = p1_boxes[i]
            p2_b = p2_boxes[i]
            det_data["frames"].append({
                "frame_index": i,
                "player_1": {"bbox": [float(p1_b.x1), float(p1_b.y1), float(p1_b.x2), float(p1_b.y2)]} if p1_b else None,
                "player_2": {"bbox": [float(p2_b.x1), float(p2_b.y1), float(p2_b.x2), float(p2_b.y2)]} if p2_b else None,
                "ball": {
                    "position_px": [float(ball_points[i].x_px), float(ball_points[i].y_px)] if ball_points[i].x_px is not None else None,
                    "confidence": float(ball_points[i].confidence) if ball_points[i].confidence is not None else None,
                    "state": ball_points[i].state.value
                }
            })
        with open(os.path.join(output_dir, "detections.json"), 'w') as f:
            json.dump(det_data, f, indent=2)

        # 2. trajectories.json
        p1_c_clean = [[float(p[0]), float(p[1])] if p is not None else None for p in p1_positions]
        p2_c_clean = [[float(p[0]), float(p[1])] if p is not None else None for p in p2_positions]
        traj_data = {
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
                    "state": p.state.value if p.state else "MISSING"
                }
                for p in ball_points
            ],
            "player_1_court_positions": p1_c_clean,
            "player_2_court_positions": p2_c_clean
        }
        with open(os.path.join(output_dir, "trajectories.json"), 'w') as f:
            json.dump(traj_data, f, indent=2)

        # 3. court_geometry.json
        court_data = {
            "court_keypoints_px": [[float(x), float(y)] for x, y in court_keypoints],
            "canonical_keypoints_m": [[float(x), float(y)] for x, y in TennisCourtGeometry.get_canonical_keypoints()],
            "homography_matrix": [[float(v) for v in row] for row in homography_matrix],
            "reprojection_error_px": float(reproj_err),
            "is_valid": bool(validate_homography(homography_matrix, court_keypoints, TennisCourtGeometry.get_canonical_keypoints()))
        }
        with open(os.path.join(output_dir, "court_geometry.json"), 'w') as f:
            json.dump(court_data, f, indent=2)

        # 4. match_events.json
        events_data = {
            "events_count": len(detected_events),
            "events": [
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type.value,
                    "frame": ev.frame_index,
                    "timestamp_s": round(ev.timestamp_s, 4),
                    "player_id": ev.player_id,
                    "ball_position_px": [round(ev.ball_position_px[0], 2), round(ev.ball_position_px[1], 2)] if ev.ball_position_px else None,
                    "court_position_m": [round(ev.court_position_m[0], 2), round(ev.court_position_m[1], 2)] if ev.court_position_m else None,
                    "confidence": round(ev.confidence, 3),
                    "trajectory_state": ev.trajectory_state,
                    "evidence": ev.evidence
                }
                for ev in detected_events
            ]
        }
        with open(os.path.join(output_dir, "match_events.json"), 'w') as f:
            json.dump(events_data, f, indent=2)

        # 5. line_calls.json
        line_calls_data = {
            "line_calls_count": len(line_call_evidences),
            "line_calls": [e.to_dict() for e in line_call_evidences]
        }
        with open(os.path.join(output_dir, "line_calls.json"), 'w') as f:
            json.dump(line_calls_data, f, indent=2)

        # 6. player_metrics.json
        player_metrics = {
            "player_1": {"total_distance_m": round(p1_dist, 2), "avg_speed_kmh": round(p1_speed, 2), "coverage_pct": p1_cov},
            "player_2": {"total_distance_m": round(p2_dist, 2), "avg_speed_kmh": round(p2_speed, 2), "coverage_pct": p2_cov}
        }
        with open(os.path.join(output_dir, "player_metrics.json"), 'w') as f:
            json.dump(player_metrics, f, indent=2)

        # 7. ball_metrics.json
        ball_metrics_data = {
            "scientific_status": "EXPERIMENTAL_2D_ESTIMATE",
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
                for s in speed_segments
            ]
        }
        with open(os.path.join(output_dir, "ball_metrics.json"), 'w') as f:
            json.dump(ball_metrics_data, f, indent=2)

        # 8. metrics.json & run_config.yaml
        summary_metrics = {
            "pipeline_fps": round(total_frames / (time.time() - start_time), 2),
            "line_calls": len(line_call_evidences),
            "events": len(detected_events),
            "court_reprojection_error_px": round(reproj_err, 4)
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(summary_metrics, f, indent=2)

        with open(os.path.join(output_dir, "run_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)

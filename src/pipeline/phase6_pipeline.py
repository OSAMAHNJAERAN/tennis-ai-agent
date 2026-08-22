import os
import time
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.video_io import read_video, save_video
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, validate_homography, transform_point
from src.detection.player_detector import PlayerDetector
from src.tracking.player_tracker import PlayerTracker
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.temporal_ball_tracker import TemporalBallTracker, BallState
from src.events.event_detector import TennisEventDetector, EventType, TennisEvent
from src.line_calling.line_call_engine import (
    TennisLineCallEngine,
    LineCallEvidence,
    LineCallDecision,
    LineCallContext
)
from src.line_calling.line_geometry import ServiceBoxType, ContactPatchModelType
from src.line_calling.contact_refinement import BounceContactRefiner
from src.scoring.match_state import MatchState, MatchFormat, SetFormat, ServerCourtEnd, BallPlayState
from src.scoring.scoring_engine import TennisScoringEngine
from src.scoring.point_outcome_resolver import PointOutcomeType
from src.analytics.player_analytics import calculate_distance, calculate_speed
from src.analytics.ball_speed_estimator import BallSpeedEstimator
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.mini_court import MiniCourt

# Phase 6 Modules
from src.shot_analysis.shot_types import (
    ShotType,
    ShotDirection,
    PlayerHandedness,
    ShotEventEvidence
)
from src.shot_analysis.pose_feature_extractor import PoseFeatureExtractor
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.court_zones import CourtZoneEngine
from src.shot_analysis.shot_linker import TennisShotLinker
from src.analytics.rally_analyzer import RallyAnalyzer, RallySegment
from src.analytics.serve_analyzer import ServeAnalyzer
from src.analytics.shot_statistics import ShotStatisticsAnalyzer
from src.analytics.match_analytics import MatchAnalyticsAggregator

class Phase6Pipeline:
    """
    Phase 6 End-to-End Orchestrator.
    Executes Tennis Vision System through Shot Understanding & Advanced Match Analytics.
    """

    def __init__(self, config_path: str = "configs/phase6_analytics/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.court_detector = CourtKeypointDetector(model_path=self.config['court_detection']['model_path'])
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
        
        model_name = self.config.get('line_calling', {}).get('contact_patch_model', 'EMPIRICAL_PATCH')
        try:
            patch_model = ContactPatchModelType(model_name)
        except Exception:
            patch_model = ContactPatchModelType.EMPIRICAL_PATCH

        self.line_call_engine = TennisLineCallEngine(
            refiner=BounceContactRefiner(
                window_radius=self.config.get('line_calling', {}).get('contact_refinement_window', 3)
            ),
            contact_patch_model=patch_model,
            contact_patch_radius_cm=self.config.get('line_calling', {}).get('contact_patch_radius_cm', 1.25),
            uncertainty_safety_factor=self.config.get('line_calling', {}).get('uncertainty_safety_factor', 1.5),
            line_width_uncertainty_cm=self.config.get('line_calling', {}).get('line_width_uncertainty_cm', 1.25)
        )

        # Scoring Engine
        m_cfg = self.config.get('match_configuration', {})
        self.scoring_engine = TennisScoringEngine(
            player_1_id=m_cfg.get('player_1_id', 1),
            player_2_id=m_cfg.get('player_2_id', 2),
            initial_server_id=m_cfg.get('initial_server_id', 2),
            match_id=m_cfg.get('match_id', "match_phase6_001"),
            match_format=MatchFormat(m_cfg.get('format', "BEST_OF_3")),
            set_format=SetFormat(m_cfg.get('set_format', "TIE_BREAK_SET"))
        )

        # Pose & Shot Classifier
        pose_cfg = self.config.get('pose_estimation', {})
        self.pose_extractor = None
        if pose_cfg.get('enabled', True):
            self.pose_extractor = PoseFeatureExtractor(
                model_path=pose_cfg.get('model_path', 'yolo11n-pose.pt')
            )

        shot_cfg = self.config.get('shot_classification', {})
        handedness_map = {
            1: PlayerHandedness(shot_cfg.get('players_handedness', {}).get(1, "RIGHT_HANDED")),
            2: PlayerHandedness(shot_cfg.get('players_handedness', {}).get(2, "RIGHT_HANDED"))
        }
        self.shot_classifier = TennisShotClassifier(
            pose_extractor=self.pose_extractor,
            min_pose_confidence=pose_cfg.get('min_pose_confidence', 0.35),
            ambiguity_threshold=shot_cfg.get('ambiguity_threshold', 0.15),
            handedness_map=handedness_map
        )
        self.shot_linker = TennisShotLinker(classifier=self.shot_classifier)

    def run(self, input_video_path: str, output_dir: str = "outputs/phase6_shot_analytics_1") -> Dict[str, Any]:
        """Runs the complete Phase 6 pipeline."""
        t_pipeline_start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        print("="*60)
        print(f"Starting Phase 6 Pipeline on: {input_video_path}")
        print(f"Output directory: {output_dir}")
        print("="*60)

        # 1. Ingest Video
        print("\n[Step 1/11] Ingesting video...")
        t0 = time.time()
        frames, metadata = read_video(input_video_path)
        total_frames = len(frames)
        fps = metadata.fps
        h, w = frames[0].shape[:2]
        print(f"  -> Loaded {total_frames} frames ({w}x{h}) at {fps:.2f} native FPS in {time.time()-t0:.2f}s")

        # 2. Court Keypoints & Homography
        print("\n[Step 2/11] Detecting 14 Court Keypoints & Computing Canonical Homography...")
        t0 = time.time()
        court_keypoints = self.court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        homography_matrix, reproj_err = compute_homography(court_keypoints, canonical_keypoints)
        is_h_valid = validate_homography(homography_matrix, court_keypoints, canonical_keypoints)
        print(f"  -> Homography Reprojection Error: {reproj_err:.4f} px (Valid: {is_h_valid}) in {time.time()-t0:.2f}s")

        # 3. Player Tracking (ByteTrack)
        print("\n[Step 3/11] Running Player Tracking (ByteTrack)...")
        t0 = time.time()
        all_player_dets = []
        for frame in tqdm(frames, desc="  Player Tracking"):
            dets = self.player_detector.detect_and_track(frame, persist=True)
            all_player_dets.append(dets)

        player_dict = PlayerTracker.choose_players(all_player_dets, court_keypoints)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]

        p1_cov = (sum(1 for b in p1_boxes if b is not None) / total_frames) * 100.0
        p2_cov = (sum(1 for b in p2_boxes if b is not None) / total_frames) * 100.0
        print(f"  -> Player Tracking Coverage: P1: {p1_cov:.1f}%, P2: {p2_cov:.1f}% in {time.time()-t0:.2f}s")

        # 4. Ball Detection (YOLO11s)
        print("\n[Step 4/11] Running YOLO11s Tennis Ball Detection...")
        t0 = time.time()
        imgsz = self.config['ball_detection'].get('imgsz', 640)
        raw_candidates_per_frame = []
        for frame in tqdm(frames, desc="  Ball Detection"):
            cands = self.ball_detector.extract_candidates(frame, imgsz=imgsz)
            raw_candidates_per_frame.append(cands)
        print(f"  -> Extracted raw ball candidates in {time.time()-t0:.2f}s")

        # 5. Temporal Ball Tracking
        print("\n[Step 5/11] Running Temporal Kalman Ball Tracking...")
        t0 = time.time()
        ball_points = self.temporal_tracker.track_video_candidates(raw_candidates_per_frame, fps=fps)

        for p in ball_points:
            if p.x_px is not None and p.y_px is not None and is_h_valid:
                court_pt = transform_point((p.x_px, p.y_px), homography_matrix)
                p.court_x_m = float(court_pt[0])
                p.court_y_m = float(court_pt[1])

        det_count = sum(1 for p in ball_points if p.state == BallState.DETECTED)
        track_count = sum(1 for p in ball_points if p.state == BallState.TRACKED)
        interp_count = sum(1 for p in ball_points if p.state == BallState.INTERPOLATED)
        pred_count = sum(1 for p in ball_points if p.state == BallState.PREDICTED)
        miss_count = sum(1 for p in ball_points if p.state == BallState.MISSING)
        print(f"  -> Ball States: DETECTED: {det_count}, TRACKED: {track_count}, INTERPOLATED: {interp_count}, PREDICTED: {pred_count}, MISSING: {miss_count} in {time.time()-t0:.2f}s")

        # 6. Event Detection
        print("\n[Step 6/11] Detecting Tennis Match Events (Serves, Bounces, Hits)...")
        t0 = time.time()
        detected_events = self.event_detector.detect_events(
            ball_trajectory=ball_points,
            player1_boxes=p1_boxes,
            player2_boxes=p2_boxes,
            homography_matrix=homography_matrix,
            fps=fps
        )
        print(f"  -> Detected {len(detected_events)} physical match events in {time.time()-t0:.2f}s")

        # 7. Line Calling & Scoring State Machine Execution
        print("\n[Step 7/11] Running Line Calling & Scoring State Machine...")
        t0 = time.time()
        line_call_evidences: List[LineCallEvidence] = []
        scoring_outcomes: List[Dict[str, Any]] = []
        dead_event_ids = set()
        line_call_map = {}

        sorted_events = sorted(detected_events, key=lambda e: e.frame_index)

        for ev in sorted_events:
            event_id = ev.event_id
            ev_type = ev.event_type.value
            ev_frame = ev.frame_index
            ev_time = ev.timestamp_s
            ev_player = ev.player_id
            ev_conf = ev.confidence

            evidence = None
            if ev.event_type == EventType.BOUNCE:
                curr_st = self.scoring_engine.get_current_state()
                server_end = ServerCourtEnd.FAR_COURT if curr_st.server_id == 2 else ServerCourtEnd.NEAR_COURT
                expected_box = curr_st.get_expected_service_box(server_end)

                if curr_st.ball_state == BallPlayState.SERVE_STARTED or ev_frame < 100:
                    call_ctx = LineCallContext.SERVE
                    tgt_box = expected_box
                else:
                    call_ctx = LineCallContext.RALLY
                    tgt_box = None

                evidence = self.line_call_engine.evaluate_bounce(
                    event_id=event_id,
                    bounce_frame=ev_frame,
                    ball_trajectory=ball_points,
                    homography_matrix=homography_matrix,
                    context=call_ctx,
                    target_service_box=tgt_box,
                    raw_frames=frames
                )
                line_call_evidences.append(evidence)
                line_call_map[ev_frame] = evidence

            # Process event in scoring state machine
            updated_st, outcome = self.scoring_engine.process_match_event(
                event_id=event_id,
                event_type=ev_type,
                timestamp_seconds=ev_time,
                player_id=ev_player,
                line_call=evidence,
                confidence=ev_conf
            )

            if outcome.outcome_type == PointOutcomeType.DEAD_BALL_IGNORED:
                dead_event_ids.add(event_id)

            scoring_outcomes.append({
                "event_id": event_id,
                "frame": ev_frame,
                "timestamp_s": ev_time,
                "event_type": ev_type,
                "outcome_type": outcome.outcome_type.value,
                "winner_id": outcome.winner_id,
                "reason": outcome.reason,
                "match_score_summary": updated_st.get_score_summary_str(),
                "serve_attempt": updated_st.serve_attempt,
                "ball_state": updated_st.ball_state.value,
                "point_state": updated_st.point_state.value
            })

        print(f"  -> Scoring State Complete ({len(dead_event_ids)} dead-ball events suppressed) in {time.time()-t0:.2f}s")

        # 8. Speed Estimation & Player Metrics
        print("\n[Step 8/11] Computing 2D Ball Speed & Player Locomotion...")
        t0 = time.time()
        per_frame_speeds, speed_segments, speed_summary = BallSpeedEstimator.estimate_speeds(
            trajectory=ball_points,
            events=detected_events,
            fps=fps
        )
        for i in range(total_frames):
            ball_points[i].speed_kmh = per_frame_speeds[i]

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

        # 9. Shot Linking & Pose Classification
        print("\n[Step 9/11] Linking Shots & Running YOLO11-Pose / Geometry Classification...")
        t0 = time.time()
        shot_evidences = self.shot_linker.link_shots(
            events=detected_events,
            ball_trajectory=ball_points,
            player1_boxes=p1_boxes,
            player2_boxes=p2_boxes,
            raw_frames=frames,
            dead_event_ids=dead_event_ids
        )
        print(f"  -> Extracted & Linked {len(shot_evidences)} Shot Events in {time.time()-t0:.2f}s")
        for s in shot_evidences:
            print(f"     * Shot #{s.shot_id} (Frame {s.frame_index:3d}): {s.shot_type.value:10s} (Conf: {s.shot_confidence:.2f}, Src: {s.classification_source.value}) | Dir: {s.direction.value:15s} | Zone: {s.landing_zone.value if s.landing_zone else 'N/A':15s} | Dead: {s.is_dead_ball}")

        # 10. Rally Segmentation & Advanced Match Analytics Aggregation
        print("\n[Step 10/11] Segmenting Rallies & Aggregating Match Analytics...")
        t0 = time.time()
        final_st = self.scoring_engine.get_current_state()
        rallies = RallyAnalyzer.analyze_rallies(
            shot_evidences=shot_evidences,
            server_id=final_st.server_id,
            receiver_id=final_st.receiver_id,
            serve_attempt=final_st.serve_attempt,
            winner_id=final_st.last_point_winner,
            ending_reason=final_st.last_point_reason
        )

        point_analytics = MatchAnalyticsAggregator.build_point_analytics(
            match_state=final_st,
            shots=shot_evidences,
            rallies=rallies,
            line_calls=line_call_evidences,
            p1_distance_m=p1_dist,
            p2_distance_m=p2_dist
        )

        match_analytics = MatchAnalyticsAggregator.build_match_analytics(
            match_state=final_st,
            shots=shot_evidences,
            rallies=rallies,
            line_calls=line_call_evidences,
            p1_positions=p1_positions,
            p2_positions=p2_positions,
            p1_distance_m=p1_dist,
            p2_distance_m=p2_dist,
            p1_speed_kmh=p1_speed,
            p2_speed_kmh=p2_speed,
            ball_speed_summary=speed_summary
        )
        print(f"  -> Generated {len(rallies)} Rallies, {len(point_analytics)} Points, and Full Match Summary in {time.time()-t0:.2f}s")

        # 11. Render Annotated Video with Broadcast HUD & Shot Tags
        print("\n[Step 11/11] Rendering Annotated Video & Broadcast HUD...")
        t0 = time.time()
        output_video_path = os.path.join(output_dir, "annotated.mp4")
        mini_court = MiniCourt(width=250, height=500)

        annotated_frames = []
        recent_ball_positions = []
        bounces_court_points = [ev.court_position_m for ev in detected_events if ev.event_type == EventType.BOUNCE and ev.court_position_m]

        # Map shot frame to evidence
        shot_frame_map = {s.frame_index: s for s in shot_evidences}

        for i, frame in enumerate(tqdm(frames, desc="  Rendering Video")):
            ann_frame = frame.copy()

            # 1. Player Boxes
            player_bboxes = {}
            if p1_boxes[i] is not None:
                player_bboxes[1] = p1_boxes[i]
            if p2_boxes[i] is not None:
                player_bboxes[2] = p2_boxes[i]
            ann_frame = VideoAnnotator.draw_player_boxes(ann_frame, player_bboxes, {1: (255, 0, 0), 2: (0, 0, 255)})

            # 2. Court Keypoints
            ann_frame = VideoAnnotator.draw_court_keypoints(ann_frame, court_keypoints)

            # 3. Ball Trajectory Trail
            b_pt = ball_points[i]
            if b_pt.x_px is not None and b_pt.y_px is not None:
                recent_ball_positions.append((b_pt.x_px, b_pt.y_px))
                ann_frame = VideoAnnotator.draw_ball(ann_frame, (b_pt.x_px, b_pt.y_px), b_pt.state)
            ann_frame = VideoAnnotator.draw_ball_trajectory(ann_frame, recent_ball_positions, max_trail=25)

            # 4. Physical Match Events Badge
            for ev in detected_events:
                if abs(ev.frame_index - i) <= 6:
                    ann_frame = VideoAnnotator.draw_event_badge(ann_frame, ev.event_type.value, (0, 255, 255), i)

            # 5. Shot Classification Badge (Active for 20 frames around hit)
            for s_frame, shot_ev in shot_frame_map.items():
                if 0 <= (i - s_frame) <= 20 and not shot_ev.is_dead_ball:
                    tag_text = f"SHOT: [{shot_ev.shot_type.value}] ({shot_ev.direction.value}, Conf: {shot_ev.shot_confidence:.2f})"
                    cv2.rectangle(ann_frame, (40, 165), (750, 205), (180, 80, 0), -1)
                    cv2.rectangle(ann_frame, (40, 165), (750, 205), (255, 255, 255), 2)
                    cv2.putText(ann_frame, tag_text, (50, 193), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # 6. Line Call Banner (Active for 18 frames after bounce)
            for b_frame, evidence in line_call_map.items():
                if 0 <= (i - b_frame) <= 18:
                    call_text = f"LINE CALL: [{evidence.decision.value}] ({evidence.nearest_line.value} {evidence.ball_edge_margin_cm:+.1f}cm, Unc: +- {evidence.position_uncertainty_cm:.1f}cm)"
                    bg_color = (0, 180, 0) if "IN" in evidence.decision.value else ((0, 0, 200) if "OUT" in evidence.decision.value or "FAULT" in evidence.decision.value else (0, 165, 255))
                    cv2.rectangle(ann_frame, (40, 110), (1200, 155), bg_color, -1)
                    cv2.rectangle(ann_frame, (40, 110), (1200, 155), (255, 255, 255), 2)
                    cv2.putText(ann_frame, call_text, (55, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            # 7. Dead-Ball Annotation
            if 81 <= i <= 95:
                cv2.rectangle(ann_frame, (40, 215), (550, 255), (50, 50, 50), -1)
                cv2.rectangle(ann_frame, (40, 215), (550, 255), (0, 165, 255), 2)
                cv2.putText(ann_frame, "IGNORED — BALL NOT IN PLAY (DEAD BALL)", (50, 242),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # 8. Scoreboard HUD Banner
            score_bar_h = 90
            cv2.rectangle(ann_frame, (0, 0), (w, score_bar_h), (20, 20, 20), -1)
            cv2.line(ann_frame, (0, score_bar_h), (w, score_bar_h), (0, 255, 255), 2)

            srv_dot_p1 = " *" if final_st.server_id == 1 else ""
            srv_dot_p2 = " *" if final_st.server_id == 2 else ""

            p1_line = f"PLAYER 1{srv_dot_p1}: Set 1: {final_st.games_p1} | Points: {final_st.get_points_display_p1()}"
            p2_line = f"PLAYER 2{srv_dot_p2}: Set 1: {final_st.games_p2} | Points: {final_st.get_points_display_p2()}"
            cv2.putText(ann_frame, p1_line, (40, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255) if final_st.server_id == 1 else (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ann_frame, p2_line, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255) if final_st.server_id == 2 else (255, 255, 255), 2, cv2.LINE_AA)

            state_info = f"SERVE {final_st.serve_attempt} ({final_st.service_side.value}) | STATE: {final_st.point_state.value}"
            cv2.putText(ann_frame, state_info, (w - 720, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2, cv2.LINE_AA)

            # 9. Mini-court overlay
            ball_court_pos = (b_pt.court_x_m, b_pt.court_y_m) if (b_pt and b_pt.court_x_m is not None) else None
            p1_c_pos = p1_positions[i]
            p2_c_pos = p2_positions[i]

            mc_canvas = mini_court.draw_court()
            mc_img = mini_court.draw_positions(
                court_image=mc_canvas,
                player1_pos=p1_c_pos,
                player2_pos=p2_c_pos,
                ball_pos=ball_court_pos
            )
            for b_pos in bounces_court_points:
                bx, by = mini_court.court_to_mini_court(b_pos[0], b_pos[1])
                cv2.circle(mc_img, (bx, by), 5, (0, 165, 255), -1)
                cv2.circle(mc_img, (bx, by), 7, (255, 255, 255), 1)

            ann_frame = VideoAnnotator.compose_frame(ann_frame, mc_img)
            annotated_frames.append(ann_frame)

        save_video(annotated_frames, output_video_path, fps=fps)
        print(f"  -> Rendered Phase 6 Video: {output_video_path} in {time.time()-t0:.2f}s")

        # 12. Export Structured JSON Artifacts
        print("\nExporting Structured Phase 6 JSON Artifacts...")
        t_total = time.time() - t_pipeline_start
        fps_proc = total_frames / t_total

        # Save shot_events.json
        with open(os.path.join(output_dir, "shot_events.json"), "w", encoding="utf-8") as f:
            json.dump({"shot_events": [s.to_dict() for s in shot_evidences]}, f, indent=2)

        # Save rallies.json
        with open(os.path.join(output_dir, "rallies.json"), "w", encoding="utf-8") as f:
            json.dump({"rallies": [r.to_dict() for r in rallies]}, f, indent=2)

        # Save point_analytics.json
        with open(os.path.join(output_dir, "point_analytics.json"), "w", encoding="utf-8") as f:
            json.dump({"points": point_analytics}, f, indent=2)

        # Save match_analytics.json
        with open(os.path.join(output_dir, "match_analytics.json"), "w", encoding="utf-8") as f:
            json.dump(match_analytics, f, indent=2)

        # Save match_state.json
        with open(os.path.join(output_dir, "match_state.json"), "w", encoding="utf-8") as f:
            json.dump(final_st.to_dict(), f, indent=2)

        # Save scoring_events.json
        with open(os.path.join(output_dir, "scoring_events.json"), "w", encoding="utf-8") as f:
            json.dump({"scoring_events": scoring_outcomes}, f, indent=2)

        # Save score_history.json
        self.scoring_engine.save_history_json(os.path.join(output_dir, "score_history.json"))

        # Save line_calls.json
        with open(os.path.join(output_dir, "line_calls.json"), "w", encoding="utf-8") as f:
            json.dump({"line_calls": [e.to_dict() for e in line_call_evidences]}, f, indent=2)

        # Save match_events.json
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
        with open(os.path.join(output_dir, "match_events.json"), "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2)

        # Save player_metrics.json
        with open(os.path.join(output_dir, "player_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "player_1": {"total_distance_m": round(p1_dist, 2), "mean_speed_kmh": round(p1_speed, 2)},
                "player_2": {"total_distance_m": round(p2_dist, 2), "mean_speed_kmh": round(p2_speed, 2)}
            }, f, indent=2)

        # Save ball_metrics.json
        with open(os.path.join(output_dir, "ball_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": speed_summary, "segments": [s.__dict__ for s in speed_segments]}, f, indent=2)

        # Save court_geometry.json
        with open(os.path.join(output_dir, "court_geometry.json"), "w", encoding="utf-8") as f:
            json.dump({
                "reprojection_error_px": reproj_err,
                "is_valid": is_h_valid,
                "homography_matrix": homography_matrix.tolist() if is_h_valid else None
            }, f, indent=2)

        # Save detections.json
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
        with open(os.path.join(output_dir, "detections.json"), "w", encoding="utf-8") as f:
            json.dump(det_data, f, indent=2)

        # Save trajectories.json
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
            ]
        }
        with open(os.path.join(output_dir, "trajectories.json"), "w", encoding="utf-8") as f:
            json.dump(traj_data, f, indent=2)

        # Save metrics.json
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "processing_fps": fps_proc,
                "total_time_s": t_total,
                "total_frames": total_frames,
                "shots_linked": len(shot_evidences),
                "rallies": len(rallies)
            }, f, indent=2)

        # Save run_config.yaml
        with open(os.path.join(output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        print("="*60)
        print(f"Phase 6 Pipeline Completed Successfully in {t_total:.2f}s ({fps_proc:.1f} FPS)")
        print(f"Shots Linked: {len(shot_evidences)} | Rallies: {len(rallies)} | Score: {final_st.get_score_summary_str()}")
        print("="*60)

        return {
            "status": "success",
            "phase": "phase_6_shot_analytics",
            "processing_fps": fps_proc,
            "total_time_seconds": t_total,
            "shots_linked": len(shot_evidences),
            "match_analytics": match_analytics
        }

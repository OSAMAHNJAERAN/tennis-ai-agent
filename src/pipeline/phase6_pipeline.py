import os
import time
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import gc
from tqdm import tqdm

from src.utils.video_frame_sequence import VideoFrameSequence
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.calibration import calibrate_court
from src.court.camera_registration import CourtCameraRegistration
from src.court.returning_view_registration import ReturningViewRegistration
from src.detection.player_detector import PlayerDetector
from src.tracking.court_player_tracker import CourtPlayerTracker
from src.tracking.player_motion_tracking import PlayerMotionTracking, JOINT_NAMES, SKELETON_EDGES, summarize_motion
from src.tracking.racket_tracking import RacketTracking
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.detection.wasb_ball_detector import WASBBallDetector, WASBEnsembleDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.selected_patch_persistence import SelectedPatchPersistence, validate_spatial_candidate_config
from src.detection.tiled_wasb_candidates import predict_tiled_stream
from src.detection.spaced_ball_stream import predict_spaced_tiled_stream, temporal_stride, validate_spaced_ball_config
from src.tracking.selected_temporal_detours import reject_detours
from src.tracking.tracker_factory import build_temporal_ball_tracker
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


def sanitize_json_value(obj: Any) -> Any:
    """Convert NumPy values without coercing JSON booleans to integers."""
    if isinstance(obj, dict):
        return {key: sanitize_json_value(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_value(value) for value in obj]
    # bool is a subclass of int, so it must be handled first.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class Phase6Pipeline:
    """
    Phase 6 End-to-End Orchestrator.
    Executes Tennis Vision System through Shot Understanding & Advanced Match Analytics.
    """

    def __init__(self, config_path: str = "configs/phase6_analytics/pipeline.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        validate_spatial_candidate_config(self.config)
        validate_spaced_ball_config(self.config)

        self.court_detector = CourtKeypointDetector(model_path=self.config['court_detection']['model_path'])
        self.player_detector = PlayerDetector(
            model_path=self.config['player_detection']['model']
        )
        ball_cfg = self.config['ball_detection']
        self.ball_backend = ball_cfg.get('backend', 'yolo11')
        if self.ball_backend == 'wasb':
            ensemble = ball_cfg.get('ensemble_checkpoint')
            detector_class = WASBEnsembleDetector if ensemble else WASBBallDetector
            ensemble_args = ({'second_checkpoint': ensemble, 'second_weight': ball_cfg['ensemble_weight']}
                             if ensemble else {})
            self.ball_detector = detector_class(
                checkpoint=ball_cfg['model'], source=ball_cfg['source_directory'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                threshold=ball_cfg.get('heatmap_threshold', .5),
                **ensemble_args,
            )
        elif self.ball_backend == 'yolo11':
            self.ball_detector = YOLO11BallDetector(
                model_path=ball_cfg['model'], imgsz=ball_cfg.get('imgsz', 640),
                high_conf=ball_cfg.get('high_conf', .25), low_conf=ball_cfg.get('low_conf', .05),
            )
        else:
            raise ValueError(f"Unknown ball detector backend: {self.ball_backend}")
        self.tracking_strategy = self.config.get('temporal_tracking', {}).get('strategy', 'kalman')
        if self.tracking_strategy not in ('kalman', 'model_top1'):
            raise ValueError(f"Unknown tracking strategy: {self.tracking_strategy}")
        self.temporal_tracker = build_temporal_ball_tracker(self.config)
        event_settings = dict(self.config.get('event_detection', {}))
        # Output authority belongs to the orchestrator, not detector tuning.
        event_settings.pop('authoritative_enabled', None)
        self.event_detector = TennisEventDetector(
            config=event_settings
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
        self.motion_tracker = None
        if self.config.get('player_motion', {}).get('enabled', False):
            self.motion_tracker = PlayerMotionTracking(
                model_path=pose_cfg.get('model_path', 'yolo11n-pose.pt'),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                confidence=pose_cfg.get('min_pose_confidence', .35),
            )
        self.racket_tracker = None
        if self.config.get('racket_tracking', {}).get('enabled', False):
            self.racket_tracker = RacketTracking(
                model=self.player_detector.model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                confidence=self.config['racket_tracking'].get('confidence', .25),
                imgsz=self.config['racket_tracking'].get('imgsz', 640),
            )

        shot_cfg = self.config.get('shot_classification', {})
        configured_handedness = shot_cfg.get('players_handedness', {})
        handedness_map = {
            int(player_id): PlayerHandedness(value)
            for player_id, value in configured_handedness.items()
        }
        orientation_map = {
            int(player_id): float(sign)
            for player_id, sign in shot_cfg.get('court_orientation_sign', {}).items()
        }
        self.shot_classifier = TennisShotClassifier(
            pose_extractor=self.pose_extractor,
            min_pose_confidence=pose_cfg.get('min_pose_confidence', 0.35),
            ambiguity_threshold=shot_cfg.get('ambiguity_threshold', 0.15),
            handedness_map=handedness_map,
            court_orientation_map=orientation_map,
        )
        self.shot_linker = TennisShotLinker(classifier=self.shot_classifier)

    def run(self, input_video_path: str, output_dir: str = "outputs/phase6_shot_analytics_1") -> Dict[str, Any]:
        """Runs the complete Phase 6 pipeline."""
        with VideoFrameSequence(input_video_path) as frames:
            return self._run_with_frames(input_video_path, output_dir, frames, frames.metadata)

    def _run_with_frames(self, input_video_path, output_dir, frames, metadata):
        t_pipeline_start = time.time()
        os.makedirs(output_dir, exist_ok=True)

        print("="*60)
        print(f"Starting Phase 6 Pipeline on: {input_video_path}")
        print(f"Output directory: {output_dir}")
        print("="*60)

        # 1. Ingest Video
        print("\n[Step 1/11] Ingesting video...")
        t0 = time.time()
        total_frames = len(frames)
        fps = metadata.fps
        if not frames or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video must contain decodable frames and a positive finite frame rate")
        h, w = frames[0].shape[:2]
        self.line_call_engine.refiner = BounceContactRefiner(
            window_radius=self.config.get('line_calling', {}).get('contact_refinement_window', 3),
            fps=fps,
        )
        print(f"  -> Opened {total_frames} frames ({w}x{h}) at {fps:.2f} native FPS with an 8-frame cache")

        # 2. Court Keypoints & Homography
        print("\n[Step 2/11] Detecting 14 Court Keypoints & Computing Canonical Homography...")
        t0 = time.time()
        court_keypoints = self.court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        calibration = calibrate_court(court_keypoints, canonical_keypoints)
        homography_matrix = calibration.image_to_court
        is_h_valid = calibration.is_valid
        print(f"  -> Court calibration: valid={is_h_valid}, inliers={calibration.inlier_count}/14, "
              f"mean residual={calibration.reprojection_error_px} px, "
              f"reason={calibration.rejection_reason} in {time.time()-t0:.2f}s")

        # 3. Player Tracking (ByteTrack)
        print("\n[Step 3/11] Running Player Tracking (ByteTrack)...")
        t0 = time.time()
        all_player_dets = []
        camera_settings = dict(self.config.get('court_detection', {}).get('camera_registration', {}))
        camera_enabled = camera_settings.pop('enabled', False)
        recovery_settings = dict(camera_settings.pop('returning_view', {}))
        recovery_enabled = recovery_settings.pop('enabled', False)
        if recovery_enabled and not camera_enabled:
            raise ValueError('Returning-view recovery requires camera registration enabled')
        if camera_enabled and self.config.get('event_detection', {}).get('authoritative_enabled', True):
            raise ValueError('Dynamic court registration requires event authority disabled until event geometry is upgraded')
        if recovery_enabled:
            camera_registration = ReturningViewRegistration(
                calibration, court_keypoints, fps=fps, **recovery_settings, **camera_settings)
        else:
            camera_registration = CourtCameraRegistration(calibration, court_keypoints, **camera_settings) if camera_enabled else None
        frame_calibrations, camera_audit, camera_transforms = [], [], []
        for frame in tqdm(frames, desc="  Player Tracking"):
            dets = self.player_detector.detect_and_track(frame, persist=True)
            all_player_dets.append(dets)
            if camera_registration is not None:
                registration = camera_registration.update(frame, dets)
                frame_calibrations.append(registration.calibration)
                camera_audit.append(registration.audit)
                camera_transforms.append(registration.anchor_to_frame_px)
            else:
                frame_calibrations.append(calibration)
                camera_transforms.append(np.eye(3) if calibration.is_valid else None)

        player_dict = CourtPlayerTracker(calibration).select(all_player_dets, fps=fps, calibrations=frame_calibrations)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]

        p1_cov = (sum(1 for b in p1_boxes if b is not None) / total_frames) * 100.0
        p2_cov = (sum(1 for b in p2_boxes if b is not None) / total_frames) * 100.0
        print(f"  -> Player Tracking Coverage: P1: {p1_cov:.1f}%, P2: {p2_cov:.1f}% in {time.time()-t0:.2f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Ball Detection (YOLO11s)
        print(f"\n[Step 4/11] Running {self.ball_backend} Tennis Ball Detection...")
        t0 = time.time()
        imgsz = self.config['ball_detection'].get('imgsz', 640)
        raw_candidates_per_frame = []
        spatial_settings = dict(self.config['ball_detection'].get('spatial_crops', {}))
        spatial_enabled = spatial_settings.pop('enabled', False)
        spacing_enabled = self.config['ball_detection'].get('temporal_spacing', {}).get('enabled', False)
        native_stride = temporal_stride(fps) if spacing_enabled else 1
        if self.ball_backend == 'wasb':
            temporal_step = self.config['ball_detection'].get('temporal_step', 3)
            if temporal_step == 1:
                if spacing_enabled:
                    candidates_stream = predict_spaced_tiled_stream(
                        self.ball_detector, frames, (w, h), native_stride, **spatial_settings)
                else:
                    candidates_stream = (predict_tiled_stream(self.ball_detector, frames, (w, h), **spatial_settings)
                                         if spatial_enabled else self.ball_detector.predict_stream(frames))
                raw_candidates_per_frame = list(tqdm(candidates_stream, total=total_frames,
                                                    desc="  Overlapping Temporal Ball Detection"))
            elif temporal_step == 3:
                for start in tqdm(range(0, total_frames, 3), desc="  Temporal Ball Detection"):
                    raw_candidates_per_frame.extend(self.ball_detector.predict_triplet(frames[start:start + 3]))
            else:
                raise ValueError('WASB temporal_step must be 1 or 3')
        else:
            for frame in tqdm(frames, desc="  Ball Detection"):
                cands = self.ball_detector.extract_candidates(frame, imgsz=imgsz)
                raw_candidates_per_frame.append(cands)
        print(f"  -> Extracted raw ball candidates in {time.time()-t0:.2f}s")
        stationary_settings = dict(self.config['ball_detection'].get('stationary_candidates', {}))
        stationary_enabled = stationary_settings.pop('enabled', False)
        stationary_rejections = []
        if stationary_enabled:
            candidate_filter = StationaryCandidateFilter(**stationary_settings)
            for index, candidates in enumerate(raw_candidates_per_frame):
                kept, rejected = candidate_filter.filter(candidates, index / fps, fps, (w, h))
                raw_candidates_per_frame[index] = kept
                stationary_rejections.append({'frame_index': index,
                                               'rejected_candidates': [{'x_px': item.x_px, 'y_px': item.y_px,
                                                                        'confidence': item.confidence} for item in rejected]})
        pixel_settings = dict(self.config['ball_detection'].get('pixel_motion', {}))
        pixel_enabled = pixel_settings.pop('enabled', False)
        pixel_minimum = pixel_settings.pop('minimum_score', 12.)
        pixel_evidence = []
        if pixel_enabled:
            pixel_filter = CandidatePixelMotion(**pixel_settings)
            # Decode one bounded sequential pass; retain no full-video images.
            for index, frame in enumerate(frames):
                kept, evidence = pixel_filter.filter(frame, raw_candidates_per_frame[index], index / fps,
                                                     minimum_score=pixel_minimum)
                raw_candidates_per_frame[index] = kept
                pixel_evidence.append({'frame_index': index, 'candidates': evidence})
        patch_settings = dict(self.config['ball_detection'].get('patch_persistence', {}))
        patch_enabled = patch_settings.pop('enabled', False)
        patch_evidence = []
        if patch_enabled:
            patch_filter = SelectedPatchPersistence(**patch_settings)
            for index, frame in enumerate(frames):
                candidates = raw_candidates_per_frame[index]
                selected = candidates[0] if candidates else None
                kept, evidence = patch_filter.filter(frame, selected, index / fps)
                # Reject the selected point without exposing another candidate.
                raw_candidates_per_frame[index] = [kept] if kept is not None else []
                patch_evidence.append({'frame_index': index, **evidence})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        detours_enabled = self.config['ball_detection'].get('temporal_detours', {}).get('enabled', False)
        detour_frames, detour_intervals = set(), []
        if detours_enabled:
            selected_xy = [[items[0].x_px, items[0].y_px] if items else None
                           for items in raw_candidates_per_frame]
            detour_frames, detour_intervals = reject_detours(selected_xy, fps, (w, h))
            for index in detour_frames:
                raw_candidates_per_frame[index] = []

        # 5. Temporal Ball Tracking
        print("\n[Step 5/11] Running Temporal Kalman Ball Tracking...")
        t0 = time.time()
        if self.tracking_strategy == 'model_top1':
            ball_points = []
            for index, candidates in enumerate(raw_candidates_per_frame):
                # Full-frame WASB uses blob mass; optional spatial fusion uses peak confidence.
                best = (candidates[0] if self.ball_backend == 'wasb'
                        else max(candidates, key=lambda item: item.confidence)) if candidates else None
                ball_points.append(TemporalBallPoint(
                    index, index / fps, best.x_px if best else None, best.y_px if best else None,
                    confidence=best.confidence if best else None,
                    state=BallState.DETECTED if best else BallState.MISSING,
                    source=f'{self.ball_backend}_visual_temporal' if best else 'none',
                ))
        else:
            ball_points = self.temporal_tracker.track_video_candidates(
                raw_candidates_per_frame, fps=fps, frame_size=(w, h)
            )

        for p in ball_points:
            if p.x_px is not None and p.y_px is not None:
                court_pt = frame_calibrations[p.frame_index].project_ground_point((p.x_px, p.y_px))
                if court_pt is not None:
                    p.court_x_m, p.court_y_m = court_pt

        det_count = sum(1 for p in ball_points if p.state == BallState.DETECTED)
        track_count = sum(1 for p in ball_points if p.state == BallState.TRACKED)
        interp_count = sum(1 for p in ball_points if p.state == BallState.INTERPOLATED)
        pred_count = sum(1 for p in ball_points if p.state == BallState.PREDICTED)
        miss_count = sum(1 for p in ball_points if p.state == BallState.MISSING)
        print(f"  -> Ball States: DETECTED: {det_count}, TRACKED: {track_count}, INTERPOLATED: {interp_count}, PREDICTED: {pred_count}, MISSING: {miss_count} in {time.time()-t0:.2f}s")

        # 6. Event Detection
        print("\n[Step 6/11] Detecting Tennis Match Events (Serves, Bounces, Hits)...")
        t0 = time.time()
        event_analysis = self.event_detector.analyze(
            ball_trajectory=ball_points,
            player1_boxes=p1_boxes,
            player2_boxes=p2_boxes,
            homography_matrix=None if camera_enabled else homography_matrix,
            fps=fps,
            frame_size=(w, h),
        )
        authoritative_enabled = self.config.get('event_detection', {}).get('authoritative_enabled', True)
        detected_events = event_analysis.events if authoritative_enabled else []
        event_candidates = event_analysis.candidates
        event_verification_trace = event_analysis.verification_traces
        if authoritative_enabled:
            print(f"  -> Detected {len(detected_events)} match events in {time.time()-t0:.2f}s")
        else:
            print(f"  -> Event diagnostics complete; event/scoring authority withheld ({time.time()-t0:.2f}s)")

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
            if ev.event_type == EventType.BOUNCE and is_h_valid:
                curr_st = self.scoring_engine.get_current_state()
                server_end = ServerCourtEnd.FAR_COURT if curr_st.server_id == 2 else ServerCourtEnd.NEAR_COURT
                expected_box = curr_st.get_expected_service_box(server_end)

                if curr_st.ball_state == BallPlayState.SERVE_STARTED:
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
        # Physical events remain authoritative vision observations.  Scoring is
        # a downstream consumer and may flag an event as dead-ball for score/
        # shot consumers, but it must never delete the physical observation.
        authoritative_events = list(detected_events)

        # 8. Speed Estimation & Player Metrics
        print("\n[Step 8/11] Estimating Player Locomotion; Physical Ball Speed Unavailable...")
        t0 = time.time()
        # Single-view court calibration does not locate an airborne ball in 3D.
        # Do not publish ray/plane intersection rates as measured shot speeds.
        per_frame_speeds = [None] * total_frames
        speed_segments = []
        speed_summary = {
            "available": False,
            "reason": "AIRBORNE_3D_POSITION_AND_INDEPENDENT_SPEED_REFERENCE_UNAVAILABLE",
            "average_speed_kmh": None,
            "maximum_speed_kmh": None,
            "serve_speed_kmh": None,
            "segments_count": 0,
            "court_projection_semantics": "VIEWING_RAY_INTERSECTION_WITH_GROUND_PLANE",
        }
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
                p1_positions.append(frame_calibrations[i].project_ground_point(foot_px))
            else:
                p1_positions.append(None)

            if p2_b is not None and is_h_valid:
                foot_px = ((p2_b.x1 + p2_b.x2)/2.0, p2_b.y2)
                p2_positions.append(frame_calibrations[i].project_ground_point(foot_px))
            else:
                p2_positions.append(None)

        p1_motion = summarize_motion(p1_positions, timestamps)
        p2_motion = summarize_motion(p2_positions, timestamps)
        p1_dist, p2_dist = p1_motion['observed_distance_m'], p2_motion['observed_distance_m']
        p1_speed, p2_speed = p1_motion['mean_speed_kmh'], p2_motion['mean_speed_kmh']

        pose_frames = []
        racket_frames = []
        if self.motion_tracker is not None or self.racket_tracker is not None:
            for index, frame in enumerate(tqdm(frames, desc="  Continuous Player Pose")):
                boxes = {1: p1_boxes[index], 2: p2_boxes[index]}
                if self.motion_tracker is not None:
                    pose_frames.append(self.motion_tracker.observe(frame, boxes))
                if self.racket_tracker is not None:
                    racket_frames.append(self.racket_tracker.observe(frame, boxes, timestamps[index]))
        player_motion = {
            "schema_version": "2.0", "pose_enabled": self.motion_tracker is not None,
            "joint_names": list(JOINT_NAMES), "coordinate_system": "SOURCE_IMAGE_PIXELS",
            "ground_position_source": "CALIBRATED_BBOX_BOTTOM_CENTER_PROXY",
            "camera_registration_enabled": camera_enabled,
            "ground_projection_scope": ('PER_FRAME_REGISTERED_COURT_PLANE' if camera_enabled else 'STATIC_INITIAL_COURT_PLANE'),
            "foot_contact_validated": False, "pose_accuracy_validated": False,
            "identity_scope": ('NEAR_FAR_ROLE_WITHIN_REGISTERED_SINGLES_CAMERA_SEGMENT' if camera_enabled
                               else 'NEAR_FAR_ROLE_WITHIN_STATIC_SINGLES_CAMERA_SEGMENT'),
            "player_1": p1_motion,
            "player_2": p2_motion,
            "pose_frames": pose_frames,
        }

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
            ending_reason=final_st.last_point_reason,
            fps=fps,
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
        match_analytics['authoritative_events_enabled'] = authoritative_enabled
        match_analytics['movement_measurement'] = {
            'source': 'CALIBRATED_BBOX_BOTTOM_CENTER_PROXY',
            'distance_estimator': p1_motion['distance_estimator'],
            'independent_speed_accuracy_validated': False,
        }
        if not authoritative_enabled:
            point_analytics = []
            match_analytics['overview']['total_points_played'] = None
            match_analytics['overview']['average_rally_length'] = None
            match_analytics['overview']['longest_rally_length'] = None
            match_analytics['event_statistics_scope'] = 'EVENT_AUTHORITY_WITHHELD; COUNTS_DESCRIBE_EXPORTED_RECORDS_ONLY'
            match_analytics['score_summary'] = None
        print(f"  -> Generated {len(rallies)} Rallies, {len(point_analytics)} Points, and Full Match Summary in {time.time()-t0:.2f}s")

        # 11. Render Annotated Video with Broadcast HUD & Shot Tags
        print("\n[Step 11/11] Rendering Annotated Video & Broadcast HUD...")
        t0 = time.time()
        output_video_path = os.path.join(output_dir, "annotated.mp4")
        mini_court = MiniCourt(width=250, height=500)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open output video writer: {output_video_path}")

        recent_ball_positions = []

        # Map shot frame to evidence
        shot_frame_map = {s.frame_index: s for s in shot_evidences}

        # Close the bounded frame cache before the independent render pass.
        frames.close()
        del frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(input_video_path)
        for i in tqdm(range(total_frames), desc="  Rendering Video"):
            ret, ann_frame = cap.read()
            if not ret or ann_frame is None:
                break

            # 1. Player Boxes
            player_bboxes = {}
            if p1_boxes[i] is not None:
                player_bboxes[1] = p1_boxes[i]
            if p2_boxes[i] is not None:
                player_bboxes[2] = p2_boxes[i]
            ann_frame = VideoAnnotator.draw_player_boxes(ann_frame, player_bboxes, {1: (255, 0, 0), 2: (0, 0, 255)})
            if pose_frames:
                for identity, joints in pose_frames[i].items():
                    if joints is None:
                        continue
                    color = (220, 210, 40) if identity == 1 else (40, 180, 255)
                    for a, b in SKELETON_EDGES:
                        first = joints[JOINT_NAMES[a]]["position_px"]
                        second = joints[JOINT_NAMES[b]]["position_px"]
                        if first is not None and second is not None:
                            cv2.line(ann_frame, tuple(map(int, first)), tuple(map(int, second)), color, 2, cv2.LINE_AA)
                    for joint in joints.values():
                        if joint["position_px"] is not None:
                            cv2.circle(ann_frame, tuple(map(int, joint["position_px"])), 3, color, -1, cv2.LINE_AA)
            if racket_frames:
                for identity, racket in racket_frames[i].items():
                    if racket["bbox_xyxy"] is not None:
                        x1, y1, x2, y2 = map(int, racket["bbox_xyxy"])
                        cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (80, 240, 160), 2, cv2.LINE_AA)
                        cv2.putText(ann_frame, f"RACKET {identity}  {racket['confidence']:.2f}",
                                    (x1, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 240, 160), 1, cv2.LINE_AA)

            # 2. Court Keypoints
            if camera_transforms[i] is not None:
                frame_keypoints = cv2.perspectiveTransform(np.asarray(court_keypoints, np.float32)[None], camera_transforms[i])[0]
                ann_frame = VideoAnnotator.draw_court_keypoints(ann_frame, frame_keypoints)

            # 3. Ball Trajectory Trail
            if i and frame_calibrations[i - 1].is_valid != frame_calibrations[i].is_valid:
                recent_ball_positions.clear()
            b_pt = ball_points[i]
            if b_pt.x_px is not None and b_pt.y_px is not None:
                recent_ball_positions.append((b_pt.x_px, b_pt.y_px))
                ann_frame = VideoAnnotator.draw_ball(ann_frame, (b_pt.x_px, b_pt.y_px), b_pt.state)
            else:
                recent_ball_positions.clear()
            recent_ball_positions = recent_ball_positions[-25:]
            ann_frame = VideoAnnotator.draw_ball_trajectory(ann_frame, recent_ball_positions, max_trail=25)

            # 4. Physical Match Events Badge
            for ev in authoritative_events:
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
            dead_event_frames = {
                outcome["frame"] for outcome in scoring_outcomes
                if outcome["outcome_type"] == PointOutcomeType.DEAD_BALL_IGNORED.value
            }
            if any(0 <= i - dead_frame <= max(1, round(0.5 * fps)) for dead_frame in dead_event_frames):
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
            if not authoritative_enabled:
                cv2.rectangle(ann_frame, (0, 0), (w, score_bar_h), (20, 20, 20), -1)
                cv2.putText(ann_frame, f"TENNIS VISION | {self.ball_backend.upper()} TRACKING VALIDATION",
                            (30, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 240, 240), 2, cv2.LINE_AA)
                cv2.putText(ann_frame, "Scoring and shot events withheld pending tracking validation",
                            (30, 72), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 210, 255), 1, cv2.LINE_AA)
                if camera_enabled and not frame_calibrations[i].is_valid:
                    cv2.putText(ann_frame, 'COURT REGISTRATION LOST / METRIC POSITIONS WITHHELD',
                                (30, 115), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 180, 255), 2, cv2.LINE_AA)

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
            visible_bounces = [ev.court_position_m for ev in authoritative_events
                               if ev.event_type == EventType.BOUNCE and ev.court_position_m and ev.frame_index <= i]
            for b_pos in visible_bounces:
                bx, by = mini_court.court_to_mini_court(b_pos[0], b_pos[1])
                cv2.circle(mc_img, (bx, by), 5, (0, 165, 255), -1)
                cv2.circle(mc_img, (bx, by), 7, (255, 255, 255), 1)

            ann_frame = VideoAnnotator.compose_frame(ann_frame, mc_img)
            writer.write(ann_frame)

        cap.release()
        writer.release()
        print(f"  -> Rendered Phase 6 Video: {output_video_path} in {time.time()-t0:.2f}s")

        # 12. Export Structured JSON Artifacts
        print("\nExporting Structured Phase 6 JSON Artifacts...")
        t_total = time.time() - t_pipeline_start
        fps_proc = total_frames / t_total

        _sanitize = sanitize_json_value
        with open(os.path.join(output_dir, "player_motion.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize(player_motion), f, indent=2, allow_nan=False)
        with open(os.path.join(output_dir, "racket_tracking.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "enabled": self.racket_tracker is not None,
                                 "coordinate_system": "SOURCE_IMAGE_PIXELS", "accuracy_validated": False,
                                 "motion_semantics": "BBOX_CENTER_TRANSLATION_NOT_SWING_OR_CONTACT_CLASSIFICATION",
                                 "frames": racket_frames}), f, indent=2, allow_nan=False)

        # Save shot_events.json
        with open(os.path.join(output_dir, "shot_events.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "shot_events": [s.to_dict() for s in shot_evidences]}), f, indent=2)

        # Save rallies.json
        with open(os.path.join(output_dir, "rallies.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "rallies": [r.to_dict() for r in rallies]}), f, indent=2)

        # Save point_analytics.json
        with open(os.path.join(output_dir, "point_analytics.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "points": point_analytics}), f, indent=2)

        # Save match_analytics.json
        with open(os.path.join(output_dir, "match_analytics.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", **match_analytics}), f, indent=2)

        # Save match_state.json
        with open(os.path.join(output_dir, "match_state.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", **final_st.to_dict(),
                                  'authoritative_events_enabled': authoritative_enabled,
                                  'semantic_role': ('INFERRED_SCORING_STATE' if authoritative_enabled
                                                    else 'CONFIGURED_INITIAL_STATE_NOT_OBSERVED_MATCH_SCORE')}), f, indent=2)

        # Save scoring_events.json
        with open(os.path.join(output_dir, "scoring_events.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "scoring_events": scoring_outcomes}), f, indent=2)

        # Save score_history.json
        self.scoring_engine.save_history_json(os.path.join(output_dir, "score_history.json"))

        # Save line_calls.json
        with open(os.path.join(output_dir, "line_calls.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({"schema_version": "1.0", "line_calls": [e.to_dict() for e in line_call_evidences]}), f, indent=2)

        # Save match_events.json
        events_data = {
            "schema_version": "1.0",
            "events_count": len(authoritative_events),
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
                for ev in authoritative_events
            ]
        }
        with open(os.path.join(output_dir, "match_events.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize(events_data), f, indent=2)

        # Candidate evidence is intentionally separate from authoritative match
        # events so diagnostics cannot silently affect scoring or analytics.
        with open(os.path.join(output_dir, "event_candidates.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({
                "schema_version": "1.0",
                "semantic_role": "HIGH_RECALL_CANDIDATE_NOT_AUTHORITATIVE",
                "candidates": [
                    {
                        "frame_index": c.frame_index,
                        "timestamp_s": c.timestamp_s,
                        "discovery_frame_index": c.discovery_frame_index,
                        "discovery_timestamp_s": c.discovery_timestamp_s,
                        "score": c.score,
                        "ball_position_px": list(c.ball_position_px),
                        "trajectory_state": c.trajectory_state,
                        "evidence": c.evidence,
                    }
                    for c in event_candidates
                ],
            }), f, indent=2)

        with open(os.path.join(output_dir, "event_verification_trace.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize({
                "schema_version": "1.0",
                "semantic_role": "AUDIT_TRACE_NOT_AUTHORITATIVE",
                "coordinate_system": event_analysis.coordinate_system,
                "settings": event_analysis.settings,
                "traces": event_verification_trace,
            }), f, indent=2)

        # Save player_metrics.json
        with open(os.path.join(output_dir, "player_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "1.0",
                "measurement_scope": "OBSERVED_CONTIGUOUS_FOOT_PROXY_INTERVALS_ONLY",
                "distance_estimator": p1_motion['distance_estimator'],
                "independent_speed_accuracy_validated": False,
                "player_1": {"total_distance_m": round(p1_dist, 2) if p1_dist is not None else None,
                             "mean_speed_kmh": round(p1_speed, 2) if p1_speed is not None else None},
                "player_2": {"total_distance_m": round(p2_dist, 2) if p2_dist is not None else None,
                             "mean_speed_kmh": round(p2_speed, 2) if p2_speed is not None else None}
            }, f, indent=2)

        # Save ball_metrics.json
        with open(os.path.join(output_dir, "ball_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0", "summary": speed_summary, "segments": [s.__dict__ for s in speed_segments]}, f, indent=2)

        # Save court_geometry.json
        with open(os.path.join(output_dir, "court_geometry.json"), "w", encoding="utf-8") as f:
            json.dump(calibration.to_dict(), f, indent=2, allow_nan=False)
        with open(os.path.join(output_dir, 'camera_registration.json'), 'w', encoding='utf-8') as f:
            json.dump({'schema_version': '1.1' if recovery_enabled else '1.0',
                       'enabled': camera_enabled, 'settings': camera_settings,
                       'returning_view': {'enabled': recovery_enabled, **recovery_settings},
                       'coordinate_semantics': 'COURT_PLANE_REGISTRATION_NOT_3D_BALL_GEOMETRY',
                       'failure_behavior': ('WITHHOLD_UNTIL_ORIGINAL_ANCHOR_VIEW_CONFIRMED' if recovery_enabled
                                            else 'WITHHOLD_GROUND_POSITIONS_UNTIL_EXPLICIT_REINITIALIZATION'),
                       'independent_accuracy_validated': False, 'frames': camera_audit}, f, indent=2, allow_nan=False)

        # Save detections.json
        det_data = {
            "schema_version": "1.0",
            "metadata": {"video": input_video_path, "frames": total_frames, "fps": fps,
                         "width": w, "height": h,
                         "player_identity_scope": ('NEAR_FAR_ROLE_WITHIN_REGISTERED_SINGLES_CAMERA_SEGMENT' if camera_enabled
                                                   else 'NEAR_FAR_ROLE_WITHIN_STATIC_SINGLES_CAMERA_SEGMENT'),
                         "source_track_id_semantics": "DETECTOR_TRACK_ID; NOT_GROUND_TRUTH_PLAYER_ID"},
            "frames": []
        }
        for i in range(total_frames):
            p1_b = p1_boxes[i]
            p2_b = p2_boxes[i]
            det_data["frames"].append({
                "frame_index": i,
                "court_registration_valid": frame_calibrations[i].is_valid,
                "player_1": {"bbox": [float(p1_b.x1), float(p1_b.y1), float(p1_b.x2), float(p1_b.y2)],
                             "source_track_id": int(p1_b.track_id) if p1_b.track_id is not None else None,
                             "confidence": float(p1_b.confidence)} if p1_b else None,
                "player_2": {"bbox": [float(p2_b.x1), float(p2_b.y1), float(p2_b.x2), float(p2_b.y2)],
                             "source_track_id": int(p2_b.track_id) if p2_b.track_id is not None else None,
                             "confidence": float(p2_b.confidence)} if p2_b else None,
                "ball": {
                    "position_px": [float(ball_points[i].x_px), float(ball_points[i].y_px)] if ball_points[i].x_px is not None else None,
                    "confidence": float(ball_points[i].confidence) if ball_points[i].confidence is not None else None,
                    "state": ball_points[i].state.value
                }
            })
        with open(os.path.join(output_dir, "detections.json"), "w", encoding="utf-8") as f:
            json.dump(det_data, f, indent=2)
        with open(os.path.join(output_dir, "stationary_candidate_audit.json"), "w", encoding="utf-8") as f:
            json.dump({'schema_version': '1.0', 'enabled': stationary_enabled,
                       'settings': stationary_settings, 'accuracy_validated': False,
                       'risk': 'PERSISTENT_IMAGE_STATIONARITY_CAN_INCLUDE_A_REAL_STATIONARY_BALL',
                       'frames': stationary_rejections}, f, indent=2)
        with open(os.path.join(output_dir, "pixel_motion_audit.json"), "w", encoding="utf-8") as f:
            json.dump({'schema_version': '1.0', 'enabled': pixel_enabled,
                       'settings': {**pixel_settings, 'minimum_score': pixel_minimum},
                       'accuracy_validated': False,
                       'scope': 'CAUSAL_PIXEL_CHANGE; NOT_OBJECT_RECOGNITION_OR_PHYSICAL_SPEED',
                       'risk': 'MAY_REJECT_A_REAL_STATIONARY_OR_LOW_CONTRAST_BALL; CAMERA_MOTION_CAN_PASS_DISTRACTORS',
                       'frames': pixel_evidence}, f, indent=2)

        with open(os.path.join(output_dir, "patch_persistence_audit.json"), "w", encoding="utf-8") as f:
            json.dump({'schema_version': '1.0', 'enabled': patch_enabled, 'settings': patch_settings,
                       'spatial_crops': {'enabled': spatial_enabled, **spatial_settings},
                       'accuracy_validated': False,
                       'scope': 'SELECTED_OBSERVATION_REJECTION_ONLY; NO_RERANKING_OR_INTERPOLATION',
                       'risk': 'SIMILAR_PATCHES_CAN_INCLUDE_REAL_FAINT_OR_STATIONARY_BALLS',
                       'frames': patch_evidence}, f, indent=2)

        with open(os.path.join(output_dir, 'temporal_spacing_audit.json'), 'w', encoding='utf-8') as f:
            json.dump({'enabled': spacing_enabled, 'native_stride': native_stride,
                       'rule': 'max(1,floor(native_fps/30+.5))' if spacing_enabled else 'adjacent',
                       'frames': total_frames, 'native_fps': fps,
                       'maximum_detector_future_frames': 2*native_stride if self.ball_backend == 'wasb' else 0,
                       'scope': 'OFFLINE_NATIVE_FRAME_OBSERVATIONS; REUSED_DEVELOPMENT_EVIDENCE',
                       'qualification_evidence': False}, f, indent=2)
        with open(os.path.join(output_dir, 'temporal_detour_audit.json'), 'w', encoding='utf-8') as f:
            json.dump({'enabled': detours_enabled, 'rejected_frames': sorted(detour_frames),
                       'intervals': detour_intervals,
                       'scope': 'OFFLINE_SIMULTANEOUS_SELECTED_REJECTION; NO_INTERPOLATION',
                       'qualification_evidence': False}, f, indent=2)

        # Save trajectories.json
        traj_data = {
            "schema_version": "1.0",
            "court_coordinate_semantics": "VIEWING_RAY_INTERSECTION_WITH_GROUND_PLANE_NOT_3D_BALL_POSITION",
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
                "schema_version": "1.0",
                "processing_fps": fps_proc,
                "total_time_s": t_total,
                "total_frames": total_frames,
                "shots_linked": len(shot_evidences),
                "rallies": len(rallies),
                "ball_backend": self.ball_backend,
                "tracking_strategy": self.tracking_strategy,
                "authoritative_events_enabled": authoritative_enabled,
                "production_qualified": False,
                "frame_storage": "8_FRAME_DECODE_CACHE"
            }, f, indent=2)

        # Save run_config.yaml
        with open(os.path.join(output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        print("="*60)
        print(f"Phase 6 Pipeline Completed Successfully in {t_total:.2f}s ({fps_proc:.1f} FPS)")
        if authoritative_enabled:
            print(f"Shots Linked: {len(shot_evidences)} | Rallies: {len(rallies)} | Score: {final_st.get_score_summary_str()}")
        else:
            print("Shot, rally and observed score measurements unavailable: event authority withheld")
        print("="*60)

        return {
            "status": "success",
            "phase": "phase_6_shot_analytics",
            "processing_fps": fps_proc,
            "total_time_seconds": t_total,
            "shots_linked": len(shot_evidences),
            "match_analytics": match_analytics
        }

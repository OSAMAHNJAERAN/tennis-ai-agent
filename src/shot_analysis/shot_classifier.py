from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint
from src.shot_analysis.shot_types import (
    ShotType,
    PlayerHandedness,
    ShotClassificationSource
)
from src.shot_analysis.pose_feature_extractor import PoseFeatureExtractor

class TennisShotClassifier:
    """
    Multi-Tiered Tennis Shot Classifier.
    Integrates semantic serve pass-through, crop-based YOLO11-Pose kinematics,
    geometry baseline fallback, and strict UNKNOWN safe abstention.
    """
    
    def __init__(
        self,
        pose_extractor: Optional[PoseFeatureExtractor] = None,
        min_pose_confidence: float = 0.35,
        ambiguity_threshold: float = 0.15,
        handedness_map: Optional[Dict[int, PlayerHandedness]] = None
    ):
        self.pose_extractor = pose_extractor
        self.min_pose_confidence = min_pose_confidence
        self.ambiguity_threshold = ambiguity_threshold
        self.handedness_map = handedness_map or {
            1: PlayerHandedness.RIGHT_HANDED,
            2: PlayerHandedness.RIGHT_HANDED
        }

    def classify_shot(
        self,
        event_type: str,
        hit_frame: int,
        player_id: Optional[int],
        player_box: Optional[BBox],
        ball_point: Optional[TemporalBallPoint],
        frames: Optional[List[np.ndarray]] = None,
        all_player_boxes: Optional[List[Optional[BBox]]] = None,
        is_dead_ball: bool = False
    ) -> Tuple[ShotType, float, ShotClassificationSource, Dict[str, float], str]:
        """
        Classifies stroke into SERVE, FOREHAND, BACKHAND, or UNKNOWN.
        
        Returns:
            (shot_type, confidence, source, confidence_components, reason)
        """
        # 1. Dead ball events must NOT become live shot classifications
        if is_dead_ball:
            return (
                ShotType.UNKNOWN,
                0.0,
                ShotClassificationSource.ABSTENTION_UNKNOWN,
                {"event": 1.0, "pose": 0.0, "ball": 0.0},
                "Dead ball event suppressed from live stroke classification."
            )

        # 2. Tier 1: Semantic Serve Event Pass-through
        if event_type == "SERVE_CONTACT":
            return (
                ShotType.SERVE,
                0.95,
                ShotClassificationSource.EVENT_PASSTHROUGH,
                {"event": 0.95, "pose": 0.90, "ball": 0.90},
                "Authoritative serve contact verified from kinematics and match state."
            )

        if player_id is None:
            return (
                ShotType.UNKNOWN,
                0.0,
                ShotClassificationSource.ABSTENTION_UNKNOWN,
                {},
                "Unattributed player hit."
            )

        handedness = self.handedness_map.get(player_id, PlayerHandedness.RIGHT_HANDED)
        court_side_sign = 1.0 if player_id == 1 else -1.0
        handedness_sign = 1.0 if handedness == PlayerHandedness.RIGHT_HANDED else (-1.0 if handedness == PlayerHandedness.LEFT_HANDED else 1.0)

        # 3. Compute Multi-Cue Kinematic and Spatial Features
        # A. Ball-Player Lateral Spatial Geometry
        geo_valid = False
        norm_geo_dx = 0.0
        if player_box is not None and ball_point is not None and ball_point.x_px is not None:
            bw = max(10.0, player_box.x2 - player_box.x1)
            player_cx = (player_box.x1 + player_box.x2) / 2.0
            raw_dx = (ball_point.x_px - player_cx) / bw
            norm_geo_dx = court_side_sign * handedness_sign * raw_dx
            geo_valid = True

        # B. YOLO11-Pose Temporal Feature Extraction
        pose_feat = None
        if self.pose_extractor is not None and frames is not None and all_player_boxes is not None:
            pose_feat = self.pose_extractor.extract_hit_window_features(
                frames=frames,
                player_boxes=all_player_boxes,
                hit_frame=hit_frame,
                player_id=player_id,
                handedness=handedness
            )

        pose_valid = (
            pose_feat is not None 
            and pose_feat.get("valid", False) 
            and pose_feat.get("mean_pose_confidence", 0.0) >= self.min_pose_confidence
        )
        norm_pose_dx = pose_feat["normalized_wrist_displacement"] if pose_valid else 0.0
        pose_conf = pose_feat.get("mean_pose_confidence", 0.0) if pose_valid else 0.0

        # 4. Multi-Cue Fusion & Classification Tiers
        # Tier 2A: Dual Consensus (Pose + Geometry Agree)
        if pose_valid and geo_valid:
            if norm_pose_dx > self.ambiguity_threshold and norm_geo_dx > 0.05:
                conf = min(0.98, 0.85 + abs(norm_pose_dx) * 0.15)
                return (
                    ShotType.FOREHAND,
                    conf,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.95, "pose": pose_conf, "geometry": abs(norm_geo_dx)},
                    f"Consensus Forehand: Pose (disp={norm_pose_dx:+.2f}) and Geometry (geo_dx={norm_geo_dx:+.2f}) agree."
                )
            elif norm_pose_dx < -self.ambiguity_threshold and norm_geo_dx < -0.05:
                conf = min(0.98, 0.85 + abs(norm_pose_dx) * 0.15)
                return (
                    ShotType.BACKHAND,
                    conf,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.95, "pose": pose_conf, "geometry": abs(norm_geo_dx)},
                    f"Consensus Backhand: Pose (disp={norm_pose_dx:+.2f}) and Geometry (geo_dx={norm_geo_dx:+.2f}) agree."
                )

        # Tier 2B: Unambiguous Lateral Ball Geometry
        if geo_valid:
            if norm_geo_dx > 0.10:
                conf = min(0.90, 0.70 + abs(norm_geo_dx) * 0.20)
                return (
                    ShotType.FOREHAND,
                    conf,
                    ShotClassificationSource.GEOMETRY_BASELINE,
                    {"event": 0.85, "pose": pose_conf, "geometry": abs(norm_geo_dx)},
                    f"Geometry: Ball offset on forehand side (geo_dx={norm_geo_dx:+.2f})."
                )
            elif norm_geo_dx < -0.10:
                conf = min(0.90, 0.70 + abs(norm_geo_dx) * 0.20)
                return (
                    ShotType.BACKHAND,
                    conf,
                    ShotClassificationSource.GEOMETRY_BASELINE,
                    {"event": 0.85, "pose": pose_conf, "geometry": abs(norm_geo_dx)},
                    f"Geometry: Ball offset on backhand side (geo_dx={norm_geo_dx:+.2f})."
                )

        # Tier 2C: High-Confidence Pose Extension Alone
        if pose_valid and pose_conf >= 0.50:
            if norm_pose_dx > self.ambiguity_threshold * 1.5:
                return (
                    ShotType.FOREHAND,
                    0.80,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.80, "pose": pose_conf, "displacement": abs(norm_pose_dx)},
                    f"Pose: Dominant wrist extension on forehand side (disp={norm_pose_dx:+.2f})."
                )
            elif norm_pose_dx < -self.ambiguity_threshold * 1.5:
                return (
                    ShotType.BACKHAND,
                    0.80,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.80, "pose": pose_conf, "displacement": abs(norm_pose_dx)},
                    f"Pose: Dominant wrist extension on backhand side (disp={norm_pose_dx:+.2f})."
                )

        # 5. Tier 4: Safe Abstention to UNKNOWN
        return (
            ShotType.UNKNOWN,
            0.50,
            ShotClassificationSource.ABSTENTION_UNKNOWN,
            {"event": 0.70, "pose": pose_conf, "geometry": abs(norm_geo_dx) if geo_valid else 0.0},
            "Ambiguous kinematic and spatial evidence. Safely abstained to UNKNOWN."
        )

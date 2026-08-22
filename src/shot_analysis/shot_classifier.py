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

        # 3. Tier 2: YOLO11-Pose Temporal Feature Extraction
        pose_feat = None
        if self.pose_extractor is not None and frames is not None and all_player_boxes is not None:
            pose_feat = self.pose_extractor.extract_hit_window_features(
                frames=frames,
                player_boxes=all_player_boxes,
                hit_frame=hit_frame,
                player_id=player_id,
                handedness=handedness
            )

        if pose_feat and pose_feat.get("valid") and pose_feat.get("mean_pose_confidence", 0.0) >= self.min_pose_confidence:
            norm_disp = pose_feat["normalized_wrist_displacement"]
            pose_conf = pose_feat["mean_pose_confidence"]

            if norm_disp > self.ambiguity_threshold:
                conf = min(0.95, 0.70 + abs(norm_disp) * 0.25)
                return (
                    ShotType.FOREHAND,
                    conf,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.90, "pose": pose_conf, "displacement": abs(norm_disp)},
                    f"Pose dominant wrist extension on forehand side (disp={norm_disp:+.2f}, conf={pose_conf:.2f})."
                )
            elif norm_disp < -self.ambiguity_threshold:
                conf = min(0.95, 0.70 + abs(norm_disp) * 0.25)
                return (
                    ShotType.BACKHAND,
                    conf,
                    ShotClassificationSource.YOLO11_POSE_TEMPORAL,
                    {"event": 0.90, "pose": pose_conf, "displacement": abs(norm_disp)},
                    f"Pose dominant wrist extension on backhand side (disp={norm_disp:+.2f}, conf={pose_conf:.2f})."
                )

        # 4. Tier 3: Geometry-Only Baseline Fallback
        if player_box is not None and ball_point is not None and ball_point.x_px is not None:
            bw = max(10.0, player_box.x2 - player_box.x1)
            player_cx = (player_box.x1 + player_box.x2) / 2.0
            raw_dx = (ball_point.x_px - player_cx) / bw

            # Normalize for court side and handedness
            norm_geo_dx = court_side_sign * handedness_sign * raw_dx

            if norm_geo_dx > 0.20:
                conf = min(0.85, 0.65 + abs(norm_geo_dx) * 0.20)
                return (
                    ShotType.FOREHAND,
                    conf,
                    ShotClassificationSource.GEOMETRY_BASELINE,
                    {"event": 0.85, "pose": 0.0, "geometry": abs(norm_geo_dx)},
                    f"Geometry baseline: Ball offset on forehand side (geo_dx={norm_geo_dx:+.2f})."
                )
            elif norm_geo_dx < -0.20:
                conf = min(0.85, 0.65 + abs(norm_geo_dx) * 0.20)
                return (
                    ShotType.BACKHAND,
                    conf,
                    ShotClassificationSource.GEOMETRY_BASELINE,
                    {"event": 0.85, "pose": 0.0, "geometry": abs(norm_geo_dx)},
                    f"Geometry baseline: Ball offset on backhand side (geo_dx={norm_geo_dx:+.2f})."
                )

        # 5. Tier 4: Safe Abstention to UNKNOWN
        return (
            ShotType.UNKNOWN,
            0.50,
            ShotClassificationSource.ABSTENTION_UNKNOWN,
            {"event": 0.70, "pose": 0.0, "geometry": 0.0},
            "Ambiguous kinematic and spatial evidence. Safely abstained to UNKNOWN."
        )

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

class ShotType(Enum):
    SERVE = "SERVE"
    FOREHAND = "FOREHAND"
    BACKHAND = "BACKHAND"
    VOLLEY = "VOLLEY"
    OVERHEAD = "OVERHEAD"
    UNKNOWN = "UNKNOWN"

class ShotDirection(Enum):
    CROSS_COURT = "CROSS_COURT"
    DOWN_THE_LINE = "DOWN_THE_LINE"
    MIDDLE = "MIDDLE"
    UNKNOWN = "UNKNOWN"

class CourtDepthZone(Enum):
    SHORT = "SHORT"
    MID = "MID"
    DEEP = "DEEP"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"

class CourtLateralZone(Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"

class CourtZone3x3(Enum):
    SHORT_LEFT = "SHORT_LEFT"
    SHORT_CENTER = "SHORT_CENTER"
    SHORT_RIGHT = "SHORT_RIGHT"
    MID_LEFT = "MID_LEFT"
    MID_CENTER = "MID_CENTER"
    MID_RIGHT = "MID_RIGHT"
    DEEP_LEFT = "DEEP_LEFT"
    DEEP_CENTER = "DEEP_CENTER"
    DEEP_RIGHT = "DEEP_RIGHT"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"

class PlayerHandedness(Enum):
    RIGHT_HANDED = "RIGHT_HANDED"
    LEFT_HANDED = "LEFT_HANDED"
    UNKNOWN_HANDEDNESS = "UNKNOWN_HANDEDNESS"

class ShotOutcome(Enum):
    IN_PLAY = "IN_PLAY"
    OUT = "OUT"
    FAULT = "FAULT"
    POINT_WINNING_SHOT = "POINT_WINNING_SHOT"
    POINT_LOSING_SHOT = "POINT_LOSING_SHOT"
    UNKNOWN = "UNKNOWN"

class ShotClassificationSource(Enum):
    EVENT_PASSTHROUGH = "EVENT_PASSTHROUGH"
    YOLO11_POSE_TEMPORAL = "YOLO11_POSE_TEMPORAL"
    GEOMETRY_BASELINE = "GEOMETRY_BASELINE"
    ABSTENTION_UNKNOWN = "ABSTENTION_UNKNOWN"

@dataclass
class ShotEventEvidence:
    """
    Structured evidence record for a single tennis stroke.
    Links perception event, player attribution, stroke classification, direction, and bounce landing.
    """
    shot_id: int
    match_event_id: int
    frame_index: int
    timestamp_s: float
    player_id: int
    shot_type: ShotType
    shot_confidence: float
    classification_source: ShotClassificationSource
    
    direction: ShotDirection
    direction_confidence: float
    direction_source: str # e.g. "GEOMETRY_DERIVED"
    
    source_court_position_m: Optional[Tuple[float, float]] = None
    landing_court_position_m: Optional[Tuple[float, float]] = None
    landing_zone: Optional[CourtZone3x3] = None
    bounce_event_id: Optional[int] = None
    bounce_frame: Optional[int] = None
    
    speed_kmh: Optional[float] = None
    outcome: ShotOutcome = ShotOutcome.UNKNOWN
    is_dead_ball: bool = False
    
    confidence_components: Optional[Dict[str, float]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['shot_type'] = self.shot_type.value
        d['direction'] = self.direction.value
        d['classification_source'] = self.classification_source.value
        d['landing_zone'] = self.landing_zone.value if self.landing_zone else None
        d['outcome'] = self.outcome.value
        return d

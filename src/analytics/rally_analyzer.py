from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from src.shot_analysis.shot_types import ShotEventEvidence, ShotType

@dataclass
class RallySegment:
    """Structured representation of a single live tennis rally."""
    rally_id: int
    point_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    server_id: int
    receiver_id: int
    serve_attempt: int
    
    total_strokes_including_serve: int
    rally_hits_excluding_serve: int
    shots_p1: int
    shots_p2: int
    
    winner_id: Optional[int] = None
    ending_reason: str = ""
    is_dead_ball_point: bool = False
    shot_ids: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RallyAnalyzer:
    """
    Rally Segmentation and Analytics Engine.
    Segments live rallies while strictly suppressing dead-ball events (e.g. Frame 84 return after fault).
    """

    @classmethod
    def analyze_rallies(
        cls,
        shot_evidences: List[ShotEventEvidence],
        server_id: int = 2,
        receiver_id: int = 1,
        serve_attempt: int = 1,
        winner_id: Optional[int] = None,
        ending_reason: str = "",
        point_id: int = 1
    ) -> List[RallySegment]:
        """
        Segments shot sequences into formal rally records.
        """
        live_shots = [s for s in shot_evidences if not s.is_dead_ball]
        
        if not live_shots:
            # All shots were dead or single serve fault
            return []

        start_s = live_shots[0]
        end_s = live_shots[-1]
        
        total_strokes = len(live_shots)
        rally_hits = len([s for s in live_shots if s.shot_type != ShotType.SERVE])
        shots_p1 = len([s for s in live_shots if s.player_id == 1])
        shots_p2 = len([s for s in live_shots if s.player_id == 2])

        dur = max(0.0, end_s.timestamp_s - start_s.timestamp_s)
        if dur == 0.0 and len(live_shots) > 1:
            dur = (end_s.frame_index - start_s.frame_index) / 30.0

        rally = RallySegment(
            rally_id=1,
            point_id=point_id,
            start_frame=start_s.frame_index,
            end_frame=end_s.bounce_frame or (end_s.frame_index + 30),
            start_time_s=start_s.timestamp_s,
            end_time_s=end_s.timestamp_s + (dur if dur > 0 else 1.0),
            duration_s=round(dur, 2),
            server_id=server_id,
            receiver_id=receiver_id,
            serve_attempt=serve_attempt,
            total_strokes_including_serve=total_strokes,
            rally_hits_excluding_serve=rally_hits,
            shots_p1=shots_p1,
            shots_p2=shots_p2,
            winner_id=winner_id,
            ending_reason=ending_reason,
            is_dead_ball_point=(total_strokes == 1 and start_s.shot_type == ShotType.SERVE and start_s.outcome == "FAULT"),
            shot_ids=[s.shot_id for s in live_shots]
        )
        return [rally]

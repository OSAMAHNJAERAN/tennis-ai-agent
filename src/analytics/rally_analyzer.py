from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from src.shot_analysis.shot_types import ShotEventEvidence, ShotType
from src.utils.shot_evidence_sequence import require_shot_result_owner

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
        point_id: int = 1,
        fps: Optional[float] = None,
        *,
        result_owner=None,
    ) -> List[RallySegment]:
        """
        Segments shot sequences into formal rally records.
        """
        owner = require_shot_result_owner(shot_evidences, result_owner) if result_owner is not None else None
        live_shots = owner.select(live_only=True) if owner is not None else shot_evidences
        start_s = end_s = None
        total_strokes = rally_hits = shots_p1 = shots_p2 = 0
        shot_ids = [] if owner is None else live_shots.field('shot_id')
        for shot in live_shots:
            if shot.is_dead_ball:
                continue
            if start_s is None:
                start_s = shot
            end_s = shot
            total_strokes += 1
            rally_hits += shot.shot_type != ShotType.SERVE
            shots_p1 += shot.player_id == 1
            shots_p2 += shot.player_id == 2
            if owner is None:
                shot_ids.append(shot.shot_id)

        if start_s is None:
            # All shots were dead or single serve fault
            return []

        end_frame = end_s.bounce_frame if end_s.bounce_frame is not None else end_s.frame_index

        if fps is not None:
            if fps <= 0:
                raise ValueError("fps must be positive")
            # Frame indices are the authoritative video time base and include a
            # terminal bounce when one is linked to the final stroke.
            start_time_s = start_s.frame_index / fps
            end_time_s = end_frame / fps
        else:
            start_time_s = start_s.timestamp_s
            end_time_s = end_s.timestamp_s
        dur = max(0.0, end_time_s - start_time_s)

        rally = RallySegment(
            rally_id=1,
            point_id=point_id,
            start_frame=start_s.frame_index,
            end_frame=end_frame,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
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
            shot_ids=shot_ids
        )
        return [rally]

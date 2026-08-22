from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple

from src.line_calling.line_geometry import ServiceBoxType

class MatchFormat(str, Enum):
    BEST_OF_3 = "BEST_OF_3"
    BEST_OF_5 = "BEST_OF_5"

class SetFormat(str, Enum):
    TIE_BREAK_SET = "TIE_BREAK_SET"  # Standard: 6-6 enters 7-point tiebreak
    ADVANTAGE_SET = "ADVANTAGE_SET"  # Advantage set: must win by 2 games

class ServiceSide(str, Enum):
    DEUCE = "DEUCE"  # Server's right half
    AD = "AD"        # Server's left half

class PointState(str, Enum):
    WAITING_FOR_SERVE = "WAITING_FOR_SERVE"
    FIRST_SERVE_IN_PROGRESS = "FIRST_SERVE_IN_PROGRESS"
    SECOND_SERVE_WAITING = "SECOND_SERVE_WAITING"
    SECOND_SERVE_IN_PROGRESS = "SECOND_SERVE_IN_PROGRESS"
    RALLY_LIVE = "RALLY_LIVE"
    POINT_COMPLETE = "POINT_COMPLETE"
    POINT_REVIEW_PENDING = "POINT_REVIEW_PENDING"

class BallPlayState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    SERVE_STARTED = "SERVE_STARTED"
    LIVE = "LIVE"
    DEAD = "DEAD"
    LET_REPLAY = "LET_REPLAY"
    POINT_COMPLETE = "POINT_COMPLETE"

class ServerCourtEnd(str, Enum):
    FAR_COURT = "FAR_COURT"   # Top baseline from camera perspective (typically Player 2)
    NEAR_COURT = "NEAR_COURT" # Bottom baseline from camera perspective (typically Player 1)

@dataclass
class CompletedSetScore:
    set_number: int
    games_p1: int
    games_p2: int
    tie_break_p1: Optional[int] = None
    tie_break_p2: Optional[int] = None
    winner_id: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MatchState:
    """
    Central authoritative match state model for tennis officiating.
    Maintains pure numeric counters for sets, games, and points,
    along with server rotation, tie-break state, and display converters.
    """
    match_id: str = "match_001"
    format: MatchFormat = MatchFormat.BEST_OF_3
    set_format: SetFormat = SetFormat.TIE_BREAK_SET
    player_1_id: int = 1
    player_2_id: int = 2
    server_id: int = 1
    receiver_id: int = 2
    initial_server_id: int = 1
    
    current_set: int = 1
    current_game: int = 1
    
    sets_won_p1: int = 0
    sets_won_p2: int = 0
    
    games_p1: int = 0
    games_p2: int = 0
    
    points_p1: int = 0
    points_p2: int = 0
    
    serve_attempt: int = 1  # 1 (First Serve) or 2 (Second Serve)
    service_side: ServiceSide = ServiceSide.DEUCE
    
    tie_break_active: bool = False
    tie_break_points_p1: int = 0
    tie_break_points_p2: int = 0
    
    point_state: PointState = PointState.WAITING_FOR_SERVE
    ball_state: BallPlayState = BallPlayState.NOT_STARTED
    
    completed_sets: List[CompletedSetScore] = field(default_factory=list)
    match_complete: bool = False
    winner_id: Optional[int] = None
    
    last_point_winner: Optional[int] = None
    last_point_reason: str = ""
    state_version: int = 0

    def get_points_display_p1(self) -> str:
        """Returns standard tennis display string for Player 1 points."""
        if self.tie_break_active:
            return str(self.tie_break_points_p1)
        return self._format_standard_points(self.points_p1, self.points_p2)

    def get_points_display_p2(self) -> str:
        """Returns standard tennis display string for Player 2 points."""
        if self.tie_break_active:
            return str(self.tie_break_points_p2)
        return self._format_standard_points(self.points_p2, self.points_p1)

    @staticmethod
    def _format_standard_points(p_self: int, p_opp: int) -> str:
        standard_map = {0: "0", 1: "15", 2: "30", 3: "40"}
        if p_self < 3 and p_opp < 3:
            return standard_map.get(p_self, "0")
        if p_self == 3 and p_opp < 3:
            return "40"
        if p_opp == 3 and p_self < 3:
            return standard_map.get(p_self, "0")
        
        # Deuce / Advantage situations (p_self >= 3 and p_opp >= 3)
        if p_self == p_opp:
            return "40"  # Displays 40-40 (Deuce)
        elif p_self == p_opp + 1:
            return "AD"
        elif p_self > p_opp + 1:
            return "GAME"
        else:
            return "40"

    def get_score_summary_str(self) -> str:
        """Returns clean human-readable score summary e.g. 'Set 1 (6-4) | Game 5: P1 40 - P2 30 | Server: P1'."""
        p1_pts = self.get_points_display_p1()
        p2_pts = self.get_points_display_p2()
        if not self.tie_break_active and self.points_p1 >= 3 and self.points_p2 >= 3 and self.points_p1 == self.points_p2:
            pts_str = "DEUCE"
        else:
            pts_str = f"P1 {p1_pts} - P2 {p2_pts}"
            
        return f"Set {self.current_set} ({self.games_p1}-{self.games_p2}) | {pts_str} | Server: P{self.server_id} ({self.service_side.value})"

    def get_expected_service_box(self, server_end: Optional[ServerCourtEnd] = None) -> ServiceBoxType:
        """
        Determines the expected target service box for the current server and service side.
        Cross-court geometry:
        - Far Server (ServerEnd = FAR_COURT, e.g. Player 2):
            - Deuce serve -> Receiver's Near Deuce box (X in [1.37, 5.485])
            - Ad serve    -> Receiver's Near Ad box (X in [5.485, 9.60])
        - Near Server (ServerEnd = NEAR_COURT, e.g. Player 1):
            - Deuce serve -> Receiver's Far Deuce box (X in [1.37, 5.485])
            - Ad serve    -> Receiver's Far Ad box (X in [5.485, 9.60])
        """
        # Default assumption: Player 2 at Far Court, Player 1 at Near Court
        if server_end is None:
            server_end = ServerCourtEnd.FAR_COURT if self.server_id == 2 else ServerCourtEnd.NEAR_COURT

        if server_end == ServerCourtEnd.FAR_COURT:
            return ServiceBoxType.NEAR_DEUCE if self.service_side == ServiceSide.DEUCE else ServiceBoxType.NEAR_AD
        else:
            return ServiceBoxType.FAR_DEUCE if self.service_side == ServiceSide.DEUCE else ServiceBoxType.FAR_AD

    def to_dict(self) -> Dict[str, Any]:
        """Serializes MatchState to structured dict for JSON export."""
        return {
            "match_id": self.match_id,
            "format": self.format.value,
            "set_format": self.set_format.value,
            "player_1": {
                "id": self.player_1_id,
                "sets_won": self.sets_won_p1,
                "games": self.games_p1,
                "points_raw": self.points_p1 if not self.tie_break_active else self.tie_break_points_p1,
                "points_display": self.get_points_display_p1()
            },
            "player_2": {
                "id": self.player_2_id,
                "sets_won": self.sets_won_p2,
                "games": self.games_p2,
                "points_raw": self.points_p2 if not self.tie_break_active else self.tie_break_points_p2,
                "points_display": self.get_points_display_p2()
            },
            "server_id": self.server_id,
            "receiver_id": self.receiver_id,
            "current_set": self.current_set,
            "current_game": self.current_game,
            "serve_attempt": self.serve_attempt,
            "service_side": self.service_side.value,
            "tie_break_active": self.tie_break_active,
            "point_state": self.point_state.value,
            "ball_state": self.ball_state.value,
            "completed_sets": [s.to_dict() for s in self.completed_sets],
            "match_complete": self.match_complete,
            "winner_id": self.winner_id,
            "last_point_winner": self.last_point_winner,
            "last_point_reason": self.last_point_reason,
            "state_version": self.state_version
        }

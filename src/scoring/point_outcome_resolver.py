from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from src.scoring.match_state import BallPlayState
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision

class PointOutcomeType(str, Enum):
    FIRST_SERVE_FAULT = "FIRST_SERVE_FAULT"
    DOUBLE_FAULT = "DOUBLE_FAULT"
    SERVICE_LET = "SERVICE_LET"
    SERVE_IN_PLAY = "SERVE_IN_PLAY"
    POINT_WON_PLAYER_1 = "POINT_WON_PLAYER_1"
    POINT_WON_PLAYER_2 = "POINT_WON_PLAYER_2"
    RALLY_CONTINUES = "RALLY_CONTINUES"
    DEAD_BALL_IGNORED = "DEAD_BALL_IGNORED"
    POINT_REVIEW_REQUIRED = "POINT_REVIEW_REQUIRED"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"

@dataclass
class PointOutcome:
    outcome_type: PointOutcomeType
    winner_id: Optional[int] = None
    reason: str = ""
    is_point_ending: bool = False
    confidence: float = 1.0
    source_event_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['outcome_type'] = self.outcome_type.value
        return d

class PointOutcomeResolver:
    """
    Decoupled point outcome resolver.
    Translates sequences of verified match events and line calls into semantic point outcomes.
    Enforces dead-ball suppression, fault tracking, and legal hitter attribution.
    """

    def __init__(self, current_server_id: int = 1, initial_serve_attempt: int = 1):
        self.current_server_id = current_server_id
        self.serve_attempt = initial_serve_attempt
        self.last_legal_hitter_id: Optional[int] = None
        self.ball_play_state: BallPlayState = BallPlayState.NOT_STARTED
        self.in_bounce_count: int = 0

    def reset_point(self, server_id: int, serve_attempt: int = 1, initial_ball_state: BallPlayState = BallPlayState.NOT_STARTED) -> None:
        """Resets state variables for a new point or second serve attempt."""
        self.current_server_id = server_id
        self.serve_attempt = serve_attempt
        self.last_legal_hitter_id = None
        self.ball_play_state = initial_ball_state
        self.in_bounce_count = 0

    def resolve_event(
        self,
        event_id: int,
        event_type: str,
        player_id: Optional[int] = None,
        line_call: Optional[LineCallEvidence] = None,
        confidence: float = 1.0
    ) -> PointOutcome:
        """
        Processes a single match event and returns the resulting PointOutcome.
        """
        # 1. Unresolved / Uncertainty Line-Call Gate
        if line_call and line_call.decision == LineCallDecision.REVIEW_REQUIRED:
            return PointOutcome(
                outcome_type=PointOutcomeType.POINT_REVIEW_REQUIRED,
                winner_id=None,
                reason=f"Line calling requires human review: {line_call.reason}",
                is_point_ending=True,
                confidence=line_call.confidence,
                source_event_id=event_id
            )

        # 2. Dead-Ball Suppression Gate (Critical Rule: dead ball cannot continue play)
        if self.ball_play_state == BallPlayState.DEAD:
            return PointOutcome(
                outcome_type=PointOutcomeType.DEAD_BALL_IGNORED,
                winner_id=None,
                reason=f"Event '{event_type}' ignored because ball is dead after fault or point conclusion.",
                is_point_ending=False,
                confidence=1.0,
                source_event_id=event_id
            )

        # 3. Serve Contact
        if event_type == "SERVE_CONTACT":
            self.ball_play_state = BallPlayState.SERVE_STARTED
            self.last_legal_hitter_id = player_id or self.current_server_id
            self.in_bounce_count = 0
            return PointOutcome(
                outcome_type=PointOutcomeType.SERVE_IN_PLAY,
                winner_id=None,
                reason=f"Player {self.last_legal_hitter_id} served ball (Attempt {self.serve_attempt}).",
                is_point_ending=False,
                confidence=confidence,
                source_event_id=event_id
            )

        # 4. Service Let
        if event_type == "SERVICE_LET":
            self.ball_play_state = BallPlayState.LET_REPLAY
            return PointOutcome(
                outcome_type=PointOutcomeType.SERVICE_LET,
                winner_id=None,
                reason="Service let: ball replayed with no score change.",
                is_point_ending=False,
                confidence=confidence,
                source_event_id=event_id
            )

        # 5. Serve Bounce Evaluation
        if event_type == "BOUNCE" and self.ball_play_state == BallPlayState.SERVE_STARTED:
            if line_call and line_call.decision in (LineCallDecision.SERVE_FAULT, LineCallDecision.OUT):
                self.ball_play_state = BallPlayState.DEAD
                if self.serve_attempt == 1:
                    return PointOutcome(
                        outcome_type=PointOutcomeType.FIRST_SERVE_FAULT,
                        winner_id=None,
                        reason=f"First serve fault by Player {self.current_server_id} ({line_call.reason}).",
                        is_point_ending=False,
                        confidence=line_call.confidence,
                        source_event_id=event_id
                    )
                else:
                    # Second fault -> Double fault -> Receiver point
                    receiver_id = 2 if self.current_server_id == 1 else 1
                    return PointOutcome(
                        outcome_type=PointOutcomeType.DOUBLE_FAULT,
                        winner_id=receiver_id,
                        reason=f"Double fault by Player {self.current_server_id}: Receiver Player {receiver_id} wins point.",
                        is_point_ending=True,
                        confidence=line_call.confidence,
                        source_event_id=event_id
                    )

            elif line_call and line_call.decision in (LineCallDecision.SERVE_IN, LineCallDecision.IN):
                self.ball_play_state = BallPlayState.LIVE
                self.in_bounce_count = 1
                return PointOutcome(
                    outcome_type=PointOutcomeType.SERVE_IN_PLAY,
                    winner_id=None,
                    reason=f"Legal serve landed in by Player {self.current_server_id}; rally live.",
                    is_point_ending=False,
                    confidence=line_call.confidence,
                    source_event_id=event_id
                )

        # 6. Player Hit during Live Rally
        if event_type in ("PLAYER_1_HIT", "PLAYER_2_HIT"):
            hitter = 1 if event_type == "PLAYER_1_HIT" else 2
            self.last_legal_hitter_id = hitter
            self.ball_play_state = BallPlayState.LIVE
            self.in_bounce_count = 0
            return PointOutcome(
                outcome_type=PointOutcomeType.RALLY_CONTINUES,
                winner_id=None,
                reason=f"Player {hitter} struck return shot.",
                is_point_ending=False,
                confidence=confidence,
                source_event_id=event_id
            )

        # 7. Rally Bounce Evaluation
        if event_type == "BOUNCE" and self.ball_play_state == BallPlayState.LIVE:
            if line_call and line_call.decision == LineCallDecision.OUT:
                self.ball_play_state = BallPlayState.DEAD
                hitter = self.last_legal_hitter_id or (1 if self.current_server_id == 2 else 2)
                winner = 2 if hitter == 1 else 1
                return PointOutcome(
                    outcome_type=PointOutcomeType.POINT_WON_PLAYER_2 if winner == 2 else PointOutcomeType.POINT_WON_PLAYER_1,
                    winner_id=winner,
                    reason=f"Player {hitter} shot landed OUT ({line_call.reason}): Player {winner} wins point.",
                    is_point_ending=True,
                    confidence=line_call.confidence,
                    source_event_id=event_id
                )

            elif line_call and line_call.decision == LineCallDecision.IN:
                self.in_bounce_count += 1
                if self.in_bounce_count >= 2:
                    self.ball_play_state = BallPlayState.DEAD
                    winner = self.last_legal_hitter_id or self.current_server_id
                    return PointOutcome(
                        outcome_type=PointOutcomeType.POINT_WON_PLAYER_1 if winner == 1 else PointOutcomeType.POINT_WON_PLAYER_2,
                        winner_id=winner,
                        reason=f"Ball bounced twice without legal return: Player {winner} wins point.",
                        is_point_ending=True,
                        confidence=line_call.confidence,
                        source_event_id=event_id
                    )
                else:
                    return PointOutcome(
                        outcome_type=PointOutcomeType.RALLY_CONTINUES,
                        winner_id=None,
                        reason="Ball landed IN court; rally continues.",
                        is_point_ending=False,
                        confidence=line_call.confidence,
                        source_event_id=event_id
                    )

        # 8. Unhandled or out-of-order event
        return PointOutcome(
            outcome_type=PointOutcomeType.UNKNOWN_EVENT,
            winner_id=None,
            reason=f"Event '{event_type}' does not trigger a deterministic rule transition.",
            is_point_ending=False,
            confidence=confidence,
            source_event_id=event_id
        )

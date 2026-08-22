from typing import Optional, Dict, Any, List, Tuple
import copy
import json

from src.scoring.match_state import (
    MatchState,
    MatchFormat,
    SetFormat,
    ServiceSide,
    PointState,
    BallPlayState
)
from src.scoring.scoring_rules import TennisScoringRules
from src.scoring.point_outcome_resolver import PointOutcomeResolver, PointOutcomeType, PointOutcome
from src.scoring.event_history import ScoreEventHistory, ScoringEventLogEntry
from src.line_calling.line_call_engine import LineCallEvidence

class TennisScoringEngine:
    """
    Authoritative, deterministic tennis match scoring engine.
    Orchestrates semantic events, point resolution, ITF scoring rules,
    event idempotency, safety pauses, and full history logging.
    """

    def __init__(
        self,
        match_id: str = "match_001",
        match_format: MatchFormat = MatchFormat.BEST_OF_3,
        set_format: SetFormat = SetFormat.TIE_BREAK_SET,
        player_1_id: int = 1,
        player_2_id: int = 2,
        initial_server_id: int = 1,
        initial_state: Optional[MatchState] = None
    ):
        if initial_state is not None:
            self.state = copy.deepcopy(initial_state)
        else:
            self.state = MatchState(
                match_id=match_id,
                format=match_format,
                set_format=set_format,
                player_1_id=player_1_id,
                player_2_id=player_2_id,
                server_id=initial_server_id,
                receiver_id=player_2_id if initial_server_id == player_1_id else player_1_id,
                initial_server_id=initial_server_id
            )

        self.resolver = PointOutcomeResolver(
            current_server_id=self.state.server_id,
            initial_serve_attempt=self.state.serve_attempt
        )
        self.history = ScoreEventHistory()

    def process_match_event(
        self,
        event_id: int,
        event_type: str,
        timestamp_seconds: float,
        player_id: Optional[int] = None,
        line_call: Optional[LineCallEvidence] = None,
        confidence: float = 1.0
    ) -> Tuple[MatchState, PointOutcome]:
        """
        Idempotently processes an incoming match event and transitions the MatchState.
        """
        # 1. Idempotency Check
        if self.history.is_processed(event_id):
            cached_outcome = PointOutcome(
                outcome_type=PointOutcomeType.UNKNOWN_EVENT,
                reason="Duplicate event ID: ignored to maintain idempotency.",
                source_event_id=event_id
            )
            return self.state, cached_outcome

        prev_state_dict = self.state.to_dict()

        # 2. Point Outcome Resolution
        outcome = self.resolver.resolve_event(
            event_id=event_id,
            event_type=event_type,
            player_id=player_id,
            line_call=line_call,
            confidence=confidence
        )

        # 3. State Machine Transitions based on Outcome
        if outcome.outcome_type == PointOutcomeType.POINT_REVIEW_REQUIRED:
            self.state.point_state = PointState.POINT_REVIEW_PENDING
            self.state.last_point_reason = outcome.reason
            self.state.state_version += 1

        elif outcome.outcome_type == PointOutcomeType.FIRST_SERVE_FAULT:
            self.state, _ = TennisScoringRules.record_fault(self.state, self.state.server_id, outcome.reason)
            self.resolver.reset_point(self.state.server_id, serve_attempt=2, initial_ball_state=BallPlayState.DEAD)

        elif outcome.outcome_type == PointOutcomeType.DOUBLE_FAULT:
            self.state, _ = TennisScoringRules.record_fault(self.state, self.state.server_id, outcome.reason)
            self.resolver.reset_point(self.state.server_id, serve_attempt=1, initial_ball_state=BallPlayState.DEAD)

        elif outcome.outcome_type == PointOutcomeType.SERVICE_LET:
            self.state = TennisScoringRules.record_service_let(self.state)
            self.resolver.reset_point(self.state.server_id, serve_attempt=self.state.serve_attempt, initial_ball_state=BallPlayState.LET_REPLAY)

        elif outcome.outcome_type == PointOutcomeType.SERVE_IN_PLAY:
            self.state.point_state = PointState.RALLY_LIVE
            self.state.ball_state = BallPlayState.LIVE
            self.state.state_version += 1

        elif outcome.outcome_type == PointOutcomeType.RALLY_CONTINUES:
            self.state.point_state = PointState.RALLY_LIVE
            self.state.ball_state = BallPlayState.LIVE
            self.state.state_version += 1

        elif outcome.outcome_type in (PointOutcomeType.POINT_WON_PLAYER_1, PointOutcomeType.POINT_WON_PLAYER_2):
            winner = 1 if outcome.outcome_type == PointOutcomeType.POINT_WON_PLAYER_1 else 2
            self.state = TennisScoringRules.award_point(self.state, winner, outcome.reason)
            self.resolver.reset_point(self.state.server_id, serve_attempt=1, initial_ball_state=BallPlayState.DEAD)

        elif outcome.outcome_type == PointOutcomeType.DEAD_BALL_IGNORED:
            # Ball is already dead; state remains unchanged
            pass

        # 4. Record to Event Sourced History
        self.history.append(
            event_id=event_id,
            timestamp_seconds=timestamp_seconds,
            event_type=event_type,
            point_outcome=outcome.outcome_type.value,
            previous_state=prev_state_dict,
            new_state=self.state.to_dict(),
            reason=outcome.reason,
            confidence=confidence
        )

        return self.state, outcome

    def get_current_state(self) -> MatchState:
        """Returns the current MatchState."""
        return self.state

    def save_state_json(self, filepath: str) -> None:
        """Saves current match state to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def save_history_json(self, filepath: str) -> None:
        """Saves history log to JSON file."""
        self.history.save_json(filepath)

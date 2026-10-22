from typing import Optional, Dict, Any, List, Tuple
import copy
import json
import math

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
            initial_serve_attempt=self.state.serve_attempt,
            player_1_id=self.state.player_1_id,
            player_2_id=self.state.player_2_id
        )
        violations = TennisScoringRules.validate_invariants(self.state)
        if violations:
            raise ValueError("Invalid initial match state: " + " ".join(violations))
        self.initial_state = copy.deepcopy(self.state)
        self.history = ScoreEventHistory()
        self.revision = 0
        self.corrections: List[Dict[str, Any]] = []
        self._inputs: List[Dict[str, Any]] = []
        self.completed_point_ids = set()

    def process_match_event(
        self,
        event_id: int,
        event_type: str,
        timestamp_seconds: float,
        player_id: Optional[int] = None,
        line_call: Optional[LineCallEvidence] = None,
        confidence: float = 1.0,
        point_id: Optional[str] = None
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

        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Event timestamp must be finite and nonnegative.")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Event confidence must be in [0, 1].")
        if self.history.entries and timestamp_seconds < self.history.entries[-1].timestamp_seconds:
            raise ValueError("Events must be processed in source-time order.")
        if point_id is not None and point_id in self.completed_point_ids:
            return self.state, PointOutcome(PointOutcomeType.DEAD_BALL_IGNORED,
                reason="Stable point ID already has a scored outcome.", source_event_id=event_id)

        prev_state_dict = self.state.to_dict()

        # 2. Point Outcome Resolution
        if self.state.point_state == PointState.POINT_REVIEW_PENDING:
            outcome = PointOutcome(PointOutcomeType.POINT_REVIEW_REQUIRED,
                reason="Confirmed score is paused pending correction of earlier evidence.",
                is_point_ending=True, source_event_id=event_id)
        elif self.state.match_complete or self.state.match_status in ("RETIRED", "SUSPENDED"):
            outcome = PointOutcome(PointOutcomeType.DEAD_BALL_IGNORED,
                reason="Match has ended or is suspended.", source_event_id=event_id)
        elif event_type in ("CONSEQUENTIAL_GAP", "POINT_OUTCOME_UNKNOWN"):
            outcome = PointOutcome(PointOutcomeType.POINT_REVIEW_REQUIRED,
                reason="Consequential point evidence is missing; confirmed score is frozen.",
                is_point_ending=True, source_event_id=event_id)
        else:
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
            self.state.score_confirmed = False
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
            winner = outcome.winner_id
            self.state = TennisScoringRules.award_point(self.state, winner, outcome.reason)
            self.resolver.reset_point(self.state.server_id, serve_attempt=1, initial_ball_state=BallPlayState.DEAD)

        if outcome.is_point_ending and outcome.winner_id is not None and point_id is not None:
            self.completed_point_ids.add(point_id)

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
        self._inputs.append(dict(event_id=event_id, event_type=event_type,
            timestamp_seconds=timestamp_seconds, player_id=player_id,
            line_call=copy.deepcopy(line_call), confidence=confidence, point_id=point_id))

        return self.state, outcome

    def correct_event(self, event_id: int, *, reason: str, expected_revision: int,
                      **replacement: Any) -> MatchState:
        """Replay all dependent events atomically; retain the previous revision's audit log."""
        if expected_revision != self.revision:
            raise ValueError("Score revision changed; refresh before correcting.")
        if not reason.strip():
            raise ValueError("Correction reason is required.")
        index = next((i for i, item in enumerate(self._inputs) if item["event_id"] == event_id), None)
        if index is None:
            raise ValueError("Unknown source event ID.")
        allowed = {"event_type", "timestamp_seconds", "player_id", "line_call", "confidence", "point_id"}
        if set(replacement) - allowed:
            raise ValueError("Unsupported correction field.")
        inputs = copy.deepcopy(self._inputs)
        inputs[index].update(replacement)
        rebuilt = TennisScoringEngine(initial_state=self.initial_state)
        for item in inputs:
            rebuilt.process_match_event(**item)
        self.corrections.append({"revision": self.revision + 1, "event_id": event_id,
                                 "reason": reason, "previous_history": self.history.to_list(),
                                 "source": "human_review"})
        self.state, self.resolver, self.history = rebuilt.state, rebuilt.resolver, rebuilt.history
        self.completed_point_ids, self._inputs = rebuilt.completed_point_ids, rebuilt._inputs
        self.revision += 1
        return self.state

    def get_current_state(self) -> MatchState:
        """Returns the current MatchState."""
        return self.state

    def save_state_json(self, filepath: str) -> None:
        """Saves current match state to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def save_history_json(self, filepath: str) -> None:
        """Saves history log to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as stream:
            json.dump({"schema_version": "2.0", "revision": self.revision,
                       "score_history": self.history.to_list(), "corrections": self.corrections}, stream, indent=2)

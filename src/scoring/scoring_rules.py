from typing import Tuple, List, Optional
import copy

from src.scoring.match_state import (
    MatchState,
    MatchFormat,
    SetFormat,
    ServiceSide,
    PointState,
    BallPlayState,
    CompletedSetScore
)

class TennisScoringRules:
    """
    Pure functional ITF 2026 tennis scoring transition rules.
    Implements deterministic standard games, 7-point tiebreaks, server rotation,
    service side alternation, sets, and match formats.
    """

    @classmethod
    def award_point(cls, state: MatchState, winner_id: int, reason: str = "") -> MatchState:
        """
        Awards a scored point to winner_id (1 or 2), updating points, games,
        tiebreaks, sets, server rotation, and service sides.
        """
        if state.match_complete:
            return state

        st = copy.deepcopy(state)
        st.last_point_winner = winner_id
        st.last_point_reason = reason
        st.serve_attempt = 1
        st.ball_state = BallPlayState.POINT_COMPLETE
        st.point_state = PointState.WAITING_FOR_SERVE

        if st.tie_break_active:
            cls._process_tie_break_point(st, winner_id)
        else:
            cls._process_standard_point(st, winner_id)

        st.state_version += 1
        return st

    @classmethod
    def record_fault(cls, state: MatchState, server_id: int, fault_reason: str = "") -> Tuple[MatchState, bool]:
        """
        Records a service fault.
        - If first serve: transitions serve_attempt 1 -> 2, ball becomes DEAD, NO POINT AWARDED.
        - If second serve: Double Fault -> receiver wins point.
        Returns: (updated_state, is_double_fault)
        """
        st = copy.deepcopy(state)
        
        if st.serve_attempt == 1:
            st.serve_attempt = 2
            st.ball_state = BallPlayState.DEAD
            st.point_state = PointState.SECOND_SERVE_WAITING
            st.last_point_reason = f"First serve fault by Player {server_id}: {fault_reason}"
            st.state_version += 1
            return st, False
        else:
            # Double Fault -> Receiver wins point
            receiver_id = st.player_2_id if server_id == st.player_1_id else st.player_1_id
            updated = cls.award_point(st, receiver_id, f"Double fault by Player {server_id}: {fault_reason}")
            return updated, True

    @classmethod
    def record_service_let(cls, state: MatchState) -> MatchState:
        """
        Records a service let. Same server replays the same service attempt with NO POINT AWARDED.
        """
        st = copy.deepcopy(state)
        st.ball_state = BallPlayState.LET_REPLAY
        st.point_state = PointState.WAITING_FOR_SERVE if st.serve_attempt == 1 else PointState.SECOND_SERVE_WAITING
        st.last_point_reason = "Service let: replay serve"
        st.state_version += 1
        return st

    @classmethod
    def _process_standard_point(cls, st: MatchState, winner_id: int) -> None:
        """Updates points in a standard tennis game according to ITF Rule 5.a."""
        if winner_id == st.player_1_id:
            st.points_p1 += 1
        else:
            st.points_p2 += 1

        p1 = st.points_p1
        p2 = st.points_p2

        # Check Game Win condition: >= 4 points AND lead >= 2 points
        if p1 >= 4 and p1 >= p2 + 2:
            cls._award_game(st, st.player_1_id)
        elif p2 >= 4 and p2 >= p1 + 2:
            cls._award_game(st, st.player_2_id)
        else:
            # Game continues -> update service side (even total points = Deuce, odd = Ad)
            total_pts = p1 + p2
            st.service_side = ServiceSide.DEUCE if (total_pts % 2 == 0) else ServiceSide.AD

    @classmethod
    def _process_tie_break_point(cls, st: MatchState, winner_id: int) -> None:
        """Updates points in a 7-point tie-break game according to ITF Rule 5.b."""
        if winner_id == st.player_1_id:
            st.tie_break_points_p1 += 1
        else:
            st.tie_break_points_p2 += 1

        tb1 = st.tie_break_points_p1
        tb2 = st.tie_break_points_p2
        total_tb = tb1 + tb2

        # Check Tie-Break Win condition: >= 7 points AND lead >= 2 points
        if tb1 >= 7 and tb1 >= tb2 + 2:
            st.games_p1 += 1
            cls._award_set(st, st.player_1_id, tb_p1=tb1, tb_p2=tb2)
        elif tb2 >= 7 and tb2 >= tb1 + 2:
            st.games_p2 += 1
            cls._award_set(st, st.player_2_id, tb_p1=tb1, tb_p2=tb2)
        else:
            # Tie-break continues -> Update server and side rotation (Rule 5.b)
            # Service rotation: 1st point by initial server, then 2 points by opponent, then alternate every 2 points
            # Formula: (total_tb + 1) // 2 is the block index
            block = (total_tb + 1) // 2
            if block % 2 == 1:
                # Opponent of initial tiebreak server
                st.server_id = st.player_2_id if st.initial_server_id == st.player_1_id else st.player_1_id
            else:
                # Initial tiebreak server
                st.server_id = st.initial_server_id

            st.receiver_id = st.player_2_id if st.server_id == st.player_1_id else st.player_1_id
            
            # Service side alternates every point: Point 0 (1st) = Deuce, Point 1 (2nd) = Ad, Point 2 (3rd) = Deuce...
            st.service_side = ServiceSide.DEUCE if (total_tb % 2 == 0) else ServiceSide.AD

    @classmethod
    def _award_game(cls, st: MatchState, game_winner_id: int) -> None:
        """Awards game to game_winner_id and rotates servers."""
        st.points_p1 = 0
        st.points_p2 = 0
        
        if game_winner_id == st.player_1_id:
            st.games_p1 += 1
        else:
            st.games_p2 += 1

        g1 = st.games_p1
        g2 = st.games_p2
        st.current_game += 1

        # Check Set Win condition (ITF Rule 6)
        if st.set_format == SetFormat.TIE_BREAK_SET:
            if g1 >= 6 and g1 >= g2 + 2:
                cls._award_set(st, st.player_1_id)
            elif g2 >= 6 and g2 >= g1 + 2:
                cls._award_set(st, st.player_2_id)
            elif g1 == 6 and g2 == 6:
                # Enter Tie-break
                st.tie_break_active = True
                st.tie_break_points_p1 = 0
                st.tie_break_points_p2 = 0
                st.initial_server_id = st.receiver_id  # Next server in rotation
                st.server_id = st.initial_server_id
                st.receiver_id = st.player_2_id if st.server_id == st.player_1_id else st.player_1_id
                st.service_side = ServiceSide.DEUCE
                return
            else:
                # Standard set continues -> rotate server
                cls._rotate_standard_server(st)
        else:
            # Advantage Set
            if g1 >= 6 and g1 >= g2 + 2:
                cls._award_set(st, st.player_1_id)
            elif g2 >= 6 and g2 >= g1 + 2:
                cls._award_set(st, st.player_2_id)
            else:
                cls._rotate_standard_server(st)

    @classmethod
    def _rotate_standard_server(cls, st: MatchState) -> None:
        """Standard game server rotation (swaps server and receiver)."""
        st.server_id, st.receiver_id = st.receiver_id, st.server_id
        st.service_side = ServiceSide.DEUCE

    @classmethod
    def _award_set(
        cls,
        st: MatchState,
        set_winner_id: int,
        tb_p1: Optional[int] = None,
        tb_p2: Optional[int] = None
    ) -> None:
        """Records completed set and checks match win condition (ITF Rule 7)."""
        set_score = CompletedSetScore(
            set_number=st.current_set,
            games_p1=st.games_p1,
            games_p2=st.games_p2,
            tie_break_p1=tb_p1,
            tie_break_p2=tb_p2,
            winner_id=set_winner_id
        )
        st.completed_sets.append(set_score)

        if set_winner_id == st.player_1_id:
            st.sets_won_p1 += 1
        else:
            st.sets_won_p2 += 1

        required_sets = 2 if st.format == MatchFormat.BEST_OF_3 else 3

        if st.sets_won_p1 >= required_sets:
            st.match_complete = True
            st.winner_id = st.player_1_id
        elif st.sets_won_p2 >= required_sets:
            st.match_complete = True
            st.winner_id = st.player_2_id
        else:
            # Match continues -> Initialize next set
            st.current_set += 1
            st.current_game = 1
            st.games_p1 = 0
            st.games_p2 = 0
            st.points_p1 = 0
            st.points_p2 = 0
            st.tie_break_active = False
            st.tie_break_points_p1 = 0
            st.tie_break_points_p2 = 0
            cls._rotate_standard_server(st)

    @classmethod
    def validate_invariants(cls, st: MatchState) -> List[str]:
        """Validates all mathematical and tennis rule invariants on MatchState."""
        violations = []
        if st.games_p1 < 0 or st.games_p2 < 0:
            violations.append("Negative game score.")
        if st.points_p1 < 0 or st.points_p2 < 0:
            violations.append("Negative point score.")
        if st.sets_won_p1 < 0 or st.sets_won_p2 < 0:
            violations.append("Negative set score.")
        if st.server_id == st.receiver_id:
            violations.append("Server cannot equal receiver.")
        if st.serve_attempt not in (1, 2):
            violations.append("Serve attempt must be 1 or 2.")
        return violations

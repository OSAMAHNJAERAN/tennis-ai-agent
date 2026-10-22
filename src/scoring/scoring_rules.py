from typing import Tuple, List, Optional
import copy

from src.scoring.match_state import (
    MatchState,
    MatchFormat,
    SetFormat,
    ServiceSide,
    PointState,
    BallPlayState,
    CompletedSetScore,
    ServerCourtEnd
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
        if winner_id not in (state.player_1_id, state.player_2_id):
            raise ValueError("Point winner must be a match participant.")
        if state.match_complete or state.match_status in ("RETIRED", "SUSPENDED"):
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
        if server_id != st.server_id:
            raise ValueError("A service fault must identify the current server.")
        if st.match_complete or st.match_status in ("RETIRED", "SUSPENDED"):
            return st, False
        
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
        if st.match_complete or st.match_status in ("RETIRED", "SUSPENDED"):
            return st
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
        target = st.active_tie_break_target()
        if tb1 >= target and tb1 >= tb2 + 2:
            st.games_p1 += 1
            cls._award_set(st, st.player_1_id, tb_p1=tb1, tb_p2=tb2)
        elif tb2 >= target and tb2 >= tb1 + 2:
            st.games_p2 += 1
            cls._award_set(st, st.player_2_id, tb_p1=tb1, tb_p2=tb2)
        else:
            if total_tb % 6 == 0:
                cls._change_ends(st)
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
        if (g1 + g2) % 2 == 1:
            cls._change_ends(st)

        # Check Set Win condition (ITF Rule 6)
        if st.active_set_format() == SetFormat.TIE_BREAK_SET:
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

    @staticmethod
    def _change_ends(st: MatchState) -> None:
        st.player_1_court_end = (ServerCourtEnd.FAR_COURT
                                if st.player_1_court_end == ServerCourtEnd.NEAR_COURT
                                else ServerCourtEnd.NEAR_COURT)

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
        was_tie_break = st.tie_break_active
        if was_tie_break:
            cls._change_ends(st)  # The tie-break is the thirteenth game.

        if set_winner_id == st.player_1_id:
            st.sets_won_p1 += 1
        else:
            st.sets_won_p2 += 1

        required_sets = 2 if st.format == MatchFormat.BEST_OF_3 else 3

        if st.sets_won_p1 >= required_sets:
            st.match_complete = True
            st.winner_id = st.player_1_id if st.initial_state_confirmed and st.score_confirmed else None
            st.match_status = "COMPLETED" if st.winner_id is not None else "PARTIAL"
        elif st.sets_won_p2 >= required_sets:
            st.match_complete = True
            st.winner_id = st.player_2_id if st.initial_state_confirmed and st.score_confirmed else None
            st.match_status = "COMPLETED" if st.winner_id is not None else "PARTIAL"
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
            if was_tie_break:
                # Rule 5(b): first tie-break server receives in the next set,
                # regardless of who served the final tie-break point.
                st.server_id = st.initial_server_id
                st.receiver_id = st.player_2_id if st.server_id == st.player_1_id else st.player_1_id
            cls._rotate_standard_server(st)

    @classmethod
    def validate_invariants(cls, st: MatchState) -> List[str]:
        """Validates all mathematical and tennis rule invariants on MatchState."""
        violations = []
        participants = (st.player_1_id, st.player_2_id)
        if st.player_1_id == st.player_2_id:
            violations.append("Match participants must be distinct.")
        if st.server_id not in participants or st.receiver_id not in participants:
            violations.append("Server and receiver must be match participants.")
        if st.initial_server_id not in participants:
            violations.append("Initial server must be a match participant.")
        if st.tie_break_points_to_win not in (7, 10) or st.deciding_set_tie_break_points_to_win not in (None, 7, 10):
            violations.append("Supported tie-break targets are 7 and 10.")
        if st.tie_break_points_p1 < 0 or st.tie_break_points_p2 < 0:
            violations.append("Negative tie-break score.")
        if st.current_set < 1 or st.current_game < 1:
            violations.append("Set and game numbers must be positive.")
        if st.tie_break_active and (st.games_p1, st.games_p2) != (6, 6) and not st.match_complete:
            violations.append("A set tie-break begins at six games all.")
        if not st.match_complete and st.active_set_format() == SetFormat.TIE_BREAK_SET and max(st.games_p1, st.games_p2) > 6:
            violations.append("A completed tie-break set cannot remain the active set.")
        if not st.match_complete and max(st.points_p1, st.points_p2) >= 4 and abs(st.points_p1 - st.points_p2) >= 2:
            violations.append("A completed standard game cannot remain active.")
        if not st.match_complete and max(st.games_p1, st.games_p2) >= 6 and abs(st.games_p1 - st.games_p2) >= 2:
            violations.append("A completed set cannot remain active.")
        if st.match_complete and st.initial_state_confirmed and st.score_confirmed and st.winner_id not in participants:
            violations.append("Confirmed match completion must identify a participant winner.")
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

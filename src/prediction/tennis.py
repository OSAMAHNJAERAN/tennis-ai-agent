"""Rule-based singles forecasts from past confirmed points only.

Probabilities assume independent points and stationary serve-point probabilities.
The beta posterior mean is a shrinkage estimate, not a calibrated forecast. Infinite
deuce/tie-break/advantage-set tails are solved analytically rather than truncated.
No video, future point, final result or whole-match aggregate is used as a feature.
"""
from __future__ import annotations

import copy
import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from src.scoring.match_state import MatchFormat, MatchState, PointState, SetFormat
from src.scoring.scoring_rules import TennisScoringRules


@dataclass(frozen=True)
class ServePointPrior:
    alpha: float = 2.0
    beta: float = 2.0
    source: str = "explicit_neutral_engineering_prior"

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.alpha, self.beta)):
            raise ValueError("Beta prior parameters must be positive finite values.")


def game_probability(p: float, points_p1: int = 0, points_p2: int = 0) -> float:
    """Probability P1 wins a standard advantage game at constant point probability p."""
    if not 0 <= p <= 1 or not math.isfinite(p):
        raise ValueError("Point probability must be in [0, 1].")
    if min(points_p1, points_p2) < 0:
        raise ValueError("Point scores cannot be negative.")
    deuce = p * p / (p * p + (1 - p) ** 2)

    @lru_cache(None)
    def solve(a, b):
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        if a >= 3 and b >= 3:
            if a == b:
                return deuce
            return p + (1 - p) * deuce if a > b else p * deuce
        return p * solve(a + 1, b) + (1 - p) * solve(a, b + 1)

    return solve(points_p1, points_p2)


def tiebreak_probability(p1_serve: float, p2_serve: float, *, first_server: int = 1,
                         points_p1: int = 0, points_p2: int = 0, target: int = 7) -> float:
    """P1 set tie-break win probability; server codes here are 1 and 2."""
    if target not in (7, 10) or first_server not in (1, 2):
        raise ValueError("Supported tie-break targets are 7/10 with two participants.")
    if min(points_p1, points_p2) < 0:
        raise ValueError("Tie-break scores cannot be negative.")
    if not all(math.isfinite(x) and 0 < x < 1 for x in (p1_serve, p2_serve)):
        raise ValueError("Serve-point probabilities must be strictly between zero and one.")
    both_win = p1_serve * (1 - p2_serve)
    both_lose = (1 - p1_serve) * p2_serve
    tied_tail = both_win / (both_win + both_lose)

    def point_p(total):
        server = first_server if ((total + 1) // 2) % 2 == 0 else 3 - first_server
        return p1_serve if server == 1 else 1 - p2_serve

    @lru_cache(None)
    def solve(a, b):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        p = point_p(a + b)
        if min(a, b) >= target - 1:
            if a == b:
                return tied_tail
            return p + (1 - p) * tied_tail if a > b else p * tied_tail
        return p * solve(a + 1, b) + (1 - p) * solve(a, b + 1)

    return solve(points_p1, points_p2)


class _RuleProbabilities:
    def __init__(self, state: MatchState, p1_serve: float, p2_serve: float):
        self.state = state
        self.p1, self.p2 = p1_serve, p2_serve
        self.game = {1: game_probability(p1_serve), 2: game_probability(1 - p2_serve)}
        self.required_sets = 2 if state.format == MatchFormat.BEST_OF_3 else 3
        self.final_set = 2 * self.required_sets - 1
        # Caches belong to this one snapshot; class-level caches would retain
        # every match state and grow across long videos/multiple sessions.
        self.set_distribution = lru_cache(None)(self.set_distribution)
        self.future_match = lru_cache(None)(self.future_match)

    def set_rules(self, set_number):
        final = set_number == self.final_set
        kind = self.state.deciding_set_format if final and self.state.deciding_set_format else self.state.set_format
        target = (self.state.deciding_set_tie_break_points_to_win
                  if final and self.state.deciding_set_tie_break_points_to_win is not None
                  else self.state.tie_break_points_to_win)
        return kind, target

    def set_distribution(self, a: int, b: int, server: int, set_number: int):
        """Joint probabilities (P1/P2 set winner, next-set server), four entries."""
        if a >= 6 and a - b >= 2:
            return self._terminal(1, server)
        if b >= 6 and b - a >= 2:
            return self._terminal(2, server)
        kind, target = self.set_rules(set_number)
        if kind == SetFormat.TIE_BREAK_SET and a == b == 6:
            q = tiebreak_probability(self.p1, self.p2, first_server=server, target=target)
            return self._blend(q, self._terminal(1, 3 - server), self._terminal(2, 3 - server))
        if kind == SetFormat.ADVANTAGE_SET and min(a, b) >= 5:
            # At game-score deuce, consecutive games have alternating servers.
            w = self.game[1] * self.game[2]
            l = (1 - self.game[1]) * (1 - self.game[2])
            q = w / (w + l)
            if a == b:
                return self._blend(q, self._terminal(1, server), self._terminal(2, server))
            p = self.game[server]
            q = p + (1 - p) * q if a > b else p * q
            return self._blend(q, self._terminal(1, 3 - server), self._terminal(2, 3 - server))
        p = self.game[server]
        return self._blend(p, self.set_distribution(a + 1, b, 3 - server, set_number),
                           self.set_distribution(a, b + 1, 3 - server, set_number))

    @staticmethod
    def _terminal(winner, next_server):
        return tuple(1.0 if (w, s) == (winner, next_server) else 0.0
                     for w in (1, 2) for s in (1, 2))

    @staticmethod
    def _blend(p, win, lose):
        return tuple(p * w + (1 - p) * l for w, l in zip(win, lose))

    def future_match(self, sets1, sets2, server):
        if sets1 >= self.required_sets:
            return 1.0
        if sets2 >= self.required_sets:
            return 0.0
        outcomes = self.set_distribution(0, 0, server, sets1 + sets2 + 1)
        return sum(q * self.future_match(sets1 + (winner == 1), sets2 + (winner == 2), next_server)
                   for q, (winner, next_server) in zip(outcomes, ((1, 1), (1, 2), (2, 1), (2, 2))))

    def current(self):
        st = self.state
        server = 1 if st.server_id == st.player_1_id else 2
        if st.tie_break_active:
            first = 1 if st.initial_server_id == st.player_1_id else 2
            q_game = tiebreak_probability(self.p1, self.p2, first_server=first,
                points_p1=st.tie_break_points_p1, points_p2=st.tie_break_points_p2,
                target=st.active_tie_break_target())
            outcomes = self._blend(q_game, self._terminal(1, 3 - first), self._terminal(2, 3 - first))
        else:
            p = self.p1 if server == 1 else 1 - self.p2
            q_game = game_probability(p, st.points_p1, st.points_p2)
            outcomes = self._blend(q_game,
                self.set_distribution(st.games_p1 + 1, st.games_p2, 3 - server, st.current_set),
                self.set_distribution(st.games_p1, st.games_p2 + 1, 3 - server, st.current_set))
        q_set = outcomes[0] + outcomes[1]
        q_match = sum(q * self.future_match(st.sets_won_p1 + (winner == 1),
                        st.sets_won_p2 + (winner == 2), next_server)
                      for q, (winner, next_server) in zip(outcomes, ((1, 1), (1, 2), (2, 1), (2, 2))))
        return {"game": q_game, "set": q_set, "match": q_match}


class TennisWinProbability:
    """Session-local beta updates; caller must supply confirmed causal point outcomes."""
    def __init__(self, player_ids=(1, 2), prior: ServePointPrior | None = None, *,
                 point_index: MutableMapping[str, tuple[int, int, float]] | None = None):
        if len(player_ids) != 2 or player_ids[0] == player_ids[1]:
            raise ValueError("Two distinct player identifiers are required.")
        if point_index is not None:
            if not isinstance(point_index, MutableMapping):
                raise TypeError("Point index must be a caller-owned mutable mapping.")
            if len(point_index):
                raise ValueError("A fresh forecast requires an empty point index.")
        self.player_ids = tuple(player_ids)
        self.prior = prior or ServePointPrior()
        self.counts = {player: [0, 0] for player in self.player_ids}
        # Native callers own this mapping's storage and lifetime. Keep even an
        # empty supplied mapping; replacing it with {} would retain all IDs in
        # process memory and disconnect duplicate checks from their disk index.
        self.point_ids = {} if point_index is None else point_index
        self.last_timestamp = None

    def observe_point(self, *, point_id: str, server_id: int, winner_id: int, timestamp_seconds: float):
        if server_id not in self.player_ids or winner_id not in self.player_ids:
            raise ValueError("Point server and winner must be match participants.")
        if not point_id or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Stable point ID and finite nonnegative timestamp are required.")
        data = (server_id, winner_id, timestamp_seconds)
        if point_id in self.point_ids:
            if self.point_ids[point_id] != data:
                raise ValueError("A point ID cannot be reused for a different outcome; replay a correction.")
            return
        if self.last_timestamp is not None and timestamp_seconds < self.last_timestamp:
            raise ValueError("Only source-time ordered points can update a causal forecast.")
        self.point_ids[point_id] = data
        self.last_timestamp = timestamp_seconds
        self.counts[server_id][0] += winner_id == server_id
        self.counts[server_id][1] += 1

    def snapshot(self, state: MatchState, *, revision: int = 0, timestamp_seconds: float | None = None):
        if (state.player_1_id, state.player_2_id) != self.player_ids:
            raise ValueError("Forecast session does not belong to the match participants.")
        if timestamp_seconds is not None and (not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("Forecast time must be finite and nonnegative.")
        if timestamp_seconds is not None and self.last_timestamp is not None and timestamp_seconds < self.last_timestamp:
            raise ValueError("Forecast would include points from its future; replay the earlier prefix.")
        served = []
        for player in self.player_ids:
            wins, total = self.counts[player]
            a, b = self.prior.alpha + wins, self.prior.beta + total - wins
            served.append({"player_id": player, "probability": a / (a + b), "points_won": wins,
                           "points_played": total, "alpha": a, "beta": b})
        result = {"schema_version": "1.0", "match_id": state.match_id, "revision": revision,
                  "score_state_version": state.state_version, "timestamp_seconds": timestamp_seconds,
                  "source": "statistical_tennis_rules", "calibration_status": "not_calibrated",
                  "sample_count": len(self.point_ids), "serve_estimates": served,
                  "prior": {"alpha": self.prior.alpha, "beta": self.prior.beta, "source": self.prior.source},
                  "assumptions": "Independent stationary serve-point probabilities; beta posterior mean plug-in.",
                  "probabilities": None, "winner_id": None}
        if not state.initial_state_confirmed or not state.score_confirmed or state.point_state == PointState.POINT_REVIEW_PENDING:
            return {**result, "status": "unavailable", "reason": "Initial score is unconfirmed or a consequential point requires review."}
        if state.match_status in ("RETIRED", "SUSPENDED"):
            return {**result, "status": "unavailable", "reason": "A rules-completion forecast is unavailable for a retired or suspended match."}
        violations = TennisScoringRules.validate_invariants(state)
        if violations:
            return {**result, "status": "unavailable", "reason": "Invalid score state: " + " ".join(violations)}
        if state.match_complete:
            return {**result, "status": "confirmed_result", "winner_id": state.winner_id,
                    "reason": "Match completion is confirmed; display the actual result instead of a forecast."}
        probabilities = _RuleProbabilities(state, served[0]["probability"], served[1]["probability"]).current()
        result["probabilities"] = {horizon: {"player_1": q, "player_2": 1 - q}
                                   for horizon, q in probabilities.items()}
        return {**result, "status": "prior_only" if not self.point_ids else "uncalibrated",
                "reason": ("Based on an explicit neutral prior and the score; no player performance points yet."
                           if not self.point_ids else "Statistical estimate from previous confirmed points; independent calibration is pending.")}


def replay_predictions(initial_state: MatchState, points: Iterable[dict[str, Any]], *, revision: int = 0,
                       prior: ServePointPrior | None = None) -> dict[str, Any]:
    """Rebuild every causal prefix after correction. Unknown points freeze score and forecasts.

    Rows: winner (participant ID or None), optional point_id, timestamp_seconds,
    status=confirmed/provisional/unknown and provenance. Legacy manual rows without
    a timestamp use their ordered point index and explicitly label that time base.
    """
    state = copy.deepcopy(initial_state)
    engine = TennisWinProbability((state.player_1_id, state.player_2_id), prior)
    history = []
    current = engine.snapshot(state, revision=revision)
    blocked = False
    seen = {}
    for index, row in enumerate(points):
        point_id = str(row.get("point_id") or f"legacy-point-{index}")
        if point_id in seen:
            if row != seen[point_id]:
                raise ValueError("Duplicate point ID with conflicting data.")
            continue
        seen[point_id] = copy.deepcopy(row)
        timestamp = row.get("timestamp_seconds", index + 1)
        if state.match_complete:
            raise ValueError("Points remain after rules-complete match; review the correction.")
        if row.get("winner") is None or row.get("status", "confirmed") != "confirmed":
            blocked = True
            state.score_confirmed = False
            state.point_state = PointState.POINT_REVIEW_PENDING
        if not blocked:
            engine.observe_point(point_id=point_id, server_id=state.server_id,
                                 winner_id=row["winner"], timestamp_seconds=timestamp)
            state = TennisScoringRules.award_point(state, row["winner"], "Confirmed point ledger outcome")
        current = engine.snapshot(state, revision=revision, timestamp_seconds=timestamp)
        history.append({**current, "point_id": point_id,
                        "time_base": "source_seconds" if "timestamp_seconds" in row else "point_sequence",
                        "provenance": row.get("provenance", "manual")})
    return {"current": current, "history": history, "state": state.to_dict()}

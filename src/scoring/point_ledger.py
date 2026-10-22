"""Stable, versioned confirmed point ledger for video review and corrections."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from src.prediction.tennis import replay_predictions
from src.scoring.match_state import MatchState


@dataclass(frozen=True)
class PointRecord:
    point_id: str
    timestamp_seconds: float
    winner: int | None = None
    status: str = "unknown"
    provenance: str = "automatic"
    evidence_ids: tuple[str, ...] = ()
    loss_reason: str | None = None

    def __post_init__(self):
        if not self.point_id or not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("Point ID and finite nonnegative source timestamp are required.")
        if self.status not in ("confirmed", "provisional", "unknown"):
            raise ValueError("Unsupported point status.")
        if self.provenance not in ("automatic", "manual", "human_review"):
            raise ValueError("Unsupported point provenance.")
        if self.status == "confirmed" and self.winner is None:
            raise ValueError("A confirmed point must have a winner.")


class PointLedger:
    def __init__(self, initial_state: MatchState | None = None):
        self.initial_state = copy.deepcopy(initial_state) if initial_state is not None else MatchState(
            initial_state_confirmed=False, score_confirmed=False, score_source="unknown_clip_initial_state")
        self.points: list[PointRecord] = []
        self.revision = 0
        self.audit: list[dict[str, Any]] = []

    def append(self, point: PointRecord) -> bool:
        existing = next((p for p in self.points if p.point_id == point.point_id), None)
        if existing is not None:
            if existing != point:
                raise ValueError("A stable point ID cannot have conflicting outcomes; use correction.")
            return False
        if self.points and point.timestamp_seconds < self.points[-1].timestamp_seconds:
            raise ValueError("Point ledger must remain in source-time order.")
        self._validate_player(point.winner)
        candidate = [*self.points, point]
        self._replay(candidate)
        self.points = candidate
        self.revision += 1
        self.audit.append({"revision": self.revision, "action": "append", "point": asdict(point)})
        return True

    def correct(self, point_id: str, *, winner: int | None, reason: str,
                expected_revision: int, evidence_ids: tuple[str, ...] = (),
                loss_reason: str | None = None) -> dict[str, Any]:
        if expected_revision != self.revision:
            raise ValueError("Score revision changed; refresh before correcting.")
        if not reason.strip():
            raise ValueError("A correction reason is required.")
        self._validate_player(winner)
        index = next((i for i, p in enumerate(self.points) if p.point_id == point_id), None)
        if index is None:
            raise ValueError("Unknown stable point ID.")
        before = self.points[index]
        after = replace(before, winner=winner, status="confirmed" if winner is not None else "unknown",
                        provenance="human_review", evidence_ids=evidence_ids, loss_reason=loss_reason)
        candidate = list(self.points)
        candidate[index] = after
        rebuilt = self._replay(candidate, revision=self.revision + 1)
        self.points = candidate
        self.revision += 1
        self.audit.append({"revision": self.revision, "action": "correct", "point_id": point_id,
                           "before": asdict(before), "after": asdict(after), "reason": reason})
        return rebuilt

    def confirm_initial_state(self, state: MatchState, *, source: str, reason: str,
                              expected_revision: int) -> dict[str, Any]:
        """Synchronize a mid-match clip to independently supplied score/server evidence."""
        if expected_revision != self.revision:
            raise ValueError("Score revision changed; refresh before synchronizing.")
        if not source.strip() or not reason.strip():
            raise ValueError("Initial-state synchronization requires source and reason.")
        state = copy.deepcopy(state)
        state.initial_state_confirmed = state.score_confirmed = True
        state.score_source = source
        rebuilt = replay_predictions(state, (asdict(p) for p in self.points), revision=self.revision + 1)
        before = self.initial_state.to_dict()
        self.initial_state = state
        self.revision += 1
        self.audit.append({"revision": self.revision, "action": "initial_state_synchronization",
                           "before": before, "after": state.to_dict(), "source": source, "reason": reason})
        return rebuilt

    def snapshot(self) -> dict[str, Any]:
        return {**self._replay(self.points), "revision": self.revision,
                "points": [asdict(p) for p in self.points], "audit": copy.deepcopy(self.audit)}

    def _replay(self, points, revision=None):
        return replay_predictions(self.initial_state, (asdict(p) for p in points),
                                  revision=self.revision if revision is None else revision)

    def _validate_player(self, player):
        if player is not None and player not in (self.initial_state.player_1_id, self.initial_state.player_2_id):
            raise ValueError("Point winner must be a match participant.")

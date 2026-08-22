from dataclasses import dataclass, asdict
from typing import List, Set, Dict, Any, Optional
import json

@dataclass
class ScoringEventLogEntry:
    sequence_number: int
    event_id: int
    timestamp_seconds: float
    event_type: str
    point_outcome: str
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    reason: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ScoreEventHistory:
    """
    Event-sourced score history log.
    Ensures idempotency, auditability, and replay capability.
    """

    def __init__(self):
        self.entries: List[ScoringEventLogEntry] = []
        self.processed_event_ids: Set[int] = set()

    def is_processed(self, event_id: int) -> bool:
        """Returns True if the event_id has already been processed."""
        return event_id in self.processed_event_ids

    def append(
        self,
        event_id: int,
        timestamp_seconds: float,
        event_type: str,
        point_outcome: str,
        previous_state: Dict[str, Any],
        new_state: Dict[str, Any],
        reason: str,
        confidence: float = 1.0
    ) -> ScoringEventLogEntry:
        """Appends a new verified transition to the event log."""
        seq = len(self.entries) + 1
        entry = ScoringEventLogEntry(
            sequence_number=seq,
            event_id=event_id,
            timestamp_seconds=timestamp_seconds,
            event_type=event_type,
            point_outcome=point_outcome,
            previous_state=previous_state,
            new_state=new_state,
            reason=reason,
            confidence=confidence
        )
        self.entries.append(entry)
        self.processed_event_ids.add(event_id)
        return entry

    def to_list(self) -> List[Dict[str, Any]]:
        """Exports all entries as a list of dicts."""
        return [entry.to_dict() for entry in self.entries]

    def save_json(self, filepath: str) -> None:
        """Saves event history log to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_list(), f, indent=2)

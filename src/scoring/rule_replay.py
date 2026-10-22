"""Atomic lossless rule-input evidence for unchanged deterministic replay.

Recorded line calls may reproduce their original interpretation. A correction
can change later serve/rally context and must recompute or explicitly review
those calls; this file is not permission to reuse them after such a change.
"""
import hashlib
import json
import re

from src.utils.json_array_artifact import JsonArrayArtifact


RULE_REPLAY_NAME = 'native_rule_replay.json'


def _plain(value):
    def numeric(item):
        if type(item).__module__.split('.')[0] == 'numpy':
            if hasattr(item, 'tolist'):
                return item.tolist()
            if hasattr(item, 'item'):
                return item.item()
        raise TypeError('Rule replay evidence requires finite JSON values')
    return json.loads(json.dumps(value, allow_nan=False, default=numeric))


class _Frames(list):
    """The stdlib encoder streams this repeatable frame view without a copy."""
    def __init__(self, frames, analysis_id):
        self.frames, self.analysis_id = frames, analysis_id

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for index in range(len(self.frames)):
            interval = _plain(self.frames[index][0])
            if not isinstance(interval, dict) or interval.get('analysis_id') != self.analysis_id:
                raise ValueError('Replay source intervals belong to a different analysis')
            if 'frame_index' in interval and interval['frame_index'] != index:
                raise ValueError('Replay source intervals are out of frame order')
            identity = _plain(self.frames[index].get(1, {}))
            if not isinstance(identity, dict):
                raise ValueError('Replay identity observations require an object')
            yield {'frame_index': index, 'interval': interval, 'identity': identity}


def timeline_record(record):
    """Memberships have their own ordered table; do not copy nested ID views."""
    return _plain({key: value for key, value in record.items()
                   if key not in ('shot_ids', 'shot_membership_scope')})


class _Rows(list):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        iterator = iter(self.rows)
        try:
            for row in iterator:
                yield timeline_record(row)
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()


def write_rule_replay(destination, engine, owner, source_frames, *, authoritative_events_enabled=True):
    """Seal raw initial state, all source intervals and accepted inputs together.

    This uses the existing owned ledger to avoid another retained event list.
    All rows must match the accepted scoring journal and point transitions.
    The atomic writer preserves an earlier destination on any failed row.
    """
    from src.scoring.match_state_codec import encode_match_state
    owner._open()
    if (not owner.sealed or engine.revision != 0 or source_frames is None
            or type(authoritative_events_enabled) is not bool
            or re.fullmatch('[0-9a-f]{64}', owner.source_fingerprint) is None):
        raise ValueError('Replay export requires a sealed original source-bound timeline')
    initial = encode_match_state(engine.initial_state)
    if engine.initial_state.match_id != owner.analysis_id:
        raise ValueError('Replay initial state belongs to a different analysis')
    records, transitions = owner.rows('replay_inputs'), owner.rows('transitions')
    if not (len(records) == len(transitions) == len(engine._inputs) == len(engine.history.entries)):
        raise ValueError('Replay input, scoring and point transition counts differ')
    if not authoritative_events_enabled and len(records):
        raise ValueError('An authority-off replay cannot contain accepted rule events')
    shots = owner.rows('replay_shots')
    if not authoritative_events_enabled and len(shots):
        raise ValueError('An authority-off replay cannot contain accepted shots')
    if len(shots) != len(owner.member_rows()):
        raise ValueError('Replay shots and accepted memberships differ')
    metadata = {'schema_version': '1.0', 'analysis_id': owner.analysis_id,
                'analysis_revision': owner.revision, 'scoring_revision': engine.revision,
                'source_sha256': owner.source_fingerprint, 'initial_state': initial,
                'final_state': encode_match_state(engine.get_current_state()),
                'authoritative_events_enabled': authoritative_events_enabled,
                'source_frame_count': len(source_frames), 'input_count': len(records),
                'shot_count': len(shots), 'shots': _Rows(shots),
                'expected_timeline': {kind: _Rows(owner.rows(kind))
                    for kind in ('points', 'rallies', 'attempts')},
                'expected_memberships': _Rows(owner.member_rows()),
                'replay_scope': 'UNCHANGED_RULE_REPRODUCTION_ONLY',
                'correction_policy': 'RECOMPUTE_OR_EXPLICITLY_REVIEW_CONTEXT_DEPENDENT_CALLS_AND_ALL_DERIVED_OUTPUTS',
                'independent_accuracy_validated': False,
                'source_frames': _Frames(source_frames, owner.analysis_id)}
    inputs, histories = iter(engine.iter_rule_inputs()), iter(engine.history.entries)
    rows, accepted = iter(records), iter(transitions)
    closed = False
    def close_readers(primary=None):
        nonlocal closed
        closed = True
        failures = []
        for iterator in (rows, accepted, inputs, histories):
            close = getattr(iterator, 'close', None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    failures.append(error)
        if primary is not None:
            for failure in failures:
                primary.add_note('Replay reader cleanup also failed: ' + str(failure))
        elif failures:
            raise failures[0]
    primary = None
    try:
        with JsonArrayArtifact(destination, metadata, 'inputs', allow_nan=False) as output:
            last_state = engine.initial_state.to_dict()
            for ordinal, (record, transition, rule_input, history) in enumerate(
                    zip(rows, accepted, inputs, histories, strict=True), 1):
                original = record['source_input']
                fingerprint = hashlib.sha256(json.dumps(original, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                event_id = original['event_id']
                if (fingerprint != record['input_fingerprint'] or transition != record['accepted_transition']
                        or transition['input_fingerprint'] != fingerprint or transition['event_id'] != event_id
                        or rule_input != record['scoring_input'] or rule_input['event_id'] != event_id
                        or history.event_id != event_id or history.previous_state != transition['pre_state']
                        or history.new_state != transition['post_state']
                        or history.point_outcome != transition['outcome_type']
                        or history.sequence_number != ordinal
                        or history.timestamp_seconds != rule_input['timestamp_seconds']
                        or history.event_type != rule_input['event_type']
                        or history.confidence != rule_input['confidence']
                        or history.reason != transition['outcome']['reason']
                        or rule_input['timestamp_seconds'] != original['timestamp_s']
                        or rule_input['event_type'] != transition['rule_event_type']
                        or rule_input['player_id'] != transition['rule_player_id']
                        or history.previous_state != last_state):
                    raise ValueError('Replay evidence differs from accepted source/rule/point history')
                output.append(record)
                last_state = history.new_state
            if engine.get_current_state().to_dict() != last_state:
                raise ValueError('Replay final state differs from its accepted history')
            # Reader teardown is part of success, before atomic publication.
            close_readers()
    except BaseException as error:
        primary = error
        raise
    finally:
        if not closed:
            close_readers(primary)

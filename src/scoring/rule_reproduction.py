"""Verify unchanged native rule/timeline reproduction in private owned storage.

This accepts an externally sealed artifact digest and original identity only.
Recorded line decisions cannot be supplied to a corrected analysis through this
interface. Verification finishes before the caller can access the rebuilt view.
No result, report or video is published here.
"""
from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import fields
import hashlib
from types import SimpleNamespace

from src.scoring.match_state_codec import encode_match_state
from src.scoring.rule_replay import timeline_record


# Keep the existing private names available to callers of the unchanged
# reproducer while both entry points share one strict source decoder.
from src.scoring.rule_replay_source import (
    open_rule_replay_source, _SHA, _FIELDS, _POLICY, _require, _plain,
    _canonical, _same, _array, _hash, _stat, _SourceFrames, _source_event,
)


class _RecordedCall:
    """A single-use decision, reachable only in a verified original context."""
    def __init__(self, evidence):
        self.evidence, self.used = evidence, False

    def evaluate_bounce(self, **request):
        value = self.evidence
        _require(not self.used and value is not None
                 and type(value.event_id) is int and value.event_id == request['event_id']
                 and type(value.bounce_frame) is int and value.bounce_frame == request['bounce_frame']
                 and value.decision_context == request['context'],
                 'recorded line decision differs from the original request')
        self.used = True
        return value


class _Shots(Sequence):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from src.shot_analysis.shot_types import (ShotEventEvidence, ShotType, ShotDirection,
            ShotClassificationSource, ShotOutcome, CourtZone3x3)
        record = _plain(self.rows[index])
        _require(isinstance(record, dict)
                 and set(record) == {field.name for field in fields(ShotEventEvidence)},
                 'raw shot fields are invalid')
        for key, kind in {'shot_type': ShotType, 'direction': ShotDirection,
                          'classification_source': ShotClassificationSource,
                          'outcome': ShotOutcome, 'landing_zone': CourtZone3x3}.items():
            if key != 'landing_zone' or record[key] is not None:
                record[key] = kind(record[key])
        return ShotEventEvidence(**record)

    def __setitem__(self, index, shot):
        self.rows[index] = shot.to_dict()


def _compare_rows(actual, expected):
    _array(expected, 'expected timeline records must be arrays')
    _require(len(actual) == len(expected), 'rebuilt timeline record counts differ')
    iterator = iter(actual)
    try:
        for row, saved in zip(iterator, expected, strict=True):
            _same(timeline_record(row), saved, 'rebuilt timeline or membership differs')
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()


@contextmanager
def reproduce_rule_timeline(path, directory, *, artifact_sha256, source_sha256,
                            analysis_id, analysis_revision):
    """Yield a verified unchanged reconstruction until this context exits.

    ``artifact_sha256`` must come from the caller's trusted checkpoint/manifest.
    Computing it from an untrusted supplied file would not establish integrity.
    The caller-owned scratch directory must already exist. Failed parsed data
    retain ResultStorage diagnostics; no published artifact is modified.
    """
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.scoring.point_timeline import PointTimeline, process_timeline_event
    from src.utils.scoring_journal import ScoringJournal
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    with ExitStack() as stack:
        source = stack.enter_context(open_rule_replay_source(path, directory,
            artifact_sha256=artifact_sha256, source_sha256=source_sha256,
            analysis_id=analysis_id, analysis_revision=analysis_revision))
        storage, document = source.storage, source.document
        initial, final, source_frames = source.initial_state, source.final_state, source.source_frames
        journal = stack.enter_context(ScoringJournal(directory))
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        owner = stack.enter_context(PointAnalysisSequence(directory, analysis_id, analysis_revision,
                                                         source_fingerprint=source_sha256))
        timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
        previous = None
        for value in document['inputs']:
            value = _plain(value)
            _canonical(value)
            _require(isinstance(value, dict) and set(value) == {'source_input', 'input_fingerprint',
                     'scoring_input', 'accepted_transition', 'rule_environment'}, 'input record fields differ')
            original = value['source_input']
            _require(hashlib.sha256(_canonical(original).encode()).hexdigest() == value['input_fingerprint'],
                     'source input fingerprint differs')
            event, context = source.decode_source_event(original)
            order = (event.frame_index, event.timestamp_s, event.event_id)
            _require(previous is None or order > previous, 'inputs are duplicated or out of source order')
            previous = order
            pre = engine.get_current_state().to_dict()
            _same(pre, value['accepted_transition']['pre_state'], 'original pre-state differs')
            translated = timeline.scoring_input(event, context, pre)
            rule = value['scoring_input']
            _same(list(translated), [rule['event_type'], rule['player_id']], 'original rule interpretation differs')
            decoded = engine._decode_input(rule)
            environment = value['rule_environment']
            _require(isinstance(environment, dict) and set(environment) == {'geometry_available'}
                     and type(environment['geometry_available']) is bool, 'rule environment differs')
            process_timeline_event(event, context, engine, timeline, _RecordedCall(decoded['line_call']),
                ball_points=(), homography_matrix=None, frames=(),
                geometry_available=environment['geometry_available'])
            accepted = owner.event(event.event_id)
            _same(owner.get('replay_inputs', accepted['uid']), value,
                  'accepted source/rule/point transition differs')
        _same(encode_match_state(engine.get_current_state()), encode_match_state(final), 'raw final state differs')
        rows = storage.series()
        for position, value in enumerate(document['shots']):
            value = _plain(value)
            _require(isinstance(value, dict) and set(value) == {'position', 'raw', 'linked'}
                     and type(value['position']) is int and value['position'] == position,
                     'raw shots are out of order')
            rows.append(value['raw'])
        shots = _Shots(rows)
        timeline.attach_shots(shots)
        timeline.finish()
        _compare_rows(owner.rows('replay_shots'), document['shots'])
        _require(isinstance(document['expected_timeline'], dict)
                 and set(document['expected_timeline']) == {'points', 'rallies', 'attempts'},
                 'expected timeline fields differ')
        for kind in ('points', 'rallies', 'attempts'):
            _compare_rows(owner.rows(kind), document['expected_timeline'][kind])
        _compare_rows(owner.member_rows(), document['expected_memberships'])
        _require(len(owner.rows('transitions')) == document['input_count'], 'accepted input counts differ')
        yield SimpleNamespace(engine=engine, timeline=timeline, owner=owner, shots=shots,
                              source_frames=source_frames,
                              authoritative_events_enabled=document['authoritative_events_enabled'])

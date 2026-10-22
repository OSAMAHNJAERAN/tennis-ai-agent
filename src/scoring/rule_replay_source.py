"""Read sealed original rule sources without reusing recorded rule decisions.

The caller supplies a trusted external artifact digest and source identity.
Large arrays remain in owned disk storage until this context closes. This
decoder checks source bindings; it does not qualify evidence or establish that
the recorded transitions are correct. Corrected processing must recompute or
explicitly review context-dependent decisions in the resulting rule context.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from src.api.result_storage import DiskSeries, ResultStorage
from src.scoring.match_state_codec import decode_match_state


_SHA = re.compile('[0-9a-f]{64}')
_FIELDS = {'schema_version', 'analysis_id', 'analysis_revision', 'scoring_revision',
           'source_sha256', 'initial_state', 'final_state', 'authoritative_events_enabled',
           'source_frame_count', 'input_count', 'shot_count', 'shots', 'expected_timeline',
           'expected_memberships', 'replay_scope', 'correction_policy',
           'independent_accuracy_validated', 'source_frames', 'inputs'}
_POLICY = 'RECOMPUTE_OR_EXPLICITLY_REVIEW_CONTEXT_DEPENDENT_CALLS_AND_ALL_DERIVED_OUTPUTS'


def _require(condition, message):
    if not condition:
        raise ValueError('Rule reproduction: ' + message)


def _plain(value):
    # Materialize only the current record, never the root arrays.
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, DiskSeries)):
        return [_plain(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _same(actual, expected, message):
    _require(_canonical(actual) == _canonical(expected), message)


def _array(value, message):
    _require(isinstance(value, (list, DiskSeries)), message)
    return value


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _stat(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _optional_integer(value):
    return value is None or type(value) is int and value >= 0


def _context_primitives(context):
    integers = {'frame_index'}
    optional_integers = {'source_frame_index'}
    optional_text = {'source_pts', 'source_time_base', 'segment_id', 'review_evidence_ref'}
    booleans = {'analysis_allowed', 'binding_audit_matches'}
    _require(set(context) == integers | optional_integers | optional_text | booleans | {'binding'},
             'unsupported context primitive fields')
    _require(all(type(context[name]) is int and context[name] >= 0 for name in integers)
             and all(_optional_integer(context[name]) for name in optional_integers)
             and all(context[name] is None or type(context[name]) is str for name in optional_text)
             and all(type(context[name]) is bool for name in booleans),
             'context primitive types are invalid')


def _binding_primitives(binding):
    integers = {'frame_index', 'role_id', 'participant_id', 'identity_revision', 'anchor_frame_index'}
    optional_integers = {'source_frame_index', 'identity_epoch'}
    optional_text = {'source_pts', 'source_time_base'}
    text = {'segment_id', 'match_identity', 'track_key', 'court_end', 'calibration_revision',
            'evidence_ref', 'analysis_id', 'source_fingerprint', 'participant_mapping_ref',
            'geometry_qualification_ref'}
    _require(set(binding) == integers | optional_integers | optional_text | text,
             'unsupported binding primitive fields')
    _require(all(type(binding[name]) is int and binding[name] >= 0 for name in integers)
             and all(_optional_integer(binding[name]) for name in optional_integers)
             and all(binding[name] is None or type(binding[name]) is str for name in optional_text)
             and all(type(binding[name]) is str for name in text),
             'binding primitive types are invalid')


def _identity_primitives(identity):
    for name in ('epoch', 'revision', 'frame_index'):
        if name in identity:
            _require(type(identity[name]) is int and identity[name] >= 0,
                     'source identity counters require exact nonnegative integers')
    if 'observations' not in identity:
        return
    observations = identity['observations']
    _require(type(observations) is list, 'source identity observations must be an array')
    for observation in observations:
        _require(type(observation) is dict, 'source identity observation must be an object')
        for name in ('player_id', 'track_key'):
            if name in observation:
                _require(observation[name] is None or type(observation[name]) is str,
                         'source identity observation primitives are invalid')
        if 'state' in observation:
            _require(type(observation['state']) is str,
                     'source identity observation primitives are invalid')
        if observation.get('state') == 'CONFIRMED':
            _require(observation.get('player_id') in ('A', 'B'),
                     'confirmed source identity requires player A or B')


class _SourceFrames(Sequence):
    def __init__(self, rows, analysis_id, *, analysis_revision=None,
                 source_sha256=None, artifact_sha256=None):
        self.rows, self.analysis_id = rows, analysis_id
        self._analysis_revision = analysis_revision
        self._source_sha256, self._artifact_sha256 = source_sha256, artifact_sha256

    @property
    def analysis_revision(self):
        return self._analysis_revision

    @property
    def source_sha256(self):
        return self._source_sha256

    @property
    def artifact_sha256(self):
        return self._artifact_sha256

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Replay source frames require an indexed read')
        if type(index) is not int:
            raise TypeError('Replay source frame index must be an integer')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = _plain(self.rows[index])
        _canonical(row)
        _require(isinstance(row, dict) and set(row) == {'frame_index', 'interval', 'identity'}
                 and type(row['frame_index']) is int and row['frame_index'] == index,
                 'source frames are omitted or out of order')
        interval, identity = row['interval'], row['identity']
        _require(isinstance(interval, dict) and interval.get('analysis_id') == self.analysis_id
                 and isinstance(identity, dict)
                 and ('analysis_id' not in identity or identity['analysis_id'] == self.analysis_id)
                 and ('frame_index' not in interval or
                      type(interval['frame_index']) is int and interval['frame_index'] == index),
                 'source interval or identity belongs to another analysis/frame')
        _identity_primitives(identity)
        return {0: interval, 1: identity}


def _source_event(record, source_frames, analysis_id, source_sha256):
    from src.scoring.point_timeline import (EventContext, ReviewedParticipantBinding,
        ReviewedRuleSignal, candidate_event_context, context_record)
    expected = {'event_id', 'frame_index', 'timestamp_s', 'player_id', 'confidence',
                'ball_position_px', 'court_position_m', 'trajectory_state', 'evidence',
                'event_type', 'reviewed_evidence_ref', 'context'}
    _require(isinstance(record, dict) and set(record) == expected,
             'original source input fields are invalid')
    _require(type(record['event_id']) is int and record['event_id'] >= 0
             and type(record['frame_index']) is int and 0 <= record['frame_index'] < len(source_frames),
             'source event identity/frame is invalid')
    _require(record['player_id'] is None or type(record['player_id']) is int
             and record['player_id'] in (1, 2), 'physical player role type is invalid')
    _require(type(record['context']) is dict
             and 'anchor_frame_index' in record['context'], 'context fields are invalid')
    context = dict(record['context'])
    anchor = context.pop('anchor_frame_index')
    _require(_optional_integer(anchor), 'context anchor type is invalid')
    _require(set(context) == {field.name for field in fields(EventContext)}, 'context fields are invalid')
    _context_primitives(context)
    binding = context['binding']
    if binding is not None:
        _require(type(binding) is dict
                 and set(binding) == {field.name for field in fields(ReviewedParticipantBinding)},
                 'binding fields are invalid')
        _binding_primitives(binding)
        context['binding'] = ReviewedParticipantBinding(**binding)
    context = EventContext(**context)
    _same(anchor, context.binding.anchor_frame_index if context.binding else None,
          'source anchor differs from its reviewed binding')
    if record['reviewed_evidence_ref'] is not None:
        event = ReviewedRuleSignal(record['event_id'], record['frame_index'], record['timestamp_s'],
            record['event_type'], record['reviewed_evidence_ref'], context)
    else:
        _require(record['event_type'] in ('SERVE_CONTACT', 'PLAYER_1_HIT', 'PLAYER_2_HIT',
                                          'BOUNCE', 'UNKNOWN_EVENT'), 'unsupported physical event type')
        event = SimpleNamespace(**{key: value for key, value in record.items()
                                   if key not in ('context', 'reviewed_evidence_ref', 'event_type')},
                                event_type=SimpleNamespace(value=record['event_type']))
    current = candidate_event_context(event, source_frames, lambda *args: context.binding,
        analysis_id=analysis_id, source_fingerprint=source_sha256)
    _same(context_record(current), record['context'], 'stored context fails current source/identity binding')
    return event, current


class RuleReplaySource:
    """Validated source data whose disk-backed arrays live inside the context.

    ``document`` retains original records as data for audit. No recorded line
    call evaluator is provided. ``decode_source_event`` also accepts a detached
    replacement record, rechecking its bindings against these same source frames.
    """

    def __init__(self, storage, document, initial_state, final_state, source_frames,
                 analysis_id, source_sha256):
        self.storage, self.document = storage, document
        self.initial_state, self.final_state = initial_state, final_state
        self.source_frames = source_frames
        self.authoritative_events_enabled = document['authoritative_events_enabled']
        self._analysis_id, self._source_sha256 = analysis_id, source_sha256

    def decode_source_event(self, record):
        """Return the source event and its currently verified source context."""
        record = _plain(record)
        _canonical(record)
        return _source_event(record, self.source_frames, self._analysis_id, self._source_sha256)


@contextmanager
def open_rule_replay_source(path, directory, *, artifact_sha256, source_sha256,
                            analysis_id, analysis_revision):
    """Yield a bounded decoder of an externally sealed original replay.

    The caller-owned scratch directory must exist. ``artifact_sha256`` must
    come from a trusted checkpoint or manifest, rather than the supplied file.
    Parsing is bracketed by hash and stat checks. A failed context retains only
    its owned storage diagnostics; successful unpublished storage is removed.
    Original transition interpretation remains the unchanged reproducer's job.
    """
    _require(isinstance(artifact_sha256, str) and _SHA.fullmatch(artifact_sha256)
             and isinstance(source_sha256, str) and _SHA.fullmatch(source_sha256)
             and isinstance(analysis_id, str) and bool(analysis_id)
             and type(analysis_revision) is int and analysis_revision >= 0, 'original identity is invalid')
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), 'artifact must be a regular owned file')
    before = _stat(path)
    _require(_hash(path) == artifact_sha256 and _stat(path) == before, 'sealed artifact digest differs')
    with ResultStorage(directory) as storage:
        def reject_constant(value):
            raise ValueError('Rule reproduction requires finite JSON: ' + value)
        document = storage.read_json(path, parse_constant=reject_constant)
        _require(_stat(path) == before and _hash(path) == artifact_sha256 and _stat(path) == before,
                 'sealed artifact changed during reading')
        _require(isinstance(document, dict) and set(document) == _FIELDS,
                 'unsupported replay envelope fields')
        _require(document['schema_version'] == '1.0' and document['analysis_id'] == analysis_id
                 and type(document['analysis_revision']) is int and document['analysis_revision'] == analysis_revision
                 and type(document['scoring_revision']) is int and document['scoring_revision'] == 0
                 and document['source_sha256'] == source_sha256
                 and document['replay_scope'] == 'UNCHANGED_RULE_REPRODUCTION_ONLY'
                 and document['correction_policy'] == _POLICY
                 and document['independent_accuracy_validated'] is False
                 and type(document['authoritative_events_enabled']) is bool,
                 'artifact is not the sealed original analysis')
        for count, rows in (('source_frame_count', 'source_frames'), ('input_count', 'inputs'), ('shot_count', 'shots')):
            _array(document[rows], rows + ' must be an array')
            _require(type(document[count]) is int and document[count] >= 0
                     and len(document[rows]) == document[count], rows + ' count differs')
        if not document['authoritative_events_enabled']:
            _require(document['input_count'] == document['shot_count'] == 0,
                     'authority-off artifacts cannot contain accepted inputs/shots')
        initial = decode_match_state(_plain(document['initial_state']))
        final = decode_match_state(_plain(document['final_state']))
        _require(initial.match_id == final.match_id == analysis_id, 'raw state analysis identity differs')
        source_frames = _SourceFrames(document['source_frames'], analysis_id,
            analysis_revision=analysis_revision, source_sha256=source_sha256,
            artifact_sha256=artifact_sha256)
        for index in range(len(source_frames)):
            source_frames[index]  # Verify unused and intervening frames too.
        yield RuleReplaySource(storage, document, initial, final, source_frames,
                               analysis_id, source_sha256)

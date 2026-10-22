"""Private corrected rule processing from one committed native generation.

Every bounce is evaluated anew in its rebuilt score and event-frame geometry.
This produces an owned, unfinished timeline for subsequent shot relinking,
analytics, forecasts and fresh rendering. It publishes no result or media and
does not enable an uploaded correction endpoint. Callers must authorize the
reviewer and eventually publish every dependent output against base_generation.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

from src.api.result_storage import DiskSeries
from src.scoring.match_state_codec import decode_match_state, encode_match_state

_SHA = re.compile(r'[0-9a-f]{64}\Z')
_FIELDS = {'schema_version', 'analysis_id', 'source_sha256', 'base_generation',
           'base_analysis_revision', 'base_rule_replay_sha256', 'correction_id',
           'reviewer_id', 'evidence_ref', 'reason', 'initial_state_amendment', 'input_amendments'}
MAX_AMENDMENTS = 128
MAX_REQUEST_BYTES = 1024 * 1024


class _CorrectedRuleResult(SimpleNamespace):
    """Native handles with a detached, immutable record of the review command.

    The audit is presentation data. Later exporters must take the actual
    admitted command from this record rather than its mutable audit copy.
    """
    def __init__(self, *, request, **handles):
        super().__init__(**handles)
        super().__setattr__('_request_json', _canonical(request))

    @property
    def request(self):
        return json.loads(self._request_json)

    def __setattr__(self, name, value):
        if name in ('request', '_request_json'):
            raise AttributeError('The admitted review command is immutable')
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ('request', '_request_json'):
            raise AttributeError('The admitted review command is immutable')
        super().__delattr__(name)


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected rule rebuild: ' + message)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, DiskSeries)):
        return [_plain(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _text(value, maximum):
    return (type(value) is str and bool(value.strip()) and len(value) <= maximum
            and not any(ord(char) < 32 for char in value))


def _request(value):
    _require(type(value) is dict and set(value) == _FIELDS, 'request fields differ')
    # Detach the bounded review command before any long-lived owned processing.
    encoded = _canonical(value).encode('utf-8')
    _require(len(encoded) <= MAX_REQUEST_BYTES, 'review command exceeds bounded size')
    value = json.loads(encoded)
    _require(type(value['schema_version']) is int and value['schema_version'] == 1,
             'unsupported review command version')
    for key in ('source_sha256', 'base_generation', 'base_rule_replay_sha256'):
        _require(type(value[key]) is str and _SHA.fullmatch(value[key]), 'invalid ' + key)
    _require(type(value['base_analysis_revision']) is int and value['base_analysis_revision'] >= 0,
             'invalid base analysis revision')
    for key, maximum in (('analysis_id', 200), ('correction_id', 128), ('reviewer_id', 200),
                         ('evidence_ref', 2000), ('reason', 4000)):
        _require(_text(value[key], maximum), 'missing or invalid ' + key)
    amendments = value['input_amendments']
    _require(type(amendments) is list and len(amendments) <= MAX_AMENDMENTS,
             'input amendments must be a bounded review list')
    seen = set()
    for item in amendments:
        _require(type(item) is dict and set(item) == {
            'event_id', 'expected_input_fingerprint', 'replacement_source_input'},
            'input amendment fields differ')
        ident = item['event_id']
        _require(type(ident) is int and ident >= 0 and ident not in seen,
                 'amended event IDs must be distinct nonnegative integers')
        seen.add(ident)
        _require(type(item['expected_input_fingerprint']) is str
                 and _SHA.fullmatch(item['expected_input_fingerprint'])
                 and type(item['replacement_source_input']) is dict,
                 'input amendment requires its source fingerprint and complete replacement')
    initial = value['initial_state_amendment']
    if initial is not None:
        _require(type(initial) is dict and set(initial) == {
            'expected_initial_state_sha256', 'replacement_initial_state'}
            and type(initial['expected_initial_state_sha256']) is str
            and _SHA.fullmatch(initial['expected_initial_state_sha256'])
            and type(initial['replacement_initial_state']) is dict,
            'initial state amendment requires a raw state and previous digest')
    _require(initial is not None or amendments, 'review command contains no amendment')
    return value


def _scratch(directory, analysis_directory):
    path = Path(directory).absolute()
    _require('..' not in path.parts, 'scratch parent traversal is forbidden')
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
                 'scratch symlinks and junctions are forbidden')
    _require(path.is_dir(), 'caller-owned scratch directory must exist')
    path = path.resolve()
    _require(not path.is_relative_to(analysis_directory)
             and not analysis_directory.is_relative_to(path),
             'scratch must be separate from the immutable analysis directory')
    return path


def _same_measurement(actual, expected):
    """Compare JSON observations without treating bools as numeric values.

    A browser serializes 0.0 as 0. Accept that exact numeric equality, never
    a tolerance, then retain the original representation in the native input.
    """
    if type(expected) in (int, float):
        return (type(actual) in (int, float)
                and (type(actual) is int or math.isfinite(actual))
                and (type(expected) is int or math.isfinite(expected))
                and actual == expected)
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_same_measurement(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_same_measurement(left, right) for left, right in zip(actual, expected)))
    return actual == expected


def _replacement(original, item):
    replacement = _plain(item['replacement_source_input'])
    _require(replacement.get('event_id') == original['event_id']
             and type(replacement.get('event_id')) is int, 'replacement changes the event ID')
    _require((replacement.get('reviewed_evidence_ref') is None) ==
             (original['reviewed_evidence_ref'] is None),
             'physical observation and reviewed signal require separate source contracts')
    # This command changes semantic interpretation/current reviewed binding.
    # Moving events or modifying observed measurements needs another evidence
    # contract; it cannot silently become a coordinate or timing correction.
    _require('frame_index' in replacement
             and _canonical(replacement['frame_index']) == _canonical(original['frame_index']),
             'replacement changes source measurement: frame_index')
    for key in ('timestamp_s', 'ball_position_px', 'court_position_m',
                'trajectory_state', 'confidence', 'evidence'):
        _require(key in replacement and _same_measurement(replacement[key], original[key]),
                 'replacement changes source measurement: ' + key)
        replacement[key] = _plain(original[key])
    original_context, context = original['context'], replacement.get('context')
    _require(type(context) is dict, 'replacement source context is missing')
    for key in ('frame_index', 'source_pts', 'source_time_base', 'source_frame_index', 'segment_id',
                'analysis_allowed', 'review_evidence_ref', 'anchor_frame_index'):
        _require(key in context and _canonical(context[key]) == _canonical(original_context[key]),
                 'replacement changes source interval: ' + key)
    _require(_canonical(replacement) != _canonical(original), 'source amendment is unchanged')
    return replacement


def _initial_state(source, amendment, analysis_id):
    original = encode_match_state(source.initial_state)
    if amendment is None:
        return source.initial_state
    _require(_digest(original) == amendment['expected_initial_state_sha256'],
             'initial state digest differs from the reviewed base')
    raw = amendment['replacement_initial_state']
    _require(_canonical(raw) != _canonical(original), 'initial state amendment is unchanged')
    state = decode_match_state(raw)
    _require(state.match_id == analysis_id and (state.player_1_id, state.player_2_id) ==
             (source.initial_state.player_1_id, source.initial_state.player_2_id),
             'initial state changes analysis or participant identities')
    if not state.match_complete:
        points = ((state.tie_break_points_p1 + state.tie_break_points_p2) if state.tie_break_active
                  else state.points_p1 + state.points_p2)
        _require(state.service_side.value == ('DEUCE' if points % 2 == 0 else 'AD'),
                 'reviewed initial service side differs from score parity')
    return state


def _line_engine(snapshot, fps):
    import yaml
    from src.line_calling.contact_refinement import BounceContactRefiner
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.line_calling.line_geometry import ContactPatchModelType
    path = snapshot.resolve_file('run_config.yaml')
    _require(path is not None and path.stat().st_size <= MAX_REQUEST_BYTES,
             'sealed bounded run configuration is required')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    _require(type(config) is dict, 'sealed run configuration is invalid')
    settings = config.get('line_calling', {})
    _require(type(settings) is dict, 'line-call settings are invalid')
    window = settings.get('contact_refinement_window', 3)
    _require(type(window) is int and 0 <= window <= 64, 'unsupported contact refinement window')
    _require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0,
             'native frame rate is unavailable')
    patch = ContactPatchModelType(settings.get('contact_patch_model', ContactPatchModelType.EMPIRICAL_PATCH.value))
    numeric = {key: settings.get(key, default) for key, default in (
        ('contact_patch_radius_cm', 1.25), ('uncertainty_safety_factor', 1.5),
        ('line_width_uncertainty_cm', 1.25))}
    for key, value in numeric.items():
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 'invalid line-call setting: ' + key)
    applied = {**numeric, 'contact_patch_model': patch.value,
               'contact_refinement_window': window, 'fps': fps}
    return TennisLineCallEngine(refiner=BounceContactRefiner(window_radius=window, fps=fps),
                               contact_patch_model=patch, **numeric), applied


@contextmanager
def rebuild_corrected_rule_timeline(analysis_directory, scratch, *, request):
    """Re-evaluate all source inputs, then yield an owned pending revision.

    This has no authentication surface: a future owner-only command handler
    must supply the reviewer identity from its session. The reviewed base must
    still be current before and after processing. No original recorded call or
    score-history correction replay is used as a new decision. The caller may
    next relink fresh shots, attach them and finish this timeline in the context;
    the eventual complete publisher must use ``expected_generation=base``.
    """
    from src.api.analysis_generation import resolve_generation
    from src.scoring.rule_replay_source import open_rule_replay_source
    from src.scoring.correction_sources import correction_sources
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.scoring.point_timeline import PointTimeline, process_timeline_event, event_input_record
    from src.utils.scoring_journal import ScoringJournal
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    command = _request(request)
    analysis_directory = Path(analysis_directory).absolute()
    snapshot = resolve_generation(analysis_directory)
    analysis_directory = analysis_directory.resolve()
    scratch = _scratch(scratch, analysis_directory)
    _require(snapshot is not None and snapshot.generation == command['base_generation']
             and snapshot.analysis_id == command['analysis_id']
             and snapshot.analysis_revision == command['base_analysis_revision'],
             'current generation differs from the reviewed base')
    _require(snapshot.file_sha256('source.mp4') == command['source_sha256'],
             'normalized source differs from the reviewed base')
    replay_path = snapshot.resolve_file('native_rule_replay.json')
    _require(replay_path is not None and snapshot.file_sha256('native_rule_replay.json') ==
             command['base_rule_replay_sha256'], 'rule replay differs from the reviewed base')
    with ExitStack() as stack:
        source = stack.enter_context(open_rule_replay_source(replay_path, scratch,
            artifact_sha256=command['base_rule_replay_sha256'], source_sha256=command['source_sha256'],
            analysis_id=command['analysis_id'], analysis_revision=snapshot.analysis_revision))
        _require(len(source.source_frames) == snapshot.frames, 'source frame count differs from generation')
        sources = stack.enter_context(correction_sources(snapshot, scratch,
            source_sha256=command['source_sha256'], source_frames=source.source_frames))
        caller, settings = _line_engine(snapshot, sources.fps)
        journal = stack.enter_context(ScoringJournal(scratch))
        initial = _initial_state(source, command['initial_state_amendment'], snapshot.analysis_id)
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        owner = stack.enter_context(PointAnalysisSequence(scratch, snapshot.analysis_id,
            snapshot.analysis_revision + 1, source_fingerprint=command['source_sha256']))
        timeline = PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source.source_frames)
        amendments = {item['event_id']: item for item in command['input_amendments']}
        used = set()
        previous = None
        bounce_count = unavailable_count = 0
        for row in source.document['inputs']:
            original = _plain(row['source_input'])
            _require(_digest(original) == row['input_fingerprint'], 'original source input digest differs')
            event, context = source.decode_source_event(original)
            _require(_canonical(event_input_record(event, context)) == _canonical(original),
                     'original source input does not round-trip through typed evidence')
            ident = event.event_id
            if ident in amendments:
                amendment = amendments[ident]
                _require(ident not in used and amendment['expected_input_fingerprint'] == row['input_fingerprint'],
                         'reviewed event fingerprint differs or is duplicated')
                replacement = _replacement(original, amendment)
                event, context = source.decode_source_event(replacement)
                _require(_canonical(event_input_record(event, context)) == _canonical(replacement),
                         'replacement source input does not round-trip through typed evidence')
                used.add(ident)
            order = (event.frame_index, event.timestamp_s, ident)
            _require(type(event.timestamp_s) in (int, float) and math.isfinite(event.timestamp_s)
                     and (previous is None or order > previous and order[1] >= previous[1])
                     and owner.event(ident) is None, 'source inputs are duplicated or out of order')
            previous = order
            geometry = sources.event_geometry(event.frame_index, context)
            kind = getattr(event, 'rule_type', None) or event.event_type.value
            if kind == 'BOUNCE':
                bounce_count += 1
                unavailable_count += not geometry.usable
            process_timeline_event(event, context, engine, timeline, caller,
                ball_points=sources.event_ball_points(event.frame_index, context,
                    window_radius=settings['contact_refinement_window']),
                homography_matrix=geometry.homography_matrix, frames=(), geometry_available=geometry.usable)
        _require(used == set(amendments), 'a reviewed source amendment was not applied')
        current = resolve_generation(analysis_directory)
        _require(current is not None and current.generation == snapshot.generation,
                 'current generation changed during corrected processing')
        _require(engine.revision == 0, 'corrected processing requires a fresh rule journal')
        audit = {'schema_version': 1, 'status': 'PRIVATE_PENDING_SHOTS_ANALYTICS_FORECASTS_AND_RENDER',
            'request': command, 'analysis_revision': owner.revision, 'scoring_revision': engine.revision,
            'initial_raw_state': encode_match_state(engine.initial_state),
            'final_raw_state': encode_match_state(engine.get_current_state()),
            'applied_event_ids': sorted(used), 'accepted_input_count': len(owner.rows('transitions')),
            'bounce_count': bounce_count, 'geometry_unavailable_bounces': unavailable_count,
            'line_call_settings': settings, 'human_review_dependent': True,
            'recorded_line_calls_reused': False, 'independent_accuracy_validated': False,
            'published': False}
        yield _CorrectedRuleResult(request=command, engine=engine, timeline=timeline, owner=owner,
            source_frames=source.source_frames, sources=sources, audit=audit, snapshot=snapshot,
            base_generation=snapshot.generation, authoritative_events_enabled=source.authoritative_events_enabled)

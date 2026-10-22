"""Owned causal statistical forecasts from newly corrected accepted rule inputs.

The prior is explicitly not calibrated. No old forecast, final outcome or
whole-match aggregate is a prediction feature. This private stage and its
scratch export cannot publish a corrected uploaded analysis.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

from src.scoring.corrected_rule_rebuild import _canonical, _digest, _scratch
from src.scoring.corrected_shot_rebuild import _current
from src.scoring.match_state_codec import encode_match_state, decode_match_state


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected forecasts: ' + message)


def _same(actual, expected, message):
    _require(_canonical(actual) == _canonical(expected), message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _model_spec():
    from src.prediction import tennis
    from src.scoring import scoring_rules
    paths = {'predictor': Path(tennis.__file__), 'rules': Path(scoring_rules.__file__),
             'native_forecast_protocol': Path(__file__)}
    return {'model': 'statistical_tennis_rules', 'protocol': 'corrected_native_beta_prefix_v1',
        'prior': {'alpha': 2., 'beta': 2., 'source': 'explicit_neutral_engineering_prior'},
        'calibration_status': 'not_calibrated', 'calibration_artifact': None,
        'code_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in paths.items()}}


def _admit(rebuilt, analysis):
    from src.api.analysis_generation import GenerationSnapshot
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.shot_evidence_sequence import ShotEvidenceSequence
    snapshot, owner = rebuilt.snapshot, rebuilt.owner
    _require(isinstance(snapshot, GenerationSnapshot) and isinstance(owner, PointAnalysisSequence),
             'native committed correction ownership is required')
    owner._open()
    rebuilt.sources._require_open()
    _require(isinstance(analysis.shots, ShotEvidenceSequence), 'native fresh shots are required')
    analysis.shots._open()
    _require(analysis.owner is owner and analysis.timeline is rebuilt.timeline
             and analysis.engine is rebuilt.engine and analysis.snapshot == snapshot
             and analysis.base_generation == rebuilt.base_generation == snapshot.generation
             and owner.sealed is True and analysis.shots._sealed is True
             and type(owner.revision) is int and owner.revision == snapshot.analysis_revision+1
             and owner.analysis_id == snapshot.analysis_id
             and type(rebuilt.engine.revision) is int and rebuilt.engine.revision == 0
             and type(rebuilt.authoritative_events_enabled) is bool,
             'fresh finished shot/timeline revision differs')
    source_frames = rebuilt.sources.source_frames
    _require(source_frames.analysis_id == owner.analysis_id
             and source_frames.analysis_revision == snapshot.analysis_revision
             and source_frames.source_sha256 == owner.source_fingerprint
             and source_frames.artifact_sha256 == snapshot.file_sha256('native_rule_replay.json')
             and len(source_frames) == snapshot.frames,
             'forecast source/revision binding differs')
    _same(encode_match_state(rebuilt.engine.initial_state), rebuilt.audit['initial_raw_state'],
          'reviewed raw initial state changed')
    _same(encode_match_state(rebuilt.engine.get_current_state()), rebuilt.audit['final_raw_state'],
          'accepted raw final state changed')
    _require(len(owner.rows('replay_inputs')) == len(owner.rows('transitions'))
             == len(rebuilt.engine._inputs) == len(rebuilt.engine.history.entries),
             'accepted input/history/transition counts differ')
    _require(rebuilt.authoritative_events_enabled or not len(rebuilt.engine._inputs),
             'authority-off source contains accepted inputs')
    _current(rebuilt)


class _Snapshots:
    """One cached prefix only; identical BEFORE/AFTER states share the math."""
    def __init__(self, predictor, revision):
        self.predictor, self.revision = predictor, revision
        self.key = self.value = None

    def take(self, state, timestamp, blocked_reason):
        _require(_number(timestamp) and timestamp >= 0, 'forecast clock is invalid')
        _require(self.predictor.last_timestamp is None
                 or timestamp >= self.predictor.last_timestamp, 'forecast includes a future point')
        raw_hash = _digest(encode_match_state(state))
        key = (raw_hash, len(self.predictor.point_ids), blocked_reason)
        if key != self.key:
            forecast_state = state
            if blocked_reason is not None:
                # Never evaluate completion mathematics on a context already
                # known to be unusable; retain its actual raw hash above.
                forecast_state = decode_match_state(encode_match_state(state))
                forecast_state.score_confirmed = False
            value = self.predictor.snapshot(forecast_state, revision=self.revision,
                                             timestamp_seconds=timestamp)
            if blocked_reason is not None:
                value = {**value, 'status': 'unavailable', 'probabilities': None, 'winner_id': None,
                    'reason': 'Native causal forecast withheld: ' + blocked_reason,
                    'withheld_reason_code': blocked_reason}
            self.key, self.value = key, value
        # Detach from the one cached value and from every stored history row.
        result = json.loads(json.dumps(self.value, allow_nan=False))
        result['timestamp_seconds'] = timestamp
        return result, raw_hash


def _terminal_context_valid(state):
    """Check terminal totals without requiring unobserved earlier set detail."""
    from src.scoring.match_state import MatchFormat
    required = 2 if state.format == MatchFormat.BEST_OF_3 else 3
    a, b = state.sets_won_p1, state.sets_won_p2
    if max(a, b) > required or a == b == required:
        return False
    terminal = max(a, b) == required
    if state.match_complete != terminal:
        return False
    if state.current_set != a+b+(0 if state.match_complete else 1):
        return False
    if state.match_complete:
        winner = state.player_1_id if a == required else state.player_2_id
        if state.initial_state_confirmed and state.score_confirmed and state.winner_id != winner:
            return False
    elif state.winner_id is not None:
        return False
    return True


def _record(binding, model_signature, phase, state, snapshots, blocked, *,
            timestamp, interval, frame, event_id=None, event_uid=None, point_uid=None,
            uses_through=None):
    forecast, raw_hash = snapshots.take(state, timestamp, blocked)
    return {**binding, 'schema_version': '1.0', 'model_signature': model_signature,
        'phase': phase, 'event_id': event_id, 'event_uid': event_uid, 'point_uid': point_uid,
        'frame_index': frame, 'timestamp_seconds': timestamp,
        'source_pts': interval['source_pts'], 'source_time_base': interval['source_time_base'],
        'source_frame_index': interval['source_frame_index'],
        'uses_points_through_event_id': uses_through,
        'raw_state_sha256': raw_hash, 'forecast': forecast}


def _completed_point(point, transition, owner, frames, server, winner):
    """Join a terminal award to its actual earlier reviewed service anchor."""
    ident = transition['event_id']
    _require(type(point['terminal_event_id']) is int and point['terminal_event_id'] == ident
             and type(point['anchor_event_id']) is int and point['anchor_event_id'] >= 0
             and type(point['server_id']) is int and point['server_id'] == server
             and type(point['winner_id']) is int and point['winner_id'] == winner
             and type(point['uid']) is str and point['point_id'] == point['uid'] == transition['point_uid'],
             'completed point differs from the causal terminal outcome')
    anchor = owner.event(point['anchor_event_id'])
    _require(anchor is not None and type(anchor.get('event_id')) is int
             and anchor['event_id'] == point['anchor_event_id']
             and anchor['physical_event_type'] == anchor['rule_event_type'] == 'SERVE_CONTACT'
             and anchor['point_uid'] == point['uid']
             and anchor['outcome_type'] == 'SERVE_IN_PLAY'
             and type(anchor['frame']) is int and _number(anchor['timestamp_s'])
             and (anchor['frame'], anchor['timestamp_s'], anchor['event_id']) <
                 (transition['frame'], transition['timestamp_s'], ident)
             and type(anchor['rule_player_id']) is int and anchor['rule_player_id'] == server
             and anchor['pre_state']['server_id'] == server
             and anchor['pre_state']['initial_state_confirmed'] is True
             and anchor['pre_state']['score_confirmed'] is True,
             'completed point lacks its earlier confirmed service anchor')
    captured = owner.get('replay_inputs', anchor['uid'])
    _require(type(captured) is dict and _digest(captured['source_input']) == captured['input_fingerprint'],
             'completed point service anchor source differs')
    _same(captured['accepted_transition'], anchor, 'completed point service anchor transition differs')
    from src.scoring.rule_replay_source import _source_event
    event, context = _source_event(captured['source_input'], frames,
                                   owner.analysis_id, owner.source_fingerprint)
    _require(event.event_id == anchor['event_id'] and context.binding_current
             and context.binding.participant_id == server
             and context.binding.role_id == event.player_id,
             'completed point service anchor has no current participant proof')


@dataclass(frozen=True)
class _ForecastResult:
    """Bound handles and detached presentation values for one owned revision."""
    owner: object
    scratch: Path
    snapshot: object
    base_generation: str
    _rebuilt: object
    _shot_analysis: object
    _history: object
    _current_json: str
    _audit_json: str

    @property
    def history(self):
        return self._history

    @property
    def current(self):
        return json.loads(self._current_json)

    @property
    def audit(self):
        return json.loads(self._audit_json)


@contextmanager
def rebuild_corrected_forecasts(rebuilt, shot_analysis, scratch):
    """Yield bounded INITIAL/BEFORE_EVENT/AFTER_EVENT forecasts for this revision.

    State replay uses this correction's new accepted inputs and calls, with
    exact history/outcome/final-raw-state checks. It never uses the original
    generation's recorded line calls. The posterior learns once at a native
    completed point's terminal event using its server before the award. An
    unanchored awarded point permanently withholds this prefix's forecasts.
    Caller authorization, full corrected exports/rendering and publication
    still belong to the future complete correction coordinator.
    """
    from src.prediction.tennis import ServePointPrior, TennisWinProbability
    from src.scoring.point_timeline import context_record, event_input_record
    from src.scoring.rule_replay_source import _source_event
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.utils.forecast_sequence import ForecastSequence
    from src.utils.scoring_journal import ScoringJournal
    _admit(rebuilt, shot_analysis)
    snapshot, owner, accepted_engine = rebuilt.snapshot, rebuilt.owner, rebuilt.engine
    scratch = _scratch(scratch, snapshot.files_directory.parent.parent.resolve())
    initial = decode_match_state(encode_match_state(accepted_engine.initial_state))
    spec = _model_spec()
    signature = _digest(spec)
    binding = {'analysis_id': owner.analysis_id, 'analysis_revision': owner.revision,
        'source_sha256': owner.source_fingerprint, 'base_generation': snapshot.generation}
    blocked = ('AUTHORITATIVE_EVENTS_UNAVAILABLE' if not rebuilt.authoritative_events_enabled else
               'INITIAL_SCORE_UNCONFIRMED' if not initial.initial_state_confirmed
               or not initial.score_confirmed else None)
    if not _terminal_context_valid(initial):
        blocked = blocked or 'INVALID_TERMINAL_SCORE'
    frames = rebuilt.sources.source_frames
    with ExitStack() as stack:
        storage = stack.enter_context(ForecastSequence(scratch,
            player_ids=(initial.player_1_id, initial.player_2_id)))
        journal = stack.enter_context(ScoringJournal(scratch))
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        predictor = TennisWinProbability((initial.player_1_id, initial.player_2_id),
            ServePointPrior(), point_index=storage.point_index)
        snapshots = _Snapshots(predictor, owner.revision)
        first_interval = frames[0][0]
        last = _record(binding, signature, 'INITIAL', engine.state, snapshots, blocked,
            timestamp=first_interval['timestamp_seconds'], interval=first_interval, frame=0)
        storage.history.append(last)
        accepted_inputs = accepted_engine.iter_rule_inputs()
        captured_inputs = iter(owner.rows('replay_inputs'))
        sentinel = object()
        previous_order = None
        used_through = None
        count = 0
        try:
            for accepted, captured in zip_longest(accepted_inputs, captured_inputs, fillvalue=sentinel):
                _require(accepted is not sentinel and captured is not sentinel,
                         'accepted source input counts differ')
                _require(type(captured) is dict and set(captured) == {'source_input',
                    'input_fingerprint', 'rule_environment', 'scoring_input', 'accepted_transition'},
                    'accepted source envelope differs')
                source = captured['source_input']
                _require(type(source) is dict and _digest(source) == captured['input_fingerprint'],
                         'accepted source fingerprint differs')
                event, context = _source_event(source, frames, owner.analysis_id, owner.source_fingerprint)
                _same(event_input_record(event, context), source, 'accepted source does not round-trip')
                ident, frame, timestamp = event.event_id, event.frame_index, event.timestamp_s
                _require(type(ident) is int and ident >= 0 and type(frame) is int
                         and 0 <= frame < snapshot.frames and _number(timestamp) and timestamp >= 0,
                         'accepted source event clock or identity differs')
                order = (frame, timestamp, ident)
                _require(previous_order is None or order > previous_order
                         and timestamp >= previous_order[1], 'accepted event order differs')
                previous_order = order
                transition = owner.event(ident)
                _same(captured['scoring_input'], accepted, 'new accepted rule input differs')
                _same(captured['accepted_transition'], transition, 'new native transition differs')
                _require(type(accepted.get('event_id')) is int and accepted['event_id'] == ident
                         and type(transition.get('event_id')) is int and transition['event_id'] == ident
                         and type(transition.get('frame')) is int and transition['frame'] == frame
                         and accepted.get('timestamp_seconds') == timestamp
                         and transition.get('timestamp_s') == timestamp
                         and transition.get('rule_event_type') == accepted.get('event_type')
                         and transition.get('rule_player_id') == accepted.get('player_id')
                         and transition.get('input_fingerprint') == captured['input_fingerprint'],
                         'new rule/source/transition join differs')
                _same(transition['context'], context_record(context), 'new source context differs')
                _same(engine.state.to_dict(), transition['pre_state'], 'new rule pre-state differs')
                interval = frames[frame][0]
                row = _record(binding, signature, 'BEFORE_EVENT', engine.state, snapshots, blocked,
                    timestamp=timestamp, interval=interval, frame=frame, event_id=ident,
                    event_uid=transition['uid'], point_uid=transition['point_uid'], uses_through=used_through)
                storage.history.append(row)
                server_before_award = engine.state.server_id
                state, outcome = engine.process_match_event(**engine._decode_input(accepted))
                _same(outcome.to_dict(), transition['outcome'], 'new replay outcome differs')
                _same(state.to_dict(), transition['post_state'], 'new rule post-state differs')
                _same(engine.history.entries[-1].to_dict(),
                      accepted_engine.history.entries[count].to_dict(), 'new accepted history differs')
                if not _terminal_context_valid(state):
                    blocked = blocked or 'INVALID_TERMINAL_SCORE'
                if outcome.is_point_ending and outcome.winner_id is not None:
                    point = owner.get('points', transition['point_uid']) if transition['point_uid'] else None
                    if point is None or point['status'] != 'COMPLETED':
                        blocked = blocked or 'UNANCHORED_POINT_OUTCOME'
                    else:
                        _completed_point(point, transition, owner, frames, server_before_award, outcome.winner_id)
                        if blocked is None and state.initial_state_confirmed and state.score_confirmed:
                            predictor.observe_point(point_id=point['uid'], server_id=server_before_award,
                                winner_id=outcome.winner_id, timestamp_seconds=timestamp)
                used_through = ident
                last = _record(binding, signature, 'AFTER_EVENT', state, snapshots, blocked,
                    timestamp=timestamp, interval=interval, frame=frame, event_id=ident,
                    event_uid=transition['uid'], point_uid=transition['point_uid'], uses_through=ident)
                storage.history.append(last)
                count += 1
        finally:
            accepted_inputs.close()
            captured_inputs.close()
        _require(count == len(owner.rows('transitions')) == len(accepted_engine._inputs),
                 'new accepted input count changed')
        _same(encode_match_state(engine.state), encode_match_state(accepted_engine.state),
              'new raw final state differs')
        _require(len(storage.history) == 1+2*count, 'forecast history cadence differs')
        if blocked is None:
            points = owner.rows('points', status='COMPLETED')
            _require(len(points) == len(storage.point_index), 'completed point posterior count differs')
            for point in points:
                stored = storage.point_index[point['uid']]
                terminal = owner.event(point['terminal_event_id'])
                _require(stored == (point['server_id'], point['winner_id'], terminal['timestamp_s']),
                         'completed point posterior differs')
        _same(_model_spec(), spec, 'statistical model/protocol source changed during processing')
        _admit(rebuilt, shot_analysis)
        storage.seal()
        audit = {**binding, 'schema_version': 1, 'status': 'PRIVATE_PENDING_EXPORTS_AND_RENDER',
            'accepted_input_count': count, 'history_count': len(storage.history),
            'posterior_point_count': len(storage.point_index), 'withheld_reason_code': blocked,
            'model_signature': signature, 'model_spec': spec,
            'calibration_status': 'not_calibrated', 'old_forecasts_reused': False,
            'history_policy': 'INITIAL_AND_BEFORE_AFTER_EVERY_ACCEPTED_EVENT',
            'point_update_policy': 'NEW_ANCHORED_COMPLETED_POINT_AT_TERMINAL_EVENT_PRE_AWARD_SERVER',
            'human_review_dependent': True, 'independent_accuracy_validated': False, 'published': False}
        forecast = _ForecastResult(storage, scratch, snapshot, snapshot.generation,
            rebuilt, shot_analysis, storage.history,
            json.dumps(last['forecast'], allow_nan=False), json.dumps(audit, allow_nan=False))
        yield forecast
        _same(_model_spec(), spec, 'statistical model/protocol source changed during consumption')
        _admit(rebuilt, shot_analysis)


def write_corrected_forecasts(destination, forecast):
    """Stream one private scratch artifact atomically; this is not publication."""
    from src.utils.json_array_artifact import JsonArrayArtifact
    from src.utils.forecast_sequence import ForecastSequence
    _require(isinstance(forecast, _ForecastResult)
             and isinstance(forecast.owner, ForecastSequence)
             and forecast.history is forecast.owner.history,
             'native bound forecast result handles are required')
    forecast.owner._open()
    _require(forecast.owner.sealed is True, 'forecast owner must be sealed before export')
    _admit(forecast._rebuilt, forecast._shot_analysis)
    directory = _scratch(forecast.scratch, forecast.snapshot.files_directory.parent.parent.resolve())
    destination = Path(destination).absolute()
    _require('..' not in destination.parts and destination.name == 'corrected_forecasts.json'
             and destination.parent.resolve() == directory and not destination.is_symlink()
             and not getattr(destination, 'is_junction', lambda: False)(),
             'forecast export must use its exact guarded scratch destination')
    _same(forecast.current, forecast.history[-1]['forecast'], 'current forecast differs from history')
    metadata = {'schema_version': '1.0', 'analysis_id': forecast.audit['analysis_id'],
        'analysis_revision': forecast.audit['analysis_revision'],
        'source_sha256': forecast.audit['source_sha256'], 'base_generation': forecast.base_generation,
        'current': forecast.current, 'audit': forecast.audit}
    rows = iter(forecast.history)
    try:
        with JsonArrayArtifact(destination, metadata, array_key='history', allow_nan=False) as output:
            for row in rows:
                output.append(row)
            _admit(forecast._rebuilt, forecast._shot_analysis)
    finally:
        close = getattr(rows, 'close', None)
        if close is not None:
            close()
    _admit(forecast._rebuilt, forecast._shot_analysis)
    return destination

"""Explicit rule transitions to revision-bound points, attempts and rallies.

Physical detector roles are never match identities. This module has no model,
geometry inference, let detector or implicit initial-score reconstruction.
"""
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import operator
import math

from src.shot_analysis.shot_types import ShotOutcome, ShotDirection


def exact_source_time(pts, time_base):
    if type(pts) is not int and (not isinstance(pts, str) or str(int(pts)) != pts):
        raise ValueError('Source PTS must be an exact integer token')
    if not isinstance(time_base,(str,Fraction)):
        raise ValueError('Source time base must retain its rational token')
    base = Fraction(time_base)
    if base <= 0:
        raise ValueError('Source time base must be positive')
    return int(pts)*base


@dataclass(frozen=True)
class ReviewedParticipantBinding:
    frame_index: int
    segment_id: str
    role_id: int
    participant_id: int
    match_identity: str
    track_key: str
    court_end: str
    identity_revision: int
    calibration_revision: str
    anchor_frame_index: int
    evidence_ref: str
    analysis_id: str = ''
    source_fingerprint: str = ''
    source_frame_index: int | None = None
    source_pts: str | None = None
    source_time_base: str | None = None
    identity_epoch: int | None = None
    participant_mapping_ref: str = ''
    geometry_qualification_ref: str = ''

    def __post_init__(self):
        if (self.match_identity not in ('A', 'B') or type(self.role_id) is not int or self.role_id not in (1, 2)
                or type(self.participant_id) is not int or self.participant_id not in (1, 2) or self.court_end not in ('NEAR_COURT', 'FAR_COURT')
                or type(self.frame_index) is not int or self.frame_index < 0
                or type(self.identity_revision) is not int or self.identity_revision < 0
                or type(self.anchor_frame_index) is not int or not 0 <= self.anchor_frame_index <= self.frame_index
                or any(not isinstance(value, str) or not value for value in
                       (self.segment_id, self.track_key, self.calibration_revision, self.evidence_ref))):
            raise ValueError('Reviewed participant/end binding requires current, explicit provenance')


@dataclass(frozen=True)
class EventContext:
    frame_index: int
    source_pts: str | None
    source_time_base: str | None
    source_frame_index: int | None
    segment_id: str | None
    analysis_allowed: bool
    review_evidence_ref: str | None
    binding: ReviewedParticipantBinding | None = None
    binding_audit_matches: bool = False

    @property
    def source_time(self):
        try:
            if self.source_pts is None or self.source_time_base is None:
                return None
            return exact_source_time(self.source_pts, self.source_time_base)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    @property
    def binding_current(self):
        return bool(self.binding_audit_matches and self.analysis_allowed and self.review_evidence_ref and self.source_time is not None
            and isinstance(self.binding, ReviewedParticipantBinding)
            and self.binding.frame_index == self.frame_index and self.binding.segment_id == self.segment_id)

    @property
    def continuity_key(self):
        if not self.binding_current:
            return None
        value = self.binding
        return (self.segment_id, value.identity_epoch, value.identity_revision,
                value.calibration_revision, value.anchor_frame_index,
                value.participant_mapping_ref, value.geometry_qualification_ref)


def candidate_event_context(event, source_frames, provider=None, *, analysis_id=None, source_fingerprint=None):
    """A provider must supply reviewed role/identity/end proof, never a role guess."""
    interval = source_frames[event.frame_index][0] if source_frames is not None else {}
    identity = source_frames[event.frame_index].get(1, {}) if source_frames is not None else {}
    binding = provider(event, interval, identity) if provider is not None else None
    if binding is not None and not isinstance(binding, ReviewedParticipantBinding):
        raise TypeError('Current reviewed binding provider must return the typed evidence contract')
    audit_matches = False
    if binding is not None:
        observation = next((item for item in identity.get('observations', ())
            if item.get('track_key') == binding.track_key and item.get('state') == 'CONFIRMED'
            and item.get('player_id') == binding.match_identity), None)
        audit_matches = bool(observation is not None and analysis_id and source_fingerprint
            and interval.get('analysis_id') == analysis_id and identity.get('analysis_id') == analysis_id
            and binding.analysis_id == analysis_id and binding.source_fingerprint == source_fingerprint
            and binding.identity_epoch == identity.get('epoch') and binding.identity_revision == identity.get('revision')
            and binding.source_frame_index == interval.get('source_frame_index')
            and binding.source_pts == interval.get('source_pts')
            and binding.source_time_base == interval.get('source_time_base')
            and binding.anchor_frame_index == interval.get('anchor_frame_index')
            and binding.participant_mapping_ref and binding.geometry_qualification_ref)
    return EventContext(event.frame_index, interval.get('source_pts'), interval.get('source_time_base'),
        interval.get('source_frame_index'), interval.get('segment_id'), interval.get('analysis_allowed') is True,
        interval.get('review_evidence_id'), binding, audit_matches)


@dataclass(frozen=True)
class ReviewedRuleSignal:
    event_id: int
    frame_index: int
    timestamp_s: float
    rule_type: str
    evidence_ref: str
    context: EventContext

    def __post_init__(self):
        if (self.rule_type not in ('SERVICE_LET', 'CONSEQUENTIAL_GAP', 'POINT_OUTCOME_UNKNOWN')
                or not isinstance(self.evidence_ref, str) or not self.evidence_ref
                or type(self.event_id) is not int or self.event_id < 0
                or self.frame_index != self.context.frame_index
                or not math.isfinite(self.timestamp_s)):
            raise ValueError('Reviewed rule signal requires an explicit supported source reference')


def merge_rule_inputs(events, signals):
    """Two ordered lookaheads; reviewed signals are not detector event types."""
    physical, reviewed = iter(events), iter(signals)
    try:
        event, signal = next(physical, None), next(reviewed, None)
        previous = None
        while event is not None or signal is not None:
            def order(value):
                if type(value.frame_index) is not int or value.frame_index < 0 or not math.isfinite(value.timestamp_s):
                    raise ValueError('Rule ordering requires a valid frame and finite source timestamp')
                return value.frame_index, value.timestamp_s, value.event_id
            if signal is not None and not isinstance(signal, ReviewedRuleSignal):
                raise TypeError('Rule ingress requires typed reviewed signals')
            if event is not None and signal is not None and order(event) == order(signal):
                raise ValueError('Physical and reviewed rule inputs have an ambiguous shared identity/order')
            take_signal = signal is not None and (event is None or order(signal) < order(event))
            value = signal if take_signal else event
            if not isinstance(value, ReviewedRuleSignal) and take_signal:
                raise TypeError('Rule ingress requires typed reviewed signals')
            current = order(value)
            if previous is not None and (current < previous or current[1] < previous[1]):
                raise ValueError('Rule inputs must remain in source frame order')
            previous = current
            yield value
            if take_signal:
                signal = next(reviewed, None)
            else:
                event = next(physical, None)
    finally:
        for iterator in (physical, reviewed):
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()


def source_duration(start, end, source_frames):
    """Physical elapsed time requires every intervening eligible source edge."""
    playback = None
    try:
        a = exact_source_time(start['source_pts'], start['source_time_base'])
        b = exact_source_time(end['source_pts'], end['source_time_base'])
        if b >= a:
            playback = float(b-a)
        if (source_frames is None or b < a or start['segment_id'] != end['segment_id']
                or end['frame_index'] < start['frame_index']):
            return None, playback, 'SOURCE_INTERVAL_UNAVAILABLE'
        for endpoint in (start, end):
            actual = source_frames[endpoint['frame_index']][0]
            if any(endpoint.get(key) != actual.get(key) for key in
                   ('source_pts', 'source_time_base', 'source_frame_index', 'segment_id', 'anchor_frame_index')):
                return None, playback, 'SOURCE_ENDPOINT_BINDING_MISMATCH'
        previous = None
        for index in range(start['frame_index'], end['frame_index']+1):
            interval = source_frames[index][0]
            current = exact_source_time(interval['source_pts'], interval['source_time_base'])
            if (Fraction(interval['source_time_base']) <= 0 or interval.get('analysis_allowed') is not True
                    or not interval.get('review_evidence_id') or interval.get('segment_id') != start['segment_id']
                    or (previous is not None and (current <= previous or interval.get('measurement_transition_allowed') is not True))
                    or interval.get('anchor_frame_index') != start['anchor_frame_index']):
                return None, playback, 'SOURCE_CONTINUITY_OR_NORMAL_SPEED_UNAVAILABLE'
            previous = current
        return float(b-a), playback, None
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError):
        return None, playback, 'SOURCE_CLOCK_UNAVAILABLE'


class PointTimeline:
    """One active point/attempt/rally in RAM; prior states live in the owner."""
    def __init__(self, owner, initial_state, *, source_frames=None):
        owner._open()
        self.owner, self.source_frames = owner, source_frames
        self.initial_state = json.loads(json.dumps(initial_state, allow_nan=False))
        self.point = self.attempt = self.rally = None
        self.replay_index = 0
        self.previous_context = None

    def uid(self, kind, anchor):
        body = [self.owner.analysis_id, self.owner.source_fingerprint, self.owner.revision, kind, anchor]
        return kind+'_'+hashlib.sha256(json.dumps(body, separators=(',', ':')).encode()).hexdigest()

    def _save(self, kind, record, *, parent=None):
        self.owner.put(kind, record['uid'], record, parent=parent, status=record.get('status'))

    def scoring_input(self, event, context, pre):
        if self.previous_context is not None:
            if (self.point is not None and context.continuity_key != self.previous_context.continuity_key):
                return 'CONSEQUENTIAL_GAP', None
            duration, _, _ = source_duration(context_record(self.previous_context), context_record(context), self.source_frames)
            if duration is None:
                return 'CONSEQUENTIAL_GAP', None
        if isinstance(event, ReviewedRuleSignal):
            if event.rule_type == 'SERVICE_LET' and not context.binding_current:
                return 'CONSEQUENTIAL_GAP', None
            return event.rule_type, None
        kind = event.event_type.value
        if kind in ('SERVE_CONTACT', 'PLAYER_1_HIT', 'PLAYER_2_HIT'):
            role = event.player_id
            if (not context.binding_current or role not in (1, 2)
                    or context.binding.role_id != role
                    or kind.startswith('PLAYER_') and kind != f'PLAYER_{role}_HIT'):
                return 'CONSEQUENTIAL_GAP', None
            participant = context.binding.participant_id
            return ('SERVE_CONTACT' if kind == 'SERVE_CONTACT' else f'PLAYER_{participant}_HIT'), participant
        if not context.binding_current:
            return 'CONSEQUENTIAL_GAP', None
        return kind, None

    def service_target(self, context, state):
        from src.scoring.match_state import ServerCourtEnd
        if (self.attempt is None or not context.binding_current
                or context.binding.participant_id != state.server_id
                or context.binding.court_end != self.attempt['server_end']
                or state.serve_attempt != self.attempt['official_attempt']
                or state.service_side.value != self.attempt['service_side']
                or list(context.continuity_key) != self.attempt['continuity_key']):
            return None
        start = self.attempt['start_context']
        end = context_record(context)
        duration, _, _ = source_duration(start, end, self.source_frames)
        if duration is None:
            return None
        return state.get_expected_service_box(ServerCourtEnd(self.attempt['server_end']))

    def observe(self, event, context, pre, outcome, post, *, rule_type, rule_player, input_fingerprint=None):
        ident = event.event_id
        if self.owner.event(ident) is not None:
            return
        self.previous_context = context
        key = self.uid('event', [ident, context.source_pts, context.source_time_base,
                                 context.source_frame_index, context.frame_index])
        kind = event.rule_type if isinstance(event, ReviewedRuleSignal) else event.event_type.value
        status = outcome.outcome_type.value
        if self.point is None and status == 'DEAD_BALL_IGNORED':
            self.owner.put('transitions', key, {'uid': key, 'event_id': ident, 'frame': event.frame_index,
                'timestamp_s': event.timestamp_s, 'physical_event_type': kind, 'rule_event_type': rule_type,
                'rule_player_id': rule_player, 'context': context_record(context), 'pre_state': pre,
                'post_state': post, 'outcome_type': status, 'outcome': outcome.to_dict(),
                'point_uid': None, 'attempt_uid': None, 'rally_uid': None,
                'reviewed_rule_evidence_ref': event.evidence_ref if isinstance(event, ReviewedRuleSignal) else None,
                'input_fingerprint': input_fingerprint or event_input_fingerprint(event,context)}, event_id=ident, frame=event.frame_index)
            return
        anchored = (kind == 'SERVE_CONTACT' and context.binding_current
                    and pre['initial_state_confirmed'] and pre['score_confirmed']
                    and rule_type == 'SERVE_CONTACT' and rule_player == pre['server_id'])
        if self.point is None:
            self.point = {'uid': self.uid('point', key), 'point_id': self.uid('point', key),
                'status': 'OPEN' if anchored else 'PARTIAL_UNANCHORED',
                'anchor_event_id': ident if anchored else None, 'first_observed_event_id': ident,
                'server_id': pre['server_id'] if anchored else None,
                'receiver_id': pre['receiver_id'] if anchored else None,
                'set_number': pre['current_set'] if anchored else None,
                'game_number': pre['current_game'] if anchored else None,
                'service_side': pre['service_side'] if anchored else None,
                'winner_id': None, 'ending_reason': None, 'terminal_event_id': None,
                'is_review_pending': False, 'pre_point_state': pre}
        if kind == 'SERVE_CONTACT' and anchored and status == 'SERVE_IN_PLAY':
            self.attempt = {'uid': self.uid('attempt', key), 'point_uid': self.point['uid'],
                'status': 'PENDING', 'official_attempt': pre['serve_attempt'], 'replay_index': self.replay_index,
                'contact_event_id': ident, 'result_event_id': None, 'server_id': pre['server_id'],
                'server_end': context.binding.court_end, 'service_side': pre['service_side'],
                'continuity_key': list(context.continuity_key), 'start_context': context_record(context),
                'rally_uid': None}
            self._save('attempts', self.attempt, parent=self.point['uid'])
        if self.attempt is not None and status in ('FIRST_SERVE_FAULT', 'DOUBLE_FAULT', 'SERVICE_LET'):
            self.attempt.update(status=status, result_event_id=ident)
            self._save('attempts', self.attempt, parent=self.point['uid'])
            self.replay_index = self.replay_index+1 if status == 'SERVICE_LET' else 0
        if (kind == 'BOUNCE' and status == 'SERVE_IN_PLAY' and self.attempt is not None
                and self.attempt['status'] == 'PENDING'):
            self.rally = {'uid': self.uid('rally', self.attempt['uid']), 'rally_id': self.uid('rally', self.attempt['uid']),
                'point_uid': self.point['uid'], 'point_id': self.point['uid'], 'status': 'OPEN',
                'server_id': self.point['server_id'], 'receiver_id': self.point['receiver_id'],
                'serve_attempt': self.attempt['official_attempt'], 'winner_id': None, 'ending_reason': None,
                'start_context': self.attempt['start_context'], 'terminal_event_id': None,
                'duration_s': None, 'source_playback_elapsed_s': None, 'duration_reason': 'RALLY_NOT_COMPLETE'}
            self.attempt.update(status='SERVE_IN_PLAY', result_event_id=ident, rally_uid=self.rally['uid'])
            self._save('attempts', self.attempt, parent=self.point['uid'])
            self._save('rallies', self.rally, parent=self.point['uid'])
        if status == 'POINT_REVIEW_REQUIRED':
            self.point.update(status='REVIEW_REQUIRED', is_review_pending=True, ending_reason=outcome.reason)
            if self.rally is not None:
                self.rally.update(status='REVIEW_REQUIRED', duration_s=None, duration_reason='REVIEW_OR_CONSEQUENTIAL_GAP')
                self._save('rallies', self.rally, parent=self.point['uid'])
            if self.attempt is not None and self.attempt['status'] == 'PENDING':
                self.attempt.update(status='REVIEW_REQUIRED', result_event_id=ident)
                self._save('attempts', self.attempt, parent=self.point['uid'])
        completed = bool(outcome.is_point_ending and outcome.winner_id is not None and self.point['anchor_event_id'] is not None)
        if completed:
            self.point.update(status='COMPLETED', winner_id=outcome.winner_id, terminal_event_id=ident,
                              ending_reason=outcome.reason)
            if self.rally is not None:
                duration, playback, reason = source_duration(self.rally['start_context'], context_record(context), self.source_frames)
                self.rally.update(status='COMPLETED', winner_id=outcome.winner_id,
                    terminal_event_id=ident, ending_reason=outcome.reason, duration_s=duration,
                    source_playback_elapsed_s=playback, duration_reason=reason)
                self._save('rallies', self.rally, parent=self.point['uid'])
        self._save('points', self.point)
        trace = {'uid': key, 'event_id': ident, 'frame': event.frame_index, 'timestamp_s': event.timestamp_s,
            'physical_event_type': kind, 'rule_event_type': rule_type, 'rule_player_id': rule_player,
            'context': context_record(context), 'pre_state': pre, 'post_state': post,
            'outcome_type': status, 'outcome': outcome.to_dict(), 'point_uid': self.point['uid'],
            'attempt_uid': self.attempt['uid'] if self.attempt else None,
            'rally_uid': self.rally['uid'] if self.rally else None,
            'reviewed_rule_evidence_ref': event.evidence_ref if isinstance(event, ReviewedRuleSignal) else None,
            'input_fingerprint': input_fingerprint or event_input_fingerprint(event,context)}
        self.owner.put('transitions', key, trace, parent=self.point['uid'], event_id=ident, frame=event.frame_index)
        if completed:
            self.point = self.attempt = self.rally = None
            self.replay_index = 0

    def attach_shots(self, shots):
        from src.scoring.rule_replay import _plain
        for position, shot in enumerate(shots):
            original_shot = _plain(shot.to_dict())
            transition = self.owner.event(shot.match_event_id)
            if transition is None:
                transition, scope, participant = {}, 'UNKNOWN', None
            else:
                participant = transition['rule_player_id']
                attempt = self.owner.get('attempts', transition['attempt_uid']) if transition['attempt_uid'] else None
                scope = 'UNKNOWN'
                if transition['outcome_type'] == 'DEAD_BALL_IGNORED':
                    scope = 'DEAD_AFTER_PLAY'
                elif transition['outcome_type'] == 'POINT_REVIEW_REQUIRED':
                    scope = 'REVIEW_REQUIRED'
                elif transition['physical_event_type'] == 'SERVE_CONTACT' and attempt:
                    scope = 'LEGAL_RALLY' if attempt['status'] == 'SERVE_IN_PLAY' else attempt['status']
                    transition['rally_uid'] = attempt['rally_uid']
                    if attempt['status'] in ('FIRST_SERVE_FAULT', 'DOUBLE_FAULT'):
                        shot.outcome = ShotOutcome.FAULT
                elif transition['outcome_type'] == 'RALLY_CONTINUES' and transition['rally_uid']:
                    scope = 'LEGAL_RALLY'
                if shot.bounce_event_id is not None:
                    bounce = self.owner.event(shot.bounce_event_id)
                    same = (bounce is not None and bounce['point_uid'] == transition['point_uid']
                            and bounce['attempt_uid'] == transition['attempt_uid']
                            and bounce['outcome_type'] not in ('DEAD_BALL_IGNORED', 'POINT_REVIEW_REQUIRED', 'UNKNOWN_EVENT'))
                    if not same:
                        shot.bounce_event_id = shot.bounce_frame = None
                        shot.landing_court_position_m = shot.landing_zone = None
                        shot.direction = ShotDirection.UNKNOWN
                        shot.direction_confidence = 0.0
                        shot.direction_source = 'POINT_OR_ATTEMPT_BOUNDARY_UNRESOLVED'
            shots[position] = shot
            self.owner.add_member(position, shot, transition, scope, participant)
            self.owner.put('replay_shots', str(position), {
                'position': position, 'raw': original_shot, 'linked': _plain(shot.to_dict())},
                frame=shot.frame_index)

    def finish(self):
        """Persist constant-size summaries without constructing all memberships."""
        from src.utils.point_analysis_sequence import SQLValues
        for kind in ('points', 'rallies'):
            keys = SQLValues(self.owner, 'SELECT uid FROM records WHERE kind=? ORDER BY position', (kind,))
            for uid in keys:
                row = self.owner.get(kind, uid)
                column = 'point_uid' if kind == 'points' else 'rally_uid'
                total = self.owner.execute(f"SELECT COUNT(*) FROM members WHERE {column}=? AND scope='LEGAL_RALLY'", (uid,), one=True)[0]
                row['total_strokes_including_serve'] = total
                serves = self.owner.execute(f"SELECT COUNT(*) FROM members WHERE {column}=? AND scope='LEGAL_RALLY' AND is_serve=1", (uid,), one=True)[0]
                row['rally_hits_excluding_serve'] = total-serves
                if kind == 'points':
                    rally = self.owner.execute("SELECT payload FROM records WHERE kind='rallies' AND parent=? ORDER BY position LIMIT 1", (uid,), one=True)
                    row['duration_s'] = json.loads(rally[0])['duration_s'] if rally else None
                self._save(kind, row, parent=row.get('point_uid'))
        self.owner.seal()


def process_timeline_event(event, context, engine, timeline, line_call_engine, *, ball_points,
                           homography_matrix, frames, geometry_available):
    """Actual scoring with pre-state capture and explicit participant translation."""
    from src.line_calling.line_call_engine import LineCallEvidence, LineCallContext
    from src.scoring.match_state import BallPlayState
    if getattr(engine, 'revision', 0) != 0:
        raise ValueError('A corrected score requires a rebuilt analysis timeline before more events')
    if context.binding is not None and (context.binding.analysis_id != timeline.owner.analysis_id
            or context.binding.source_fingerprint != timeline.owner.source_fingerprint):
        raise ValueError('Event context belongs to a different analysis/source owner')
    original_input = event_input_record(event, context)
    fingerprint = hashlib.sha256(json.dumps(original_input, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    duplicate = timeline.owner.event(event.event_id)
    if duplicate is not None:
        physical_kind = event.rule_type if isinstance(event, ReviewedRuleSignal) else event.event_type.value
        if (duplicate.get('input_fingerprint') != fingerprint
                or duplicate['frame'] != event.frame_index or duplicate['physical_event_type'] != physical_kind
                or any(duplicate['context'].get(key) != getattr(context,key) for key in
                       ('source_pts','source_time_base','source_frame_index','segment_id'))):
            raise ValueError('An event identity cannot refer to different source evidence')
        from src.scoring.point_outcome_resolver import PointOutcome, PointOutcomeType
        outcome = duplicate['outcome']
        return engine.get_current_state(), PointOutcome(
            outcome_type=PointOutcomeType(outcome['outcome_type']), winner_id=outcome['winner_id'],
            reason=outcome['reason'], is_point_ending=outcome['is_point_ending'],
            confidence=outcome['confidence'], source_event_id=outcome['source_event_id']), None
    pre = json.loads(json.dumps(engine.get_current_state().to_dict(), allow_nan=False))
    rule_type, player = timeline.scoring_input(event, context, pre)
    kind = event.rule_type if isinstance(event, ReviewedRuleSignal) else event.event_type.value
    evidence = None
    if kind == 'BOUNCE':
        state = engine.get_current_state()
        call_context = LineCallContext.SERVE if state.ball_state == BallPlayState.SERVE_STARTED else LineCallContext.RALLY
        target = timeline.service_target(context, state) if call_context == LineCallContext.SERVE else None
        if (not geometry_available or not context.binding_current
                or call_context == LineCallContext.SERVE and target is None):
            evidence = LineCallEvidence.unavailable(event.event_id, event.frame_index, call_context,
                'Current reviewed participant, court end, source continuity or qualified geometry unavailable.',
                reason_code='SERVICE_CONTEXT_UNAVAILABLE' if call_context == LineCallContext.SERVE else 'EVENT_GEOMETRY_CONTEXT_UNAVAILABLE')
        else:
            evidence = line_call_engine.evaluate_bounce(event_id=event.event_id, bounce_frame=event.frame_index,
                ball_trajectory=ball_points, homography_matrix=homography_matrix, context=call_context,
                target_service_box=target, raw_frames=frames)
        if (not isinstance(evidence, LineCallEvidence)
                or type(evidence.event_id) is not int or evidence.event_id != event.event_id
                or type(evidence.bounce_frame) is not int or evidence.bounce_frame != event.frame_index):
            raise ValueError('Line-call evidence must bind the source event ID and bounce frame before acceptance')
    state, outcome = engine.process_match_event(event_id=event.event_id, event_type=rule_type,
        timestamp_seconds=event.timestamp_s, player_id=player, line_call=evidence,
        confidence=getattr(event, 'confidence', 1.0))
    post = json.loads(json.dumps(state.to_dict(), allow_nan=False))
    timeline.observe(event, context, pre, outcome, post, rule_type=rule_type, rule_player=player,
                     input_fingerprint=fingerprint)
    transition = timeline.owner.event(event.event_id)
    timeline.owner.put('replay_inputs', transition['uid'], {
        'source_input': original_input, 'input_fingerprint': fingerprint,
        'rule_environment': {'geometry_available': bool(geometry_available)},
        'scoring_input': engine.accepted_rule_input(event.event_id),
        'accepted_transition': transition}, event_id=event.event_id, frame=event.frame_index)
    return state, outcome, evidence


def context_record(context):
    result = asdict(context)
    result['anchor_frame_index'] = context.binding.anchor_frame_index if context.binding else None
    return result


def event_input_fingerprint(event, context):
    """Retries bind the entire physical/reviewed input and current proof."""
    return hashlib.sha256(json.dumps(event_input_record(event, context), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def event_input_record(event, context):
    """Freeze original unrounded evidence before rule/landing/shot mutation."""
    reviewed = isinstance(event,ReviewedRuleSignal)
    payload = {key:getattr(event,key,None) for key in ('event_id','frame_index','timestamp_s','player_id',
        'confidence','ball_position_px','court_position_m','trajectory_state','evidence')}
    payload['event_type'] = event.rule_type if reviewed else event.event_type.value
    payload['reviewed_evidence_ref'] = event.evidence_ref if reviewed else None
    payload['context'] = context_record(context)
    def numeric(value):
        if type(value).__module__.split('.')[0] == 'numpy':
            if hasattr(value,'tolist'): return value.tolist()
            if hasattr(value,'item'): return value.item()
        raise TypeError('Event proof must have a finite JSON representation')
    serialized = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False,default=numeric)
    return json.loads(serialized)


class EligibleShots(Sequence):
    def __init__(self, ledger, shots, *, landings=False):
        self.ledger, self.shots, self.positions = ledger, shots, ledger.eligible_positions(landings=landings)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            from src.utils.mapped_sequence_view import MappedSequenceView
            return MappedSequenceView(self, lambda value, _: value)[index]
        position = self.positions[index]
        return replace(self.shots[position], player_id=self.ledger.participant(position))

    def __iter__(self):
        iterator = iter(self.positions)
        try:
            for position in iterator:
                yield replace(self.shots[position], player_id=self.ledger.participant(position))
        finally:
            iterator.close()


def native_match_analytics(ledger, shots, final_state, *, authoritative):
    """Reduce explicit completed/partial records and resolved serve slots."""
    from src.utils.mapped_sequence_view import MappedSequenceView
    players = {}
    for participant in (1, 2):
        players['player_'+str(participant)] = {
            'id': participant, 'movement': {'distance_meters': None, 'average_speed_kmh': None},
            'shots': dict.fromkeys(('total','forehand','backhand','serve','volley','overhead','unknown'),0),
            'direction': dict.fromkeys(('cross_court','down_the_line','middle','unknown'),0),
            'landing_zones': {}, 'court_occupancy_heatmap': []}
    for shot in EligibleShots(ledger, shots):
        player = players['player_'+str(shot.player_id)]
        player['shots']['total'] += 1
        player['shots'][shot.shot_type.value.lower()] += 1
        player['direction'][shot.direction.value.lower()] += 1
        if shot.landing_zone is not None:
            zone = shot.landing_zone.value
            player['landing_zones'][zone] = player['landing_zones'].get(zone,0)+1
    serve = {str(p): dict(server_id=p, contact_executions=0, service_lets=0, first_serves_attempted=0,
        second_serves_attempted=0, first_serves_in=0, first_serves_faulted=0,
        second_serves_in=0, double_faults=0, unresolved_executions=0) for p in (1,2)}
    for attempt in ledger.rows('attempts'):
        values = serve[str(attempt['server_id'])]
        values['contact_executions'] += 1
        slot = attempt['official_attempt']
        if attempt['replay_index'] == 0:
            values['first_serves_attempted' if slot == 1 else 'second_serves_attempted'] += 1
        status = attempt['status']
        if status == 'SERVICE_LET': values['service_lets'] += 1
        elif status == 'FIRST_SERVE_FAULT': values['first_serves_faulted'] += 1
        elif status == 'DOUBLE_FAULT': values['double_faults'] += 1
        elif status == 'SERVE_IN_PLAY': values['first_serves_in' if slot == 1 else 'second_serves_in'] += 1
        else: values['unresolved_executions'] += 1
    for values in serve.values():
        denominator = values['first_serves_in']+values['first_serves_faulted']
        values['first_serve_percentage'] = values['first_serves_in']/denominator*100 if denominator else None
        values['first_serve_percentage_denominator'] = denominator
        values['percentage_definition'] = 'RESOLVED_OFFICIAL_FIRST_SLOTS_IN_OVER_IN_PLUS_FAULT; LET_EXECUTIONS_EXCLUDED'
    rally_count = rally_total = longest = known_duration_count = 0
    effective = 0.
    for rally in ledger.rows('rallies', status='COMPLETED'):
        rally_count += 1
        count = rally['total_strokes_including_serve']
        rally_total += count
        longest = max(longest,count)
        if rally['duration_s'] is not None:
            known_duration_count += 1
            effective += rally['duration_s']
    landing_shots = EligibleShots(ledger, shots, landings=True)
    def landing(shot, position):
        return {'shot_id':shot.shot_id,'player_id':shot.player_id,'shot_type':shot.shot_type.value,
                'x_m':shot.landing_court_position_m[0],'y_m':shot.landing_court_position_m[1],
                'landing_zone':shot.landing_zone.value if shot.landing_zone else None,'is_dead_ball':False}
    completed = len(ledger.rows('points',status='COMPLETED'))
    return {'schema_version':'2.0', 'match_id':final_state.match_id, 'format':final_state.format.value,
        'match_complete':final_state.match_complete if authoritative else None,
        'winner_id':final_state.winner_id if authoritative and final_state.score_confirmed else None,
        'score_summary':final_state.get_score_summary_str() if authoritative and final_state.score_confirmed else None,
        'authoritative_events_enabled':authoritative,
        'event_statistics_scope':'REVIEW_BOUND_POINT_ATTEMPT_TIMELINE; PHYSICAL_ROLE_OBSERVATIONS_SEPARATE',
        'overview':{'total_points_played':completed if authoritative else None,
            'partial_or_review_points':len(ledger.rows('points'))-completed,
            'total_shots_recorded':len(shots),'live_shots_recorded':len(EligibleShots(ledger,shots)),
            'total_rallies':rally_count,'average_rally_length':rally_total/rally_count if rally_count else None,
            'longest_rally_length':longest if rally_count else None,
            'effective_play_seconds':effective if known_duration_count else None,
            'timed_completed_rallies':known_duration_count, 'completed_rallies_duration_denominator':rally_count},
        'players':players,'serve_analytics':{'by_player':serve},
        'ball_landing_heatmap':MappedSequenceView(landing_shots,landing),
        'ball_speed_analytics':{'available':False,'reason':'INDEPENDENT_PHYSICAL_SPEED_QUALIFICATION_UNAVAILABLE'},
        'remaining_capacity_limits':['WHOLE_ANALYSIS_CORRECTION_PUBLICATION','DURABLE_INFERENCE_CHECKPOINTS'],
        'independent_accuracy_validated':False}

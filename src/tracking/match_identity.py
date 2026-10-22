"""Causal A/B identity from explicit anchors and conservative local continuity.

``track_id`` is a detector identifier inside one continuity epoch. A/B are
match identities established by reviewed anchors, never by near/far position.
Colour and court continuity are engineering compatibility scores, not calibrated
probabilities. Reconnection hypotheses never become confirmed automatically.

The caller persists returned events/audit records and resets its actual tracker
when ``local_reset`` is true. This module contains no network or ReID model.
Memory holds two identity descriptors, current local bindings and at most 128
recent correction commands. Motion continuity only permits an identity/timing
bridge; independent ground-position and metric quality gates still apply.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Timestamp must be finite nonnegative source seconds.')
    return float(value)


def _vector(value, size, unit=False):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'Expected {size} finite coordinates.')
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and (not unit or 0 <= item <= 1) for item in result):
        raise ValueError('Coordinates must be finite; colour channels use [0,1].')
    return result


def _id(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('A local track ID must be a nonnegative integer.')
    return value


def _segment(value):
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError('A nonempty continuity segment ID is required.')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class IdentitySettings:
    max_gap_seconds: float = 1.0
    max_ground_speed_mps: float = 12.0
    ground_slack_m: float = .5
    max_colour_distance: float = .35
    hypothesis_min_score: float = .65
    hypothesis_margin: float = .2
    max_observations: int = 32

    def __post_init__(self):
        values = asdict(self)
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in values.values()):
            raise ValueError('Identity guards must be finite numbers.')
        if self.max_gap_seconds <= 0 or self.max_ground_speed_mps <= 0 or self.ground_slack_m < 0:
            raise ValueError('Positive time/speed guards and nonnegative slack are required.')
        if not 0 < self.max_colour_distance <= 1 or not 0 <= self.hypothesis_min_score <= 1 or not 0 < self.hypothesis_margin <= 1:
            raise ValueError('Compatibility settings must use the documented unit ranges.')
        if not isinstance(self.max_observations, int) or not 2 <= self.max_observations <= 128:
            raise ValueError('Bound current observations between two and 128.')


@dataclass(frozen=True)
class IdentityObservation:
    track_id: int | None
    court_position_m: tuple[float, float] | None = None
    colour_rgb: tuple[float, float, float] | None = None
    participant: bool | None = None
    overlap: bool = False
    ground_point_usable: bool = False

    def __post_init__(self):
        if self.track_id is not None:
            _id(self.track_id)
        object.__setattr__(self, 'court_position_m', _vector(self.court_position_m, 2))
        object.__setattr__(self, 'colour_rgb', _vector(self.colour_rgb, 3, True))
        if (self.participant is not None and not isinstance(self.participant, bool)) or not isinstance(self.overlap, bool) or not isinstance(self.ground_point_usable, bool):
            raise ValueError('Participant/overlap/ground flags must be booleans or unknown participant.')


@dataclass(frozen=True)
class IdentityAnchor:
    anchor_id: str
    player_id: str
    track_id: int
    timestamp: float
    segment_id: str
    reason: str
    court_position_m: tuple[float, float] | None = None
    colour_rgb: tuple[float, float, float] | None = None
    source: str = 'manual_review'

    def __post_init__(self):
        if self.player_id not in {'A', 'B'} or not isinstance(self.anchor_id, str) or not 1 <= len(self.anchor_id) <= 128:
            raise ValueError('A reviewed anchor needs a stable command ID and player A or B.')
        _id(self.track_id)
        _segment(self.segment_id)
        object.__setattr__(self, 'timestamp', _time(self.timestamp))
        object.__setattr__(self, 'court_position_m', _vector(self.court_position_m, 2))
        object.__setattr__(self, 'colour_rgb', _vector(self.colour_rgb, 3, True))
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 1000:
            raise ValueError('Document why this identity association was confirmed.')
        if self.source not in {'manual_review', 'independent_verified_identity'}:
            raise ValueError('Detector output is not a verified identity anchor.')


class MatchIdentity:
    def __init__(self, analysis_id: str, settings: IdentitySettings | None = None):
        if not isinstance(analysis_id, str) or not analysis_id or len(analysis_id) > 160:
            raise ValueError('A unique analysis/session ID is required.')
        self.analysis_id, self.settings = analysis_id, settings or IdentitySettings()
        self.epoch, self.revision, self.segment_id, self.last_timestamp = 0, 0, None, None
        self.identities, self.bindings, self.commands = {}, {}, {}
        self.anchor_floor = 0.
        self.last_input = self.last_result = None

    def correct(self, anchor: IdentityAnchor | dict):
        """Confirm at the current source time; older corrections require replay.

        An explicit swap needs one anchor per player. Moving A onto B's local
        track revokes B there; it never assigns B to A's previous track by guess.
        Recent duplicate command IDs are idempotent; persist the full audit in
        the caller for durable deduplication beyond the bounded recent window.
        """
        anchor = IdentityAnchor(**anchor) if isinstance(anchor, dict) else anchor
        if not isinstance(anchor, IdentityAnchor):
            raise ValueError('Supply an IdentityAnchor or its documented fields.')
        payload = json.loads(json.dumps(asdict(anchor)))
        command = self.commands.get(anchor.anchor_id)
        if command:
            if command['payload'] != payload:
                raise ValueError('An anchor command ID cannot represent a different correction.')
            return json.loads(json.dumps(command['event']))
        if self.segment_id is not None and anchor.segment_id != self.segment_id:
            raise ValueError('Update the new continuity segment before anchoring its local tracks.')
        if self.last_timestamp is not None and anchor.timestamp != self.last_timestamp:
            raise ValueError('Confirm at the current source timestamp; replay an earlier checkpoint for historical corrections.')
        self.segment_id = anchor.segment_id
        self.anchor_floor = max(self.anchor_floor, anchor.timestamp)
        replaced = self.bindings.get(anchor.track_id, {}).get('player_id')
        self.bindings = {track: item for track, item in self.bindings.items()
                         if item['player_id'] != anchor.player_id and track != anchor.track_id}
        self.bindings[anchor.track_id] = {'player_id': anchor.player_id, 'epoch': self.epoch, 'last_seen': anchor.timestamp,
                                        'court_position_m': anchor.court_position_m, 'colour_rgb': anchor.colour_rgb,
                                        'ground_point_usable': False, 'anchor_id': anchor.anchor_id}
        self.identities[anchor.player_id] = {'colour_rgb': anchor.colour_rgb, 'court_position_m': anchor.court_position_m,
                                           'ground_point_usable': False, 'epoch': self.epoch,
                                           'last_seen': anchor.timestamp, 'anchor_id': anchor.anchor_id}
        self.revision += 1
        event = {'event_id': 'identity_anchor_' + _digest([self.analysis_id, anchor.anchor_id])[:32],
                 'type': 'IDENTITY_CORRECTION', 'analysis_id': self.analysis_id, 'epoch': self.epoch,
                 'revision': self.revision, 'anchor': payload, 'replaced_player_id': replaced,
                 'provenance': anchor.source}
        self.commands[anchor.anchor_id] = {'payload': payload, 'event': event}
        if len(self.commands) > 128:
            del self.commands[next(iter(self.commands))]
        self.last_input = self.last_result = None
        return json.loads(json.dumps(event))

    def _support(self, observation, descriptor, timestamp, allow_ground):
        evidence, scores = [], []
        old_colour = descriptor.get('colour_rgb')
        if observation.colour_rgb is not None and old_colour is not None:
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(observation.colour_rgb, old_colour)) / 3)
            scores.append(max(0., 1 - distance / self.settings.max_colour_distance))
            evidence.append({'signal': 'colour', 'distance': distance, 'compatible': distance <= self.settings.max_colour_distance})
        old_ground = descriptor.get('court_position_m')
        elapsed = timestamp - descriptor['last_seen']
        if (allow_ground and observation.ground_point_usable and descriptor.get('ground_point_usable', False)
                and descriptor.get('epoch') == self.epoch and observation.court_position_m is not None
                and old_ground is not None and 0 <= elapsed <= self.settings.max_gap_seconds):
            distance = math.dist(observation.court_position_m, old_ground)
            limit = self.settings.ground_slack_m + self.settings.max_ground_speed_mps * elapsed
            scores.append(max(0., 1 - distance / max(limit, 1e-12)))
            evidence.append({'signal': 'court_continuity', 'distance_m': distance, 'limit_m': limit, 'compatible': distance <= limit})
        return {'compatibility_score': sum(scores) / len(scores) if scores else None,
                'evidence': evidence, 'calibrated_probability': None}

    def update(self, observations, timestamp, segment_id):
        """Update once per source timestamp; identical overlap replays deduplicate.

        Inputs are IdentityObservation values or their documented dictionaries.
        Unknown cuts/gaps reset local associations and prohibit a motion bridge.
        A new track's colour/court candidate is always PROVISIONAL or AMBIGUOUS.
        The caller retains original PTS/source seconds rather than assumed FPS.
        """
        timestamp, segment_id = _time(timestamp), _segment(segment_id)
        bounded = []
        for index, item in enumerate(observations):
            if index >= self.settings.max_observations:
                raise ValueError('Supply a bounded set of typed identity observations.')
            observation = IdentityObservation(**item) if isinstance(item, dict) else item
            if not isinstance(observation, IdentityObservation):
                raise ValueError('Supply a bounded set of typed identity observations.')
            bounded.append(observation)
        observations = bounded
        track_ids = [item.track_id for item in observations if item.track_id is not None]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError('Local track IDs must be unique within a frame.')
        if timestamp < self.anchor_floor or (self.last_timestamp is not None and timestamp < self.last_timestamp):
            raise ValueError('Source time cannot move backwards or precede a future anchor.')
        fingerprint = _digest([timestamp, segment_id, self.revision, [asdict(item) for item in observations]])
        if fingerprint == self.last_input:
            return json.loads(json.dumps(self.last_result))
        if self.last_timestamp == timestamp and self.segment_id == segment_id and self.last_input is not None:
            raise ValueError('Different evidence at the same source time requires replay from a checkpoint.')
        reset_reason = None
        if self.segment_id is not None and segment_id != self.segment_id:
            reset_reason = 'camera_or_continuity_segment_changed'
        elif self.last_timestamp is not None and timestamp - self.last_timestamp > self.settings.max_gap_seconds:
            reset_reason = 'source_timestamp_gap'
        if reset_reason:
            self.epoch += 1
            self.bindings.clear()
        self.segment_id = segment_id
        expired = [track for track, binding in self.bindings.items() if timestamp - binding['last_seen'] > self.settings.max_gap_seconds]
        for track in expired:
            del self.bindings[track]
        previous = self.bindings.copy()
        records = []
        for observation_index, observation in enumerate(observations):
            key = observation.track_id
            binding = previous.get(key)
            player_id, state, reason, bridge = None, 'UNKNOWN', 'identity_has_not_been_confirmed', False
            support, hypotheses = {'compatibility_score': None, 'evidence': [], 'calibrated_probability': None}, []
            if observation.participant is False:
                state, reason = 'EXCLUDED', 'observed_person_is_outside_play'
                self.bindings.pop(key, None)
            elif key is None:
                reason = 'local_track_id_missing'
            elif observation.overlap:
                state, reason = 'AMBIGUOUS', 'overlap_requires_identity_review'
                self.bindings.pop(key, None)
            elif binding is not None:
                support = self._support(observation, binding, timestamp, True)
                conflicts = [item for item in support['evidence'] if not item['compatible']]
                if conflicts:
                    state, reason = 'AMBIGUOUS', 'local_track_continuity_conflict'
                    self.bindings.pop(key, None)
                else:
                    player_id, state, reason = binding['player_id'], 'CONFIRMED', 'local_track_continues_reviewed_anchor'
                    bridge = (not reset_reason and 0 < timestamp - binding['last_seen'] <= self.settings.max_gap_seconds
                              and binding['ground_point_usable'] and observation.ground_point_usable
                              and binding['court_position_m'] is not None and observation.court_position_m is not None)
                    refreshed = {**binding, 'last_seen': timestamp,
                                 'court_position_m': observation.court_position_m,
                                 'colour_rgb': observation.colour_rgb or binding['colour_rgb'],
                                 'ground_point_usable': observation.ground_point_usable}
                    self.bindings[key] = refreshed
                    self.identities[player_id] = {field: refreshed[field] for field in ('colour_rgb', 'court_position_m', 'ground_point_usable', 'epoch', 'last_seen', 'anchor_id')}
            if state == 'UNKNOWN' and key is not None:
                for candidate, descriptor in self.identities.items():
                    evidence = self._support(observation, descriptor, timestamp, not reset_reason)
                    if evidence['compatibility_score'] is not None:
                        hypotheses.append({'player_id': candidate, **evidence})
                hypotheses.sort(key=lambda item: (-item['compatibility_score'], item['player_id']))
                if hypotheses:
                    top = hypotheses[0]['compatibility_score']
                    margin = top - hypotheses[1]['compatibility_score'] if len(hypotheses) > 1 else top
                    if top >= self.settings.hypothesis_min_score and margin >= self.settings.hypothesis_margin:
                        state, reason = 'PROVISIONAL', 'candidate_identity_requires_reviewed_anchor'
                    else:
                        state, reason = 'AMBIGUOUS', 'support_does_not_separate_match_identities'
            event_key = key if key is not None else f'untracked:{observation_index}'
            event_id = 'identity_' + _digest([self.analysis_id, self.epoch, segment_id, timestamp, event_key])[:32]
            records.append({'event_id': event_id, 'analysis_id': self.analysis_id, 'epoch': self.epoch,
                            'segment_id': segment_id, 'timestamp': timestamp, 'revision': self.revision,
                            'track_id': key, 'track_key': f'{self.epoch}:{key}' if key is not None else None,
                            'player_id': player_id, 'state': state, 'reason': reason,
                            'anchor_id': binding['anchor_id'] if binding and state == 'CONFIRMED' else None,
                            'motion_continuity': bridge, 'support': support, 'hypotheses': hypotheses})
        # Two local boxes may suggest the same person after fragmentation; they
        # remain hypotheses, never duplicate confirmed player assignments.
        for track, binding in self.bindings.items():
            if track not in track_ids:
                binding['ground_point_usable'] = False
        self.last_timestamp = timestamp
        result = {'analysis_id': self.analysis_id, 'epoch': self.epoch, 'revision': self.revision,
                  'local_reset': bool(reset_reason), 'reset_reason': reset_reason, 'observations': records}
        self.last_input, self.last_result = fingerprint, result
        return json.loads(json.dumps(result))

    def checkpoint(self):
        """A bounded JSON-compatible state; the caller commits it with event IDs."""
        state = {'schemaVersion': 1, 'analysis_id': self.analysis_id, 'settings': asdict(self.settings),
                 'epoch': self.epoch, 'revision': self.revision, 'segment_id': self.segment_id,
                 'last_timestamp': self.last_timestamp, 'anchor_floor': self.anchor_floor,
                 'identities': self.identities, 'bindings': {str(key): value for key, value in self.bindings.items()},
                 'commands': self.commands, 'last_input': self.last_input, 'last_result': self.last_result}
        return json.loads(json.dumps(state, allow_nan=False))

    @classmethod
    def restore(cls, state, analysis_id):
        try:
            return cls._restore_checked(state, analysis_id)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Malformed identity checkpoint.') from error

    @classmethod
    def _restore_checked(cls, state, analysis_id):
        if not isinstance(state, dict) or state.get('schemaVersion') != 1 or state.get('analysis_id') != analysis_id:
            raise ValueError('Checkpoint belongs to another analysis or unsupported schema.')
        instance = cls(analysis_id, IdentitySettings(**state['settings']))
        if len(state['identities']) > 2 or len(state['bindings']) > 2 or len(state['commands']) > 128:
            raise ValueError('Identity checkpoint exceeds its bounded state.')
        if set(state['identities']) - {'A', 'B'}:
            raise ValueError('Checkpoint contains an invalid match identity.')
        if type(state['epoch']) is not int or state['epoch'] < 0 or type(state['revision']) is not int or state['revision'] < 0:
            raise ValueError('Invalid checkpoint epoch or correction revision.')
        if state['segment_id'] is not None:
            _segment(state['segment_id'])
        if state['last_timestamp'] is not None:
            _time(state['last_timestamp'])
        _time(state['anchor_floor'])
        # Round-trip through strict JSON rejects nonfinite/cross-object aliases.
        state = json.loads(json.dumps(state, allow_nan=False))
        for name in ('epoch', 'revision', 'segment_id', 'last_timestamp', 'anchor_floor', 'identities', 'commands', 'last_input', 'last_result'):
            setattr(instance, name, state[name])
        if any(not isinstance(key, str) or not key.isdigit() or str(int(key)) != key for key in state['bindings']):
            raise ValueError('Checkpoint local keys must be canonical integer IDs.')
        instance.bindings = {_id(int(key)): value for key, value in state['bindings'].items()}
        for descriptor in [*instance.identities.values(), *instance.bindings.values()]:
            _time(descriptor['last_seen'])
            if (type(descriptor.get('epoch')) is not int or not 0 <= descriptor['epoch'] <= instance.epoch
                    or not isinstance(descriptor.get('ground_point_usable'), bool)
                    or not isinstance(descriptor.get('anchor_id'), str) or not 1 <= len(descriptor['anchor_id']) <= 128
                    or descriptor['last_seen'] > max(instance.last_timestamp or 0., instance.anchor_floor)):
                raise ValueError('Invalid checkpoint identity descriptor.')
            _vector(descriptor.get('colour_rgb'), 3, True)
            _vector(descriptor.get('court_position_m'), 2)
        if any(item.get('player_id') not in {'A', 'B'} for item in instance.bindings.values()) or len({item['player_id'] for item in instance.bindings.values()}) != len(instance.bindings):
            raise ValueError('Checkpoint contains conflicting confirmed associations.')
        if any(item['epoch'] != instance.epoch for item in instance.bindings.values()):
            raise ValueError('Local associations cannot survive a continuity epoch change.')
        if instance.last_result is not None and (not isinstance(instance.last_result, dict)
                or instance.last_result.get('analysis_id') != analysis_id
                or len(instance.last_result.get('observations', [])) > instance.settings.max_observations):
            raise ValueError('Invalid checkpoint replay result.')
        for command_id, command in instance.commands.items():
            validated = IdentityAnchor(**command['payload'])
            event = command['event']
            expected_id = 'identity_anchor_' + _digest([analysis_id, command_id])[:32]
            if (validated.anchor_id != command_id or event.get('event_id') != expected_id
                    or event.get('analysis_id') != analysis_id or event.get('anchor') != command['payload']
                    or type(event.get('revision')) is not int or not 1 <= event['revision'] <= instance.revision):
                raise ValueError('Invalid checkpoint correction provenance.')
        return instance

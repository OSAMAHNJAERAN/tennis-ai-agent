"""Owned physical events and selected image boxes for corrected shot linking.

These views consume a pending rule rebuild, never original shot links or the
mutable analysis-root aliases. They retain only the current record in Python;
duration-long inputs and reconstructed native events live in owned SQLite.
Court coordinates on contacts are withheld: a court-plane mapping of an
airborne ball is not a recovered physical contact position. Bounce positions
come exclusively from this rebuild's newly accepted line calls.
"""
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
import hashlib
from itertools import zip_longest
import json
import math
import operator
from pathlib import Path

from src.api.analysis_generation import GenerationSnapshot, resolve_generation
from src.api.result_storage import DiskSeries, ResultStorage
from src.utils.bbox_utils import BBox
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.point_analysis_sequence import PointAnalysisSequence


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected shot sources: ' + message)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, DiskSeries)):
        return [_plain(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pair(value, name):
    value = _plain(value)
    _require(type(value) is list and len(value) == 2 and all(_number(item) for item in value),
             name + ' must contain two finite coordinates')
    return tuple(value)


def _scratch(directory, analysis_directory):
    path = Path(directory).absolute()
    _require('..' not in path.parts, 'scratch parent traversal is forbidden')
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
                 'scratch symlinks and junctions are forbidden')
    _require(path.is_dir(), 'private scratch directory must already exist')
    path = path.resolve()
    _require(not path.is_relative_to(analysis_directory)
             and not analysis_directory.is_relative_to(path),
             'scratch must be separate from the immutable analysis directory')
    return path


@contextmanager
def _ephemeral_storage(directory):
    """Close then remove only this adapter's unpublished parser diagnostics.

    Generic ResultStorage deliberately retains failed parsing diagnostics.
    These private reconstructed inputs are reproducible from the retained base
    and command; this adapter instead owns ephemeral files on every exit.
    """
    storage = ResultStorage(directory)
    try:
        with storage:
            yield storage
    except BaseException as primary:
        path = storage.path
        if path is not None:
            try:
                _require(path.parent == directory and path.name.startswith('.result_storage_')
                         and path.suffix == '.sqlite3', 'parser cleanup escaped its owned directory')
                for parent in (directory, *directory.parents):
                    _require(not parent.is_symlink()
                             and not getattr(parent, 'is_junction', lambda: False)(),
                             'parser cleanup parent changed to a link')
                for suffix in ('-journal', '-wal', '-shm', ''):
                    owned = Path(str(path) + suffix)
                    _require(not owned.is_symlink() and owned.resolve().parent == directory,
                             'parser cleanup escaped its owned directory')
                    owned.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                primary.add_note('Corrected shot parser cleanup also failed: ' + str(cleanup_error))
        raise


def _box(value):
    if value is None:
        return None
    value = _plain(value)
    _require(type(value) is dict and set(value) == {'bbox', 'source_track_id', 'confidence'},
             'selected player box fields differ')
    coordinates = value['bbox']
    _require(type(coordinates) is list and len(coordinates) == 4
             and all(_number(item) for item in coordinates)
             and coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1],
             'selected player box requires finite positive extent')
    confidence, track = value['confidence'], value['source_track_id']
    _require(_number(confidence) and 0 <= confidence <= 1,
             'selected player confidence is invalid')
    _require(track is None or type(track) is int and track >= 0,
             'selected player track ID is invalid')
    # The native detection export does not retain class/pose/ground-point
    # attributes. Do not synthesize them from filtered full-frame pose exports.
    return BBox(*coordinates, confidence=confidence, track_id=track)


class _Boxes(Sequence):
    def __init__(self, owner, role, indices=None):
        self.owner, self.role = owner, role
        self.indices = range(owner.frame_count) if indices is None else indices

    def __len__(self):
        self.owner._require_open()
        return len(self.indices)

    def __getitem__(self, index):
        self.owner._require_open()
        if isinstance(index, slice):
            return _Boxes(self.owner, self.role, self.indices[index])
        if isinstance(index, bool):
            raise TypeError('Player box index must be an integer')
        index = operator.index(index)
        return _box(self.owner._frames[self.indices[index]]['player_' + str(self.role)])


class CorrectedShotSources:
    """The owning context must outlive every event and image-box consumer."""

    def __init__(self, event_owner, frames, frame_count, audit, *, image_size):
        self.event_owner = event_owner
        self.events = event_owner.events
        self._frames, self.frame_count = frames, frame_count
        self.image_size = image_size
        self.audit, self._open = audit, True
        self.player1_boxes, self.player2_boxes = _Boxes(self, 1), _Boxes(self, 2)

    def _require_open(self):
        if not self._open:
            raise RuntimeError('Corrected shot sources are closed')


def _validate_detections(document, rebuilt):
    snapshot, source_frames = rebuilt.snapshot, rebuilt.sources.source_frames
    _require(type(document) is dict and document.get('schema_version') == '1.0'
             and type(document.get('metadata')) is dict, 'unsupported detections envelope')
    metadata = document['metadata']
    for value in (document, metadata):
        if 'analysis_id' in value:
            _require(value['analysis_id'] == snapshot.analysis_id, 'detection analysis differs')
        if 'analysis_revision' in value:
            _require(type(value['analysis_revision']) is int
                     and value['analysis_revision'] == snapshot.analysis_revision,
                     'detection revision differs')
        if 'source_sha256' in value:
            _require(value['source_sha256'] == source_frames.source_sha256, 'detection source differs')
    _require(type(metadata.get('frames')) is int and metadata['frames'] == snapshot.frames
             and _number(metadata.get('fps')) and metadata['fps'] > 0
             and metadata['fps'] == rebuilt.sources.fps
             and all(type(metadata.get(key)) is int and metadata[key] > 0 for key in ('width', 'height')),
             'detection frame count, clock or dimensions differ')
    frames = document.get('frames')
    _require(isinstance(frames, (list, DiskSeries)) and len(frames) == snapshot.frames,
             'selected player frame count differs')
    for index, frame in enumerate(frames):
        _require(type(frame) is dict and type(frame.get('frame_index')) is int
                 and frame['frame_index'] == index and 'player_1' in frame and 'player_2' in frame
                 and type(frame.get('court_registration_valid')) is bool
                 and _canonical(frame.get('camera_interval')) == _canonical(source_frames[index][0]),
                 'selected player/source frame join differs')
        if 'match_identity' in frame:
            _require(_canonical(frame['match_identity']) ==
                     _canonical(source_frames[index][1].get('observations')),
                     'selected player identity observation join differs')
        _box(frame['player_1'])
        _box(frame['player_2'])
    return frames


def _append_physical_events(rebuilt, event_owner):
    # Use the same strict primitive/context decoder as rule replay. It returns
    # an untyped physical record, which is materialized into the native event
    # dataclass only after its fresh accepted rule input has been checked.
    from src.scoring.rule_replay_source import _source_event
    from src.events.event_detector import EventType, TennisEvent
    owner, engine = rebuilt.owner, rebuilt.engine
    records = owner.rows('replay_inputs')
    _require(len(records) == len(owner.rows('transitions')) == len(engine._inputs),
             'accepted rule/source/transition counts differ')
    _require(rebuilt.authoritative_events_enabled or not len(records),
             'authority-off rebuild contains accepted source inputs')
    audit = {'schema_version': 1, 'accepted_input_count': 0, 'physical_event_count': 0,
        'reviewed_signal_count': 0, 'bounce_positions_available': 0,
        'bounce_positions_unavailable': 0, 'nonbounce_positions_withheld': 0,
        'authoritative_events_enabled': rebuilt.authoritative_events_enabled,
        'contact_court_position_policy': 'WITHHELD_AIRBORNE_BALL_IS_NOT_COURT_PLANE_CONTACT',
        'bounce_court_position_source': 'NEW_ACCEPTED_CORRECTED_RULE_LINE_CALL',
        'player_box_source': 'SEALED_SELECTED_IMAGE_DETECTIONS',
        'pose_or_ground_evidence_reconstructed': False,
        'old_shot_links_reused': False, 'independent_accuracy_validated': False}
    sentinel, previous = object(), None
    inputs, rows = engine.iter_rule_inputs(), iter(records)
    try:
        for row, accepted in zip_longest(rows, inputs, fillvalue=sentinel):
            _require(row is not sentinel and accepted is not sentinel,
                     'accepted rule/source sequence counts differ')
            _require(type(row) is dict and set(row) == {'source_input', 'input_fingerprint',
                'rule_environment', 'scoring_input', 'accepted_transition'},
                'accepted replay source fields differ')
            source = row['source_input']
            _require(type(source) is dict and hashlib.sha256(_canonical(source).encode()).hexdigest() ==
                     row['input_fingerprint'], 'source input fingerprint differs')
            event, context = _source_event(source, rebuilt.sources.source_frames,
                owner.analysis_id, owner.source_fingerprint)
            ident = event.event_id
            _require(_canonical(row['scoring_input']) == _canonical(accepted)
                     and type(accepted.get('event_id')) is int and accepted['event_id'] == ident
                     and _number(accepted.get('timestamp_seconds'))
                     and accepted['timestamp_seconds'] == event.timestamp_s,
                     'new accepted rule input differs from the rule journal')
            transition = owner.event(ident)
            _require(transition is not None
                     and _canonical(row['accepted_transition']) == _canonical(transition)
                     and transition.get('input_fingerprint') == row['input_fingerprint']
                     and type(transition.get('frame')) is int and transition['frame'] == event.frame_index
                     and transition.get('timestamp_s') == event.timestamp_s,
                     'accepted transition differs from its source input')
            _require(_number(event.timestamp_s) and event.timestamp_s >= 0,
                     'physical event timestamp is invalid')
            order = (event.frame_index, event.timestamp_s, ident)
            _require(previous is None or order > previous and order[1] >= previous[1],
                     'source input order differs')
            previous = order
            audit['accepted_input_count'] += 1
            if source['reviewed_evidence_ref'] is not None:
                audit['reviewed_signal_count'] += 1
                continue
            _require(type(event.evidence) is dict, 'native physical event evidence must be an object')
            _require(_number(event.confidence) and 0 <= event.confidence <= 1
                     and type(event.trajectory_state) is str,
                     'physical event confidence or trajectory state is invalid')
            ball_position = _pair(event.ball_position_px, 'physical ball position')
            position = None
            if event.event_type.value == 'BOUNCE':
                call = accepted.get('line_call')
                _require(type(call) is dict and type(call.get('event_id')) is int
                         and call['event_id'] == ident and type(call.get('bounce_frame')) is int
                         and call['bounce_frame'] == event.frame_index,
                         'new bounce call belongs to another event/frame')
                if call.get('bounce_position_m') is not None:
                    environment = row['rule_environment']
                    _require(type(environment) is dict
                             and environment.get('geometry_available') is True and context.binding_current,
                             'bounce position lacks current accepted geometry')
                    position = _pair(call['bounce_position_m'], 'new accepted bounce position')
                audit['bounce_positions_available' if position is not None
                      else 'bounce_positions_unavailable'] += 1
            else:
                audit['nonbounce_positions_withheld'] += 1
            event_owner.events.append(TennisEvent(event_id=ident, event_type=EventType(event.event_type.value),
                frame_index=event.frame_index, timestamp_s=event.timestamp_s, player_id=event.player_id,
                ball_position_px=ball_position, court_position_m=position, confidence=event.confidence,
                trajectory_state=event.trajectory_state, evidence=event.evidence))
            audit['physical_event_count'] += 1
    finally:
        rows.close()
        inputs.close()
    event_owner.flush()
    return audit


@contextmanager
def corrected_shot_sources(rebuilt, directory):
    """Yield native physical events and selected boxes for a pending revision.

    ``rebuilt`` is the live result of ``rebuild_corrected_rule_timeline``. The
    caller must keep that context open. This adapter owns only fresh scratch
    storage; it neither changes the pending rule ledger nor publishes files.
    """
    snapshot, owner, sources = rebuilt.snapshot, rebuilt.owner, rebuilt.sources
    _require(isinstance(snapshot, GenerationSnapshot) and isinstance(owner, PointAnalysisSequence),
             'a committed generation and native pending owner are required')
    owner._open()
    sources._require_open()
    _require(not owner.sealed and type(rebuilt.authoritative_events_enabled) is bool
             and owner.analysis_id == snapshot.analysis_id
             and type(owner.revision) is int and owner.revision == snapshot.analysis_revision + 1
             and owner.source_fingerprint == sources.source_frames.source_sha256
             and sources.snapshot == snapshot and rebuilt.base_generation == snapshot.generation
             and len(sources.source_frames) == snapshot.frames
             and sources.source_frames.analysis_id == snapshot.analysis_id
             and sources.source_frames.analysis_revision == snapshot.analysis_revision
             and sources.source_frames.artifact_sha256 == snapshot.file_sha256('native_rule_replay.json')
             and type(getattr(rebuilt.engine, 'revision', None)) is int and rebuilt.engine.revision == 0
             and not len(owner.member_rows()) and not len(owner.rows('replay_shots')),
             'pending source/revision is stale, sealed or already linked')
    analysis_directory = snapshot._root.parent.resolve()
    _require(resolve_generation(analysis_directory) == snapshot,
             'current generation differs from the pending base')
    directory = _scratch(directory, analysis_directory)
    path = snapshot.resolve_file('detections.json')
    _require(path is not None, 'sealed selected player detections are absent')
    with ExitStack() as stack:
        storage = stack.enter_context(_ephemeral_storage(directory))
        def finite(token):
            raise ValueError('Corrected shot sources require finite JSON: ' + token)
        detections = storage.read_json(path, parse_constant=finite)
        _require(snapshot.resolve_file('detections.json') == path,
                 'sealed selected player detections changed during parsing')
        frames = _validate_detections(detections, rebuilt)
        events = stack.enter_context(EventAnalysisSequence(directory))
        audit = _append_physical_events(rebuilt, events)
        _require(resolve_generation(analysis_directory) == snapshot,
                 'current generation changed during shot source preparation')
        result = CorrectedShotSources(events, frames, snapshot.frames, audit,
            image_size=(detections['metadata']['width'], detections['metadata']['height']))
        try:
            yield result
        finally:
            result._open = False

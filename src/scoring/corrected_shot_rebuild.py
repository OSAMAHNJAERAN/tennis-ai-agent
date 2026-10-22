"""Private fresh shot linking and native analytics for an owned correction.

This stage finishes the rebuilt point timeline. It never publishes a result,
renders media, or gives a trusted crop-feature callback model qualification.
The inference worker must bind that callback to its already selected player
model before this can become part of whole-analysis correction publication.
"""
from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import json
import math
import operator
from types import SimpleNamespace

from src.scoring.corrected_rule_rebuild import _scratch, MAX_REQUEST_BYTES


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected shot rebuild: ' + message)


def _current(rebuilt):
    from src.api.analysis_generation import resolve_generation
    snapshot = rebuilt.snapshot
    current = resolve_generation(snapshot.files_directory.parent.parent)
    _require(current is not None and current.generation == rebuilt.base_generation
             and current.generation == snapshot.generation,
             'current generation changed during fresh shot processing')
    # Revalidate immutable source/config assets, including after a callback.
    snapshot.resolve_file('source.mp4')
    snapshot.resolve_file('run_config.yaml')


def _settings(snapshot):
    import yaml
    from src.shot_analysis.shot_types import PlayerHandedness
    path = snapshot.resolve_file('run_config.yaml')
    _require(path is not None and path.stat().st_size <= MAX_REQUEST_BYTES,
             'sealed bounded run configuration is required')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    _require(type(config) is dict, 'run configuration must be an object')
    sections = {}
    for name in ('pose_estimation', 'shot_classification', 'player_detection', 'model_system'):
        value = config.get(name, {})
        _require(type(value) is dict, 'invalid configuration section: ' + name)
        sections[name] = value
    pose, shot = sections['pose_estimation'], sections['shot_classification']
    enabled = pose.get('enabled', True)
    _require(type(enabled) is bool, 'pose enabled must be boolean')
    thresholds = {}
    for name, value in (('min_pose_confidence', pose.get('min_pose_confidence', .35)),
                        ('ambiguity_threshold', shot.get('ambiguity_threshold', .15))):
        _require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1,
                 'invalid classifier threshold: ' + name)
        thresholds[name] = value
    maps = {}
    for name in ('players_handedness', 'court_orientation_sign'):
        raw = shot.get(name, {})
        _require(type(raw) is dict and len(raw) <= 2, 'invalid classifier map: ' + name)
        values = {}
        for role, value in raw.items():
            _require((type(role) is int and role in (1, 2))
                     or (type(role) is str and role in ('1', '2')), 'invalid physical player role')
            role = int(role)
            _require(role not in values, 'duplicate physical player role')
            if name == 'players_handedness':
                _require(type(value) is str, 'handedness must use native enum values')
                value = PlayerHandedness(value)
            else:
                _require(type(value) in (int, float) and value in (-1, 1),
                         'court orientation must be an explicit signed unit')
                value = float(value)
            values[role] = value
        maps[name] = values
    return sections, {**thresholds, **maps, 'pose_enabled': enabled,
        # The existing Phase6 constructor does not forward the YAML crop/window
        # fields. These are the actual extractor defaults used by that producer.
        'effective_window_half_size': 3, 'effective_crop_margin': .65}


class _LazyFrames(Sequence):
    """Open the sealed normalized video only when a crop actually reads it."""
    def __init__(self, rebuilt, stack, *, image_size):
        self.rebuilt, self.stack = rebuilt, stack
        self.image_size = image_size
        self.sequence = None
        self.open = True

    def __len__(self):
        if not self.open:
            raise RuntimeError('Corrected crop frames are closed')
        return self.rebuilt.snapshot.frames

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Read one bounded crop frame by global index')
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.sequence is None:
            from src.utils.video_frame_sequence import VideoFrameSequence
            sequence = self.stack.enter_context(VideoFrameSequence(
                str(self.rebuilt.snapshot.resolve_file('source.mp4')), cache_size=8))
            _require(len(sequence) == len(self)
                     and (sequence.metadata.width, sequence.metadata.height) == self.image_size
                     and math.isclose(sequence.metadata.fps, self.rebuilt.sources.fps,
                                      rel_tol=1e-6, abs_tol=1e-6),
                     'decoded normalized video differs from retained frame clock or image dimensions')
            self.sequence = sequence
        return self.sequence[index]


class _WindowBoxes(Sequence):
    """At most seven eligible indices; do not crop across source/identity cuts."""
    def __init__(self, boxes, source_frames, hit, radius):
        self.boxes = boxes
        self.allowed = set()
        hit_box = boxes[hit]
        hit_track = getattr(hit_box, 'track_id', None)
        interval, identity = source_frames[hit][0], source_frames[hit][1]
        def identity_signature(value):
            return (value.get('analysis_id'), value.get('epoch'), value.get('revision'),
                    tuple((item.get('track_key'), item.get('player_id'), item.get('state'))
                          for item in value.get('observations', ())))
        identity_key = identity_signature(identity)
        for frame in range(max(0, hit-radius), min(len(boxes), hit+radius+1)):
            current, observed = source_frames[frame][0], source_frames[frame][1]
            box = boxes[frame]
            # A local role can select another person without changing the
            # complete identity observation list. Do not average their crops.
            # Unknown local tracking supports the event frame alone.
            if frame != hit and (hit_track is None or box is None
                                 or getattr(box, 'track_id', None) != hit_track):
                continue
            if (current.get('analysis_allowed') is not True
                    or current.get('segment_id') != interval.get('segment_id')
                    or current.get('anchor_frame_index') != interval.get('anchor_frame_index')
                    or identity_signature(observed) != identity_key):
                continue
            contiguous = True
            for step in range(min(hit, frame)+1, max(hit, frame)+1):
                before, after = source_frames[step-1][0], source_frames[step][0]
                step_box = boxes[step]
                if (after.get('analysis_allowed') is not True
                        or after.get('segment_id') != interval.get('segment_id')
                        or after.get('anchor_frame_index') != interval.get('anchor_frame_index')
                        or identity_signature(source_frames[step][1]) != identity_key
                        or step_box is None or (step != hit
                            and getattr(step_box, 'track_id', None) != hit_track)
                        or after.get('measurement_transition_allowed') is not True
                        or after.get('source_frame_index') != before.get('source_frame_index')+1
                        or after.get('source_time_base') != before.get('source_time_base')
                        or int(after['source_pts']) <= int(before['source_pts'])):
                    contiguous = False
                    break
            if contiguous:
                self.allowed.add(frame)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Read one crop box by global index')
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.boxes[index] if index in self.allowed else None


class _CropFrames(Sequence):
    def __init__(self, frames, boxes):
        self.frames, self.boxes = frames, boxes

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if type(index) is not int or index not in self.boxes.allowed:
            raise ValueError('Crop frame is outside this continuous hit window')
        return self.frames[index]


@dataclass(frozen=True)
class CorrectedPoseRequest:
    """Trusted local callback inputs; returned features remain unqualified."""
    analysis_id: str
    analysis_revision: int
    base_generation: str
    source_sha256: str
    run_config_sha256: str
    selected_player_model_path: str
    frames: Sequence
    player_boxes: Sequence
    hit_frame: int
    player_id: int
    handedness: object
    court_orientation_sign: float
    window_half_size: int = 3
    crop_margin: float = .65


class _RequiredCropFeatures:
    def __init__(self, rebuilt, frames, sections, provider):
        self.rebuilt, self.frames = rebuilt, frames
        self.sections, self.provider = sections, provider
        self.calls = 0

    def extract_hit_window_features(self, *, frames, player_boxes, hit_frame, player_id,
                                    handedness, court_orientation_sign):
        if self.provider is None:
            raise RuntimeError('Corrected live hit requires the configured selected-player '
                               'crop pose provider; classification cannot skip that branch')
        _require(callable(self.provider), 'crop pose provider must be a trusted local callable')
        system, pose = self.sections['model_system'], self.sections['pose_estimation']
        path = self.sections['player_detection'].get('model')
        _require(system.get('family') == 'yolo26' and system.get('vision_model_count') == 3
                 and pose.get('source') == 'player_model' and type(path) is str and bool(path)
                 and pose.get('model_path') in (None, path),
                 'crop features must use the selected three-model player role')
        _require(frames is self.frames, 'crop frames differ from the sealed source')
        boxes = _WindowBoxes(player_boxes, self.rebuilt.sources.source_frames, hit_frame, 3)
        request = CorrectedPoseRequest(self.rebuilt.snapshot.analysis_id,
            self.rebuilt.owner.revision, self.rebuilt.base_generation,
            self.rebuilt.owner.source_fingerprint,
            self.rebuilt.snapshot.file_sha256('run_config.yaml'), path,
            _CropFrames(frames, boxes), boxes, hit_frame, player_id,
            handedness, court_orientation_sign)
        self.calls += 1
        value = self.provider(request)
        _require(type(value) is dict, 'crop provider must return native feature fields')
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)
        _require(len(encoded.encode('utf-8')) <= 16384, 'crop feature response exceeds bounded size')
        value = json.loads(encoded)
        _require(type(value.get('valid')) is bool, 'crop feature validity must be boolean')
        if value['valid']:
            for name in ('normalized_wrist_displacement', 'mean_pose_confidence'):
                number = value.get(name)
                _require(type(number) in (int, float) and math.isfinite(number),
                         'invalid crop feature: ' + name)
            _require(0 <= value['mean_pose_confidence'] <= 1, 'invalid pose confidence')
        _current(self.rebuilt)
        return value


@contextmanager
def relink_corrected_shots(rebuilt, scratch, *, pose_provider=None):
    """Finish one private corrected timeline with newly linked physical shots.

    A configured live pose branch requires a trusted local callable accepting
    ``CorrectedPoseRequest``. It should reuse the selected player model through
    ``PoseFeatureExtractor.extract_hit_window_features`` with the supplied
    arguments. This seam does not verify in-memory checkpoint identity; the
    eventual inference worker must do that before publication. Without that
    provider the reachable branch fails explicitly, rather than falling back
    to geometry merely because a model or source frames were omitted.
    """
    from src.scoring.corrected_shot_sources import corrected_shot_sources
    from src.scoring.point_timeline import native_match_analytics
    from src.shot_analysis.shot_classifier import TennisShotClassifier
    from src.shot_analysis.shot_linker import TennisShotLinker
    from src.utils.shot_evidence_sequence import ShotEvidenceSequence
    snapshot, owner = rebuilt.snapshot, rebuilt.owner
    scratch = _scratch(scratch, snapshot.files_directory.parent.parent.resolve())
    _current(rebuilt)
    _require(rebuilt.timeline.owner is owner and not owner.sealed
             and owner.analysis_id == snapshot.analysis_id
             and owner.revision == snapshot.analysis_revision+1
             and owner.source_fingerprint == rebuilt.source_frames.source_sha256
             and rebuilt.engine.revision == 0,
             'fresh pending correction timeline is required')
    _require(not len(owner.member_rows()) and not len(owner.rows('replay_shots')),
             'correction shots were already attached')
    sections, settings = _settings(snapshot)
    with ExitStack() as stack:
        source = stack.enter_context(corrected_shot_sources(rebuilt, scratch))
        frames = _LazyFrames(rebuilt, stack, image_size=source.image_size)
        stack.callback(setattr, frames, 'open', False)
        extractor = _RequiredCropFeatures(rebuilt, frames, sections, pose_provider)
        classifier = TennisShotClassifier(
            pose_extractor=extractor if settings['pose_enabled'] else None,
            min_pose_confidence=settings['min_pose_confidence'],
            ambiguity_threshold=settings['ambiguity_threshold'],
            handedness_map=settings['players_handedness'],
            court_orientation_map=settings['court_orientation_sign'])
        shots = stack.enter_context(ShotEvidenceSequence(scratch))
        TennisShotLinker(classifier).link_shots(source.events, rebuilt.sources.ball_points,
            source.player1_boxes, source.player2_boxes, raw_frames=frames,
            dead_event_ids=owner.dead_event_ids(), result_output=shots,
            event_output=source.event_owner)
        rebuilt.timeline.attach_shots(shots)
        shots.seal()
        rebuilt.timeline.finish()
        analytics = native_match_analytics(owner, shots, rebuilt.engine.get_current_state(),
                                           authoritative=rebuilt.authoritative_events_enabled)
        _current(rebuilt)
        audit = {'schema_version': 1, 'status': 'PRIVATE_PENDING_FORECASTS_EXPORTS_AND_RENDER',
            'analysis_id': owner.analysis_id, 'analysis_revision': owner.revision,
            'base_generation': rebuilt.base_generation,
            'source_sha256': owner.source_fingerprint,
            'run_config_sha256': snapshot.file_sha256('run_config.yaml'),
            'physical_sources': dict(source.audit),
            'shot_count': len(shots), 'pose_provider_calls': extractor.calls,
            'source_video_opened_for_crops': frames.sequence is not None,
            'selected_checkpoint_execution_verified': False,
            'crop_provider_scope': 'TRUSTED_LOCAL_CALLBACK; WORKER_MODEL_IDENTITY_ADMISSION_PENDING',
            'effective_pose_window_half_size': 3, 'effective_pose_crop_margin': .65,
            'old_shots_reused': False, 'human_review_dependent': True,
            'independent_accuracy_validated': False, 'published': False}
        yield SimpleNamespace(shots=shots, match_analytics=analytics, events=source.events,
            event_owner=source.event_owner, owner=owner, timeline=rebuilt.timeline,
            engine=rebuilt.engine, snapshot=snapshot, base_generation=rebuilt.base_generation,
            audit=audit)
        _current(rebuilt)

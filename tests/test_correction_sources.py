"""Sealed synthetic observations; no model inference or accuracy qualification."""
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
import builtins
import hashlib
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import tracemalloc

import pytest

from src.api.analysis_generation import publish_analysis_generation, resolve_generation
from src.api.result_storage import ResultStorage, DiskSeries
from src.scoring.correction_sources import correction_sources
from src.scoring.point_timeline import ReviewedParticipantBinding, candidate_event_context


SOURCE_SHA = 'a' * 64
CAPACITY_OBSERVATIONS = []


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, DiskSeries)):
        return [plain(item) for item in value]
    return value


def write_observation_assets(directory, source_frames, *, homography=None, positions=None, fps=30.0):
    """Three actual native-shaped assets reusable by correction integration tests.

    The caller supplies indexed ``{0: interval, 1: identity}`` source frames
    from its rule fixture, adds native_rule_replay.json, then publishes them
    together through the real immutable generation publisher. These are purely
    synthetic coordinates and calibration, without any physical qualification.
    """
    directory = Path(directory)
    homography = homography if homography is not None else [[.01, 0., 0.], [0., .01, 0.], [0., 0., 1.]]
    detections, camera = [], []
    for index in range(len(source_frames)):
        interval = plain(source_frames[index][0])
        usable = interval.get('analysis_allowed') is True
        position = [300.1234567890123, 800.9876543210987] if positions is None else positions[index]
        detections.append({'frame_index': index, 'court_registration_valid': usable,
            'player_1': None, 'player_2': None, 'camera_interval': interval,
            'ball': {'position_px': position, 'confidence': .98765432101234,
                     'state': 'MISSING' if position is None else 'DETECTED'}})
        camera.append({'frame_index': index, 'is_valid': usable,
            'rejection_reason': None if usable else 'SYNTHETIC_UNAVAILABLE',
            'image_to_court': homography if usable else None,
            'anchor_to_frame_px': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] if usable else None,
            'camera_interval': interval})
    values = {
        'detections.json': {'schema_version': '1.0',
            'metadata': {'frames': len(source_frames), 'fps': fps, 'width': 1920, 'height': 1080},
            'frames': detections},
        'camera_registration.json': {'schema_version': '1.0', 'enabled': True,
            'coordinate_semantics': 'COURT_PLANE_REGISTRATION_NOT_3D_BALL_GEOMETRY',
            'independent_accuracy_validated': False, 'frames': camera},
        'court_geometry.json': {'schema_version': '2.0', 'is_valid': True,
            'rejection_reason': None, 'homography_matrix': homography,
            'independent_accuracy_validated': False},
    }
    files = {}
    for name, value in values.items():
        path = directory / name
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        files[name] = path
    return files


class SyntheticSourceFrames(Sequence):
    """Test stand-in for a separately validated native rule-source decoder."""

    def __init__(self, frame_count=11):
        self.analysis_id, self.analysis_revision, self.source_sha256 = 'fixture', 2, SOURCE_SHA
        self.artifact_sha256 = None
        self.rows = []
        for index in range(frame_count):
            # Source clock intentionally starts at 10 s; playback starts at 0.
            interval = {'analysis_id': self.analysis_id, 'frame_index': index,
                'timestamp_seconds': index / 30., 'source_pts': str(900000 + index * 3000),
                'source_time_base': '1/90000', 'source_frame_index': index,
                'source_timestamp_seconds': float(Fraction(900000 + index * 3000, 90000)),
                'segment_id': 'camera_fixture_0', 'anchor_frame_index': 0,
                'analysis_allowed': True, 'measurement_transition_allowed': index > 0,
                'reanchor_required': False, 'review_evidence_id': 'synthetic-normal-play'}
            identity = {'analysis_id': self.analysis_id, 'revision': 0, 'epoch': 0,
                'observations': [{'track_key': 'fixture-track', 'state': 'CONFIRMED', 'player_id': 'A'}]}
            self.rows.append({0: interval, 1: identity})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def current_context(source, frame=5):
    interval = source[frame][0]
    binding = ReviewedParticipantBinding(frame, interval['segment_id'], 1, 1, 'A', 'fixture-track',
        'NEAR_COURT', 0, 'synthetic-reviewed-calibration', interval['anchor_frame_index'],
        'synthetic-reviewed-identity', analysis_id=source.analysis_id, source_fingerprint=source.source_sha256,
        source_frame_index=interval['source_frame_index'], source_pts=interval['source_pts'],
        source_time_base=interval['source_time_base'], identity_epoch=0,
        participant_mapping_ref='synthetic-mapping', geometry_qualification_ref='synthetic-geometry')
    return candidate_event_context(SimpleNamespace(frame_index=frame), source, lambda *_: binding,
        analysis_id=source.analysis_id, source_fingerprint=source.source_sha256)


def sealed_fixture(directory, *, source=None, mutate=None, omit=()):
    directory.mkdir()
    source = source if source is not None else SyntheticSourceFrames()
    working = directory / 'working'
    working.mkdir()
    files = write_observation_assets(working, source)
    replay = working / 'native_rule_replay.json'
    replay.write_text('{"scope":"SYNTHETIC_EXTERNAL_DECODER_STAND_IN"}', encoding='utf-8')
    source.artifact_sha256 = hashlib.sha256(replay.read_bytes()).hexdigest()
    files['native_rule_replay.json'] = replay
    if mutate is not None:
        mutate(files)
    for name in omit:
        files.pop(name)
    root = directory / '.result'
    root.mkdir()
    with ResultStorage(root) as storage:
        publish_analysis_generation(storage,
            {'summary': {'id': source.analysis_id, 'frames': len(source)},
             'analysisRevision': source.analysis_revision}, directory,
            files=files, analysis_revision=source.analysis_revision)
    scratch = directory / 'scratch'
    scratch.mkdir()
    return resolve_generation(directory), source, scratch


def change(files, name, mutate):
    value = json.loads(files[name].read_text(encoding='utf-8'))
    mutate(value)
    files[name].write_text(json.dumps(value), encoding='utf-8')


@contextmanager
def opened(fixture, **kwargs):
    snapshot, source, scratch = fixture
    options = {'source_sha256': SOURCE_SHA, 'source_frames': source}
    options.update(kwargs)
    with correction_sources(snapshot, scratch, **options) as sources:
        yield sources


def test_actual_typed_unrounded_points_and_two_distinct_clocks(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run')
    with opened(fixture) as sources:
        from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
        point = sources.ball_points[5]
        assert isinstance(point, TemporalBallPoint) and point.state is BallState.DETECTED
        assert point.x_px == 300.1234567890123 and point.y_px == 800.9876543210987
        assert point.confidence == .98765432101234 and point.timestamp_seconds == 5 / 30.
        assert point.timestamp_seconds != float(Fraction(fixture[1][5][0]['source_pts']) / 90000)
        assert point.court_x_m is point.court_y_m is point.speed_kmh is point.velocity_px_per_sec is None
        assert sources.fps == 30.
        assert sources.ball_points[-1].frame_index == 10
        point.x_px = -100.
        assert sources.ball_points[5].x_px == 300.1234567890123
        retained = sources.ball_points[:]
        assert len(sources._detections['frames']) == 11
    assert not list(fixture[2].glob('.result_storage_*.sqlite3*'))
    with pytest.raises(RuntimeError, match='closed'):
        retained[0]


def test_current_homography_is_independent_and_readonly(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run', mutate=lambda files: change(files, 'camera_registration.json',
        lambda data: data['frames'][5].__setitem__('image_to_court', [[.02, 0., 2.], [0., .03, 3.], [0., 0., 1.]])))
    with opened(fixture) as sources:
        geometry = sources.event_geometry(5, current_context(fixture[1]))
        assert geometry.usable and geometry.reason == 'CURRENT_REVIEW_BOUND_FRAME_REGISTRATION'
        assert geometry.homography_matrix.tolist() == [[.02, 0., 2.], [0., .03, 3.], [0., 0., 1.]]
        with pytest.raises(ValueError):
            geometry.homography_matrix[0, 0] = 100.


def test_reanchored_valid_current_geometry_does_not_use_invalid_initial_fit(tmp_path):
    source = SyntheticSourceFrames()
    for index in range(5, len(source)):
        source[index][0].update(segment_id='camera_fixture_1', anchor_frame_index=5,
                               measurement_transition_allowed=index > 5)
    def mutate(files):
        change(files, 'court_geometry.json', lambda data: data.update(
            is_valid=False, rejection_reason='INITIAL_FIT_REJECTED', homography_matrix=None))
        def reanchored(data):
            for row in data['frames'][5:]:
                row['image_to_court'] = [[.02, 0., 2.], [0., .03, 3.], [0., 0., 1.]]
        change(files, 'camera_registration.json', reanchored)
    fixture = sealed_fixture(tmp_path / 'run', source=source, mutate=mutate)
    with opened(fixture) as sources:
        geometry = sources.event_geometry(5, current_context(source))
        assert geometry.usable and geometry.homography_matrix[0, 2] == 2.
        points = sources.event_ball_points(5, current_context(source))
        assert points[4].x_px is None and points[5].x_px is not None and points[6].x_px is not None


def test_mask_stops_at_gap_and_bounds_slices_without_allocating_all_points(tmp_path):
    source = SyntheticSourceFrames()
    source[6][0]['measurement_transition_allowed'] = False
    fixture = sealed_fixture(tmp_path / 'run', source=source)
    with opened(fixture) as sources:
        points = sources.event_ball_points(5, current_context(source))
        assert len(points[:]) == len(source) and not isinstance(points[:], list)
        assert [point.x_px is not None for point in points[2:9]] == [True, True, True, True, False, False, False]
        assert points[1].x_px is None and points[9].x_px is None
        assert sources.ball_points[6].x_px is not None
        assert len(points[::-1]) == len(source) and points[::-1][5].frame_index == 5


@pytest.mark.parametrize('field,value', [
    ('source_pts', '1'), ('source_time_base', '1/1'), ('source_frame_index', 10),
    ('segment_id', 'another'), ('analysis_allowed', False), ('binding_audit_matches', False),
])
def test_context_mismatch_withholds_geometry_and_masks_points(tmp_path, field, value):
    fixture = sealed_fixture(tmp_path / 'run')
    context = replace(current_context(fixture[1]), **{field: value})
    with opened(fixture) as sources:
        assert not sources.event_geometry(5, context).usable
        assert all(point.x_px is None for point in sources.event_ball_points(5, context)[2:9])


def test_binding_claim_is_recomputed_against_actual_identity_audit(tmp_path):
    source = SyntheticSourceFrames()
    context = current_context(source)
    source[5][1]['observations'][0]['state'] = 'UNKNOWN'
    fixture = sealed_fixture(tmp_path / 'run', source=source)
    with opened(fixture) as sources:
        assert context.binding_current  # The copied claim alone is insufficient.
        assert not sources.event_geometry(5, context).usable


def test_explicit_camera_calibration_revision_must_match_review(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run', mutate=lambda files: change(files, 'camera_registration.json',
        lambda data: data['frames'][5].__setitem__('calibration_revision', 'another-reviewed-fit')))
    with opened(fixture) as sources:
        geometry = sources.event_geometry(5, current_context(fixture[1]))
        assert not geometry.usable and geometry.reason == 'CALIBRATION_REVISION_DIFFERS'


@pytest.mark.parametrize('missing', ['detections.json', 'camera_registration.json', 'court_geometry.json', 'native_rule_replay.json'])
def test_missing_sealed_assets_never_fall_back_to_working_root(tmp_path, missing):
    fixture = sealed_fixture(tmp_path / 'run', omit=(missing,))
    (fixture[0]._root.parent / missing).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        with opened(fixture):
            pytest.fail('missing sealed source was admitted')


@pytest.mark.parametrize('field,value', [
    ('analysis_id', 'other'), ('analysis_revision', 1), ('analysis_revision', True),
    ('source_sha256', 'b' * 64), ('artifact_sha256', 'b' * 64),
])
def test_decoder_provenance_must_bind_exact_committed_generation(tmp_path, field, value):
    fixture = sealed_fixture(tmp_path / 'run')
    setattr(fixture[1], field, value)
    with pytest.raises(ValueError, match='generation/revision/source'):
        with opened(fixture):
            pytest.fail('mixed decoder view was admitted')


@pytest.mark.parametrize('name,mutate', [
    ('detections.json', lambda d: d['metadata'].__setitem__('frames', 12)),
    ('detections.json', lambda d: d['metadata'].__setitem__('fps', True)),
    ('detections.json', lambda d: d.__setitem__('analysis_revision', True)),
    ('detections.json', lambda d: d['frames'].pop()),
    ('detections.json', lambda d: d['frames'][5].__setitem__('frame_index', 6)),
    ('detections.json', lambda d: d['frames'][5]['camera_interval'].__setitem__('source_pts', '1')),
    ('detections.json', lambda d: d['frames'][5]['ball'].__setitem__('position_px', [1., None])),
    ('detections.json', lambda d: d['frames'][5]['ball'].__setitem__('confidence', True)),
    ('detections.json', lambda d: d['frames'][5]['ball'].__setitem__('confidence', 1.01)),
    ('detections.json', lambda d: d['frames'][5]['ball'].__setitem__('state', 'UNKNOWN')),
    ('camera_registration.json', lambda d: d['frames'].pop()),
    ('camera_registration.json', lambda d: d['frames'][5].__setitem__('frame_index', True)),
    ('camera_registration.json', lambda d: d['frames'][5]['camera_interval'].__setitem__('anchor_frame_index', 3)),
    ('camera_registration.json', lambda d: d['frames'][5].__setitem__('image_to_court', [[0., 0., 0.]] * 3)),
    ('camera_registration.json', lambda d: d['frames'][5].__setitem__('is_valid', False)),
    ('court_geometry.json', lambda d: d.__setitem__('schema_version', '1.0')),
    ('court_geometry.json', lambda d: d.__setitem__('homography_matrix', [[1., 2.], [3., 4.]])),
])
def test_malformed_or_mixed_native_artifacts_fail_before_yield(tmp_path, name, mutate):
    fixture = sealed_fixture(tmp_path / 'run', mutate=lambda files: change(files, name, mutate))
    with pytest.raises(ValueError):
        with opened(fixture):
            pytest.fail('invalid artifact was admitted')


def test_missing_ball_and_unavailable_current_registration_abstain(tmp_path):
    source = SyntheticSourceFrames()
    source[6][0].update(analysis_allowed=False, measurement_transition_allowed=False)
    source[7][0]['measurement_transition_allowed'] = False
    def mutate(files):
        change(files, 'detections.json', lambda d: d['frames'][5]['ball'].update(position_px=None, state='MISSING'))
    fixture = sealed_fixture(tmp_path / 'run', source=source, mutate=mutate)
    with opened(fixture) as sources:
        geometry = sources.event_geometry(5, current_context(source))
        assert not geometry.usable and geometry.reason == 'BOUNCE_OBSERVATION_UNAVAILABLE'
        assert not sources.event_geometry(6, current_context(source, 6)).usable


def test_live_snapshots_reject_mutated_assets_even_after_first_resolve(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run')
    path = fixture[0].resolve_file('detections.json')
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        with opened(fixture):
            pytest.fail('mutated sealed content was admitted')


def test_source_mutations_after_open_never_rewrite_private_frozen_observations(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run')
    context = current_context(fixture[1])
    with opened(fixture) as sources:
        fixture[1][5][0]['timestamp_seconds'] = 100.
        fixture[1][5][1]['observations'][0]['state'] = 'UNKNOWN'
        assert sources.ball_points[5].timestamp_seconds == 5 / 30.
        assert sources.event_geometry(5, context).usable


def test_camera_disabled_static_fit_is_tolerated_but_cannot_enable_current_geometry(tmp_path):
    fixture = sealed_fixture(tmp_path / 'run', mutate=lambda files: change(files, 'camera_registration.json',
        lambda data: data.update(enabled=False, frames=[])))
    with opened(fixture) as sources:
        geometry = sources.event_geometry(5, current_context(fixture[1]))
        assert not geometry.usable and geometry.reason == 'CURRENT_FRAME_REGISTRATION_UNAVAILABLE'
        assert sources.ball_points[5].x_px is not None


def test_open_and_unavailable_context_need_no_numerical_or_model_imports(tmp_path, monkeypatch):
    fixture = sealed_fixture(tmp_path / 'run')
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'numpy', 'pandas', 'cv2', 'torch', 'ultralytics'}:
            raise AssertionError('Unavailable source path must stay numerical/model free: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with opened(fixture) as sources:
        assert sources.fps == 30. and len(sources.ball_points[:]) == 11
        assert not sources.event_geometry(5, None).usable


@pytest.mark.parametrize('values', [
    {'source_timestamp_seconds': 10.},
    {'local_reset': True, 'discontinuity_reasons': ['reviewed_cut']},
])
def test_internally_joined_but_contradictory_source_clock_or_transition_fails_closed(tmp_path, values):
    source = SyntheticSourceFrames()
    source[5][0].update(values)
    fixture = sealed_fixture(tmp_path / 'run', source=source)
    with pytest.raises(ValueError):
        with opened(fixture):
            pytest.fail('contradictory source clock/continuity was admitted')


@pytest.mark.parametrize('state', ['PREDICTED', 'INTERPOLATED', 'OCCLUDED', 'MISSING'])
def test_nonoptical_base_abstains_and_neighbours_cannot_influence_contact_fit(tmp_path, state):
    def mutate(files):
        def edit(data):
            data['frames'][5]['ball']['state'] = state
            data['frames'][7]['ball']['state'] = state
            if state == 'MISSING':
                data['frames'][5]['ball']['position_px'] = None
                data['frames'][7]['ball']['position_px'] = None
        change(files, 'detections.json', edit)
    fixture = sealed_fixture(tmp_path / 'run', mutate=mutate)
    with opened(fixture) as sources:
        assert sources.ball_points[5].state.value == state
        assert not sources.event_geometry(5, current_context(fixture[1])).usable
        points = sources.event_ball_points(6, current_context(fixture[1], 6))
        assert points[5].x_px is None and points[7].x_px is None
        assert points[6].x_px is not None


@pytest.mark.parametrize('kind', ['missing_confidence', 'registration_gap'])
def test_unqualified_neighbours_do_not_enter_contact_refinement(tmp_path, kind):
    def mutate(files):
        if kind == 'missing_confidence':
            change(files, 'detections.json', lambda d: d['frames'][6]['ball'].__setitem__('confidence', None))
        else:
            change(files, 'detections.json', lambda d: d['frames'][6].__setitem__('court_registration_valid', False))
            change(files, 'camera_registration.json', lambda d: d['frames'][6].update(
                is_valid=False, image_to_court=None, rejection_reason='SYNTHETIC_REGISTRATION_GAP'))
    fixture = sealed_fixture(tmp_path / 'run', mutate=mutate)
    with opened(fixture) as sources:
        points = sources.event_ball_points(5, current_context(fixture[1]))
        assert sources.ball_points[6].x_px is not None
        assert points[6].x_px is None and points[5].x_px is not None
        if kind == 'registration_gap':
            assert points[7].x_px is None
        else:
            assert points[7].x_px is not None


def test_real_contact_refiner_cannot_mix_camera_shifted_neighbour_pixels(tmp_path):
    def mutate(files):
        def pixels(data):
            for index, row in enumerate(data['frames']):
                row['ball']['position_px'] = [300. if index == 5 else 312., 800. - 2. * abs(index - 5)]
        def geometry(data):
            for index, row in enumerate(data['frames']):
                row['image_to_court'] = [[.01, 0., 0. if index == 5 else -.12], [0., .01, 0.], [0., 0., 1.]]
        change(files, 'detections.json', pixels)
        change(files, 'camera_registration.json', geometry)
    fixture = sealed_fixture(tmp_path / 'run', mutate=mutate)
    from src.line_calling.contact_refinement import BounceContactRefiner
    with opened(fixture) as sources:
        refiner = BounceContactRefiner(window_radius=3, fps=sources.fps)
        unsafe = refiner.refine_bounce_contact(5, sources.ball_points)
        assert unsafe.x_px == pytest.approx(312.)
        assert unsafe.refinement_method == 'PIECEWISE_IMPACT_INTERSECTION'
        actual = refiner.refine_bounce_contact(5, sources.event_ball_points(5, current_context(fixture[1])))
        assert actual.x_px == 300. and actual.refinement_method == 'RAW_DISCRETE_FRAME'
        assert sources.event_geometry(5, current_context(fixture[1])).usable


def test_positive_and_negative_scalar_homographies_retain_same_view_neighbours(tmp_path):
    def mutate(files):
        def geometry(data):
            for index, row in enumerate(data['frames']):
                scale = -2. if index % 2 else 3.
                row['image_to_court'] = [[value * scale for value in line] for line in row['image_to_court']]
        change(files, 'camera_registration.json', geometry)
    fixture = sealed_fixture(tmp_path / 'run', mutate=mutate)
    with opened(fixture) as sources:
        points = sources.event_ball_points(5, current_context(fixture[1]))
        assert all(point.x_px is not None for point in points[2:9])


@pytest.mark.parametrize('frame_count', [257, 4097])
def test_spilled_sources_and_lazy_slices_keep_duration_out_of_retained_memory(tmp_path, frame_count):
    fixture = sealed_fixture(tmp_path / 'run', source=SyntheticSourceFrames(frame_count))
    # Load actual numeric point classes before measuring the bounded parser.
    from src.tracking.temporal_ball_tracker import TemporalBallPoint
    gc.collect()
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        with opened(fixture) as sources:
            assert isinstance(sources._detections['frames'], DiskSeries)
            assert isinstance(sources._camera['frames'], DiskSeries)
            assert isinstance(sources.source_frames._rows, DiskSeries)
            view = sources.event_ball_points(frame_count // 2, current_context(fixture[1], frame_count // 2))[:]
            assert not isinstance(view, list) and len(view) == frame_count
            assert isinstance(view[frame_count // 2], TemporalBallPoint)
            assert view[0].x_px is None and view[-1].x_px is None
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            retained, peak_delta = current - before, peak - before
            CAPACITY_OBSERVATIONS.append({'frames': frame_count, 'retained_bytes': retained,
                                          'peak_bytes': peak_delta})
            assert retained < 2_000_000
            assert peak_delta < 5_000_000
    finally:
        tracemalloc.stop()

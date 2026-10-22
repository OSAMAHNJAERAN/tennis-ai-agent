"""Actual encoded private corrected replay, using synthetic source imagery.

Native rule/shot/forecast/export producers and CPU video codecs execute. No
vision inference, independent accuracy, long-match or live claim is made.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tests.test_corrected_exports import export, fresh_forecast, digests, read
from tests.test_corrected_rule_rebuild import unchanged
from tests.test_corrected_shot_rebuild import (
    advance_equivalent_generation, build_shot_base, no_model_imports,
)


OBSERVATIONS = []
RENDER_FILES = {'annotated.mp4', 'annotated_web.mp4', 'corrected_render_manifest.json'}


def test_normal_per_frame_identity_metadata_does_not_reset_a_stable_reviewed_binding():
    from src.scoring.corrected_rendering import _continuous
    matrix = [[.01, 0., 0.], [0., .01, 0.], [0., 0., 1.]]
    rows = []
    for frame in range(2):
        interval = {'analysis_allowed': True, 'measurement_transition_allowed': frame > 0,
            'segment_id': 'same-camera', 'anchor_frame_index': 0, 'local_reset': False,
            'discontinuity_reasons': []}
        identity = {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0,
            'local_reset': False, 'observations': [
                {'track_key': '0:11', 'player_id': 'A', 'state': 'CONFIRMED',
                 'event_id': 'frame-'+str(frame), 'timestamp': frame/30.,
                 'support': .9-frame*.01, 'hypotheses': {'A': .9-frame*.01}},
                {'track_key': '0:22', 'player_id': 'B', 'state': 'CONFIRMED',
                 'event_id': 'other-frame-'+str(frame), 'timestamp': frame/30.,
                 'support': .8+frame*.01, 'hypotheses': {'B': .8+frame*.01}},
            ]}
        rows.append({0: interval, 1: identity})
    sources = SimpleNamespace(source_frames=rows, _camera={'enabled': True,
        'frames': [{'is_valid': True, 'image_to_court': matrix} for _ in rows]})
    assert _continuous(sources, (rows[0][0], rows[0][1]), 1)
    rows[1][1]['observations'][0]['player_id'] = 'B'
    assert not _continuous(sources, (rows[0][0], rows[0][1]), 1)
    rows[1][1]['observations'][0]['player_id'] = 'A'
    rows[1][1]['local_reset'] = True
    assert not _continuous(sources, (rows[0][0], rows[0][1]), 1)


def render(rebuilt, shots, forecast, exports, scratch):
    from src.scoring.corrected_rendering import render_corrected_video
    return render_corrected_video(rebuilt, shots, forecast, exports, scratch)


def decode(path, *, retain=()):
    import cv2
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), str(path)
    metadata = {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': capture.get(cv2.CAP_PROP_FPS)}
    count, frames = 0, {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame is not None and frame.shape == (960, 1280, 3)
            if count in retain:
                frames[count] = frame
            count += 1
    finally:
        capture.release()
    return {**metadata, 'frames': count}, frames


@contextmanager
def fresh_exports(fixture):
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        exports = export(rebuilt, shots, forecast, fixture.scratch)
        yield rebuilt, shots, forecast, exports


def assert_render_result(result, fixture, exports):
    assert isinstance(result.directory, Path)
    assert result.directory.parent == fixture.scratch.resolve()
    assert result.directory != exports.directory
    assert set(result.files) == RENDER_FILES
    assert set(digests(result.directory)) == RENDER_FILES
    assert all(path == result.directory/name for name, path in result.files.items())
    with pytest.raises(TypeError):
        result.files['injected'] = Path('injected')
    seal = read(result.files['corrected_render_manifest.json'])
    assert seal['schema_version'] == '1.0'
    assert seal['analysis_id'] == 'fixture' and seal['analysis_revision'] == 1
    assert seal['base_generation'] == fixture.snapshot.generation
    assert seal['source_sha256'] == fixture.source_sha
    assert seal['export_manifest_sha256'] == hashlib.sha256(
        exports.files['corrected_export_manifest.json'].read_bytes()).hexdigest()
    assert seal['status'] == 'PRIVATE_PENDING_REPORT_AND_PUBLICATION'
    assert seal['published'] is False and seal['independent_accuracy_validated'] is False
    assert seal['frames'] == 12 and seal['fps'] == pytest.approx(30.)
    assert seal['width'] == 1280 and seal['height'] == 960
    assert set(seal['artifacts']) == {'annotated.mp4', 'annotated_web.mp4'}
    for name, record in seal['artifacts'].items():
        path = result.files[name]
        assert record == {'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return seal


def test_actual_two_encoded_videos_show_corrected_causal_hud_fault_and_dead_receiver(tmp_path, monkeypatch):
    import cv2
    from src.line_calling.line_call_engine import LineCallEvidence
    from src.visualization.video_annotator import VideoAnnotator
    fixture = build_shot_base(tmp_path, playable_source=True)
    text, text_frames, calls, ball_draws = [], [], [], []
    original_text, original_call = cv2.putText, LineCallEvidence.overlay_text
    original_ball = VideoAnnotator.draw_ball
    def observed_text(image, value, *args, **kwargs):
        text.append(str(value))
        text_frames.append((len(ball_draws)-1, str(value)))
        return original_text(image, value, *args, **kwargs)
    def observed_call(call):
        calls.append((call.event_id, call.decision.value))
        return original_call(call)
    def observed_ball(frame, position, state):
        ball_draws.append((position, state.value))
        return original_ball(frame, position, state)
    monkeypatch.setattr(cv2, 'putText', observed_text)
    monkeypatch.setattr(LineCallEvidence, 'overlay_text', observed_call)
    monkeypatch.setattr(VideoAnnotator, 'draw_ball', staticmethod(observed_ball))
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        result = render(rebuilt, shots, forecast, exports, fixture.scratch)
        seal = assert_render_result(result, fixture, exports)
        source_meta, source_frames = decode(exports.files['source.mp4'], retain=(0, 4, 8))
        for name in ('annotated.mp4', 'annotated_web.mp4'):
            metadata, frames = decode(result.files[name], retain=(0, 4, 8))
            assert metadata == source_meta == {'width': 1280, 'height': 960, 'fps': 30., 'frames': 12}
            # Native HUD covers this source region; compression may alter exact pixels.
            assert (frames[0][:90] != source_frames[0][:90]).any()
            # Fresh SERVE_FAULT banner red interior, away from text and borders.
            pixel = frames[4][120, 950]
            assert int(pixel[2]) > int(pixel[0])+70 and int(pixel[2]) > int(pixel[1])+70
            # The late receiver is now explicitly dead rather than a new live hit.
            assert not (frames[8][220:250, 40:550] == source_frames[8][220:250, 40:550]).all()
        assert len(ball_draws) == 12
        assert len(calls) == 8 and set(calls) == {(2, 'SERVE_FAULT')}
        assert any('Points 15' in value for value in text)
        assert any('Points 30' in value for value in text)
        assert any(frame == 0 and 'PLAYER 2' in value and 'Points 15' in value
                   for frame, value in text_frames)
        assert not any(frame < 4 and 'Points 30' in value for frame, value in text_frames)
        assert any(frame == 4 and 'PLAYER 2' in value and 'Points 30' in value
                   for frame, value in text_frames)
        assert any('DEAD BALL' in value.upper() for value in text)
        assert any('uncalibrated' in value.lower() for value in text)
        assert any(frame == 0 and 'completed samples 0' in value for frame, value in text_frames)
        assert not any(frame < 4 and 'completed samples 1' in value for frame, value in text_frames)
        assert any(frame == 4 and 'completed samples 1' in value for frame, value in text_frames)
        assert not any('FOREHAND' in value or 'BACKHAND' in value for value in text)
        assert digests(exports.directory) == before
        OBSERVATIONS.append({'actual_toy_render': {'decoded_source_frames': 12,
            'decoded_native_frames': 12, 'decoded_browser_frames': 12,
            'fresh_line_decision': 'SERVE_FAULT', 'original_legal_receiver_suppressed': True,
            'output_size': [seal['width'], seal['height']], 'published': False,
            'independent_accuracy_validated': False}})
    assert_render_result(result, fixture, exports)
    unchanged(fixture)


def test_reviewed_landing_is_drawn_from_new_fault_geometry_in_actual_video(tmp_path, monkeypatch):
    from src.events.event_detector import EventType
    from src.scoring import point_timeline
    from src.visualization import landing_zones
    native = point_timeline.process_timeline_event
    def reviewed(event, *args, **kwargs):
        if getattr(event, 'event_type', None) is EventType.BOUNCE:
            event.evidence.update(review_status='confirmed', active_play=True,
                position_type='ground_contact', position_uncertainty_m=.1,
                ground_contact_source='EXPLICIT_SYNTHETIC_CONTACT_NOT_INDEPENDENT_GROUND_TRUTH')
        return native(event, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(point_timeline, 'process_timeline_event', reviewed)
        fixture = build_shot_base(tmp_path, playable_source=True, config_tail=(
            'visualization:\n  landing_zones:\n    enabled: true\n'
            '    partition: tactical_3x3\n    duration_seconds: 0.8\n'))
    observed = []
    native_draw = landing_zones.draw_landing
    def drawing(frame, record, **kwargs):
        observed.append((record['event_id'], record['analysis_revision'], record['line_call'],
                         kwargs['timestamp_seconds'], record['contact_position_px']))
        return native_draw(frame, record, **kwargs)
    monkeypatch.setattr(landing_zones, 'draw_landing', drawing)
    from src.scoring import corrected_rendering as renderer
    if hasattr(renderer, 'draw_landing'):
        monkeypatch.setattr(renderer, 'draw_landing', drawing)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        assert {row['event_id'] for row in read(exports.files['landing_zones.json'])['records']} == {2}
        result = render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert_render_result(result, fixture, exports)
        assert observed and all(row[:3] == (2, 1, 'out') for row in observed)
        assert all(row[3] >= 4/30. and row[4] == pytest.approx([300., 800.]) for row in observed)
        _, frames = decode(result.files['annotated.mp4'], retain=(4,))
        _, source = decode(exports.files['source.mp4'], retain=(4,))
        # A large interior part of the new far-left zone receives polygon fill.
        assert not (frames[4][450:600, 250:350] == source[4][450:600, 250:350]).all()
    unchanged(fixture)


@pytest.mark.parametrize('name', ['source.mp4', 'match_state.json', 'corrected_export_manifest.json'])
def test_corrupt_export_artifact_rejects_before_render_seal(tmp_path, name):
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        path = exports.files[name]
        original = path.read_bytes()
        try:
            path.write_bytes(b'SYNTHETIC_CHANGED_EXPORT_BYTES')
            with pytest.raises((ValueError, json.JSONDecodeError), match='artifact|export|source|manifest|bytes|changed|Expecting'):
                render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
        finally:
            path.write_bytes(original)
    unchanged(fixture)


def test_self_consistent_export_reseal_cannot_replace_the_frozen_producer_digest(tmp_path):
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        path = exports.files['match_state.json']
        manifest_path = exports.files['corrected_export_manifest.json']
        original, original_manifest = path.read_bytes(), manifest_path.read_bytes()
        try:
            document = read(path)
            document['score_source'] = 'SYNTHETIC_SELF_CONSISTENT_SUBSTITUTION'
            path.write_text(json.dumps(document), encoding='utf-8')
            manifest = read(manifest_path)
            manifest['artifacts'][path.name] = {'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() != exports.manifest_sha256
            with pytest.raises(ValueError, match='manifest|digest|hash|seal|export'):
                render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
        finally:
            path.write_bytes(original)
            manifest_path.write_bytes(original_manifest)
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['missing_file', 'foreign_path', 'outside_scratch'])
def test_render_requires_exact_sealed_inventory_and_guarded_stage(tmp_path, damage):
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        argument, scratch = exports, fixture.scratch
        if damage == 'missing_file':
            files = dict(exports.files)
            files.pop('match_state.json')
            argument = replace(exports, files=MappingProxyType(files))
        elif damage == 'foreign_path':
            files = dict(exports.files)
            files['source.mp4'] = fixture.snapshot.resolve_file('source.mp4')
            argument = replace(exports, files=MappingProxyType(files))
        else:
            scratch = fixture.directory
        with pytest.raises((ValueError, TypeError), match='inventory|export|source|path|scratch|directory|binding|file'):
            render(rebuilt, shots, forecast, argument, scratch)
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
    unchanged(fixture)


@pytest.mark.parametrize('count', [11, 13])
def test_actual_source_decode_requires_exact_sealed_analysis_frame_count(tmp_path, monkeypatch, count):
    import cv2
    native_writer = cv2.VideoWriter
    class DifferentCountWriter:
        def __init__(self, *args, **kwargs):
            self.writer = native_writer(*args, **kwargs)
            self.count, self.last = 0, None
        def isOpened(self):
            return self.writer.isOpened()
        def write(self, frame):
            self.count += 1
            self.last = frame
            if count == 13 or self.count <= 11:
                self.writer.write(frame)
        def release(self):
            if count == 13:
                self.writer.write(self.last)
            self.writer.release()
    with monkeypatch.context() as scoped:
        scoped.setattr(cv2, 'VideoWriter', DifferentCountWriter)
        fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        metadata, _ = decode(exports.files['source.mp4'])
        assert metadata['frames'] == count and fixture.snapshot.frames == 12
        with pytest.raises((ValueError, RuntimeError), match='frame|count|ended|extra|align'):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
    unchanged(fixture)


def test_actual_source_dimensions_must_match_sealed_observation_metadata(tmp_path):
    fixture = build_shot_base(tmp_path, playable_source=True, metadata_size=(640, 480))
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        with pytest.raises((ValueError, RuntimeError), match='dimension|width|height|size|metadata'):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
    unchanged(fixture)


def test_actual_source_fps_must_match_the_retained_normalized_clock(tmp_path, monkeypatch):
    import cv2
    native_writer = cv2.VideoWriter
    def writer(*args, **kwargs):
        args = list(args)
        args[2] = 25.
        return native_writer(*args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(cv2, 'VideoWriter', writer)
        fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        metadata, _ = decode(exports.files['source.mp4'])
        assert metadata['fps'] == 25.
        with pytest.raises((ValueError, RuntimeError), match='fps|FPS|rate|clock|timing'):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
    unchanged(fixture)


def test_failed_later_render_leaves_successful_videos_and_exports_unchanged(tmp_path, monkeypatch):
    import cv2
    from src.visualization.video_annotator import VideoAnnotator
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        first = render(rebuilt, shots, forecast, exports, fixture.scratch)
        before_render, before_export = digests(first.directory), digests(exports.directory)
        handles = []
        capture_factory, writer_factory = cv2.VideoCapture, cv2.VideoWriter
        def capture(*args, **kwargs):
            handle = capture_factory(*args, **kwargs)
            handles.append(handle)
            return handle
        def writer(*args, **kwargs):
            handle = writer_factory(*args, **kwargs)
            handles.append(handle)
            return handle
        monkeypatch.setattr(cv2, 'VideoCapture', capture)
        monkeypatch.setattr(cv2, 'VideoWriter', writer)
        native = VideoAnnotator.draw_ball
        def fail(frame, position, state):
            native(frame, position, state)
            raise RuntimeError('SYNTHETIC_RENDER_FAILURE')
        monkeypatch.setattr(VideoAnnotator, 'draw_ball', staticmethod(fail))
        with pytest.raises(RuntimeError, match='SYNTHETIC_RENDER_FAILURE'):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert handles and all(not handle.isOpened() for handle in handles)
        assert digests(first.directory) == before_render
        assert digests(exports.directory) == before_export
        assert set(fixture.scratch.rglob('corrected_render_manifest.json')) == {
            first.files['corrected_render_manifest.json']}
    unchanged(fixture)


@pytest.mark.parametrize('failure', ['encoder', 'corrupt_output'])
def test_browser_encode_or_full_decode_failure_cannot_seal_a_native_only_result(tmp_path, monkeypatch, failure):
    from src.api.worker import AnalysisError
    from src.scoring import corrected_rendering as renderer
    fixture = build_shot_base(tmp_path, playable_source=True)
    attempted = []
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        native_execute = renderer.execute
        def execute(args, *positional, **kwargs):
            if 'libx264' not in args:
                return native_execute(args, *positional, **kwargs)
            attempted.append(list(args))
            if failure == 'encoder':
                raise RuntimeError('SYNTHETIC_BROWSER_ENCODER_FAILURE')
            value = native_execute(args, *positional, **kwargs)
            Path(args[-1]).write_bytes(b'SYNTHETIC_INVALID_BROWSER_VIDEO')
            return value
        monkeypatch.setattr(renderer, 'execute', execute)
        with pytest.raises((ValueError, RuntimeError, AnalysisError)):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert len(attempted) == 1
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
        assert digests(exports.directory) == before
        native_outputs = list(fixture.scratch.rglob('annotated.mp4'))
        assert len(native_outputs) == 1
        metadata, _ = decode(native_outputs[0])
        assert metadata['frames'] == 12
    unchanged(fixture)


def test_generation_advance_during_actual_frame_render_rejects_final_seal(tmp_path, monkeypatch):
    from src.api.analysis_generation import resolve_generation
    from src.visualization.video_annotator import VideoAnnotator
    fixture = build_shot_base(tmp_path, playable_source=True)
    advanced = []
    with pytest.raises(ValueError, match='current generation|base'):
        with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
            before = digests(exports.directory)
            native = VideoAnnotator.draw_ball
            def advance(frame, position, state):
                if not advanced:
                    advanced.append(advance_equivalent_generation(fixture))
                return native(frame, position, state)
            monkeypatch.setattr(VideoAnnotator, 'draw_ball', staticmethod(advance))
            with pytest.raises(ValueError, match='current generation|base'):
                render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
            assert digests(exports.directory) == before
    assert len(advanced) == 1
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert digests(fixture.snapshot.files_directory) == fixture.manifest


def test_changed_live_physical_event_after_export_rejects_before_video_seal(tmp_path):
    from src.events.event_detector import EventType
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        original = shots.events[1]
        changed = copy.deepcopy(original)
        changed.event_type = EventType.UNKNOWN_EVENT
        shots.events[1] = changed
        try:
            assert read(exports.files['match_events.json'])['events'][1]['event_type'] == 'BOUNCE'
            with pytest.raises(ValueError, match='physical event|sealed export'):
                render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
            assert digests(exports.directory) == before
        finally:
            shots.events[1] = original
    unchanged(fixture)


def test_temporary_live_event_mutation_and_restore_during_frame_render_invalidates_candidate(tmp_path, monkeypatch):
    from src.events.event_detector import EventType
    from src.visualization.video_annotator import VideoAnnotator
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        original = shots.events[1]
        initial_generation = shots.event_owner.events_generation
        mutated = []
        native_draw = VideoAnnotator.draw_ball
        def drawing(frame, position, state):
            if not mutated:
                changed = copy.deepcopy(original)
                changed.event_type = EventType.UNKNOWN_EVENT
                shots.events[1] = changed
                shots.events[1] = original
                mutated.append(shots.event_owner.events_generation)
            return native_draw(frame, position, state)
        monkeypatch.setattr(VideoAnnotator, 'draw_ball', staticmethod(drawing))
        try:
            with pytest.raises(ValueError, match='physical events changed'):
                render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert mutated == [initial_generation+2]
            assert shots.events[1] == original
            assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
            assert digests(exports.directory) == before
        finally:
            shots.events[1] = original
    unchanged(fixture)


def test_mutable_timeline_initial_presentation_cannot_replace_admitted_hud_fallback(tmp_path, monkeypatch):
    fixture = build_shot_base(tmp_path, playable_source=True)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        accepted_initial = rebuilt.engine.initial_state.to_dict()
        original_initial = copy.deepcopy(rebuilt.timeline.initial_state)
        rebuilt.timeline.initial_state['score_confirmed'] = False
        rebuilt.timeline.initial_state['player_2']['points_display'] = '40'
        assert rebuilt.timeline.initial_state != accepted_initial
        calls = []
        native = rebuilt.owner.frame_state
        def frame_state(frame, initial):
            calls.append((frame, copy.deepcopy(initial)))
            return native(frame, initial)
        monkeypatch.setattr(rebuilt.owner, 'frame_state', frame_state)
        try:
            result = render(rebuilt, shots, forecast, exports, fixture.scratch)
            assert_render_result(result, fixture, exports)
            assert [frame for frame, _ in calls] == list(range(12))
            assert all(initial == accepted_initial for _, initial in calls)
            metadata, _ = decode(result.files['annotated_web.mp4'])
            assert metadata['frames'] == 12
        finally:
            rebuilt.timeline.initial_state.clear()
            rebuilt.timeline.initial_state.update(original_initial)
    unchanged(fixture)


@pytest.mark.parametrize('name', ['annotated.mp4', 'annotated_web.mp4'])
def test_movie_changed_after_actual_full_decode_cannot_receive_a_render_seal(tmp_path, monkeypatch, name):
    from src.scoring import corrected_rendering as renderer
    fixture = build_shot_base(tmp_path, playable_source=True)
    checked = []
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        native = renderer.full_video_check
        def full_video_check(path, **kwargs):
            value = native(path, **kwargs)
            checked.append(path.name)
            if path.name == name:
                path.write_bytes(b'SYNTHETIC_MOVIE_CHANGED_AFTER_SUCCESSFUL_FULL_DECODE')
            return value
        monkeypatch.setattr(renderer, 'full_video_check', full_video_check)
        with pytest.raises(ValueError, match='video artifact changed during full decode'):
            render(rebuilt, shots, forecast, exports, fixture.scratch)
        assert name in checked
        assert not list(fixture.scratch.rglob('corrected_render_manifest.json'))
        assert digests(exports.directory) == before
    unchanged(fixture)


def test_one_horizon_crossing_display_frame_withholds_geometry_then_recovers(tmp_path, monkeypatch):
    import cv2
    import tests.test_correction_sources as assets
    from src.court.court_geometry import TennisCourtGeometry
    from src.scoring import corrected_rendering as renderer
    from src.visualization.video_annotator import VideoAnnotator
    matrix = [[.01, 0., 0.], [0., .01, 0.], [0., .001, 1.]]
    native_assets = assets.write_observation_assets
    def observations(*args, **kwargs):
        files = native_assets(*args, **kwargs)
        path = files['camera_registration.json']
        camera = read(path)
        camera['frames'][9]['image_to_court'] = matrix
        path.write_text(json.dumps(camera, allow_nan=False), encoding='utf-8')
        path = files['court_geometry.json']
        court = read(path)
        court['canonical_keypoints_m'] = TennisCourtGeometry.get_canonical_keypoints().tolist()
        path.write_text(json.dumps(court, allow_nan=False), encoding='utf-8')
        return files
    with monkeypatch.context() as scoped:
        scoped.setattr(assets, 'write_observation_assets', observations)
        fixture = build_shot_base(tmp_path, playable_source=True)
    active_frame, completed, projected, messages = [None], [], [], []
    native_frame, native_points = renderer._draw_frame, VideoAnnotator.draw_court_keypoints
    native_text = cv2.putText
    def draw_frame(image, frame, *args, **kwargs):
        active_frame[0] = frame
        value = native_frame(image, frame, *args, **kwargs)
        completed.append(frame)
        return value
    def draw_points(image, points):
        projected.append(active_frame[0])
        return native_points(image, points)
    def text(image, value, *args, **kwargs):
        messages.append((active_frame[0], str(value)))
        return native_text(image, value, *args, **kwargs)
    monkeypatch.setattr(renderer, '_draw_frame', draw_frame)
    monkeypatch.setattr(VideoAnnotator, 'draw_court_keypoints', staticmethod(draw_points))
    monkeypatch.setattr(cv2, 'putText', text)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        before = digests(exports.directory)
        assert rebuilt.sources._camera['frames'][9]['is_valid'] is True
        assert rebuilt.sources._camera['frames'][9]['image_to_court'] == matrix
        result = render(rebuilt, shots, forecast, exports, fixture.scratch)
        seal = assert_render_result(result, fixture, exports)
        assert completed == list(range(12))
        assert set(projected) == set(range(12))-{9}
        assert any(frame == 9 and 'geometry' in value.lower() and 'unavailable' in value.lower()
                   for frame, value in messages)
        assert 10 in projected  # A later safe display frame recovers normally.
        assert seal['geometry_withheld_frame_count'] == 1
        reasons = seal['geometry_withheld_reason_counts']
        assert isinstance(reasons, dict) and reasons
        assert all(isinstance(reason, str) and reason and type(count) is int and count > 0
                   for reason, count in reasons.items())
        assert sum(reasons.values()) >= seal['geometry_withheld_frame_count']
        for name in ('annotated.mp4', 'annotated_web.mp4'):
            metadata, _ = decode(result.files[name])
            assert metadata['frames'] == 12
        assert rebuilt.sources._camera['frames'][9]['is_valid'] is True
        assert rebuilt.sources._camera['frames'][9]['image_to_court'] == matrix
        assert digests(exports.directory) == before
    unchanged(fixture)

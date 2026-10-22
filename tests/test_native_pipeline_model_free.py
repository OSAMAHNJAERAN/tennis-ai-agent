"""Complete native software/codec flow, with explicit synthetic vision inputs.

These fixtures bypass model loading/qualification and insert a synthetic review
gate. Recovery controls exercise the actual pipeline storage/continuation with
injected results; they are not real checkpoint, worker-process or match evidence.
"""
from contextlib import closing
import builtins
import hashlib
import json
import shutil
import sqlite3

import cv2
import numpy as np
import pytest

from src.api.analysis_resume import REQUIRED_PIPELINE_FILES
from tests.helpers.native_pipeline_source import (
    FORBIDDEN_ROOTS, FPS, HEIGHT, WIDTH, injected_native_pipeline, no_inference_imports,
)


PLAYER_AUDITS = ('camera_registration.json', 'camera_intervals.json', 'match_identity.json', 'image_people.json')
INPUT_BOUNDARY = {'source.mp4', 'media_metadata.json', 'inference_config.yaml'}
GUARD_CONTROL_OBSERVATIONS = []
WORKFLOW_IMPORT_OBSERVATIONS = []
RECOVERY_IMPORT_OBSERVATIONS = []


@pytest.mark.parametrize('downstream_state', ['would_return_cached', 'would_raise_missing'])
def test_caught_neural_import_still_fails_guard_context_without_forwarding(monkeypatch, downstream_state):
    """Synthetic delegate controls; never create or import a torch module.

    Both hypothetical cache states sit behind the guard. The real importer is
    never reached for the deliberate forbidden name, even when caught inside.
    These intentional attempts are separate from workflow no-attempt evidence.
    """
    native_import = builtins.__import__
    forwarded = []

    def downstream(name, *args, **kwargs):
        if name == 'torch':
            forwarded.append(name)
            if downstream_state == 'would_return_cached':
                return None  # Unreachable synthetic cache-return branch.
            raise ModuleNotFoundError('Unreachable synthetic missing-import branch')
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', downstream)
    with pytest.raises(AssertionError, match='retained forbidden import attempts'):
        with no_inference_imports() as attempts:
            try:
                __import__('torch')
            except AssertionError:
                pass
    assert forwarded == []
    assert attempts == [{'mechanism': 'builtins.__import__', 'name': 'torch', 'root': 'torch'}]
    GUARD_CONTROL_OBSERVATIONS.append({'case': downstream_state, 'deliberate_blocked_attempts': list(attempts),
                                       'actual_import_delegations': len(forwarded)})


def test_guard_preserves_primary_body_interrupt_and_notes_caught_attempt():
    failure = KeyboardInterrupt('Synthetic body interruption after deliberate guard control')
    with pytest.raises(KeyboardInterrupt) as caught:
        with no_inference_imports() as attempts:
            try:
                __import__('torchvision')
            except AssertionError:
                pass
            raise failure
    assert caught.value is failure
    assert failure.__notes__ == ['Blocked neural import attempts retained by fixture guard: torchvision']
    assert attempts == [{'mechanism': 'builtins.__import__', 'name': 'torchvision', 'root': 'torchvision'}]
    GUARD_CONTROL_OBSERVATIONS.append({'case': 'preserved_primary_interrupt',
        'deliberate_blocked_attempts': list(attempts), 'actual_import_delegations': 0})


def decoded_count(path):
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (HEIGHT, WIDTH, 3)
            count += 1
        return count
    finally:
        capture.release()


def decoded_pixel_hashes(path):
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        values = []
        while True:
            ok, frame = capture.read()
            if not ok:
                return values
            values.append(hashlib.sha256(np.asarray(frame).tobytes()).hexdigest())
    finally:
        capture.release()


def test_actual_native_source_completes_all_nonmodel_stages_and_exports(tmp_path):
    with injected_native_pipeline(tmp_path / 'complete') as fixture:
        result = fixture.pipeline.run(str(fixture.source), str(fixture.directory))
        assert result['status'] == 'success'
        assert result['shots_linked'] == 0
        assert fixture.pipeline.yolo26 is True and fixture.pipeline.pose_extractor is None
        assert fixture.pipeline.motion_tracker is fixture.pipeline.racket_tracker is None
        assert fixture.pipeline.player_detector.model is fixture.models['players']
        assert len(fixture.models['players'].tracking_calls) == fixture.frame_count
        assert len(fixture.models['court'].calls) == fixture.frame_count + 1
        assert len(fixture.models['ball'].calls) == fixture.frame_count
        assert fixture.models['players'].calls == []
        assert fixture.models['players'].callback_resets == 1
        assert fixture.models['players'].predictor is None
        first_ball = next(index for index, row in enumerate(fixture.trace) if row[0] == 'ball')
        assert sum(row[0] == 'players' for row in fixture.trace[:first_ball]) == fixture.frame_count
        assert all(row[0] == 'ball' for row in fixture.trace[first_ball:])

        for name in set(REQUIRED_PIPELINE_FILES) - INPUT_BOUNDARY:
            assert (fixture.directory / name).is_file(), name
        for name in PLAYER_AUDITS:
            value = json.loads((fixture.directory / name).read_text())
            assert len(value['frames']) == fixture.frame_count
            assert [row['frame_index'] for row in value['frames']] == list(range(fixture.frame_count))

        detections = json.loads((fixture.directory / 'detections.json').read_text())
        assert detections['metadata']['fps'] == FPS
        assert len(detections['frames']) == fixture.frame_count
        # The actual global selector retains two roles from the injected people.
        assert all(row['player_1'] is not None and row['player_2'] is not None for row in detections['frames'])
        metrics = json.loads((fixture.directory / 'metrics.json').read_text())
        crop = metrics['pose_crop_execution']
        assert crop == result['pose_crop_execution']
        assert crop['prediction_calls'] == crop['completed_prediction_calls'] == 0
        assert crop['failed'] is False and crop['resolved_device'] == 'cpu'
        assert metrics['production_qualified'] is False and metrics['authoritative_events_enabled'] is False
        assert json.loads((fixture.directory / 'player_metrics.json').read_text())['available'] is False
        assert json.loads((fixture.directory / 'ball_metrics.json').read_text())['summary']['available'] is False
        assert decoded_count(fixture.directory / 'annotated.mp4') == fixture.frame_count
        assert decoded_count(fixture.source) == fixture.frame_count
        assert not list(fixture.directory.glob('.*.tmp'))
        assert not list(fixture.directory.glob('*.sqlite3'))
    WORKFLOW_IMPORT_OBSERVATIONS.append({'case': 'complete_native_flow',
        'forbidden_roots': sorted(FORBIDDEN_ROOTS), 'workflow_attempts': list(fixture.import_attempts),
        'source_loader_attempts': list(fixture.module.fixture_source_import_attempts)})
    assert fixture.import_attempts == fixture.module.fixture_source_import_attempts == []


def test_native_ball_failure_preserves_completed_player_audits_but_no_results(tmp_path):
    with injected_native_pipeline(tmp_path / 'failure', fail_ball_at=2) as fixture:
        with pytest.raises(RuntimeError, match='Injected ball result boundary failure'):
            fixture.pipeline.run(str(fixture.source), str(fixture.directory))
        assert len(fixture.models['players'].tracking_calls) == fixture.frame_count
        assert len(fixture.models['court'].calls) == fixture.frame_count + 1
        assert len(fixture.models['ball'].calls) == 3
        assert fixture.models['players'].calls == []
        for name in PLAYER_AUDITS:
            value = json.loads((fixture.directory / name).read_text())
            assert len(value['frames']) == fixture.frame_count
        for name in ('metrics.json', 'annotated.mp4', 'native_rule_replay.json', 'point_timeline_manifest.json'):
            assert not (fixture.directory / name).exists(), name
        journals = list(fixture.directory.glob('ball_candidates_*.sqlite3'))
        assert len(journals) == 1
        with closing(sqlite3.connect(journals[0])) as connection:
            assert connection.execute('SELECT COUNT(*) FROM candidate_frames').fetchone()[0] == 2
        assert not list(fixture.directory.glob('.*.tmp'))
    WORKFLOW_IMPORT_OBSERVATIONS.append({'case': 'failed_ball_flow',
        'forbidden_roots': sorted(FORBIDDEN_ROOTS), 'workflow_attempts': list(fixture.import_attempts),
        'source_loader_attempts': list(fixture.module.fixture_source_import_attempts)})
    assert fixture.import_attempts == fixture.module.fixture_source_import_attempts == []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_recovery_workflow(case, fixture):
    assert fixture.import_attempts == fixture.module.fixture_source_import_attempts == []
    RECOVERY_IMPORT_OBSERVATIONS.append({'case': case, 'forbidden_roots': sorted(FORBIDDEN_ROOTS),
        'workflow_attempts': list(fixture.import_attempts),
        'source_loader_attempts': list(fixture.module.fixture_source_import_attempts)})


def test_native_prefix_recovery_skips_completed_players_and_infers_exact_ball_suffix(tmp_path):
    identity = {'analysis_id': 'synthetic_native_pipeline_only', 'fixture_inputs': 'same-sealed-inputs-v1'}
    recovered_dir = tmp_path / 'recovered'
    failure = KeyboardInterrupt('Injected process-loss boundary at third raw ball frame')
    with injected_native_pipeline(recovered_dir, inference_identity=identity,
                                  fail_ball_at=2, fail_ball_error=failure) as first:
        with pytest.raises(KeyboardInterrupt) as caught:
            first.pipeline.run(str(first.source), str(first.directory))
        assert caught.value is failure
        assert len(first.models['players'].tracking_calls) == first.frame_count
        assert len(first.models['court'].calls) == first.frame_count + 1
        assert first.models['ball'].frame_indices == [0, 1, 2]
        for name in ('.native_inference_prefix.json', '.native_inference_prefix.sqlite3',
                     '.ball_inference.sqlite3', *PLAYER_AUDITS):
            assert (recovered_dir / name).is_file(), name
        with closing(sqlite3.connect(recovered_dir / '.ball_inference.sqlite3')) as connection:
            assert connection.execute('SELECT next_frame FROM journal_metadata WHERE id=1').fetchone() == (2,)
            assert connection.execute('SELECT frame_index FROM raw_frames ORDER BY frame_index').fetchall() == [(0,), (1,)]
        assert not (recovered_dir / 'metrics.json').exists()
    record_recovery_workflow('interrupted_after_two_raw_frames', first)

    retained_names = ('source.mp4', 'source_frame_mapping.jsonl', 'inference_config.yaml',
                      '.native_inference_prefix.json', '.native_inference_prefix.sqlite3', *PLAYER_AUDITS)
    retained = {name: digest(recovered_dir / name) for name in retained_names}
    with injected_native_pipeline(recovered_dir, existing=True,
            inference_identity={**identity, 'fixture_inputs': 'changed-input-identity'}) as rejected:
        with pytest.raises(ValueError):
            rejected.pipeline.run(str(rejected.source), str(rejected.directory))
        assert all(not model.calls and not model.tracking_calls for model in rejected.models.values())
        assert {name: digest(recovered_dir / name) for name in retained_names} == retained
        assert not (recovered_dir / 'metrics.json').exists()
    record_recovery_workflow('rejected_changed_identity_before_model_calls', rejected)
    with injected_native_pipeline(recovered_dir, existing=True, inference_identity=identity) as rejected_authority:
        rejected_authority.pipeline.config['event_detection']['authoritative_enabled'] = True
        with pytest.raises(ValueError, match='Dynamic court registration requires event authority disabled'):
            rejected_authority.pipeline.run(str(rejected_authority.source), str(rejected_authority.directory))
        assert all(not model.calls and not model.tracking_calls for model in rejected_authority.models.values())
        assert {name: digest(recovered_dir / name) for name in retained_names} == retained
    record_recovery_workflow('rejected_event_authority_before_restored_model_calls', rejected_authority)
    # Copy only normalized inputs to a clean output directory. Both executions
    # consume byte-identical media/clock/config; no prefix or raw journal copies.
    continuous_dir = tmp_path / 'continuous'
    continuous_dir.mkdir()
    for name in ('source.mp4', 'source_frame_mapping.jsonl', 'inference_config.yaml'):
        shutil.copyfile(recovered_dir / name, continuous_dir / name)

    with injected_native_pipeline(recovered_dir, existing=True, inference_identity=identity) as resumed:
        result = resumed.pipeline.run(str(resumed.source), str(resumed.directory))
        assert resumed.models['players'].tracking_calls == resumed.models['players'].calls == []
        assert resumed.models['court'].calls == []
        assert resumed.models['ball'].frame_indices == list(range(2, resumed.frame_count))
        assert result['inference_resume'] == {'prefix_reused': True, 'raw_frames_reused': 2,
            'raw_frames_inferred': resumed.frame_count - 2, 'total_frames': resumed.frame_count}
        assert {name: digest(recovered_dir / name) for name in retained_names} == retained
        assert resumed.models['players'].device == resumed.models['ball'].device == 'cpu'
        assert all(options.get('device') == 'cpu' for options in resumed.models['ball'].calls)
    record_recovery_workflow('resumed_exact_raw_suffix', resumed)

    with injected_native_pipeline(continuous_dir, existing=True, inference_identity=identity) as continuous:
        fresh = continuous.pipeline.run(str(continuous.source), str(continuous.directory))
        assert continuous.models['ball'].frame_indices == list(range(continuous.frame_count))
        assert fresh['inference_resume'] == {'prefix_reused': False, 'raw_frames_reused': 0,
            'raw_frames_inferred': continuous.frame_count, 'total_frames': continuous.frame_count}
    record_recovery_workflow('continuous_reference', continuous)

    # Every emitted semantic artifact is byte-identical except the intentional
    # absolute source pathname and execution-duration/resume records.
    for name in set(REQUIRED_PIPELINE_FILES) - INPUT_BOUNDARY - {'metrics.json', 'detections.json'}:
        if name == 'annotated.mp4':
            continue
        assert (recovered_dir / name).read_bytes() == (continuous_dir / name).read_bytes(), name
    recovered_detections = json.loads((recovered_dir / 'detections.json').read_text())
    continuous_detections = json.loads((continuous_dir / 'detections.json').read_text())
    recovered_detections['metadata'].pop('video')
    continuous_detections['metadata'].pop('video')
    assert recovered_detections == continuous_detections
    metrics = json.loads((recovered_dir / 'metrics.json').read_text())
    assert metrics['inference_resume'] == result['inference_resume']
    assert metrics['pose_crop_execution']['prediction_calls'] == 0
    assert decoded_count(recovered_dir / 'annotated.mp4') == resumed.frame_count
    assert decoded_count(continuous_dir / 'annotated.mp4') == continuous.frame_count
    assert decoded_pixel_hashes(recovered_dir / 'annotated.mp4') == decoded_pixel_hashes(continuous_dir / 'annotated.mp4')

"""Actual worker process loss during native ball inference, injected models.

The two recovery children execute real codecs, source normalization, source map,
native numerical/storage/render code and completed-result publication. Vision
result/asset admission are explicit synthetic boundaries, not model accuracy.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import cv2
import yaml

from src.api.analysis_resume import REQUIRED_PIPELINE_FILES
from src.api.jobs import JobStore
from src.api.result_document import ResultDocument
from tests.helpers.native_pipeline_source import ROOT, _write_source


CHILD = Path(__file__).parent / 'helpers/native_pipeline_resume_child.py'
SCOPE = 'NATIVE_WORKER_PROCESS_RECOVERY_WITH_SYNTHETIC_SELECTED_VISION'
PROCESS_OBSERVATIONS = []
ANALYSIS_ID = 'match_26ba1126ba1126ba1126ba1126ba1126'
AUDITS = ('camera_registration.json', 'camera_intervals.json', 'match_identity.json', 'image_people.json')
BALL_VIEW_REGIONS = [[0, 0, 320, 240], [0, 0, 160, 160], [160, 0, 320, 160],
                     [0, 80, 160, 240], [160, 80, 320, 240]]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels(path):
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        result = []
        while True:
            ok, frame = capture.read()
            if not ok:
                return result
            result.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:
        capture.release()


def observations(storage):
    return [json.loads(line) for line in (storage / 'native_worker_observations.jsonl').read_text(encoding='utf-8').splitlines()]


def make_job(storage, original, *, tiles=False):
    config = yaml.safe_load((ROOT / 'configs/production/yolo26.yaml').read_text(encoding='utf-8'))
    config['scope'] = SCOPE
    config['model_system']['device'] = 'cpu'
    config['pose_estimation']['enabled'] = False
    config['player_motion']['enabled'] = False
    config['player_detection']['model'] = 'SYNTHETIC-selected-player.pt'
    config['player_detection']['tracker'] = 'SYNTHETIC-pinned-tracker.yaml'
    config['ball_detection']['model'] = 'SYNTHETIC-selected-ball.pt'
    config['court_detection']['model_path'] = 'SYNTHETIC-selected-court.pt'
    config['ball_detection']['tiles']['enabled'] = tiles
    if tiles:
        config['ball_detection']['tiles'].update(size_px=160, overlap=0)
    config['camera_intervals'] = {'reviews': [{'evidence_id': 'SYNTHETIC_REVIEW_GATE_NOT_GROUND_TRUTH',
        'start_source_seconds': 0., 'end_source_seconds': 100., 'label': 'match_play',
        'viewpoint': 'elevated_endcourt_singles', 'normal_speed_verified': True,
        'provenance': 'manual_review'}]}
    store = JobStore(storage)
    config_path = storage / 'synthetic_selected_config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    store.create(ANALYSIS_ID, 'synthetic-native-process-resume.mp4', '.mp4', original.stat().st_size,
                 config_path, owner_id='native-process-recovery-owner')
    uploaded = store.upload_dir(ANALYSIS_ID) / 'original.mp4'
    uploaded.parent.mkdir(parents=True)
    shutil.copyfile(original, uploaded)
    return store, config_path


def run_child(mode, storage, config):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TENNIS_STORAGE_ROOT=str(storage))
    return subprocess.run([sys.executable, '-B', str(CHILD.resolve()), mode,
                           str(storage), str(config), ANALYSIS_ID],
                          capture_output=True, text=True, timeout=120, env=environment, cwd=ROOT,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def assert_ball_views(row, frame_indices, *, interrupted=False):
    expected = [(frame, view, region) for frame in frame_indices
                for view, region in enumerate(BALL_VIEW_REGIONS)]
    if interrupted:
        expected += [(2, view, BALL_VIEW_REGIONS[view]) for view in (0, 1)]
    actual = row['ball_view_calls']
    assert [(view['frame_index'], view['view_index'], view['region']) for view in actual] == expected
    assert row['ball_complete_view_frame_indices'] == frame_indices
    assert row['ball_predict_calls'] == len(expected)
    for position, view in enumerate(actual):
        assert view['result_prepared'] is True
        assert view['returned_to_adapter'] is (not interrupted or position != len(actual) - 1)
        assert view['candidates'] == (1 if view['frame_index'] != 4 and view['view_index'] in (0, 1, 3) else 0)


def _exercise_worker_recovery(tmp_path, *, mid_tile=False):
    # The actual media tools are required by this test; absence is not a pass.
    assert shutil.which(os.getenv('TENNIS_FFMPEG', 'ffmpeg')) is not None
    assert shutil.which(os.getenv('TENNIS_FFPROBE', 'ffprobe')) is not None
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    original = _write_source(input_dir, 8)
    storage = tmp_path / 'recovered'
    store, config = make_job(storage, original, tiles=mid_tile)
    output = store.output_dir(ANALYSIS_ID)

    first_mode = 'crash_mid_tile' if mid_tile else 'crash'
    first = run_child(first_mode, storage, config)
    assert first.returncode == 73, (first.returncode, first.stdout, first.stderr)
    interrupted = JobStore(storage).get(ANALYSIS_ID)
    assert interrupted['status'] == 'PROCESSING' and interrupted['attempt'] == 1
    assert interrupted['completed_at'] is None
    assert not (output / '.pipeline_checkpoint.json').exists()
    assert not (output / '.result/current.store').exists()
    assert not (output / 'metrics.json').exists()
    with closing(sqlite3.connect(output / '.ball_inference.sqlite3')) as connection:
        assert connection.execute('SELECT next_frame FROM journal_metadata WHERE id=1').fetchone() == (2,)
        assert connection.execute('SELECT frame_index FROM raw_frames ORDER BY frame_index').fetchall() == [(0,), (1,)]
        retained_raw_rows = connection.execute(
            'SELECT frame_index,payload,payload_sha256 FROM raw_frames ORDER BY frame_index').fetchall()
    retained_names = ('source.mp4', 'media_metadata.json', 'inference_config.yaml',
        'source_frame_mapping.jsonl', 'normalization_frame_stats.txt', '.native_inference_source.json',
        '.native_inference_prefix.json', '.native_inference_prefix.sqlite3', *AUDITS)
    retained = {name: digest(output / name) for name in retained_names}
    before = observations(storage)
    exit_event = 'hard_exit_mid_tile' if mid_tile else 'hard_exit_after_raw_frontier'
    exited = next(row for row in before if row['event'] == exit_event)
    assert exited['next_frame'] == 2 and exited['players_track_calls'] == 8
    assert exited['court_predict_calls'] == 9
    assert [row['frame_index'] for row in before if row['event'] == 'committed_raw_frame'] == [0, 1]
    if mid_tile:
        assert exited['incomplete_frame_index'] == 2 and exited['incomplete_view_index'] == 1
        assert exited['full_frame_prediction_returned'] is exited['tile_prediction_result_prepared'] is True
        assert exited['ball_frame_indices'] == [0] * 5 + [1] * 5 + [2, 2]
        assert exited['ball_full_frame_indices'] == [0, 1, 2]
        assert_ball_views(exited, [0, 1], interrupted=True)
    else:
        assert exited['ball_predict_calls'] == 2 and exited['ball_frame_indices'] == [0, 1]
    assert exited['selected_models_constructed'] == 3 and exited['bundles_constructed'] == 1
    assert exited['import_attempts'] == exited['forbidden_loaded'] == exited['source_loader_import_attempts'] == []

    second = run_child('resume', storage, config)
    assert second.returncode == 0, (second.returncode, second.stdout, second.stderr,
                                   (output / 'processing.log').read_text(encoding='utf-8')[-10000:])
    completed = JobStore(storage).get(ANALYSIS_ID)
    assert completed['status'] == 'COMPLETED' and completed['attempt'] == 2
    assert completed['owner_id'] == 'native-process-recovery-owner'
    assert {name: digest(output / name) for name in retained_names} == retained
    rows = observations(storage)
    final = next(row for row in rows if row['event'] == 'completed')
    assert final['players_track_calls'] == final['players_predict_calls'] == final['court_predict_calls'] == 0
    assert final['ball_full_frame_indices'] == [2, 3, 4, 5, 6, 7]
    if mid_tile:
        assert final['ball_frame_indices'] == [frame for frame in range(2, 8) for _ in BALL_VIEW_REGIONS]
        assert_ball_views(final, list(range(2, 8)))
    else:
        assert final['ball_predict_calls'] == 6 and final['ball_frame_indices'] == [2, 3, 4, 5, 6, 7]
    assert final['selected_models_constructed'] == 3 and final['bundles_constructed'] == 1
    assert final['pid'] != exited['pid'] and final['pid'] != os.getpid()
    assert [row['event'] for row in rows].count('normalizing_source') == 1
    assert [row['event'] for row in rows].count('encoding_browser_video') == 1
    assert [row['attempt'] for row in rows if row['event'] == 'claimed'] == [1, 2]
    assert final['import_attempts'] == final['forbidden_loaded'] == final['source_loader_import_attempts'] == []
    with closing(sqlite3.connect(output / '.ball_inference.sqlite3')) as connection:
        assert connection.execute('SELECT next_frame FROM journal_metadata WHERE id=1').fetchone() == (8,)
        assert connection.execute('SELECT frame_index FROM raw_frames ORDER BY frame_index').fetchall() == [(i,) for i in range(8)]
        assert connection.execute('SELECT frame_index,payload,payload_sha256 FROM raw_frames '
                                  'WHERE frame_index<2 ORDER BY frame_index').fetchall() == retained_raw_rows
        if mid_tile:
            assert json.loads(connection.execute('SELECT payload FROM raw_frames WHERE frame_index=4').fetchone()[0]) == []
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['inference_resume'] == {'prefix_reused': True, 'raw_frames_reused': 2,
        'raw_frames_inferred': 6, 'total_frames': 8}
    with ResultDocument(output, ANALYSIS_ID) as document:
        assert document.value['summary']['frames'] == 8
        assert document.value['provenance']['productionQualified'] is False
        assert document.generation == final['publication']['generation']
    assert JobStore(storage).claim() is None

    continuous_storage = tmp_path / 'continuous'
    continuous_store, continuous_config = make_job(continuous_storage, original, tiles=mid_tile)
    reference = run_child('continuous', continuous_storage, continuous_config)
    continuous = continuous_store.output_dir(ANALYSIS_ID)
    assert reference.returncode == 0, (reference.returncode, reference.stdout, reference.stderr,
                                      (continuous / 'processing.log').read_text(encoding='utf-8')[-10000:])
    continuous_rows = observations(continuous_storage)
    continuous_final = next(row for row in continuous_rows if row['event'] == 'completed')
    assert continuous_final['players_track_calls'] == 8 and continuous_final['court_predict_calls'] == 9
    assert continuous_final['ball_full_frame_indices'] == list(range(8))
    if mid_tile:
        assert continuous_final['ball_frame_indices'] == [frame for frame in range(8) for _ in BALL_VIEW_REGIONS]
        assert_ball_views(continuous_final, list(range(8)))
        with closing(sqlite3.connect(continuous / '.ball_inference.sqlite3')) as connection:
            assert json.loads(connection.execute('SELECT payload FROM raw_frames WHERE frame_index=4').fetchone()[0]) == []
    else:
        assert continuous_final['ball_frame_indices'] == list(range(8))
    assert continuous_final['import_attempts'] == continuous_final['forbidden_loaded'] == continuous_final['source_loader_import_attempts'] == []
    assert digest(output / 'source.mp4') == digest(continuous / 'source.mp4')
    exclusions = {'source.mp4', 'media_metadata.json', 'inference_config.yaml', 'annotated.mp4',
                  'detections.json', 'metrics.json'}
    for name in set(REQUIRED_PIPELINE_FILES) - exclusions:
        assert (output / name).read_bytes() == (continuous / name).read_bytes(), name
    left = json.loads((output / 'detections.json').read_text(encoding='utf-8'))
    right = json.loads((continuous / 'detections.json').read_text(encoding='utf-8'))
    left['metadata'].pop('video')
    right['metadata'].pop('video')
    assert left == right
    assert len(pixels(output / 'annotated.mp4')) == len(pixels(continuous / 'annotated.mp4')) == 8
    assert pixels(output / 'annotated.mp4') == pixels(continuous / 'annotated.mp4')
    assert pixels(output / 'annotated_web.mp4') == pixels(continuous / 'annotated_web.mp4')
    evidence = {'scope': SCOPE, 'analysis_id': ANALYSIS_ID,
        'recovery_case': 'during_first_tile_of_frame_2' if mid_tile else 'after_two_complete_frames',
        'first_child_mode': first_mode, 'synthetic_video_frames': 8,
        'recovery_child_exit_codes': [first.returncode, second.returncode],
        'recovery_process_ids': [exited['pid'], final['pid']], 'attempts': [1, 2],
        'normalizations_across_recovery': 1, 'repeated_player_or_court_predictions': 0,
        'reused_raw_frames': 2, 'resumed_ball_frame_indices': final['ball_full_frame_indices'],
        'continuous_ball_frame_indices': continuous_final['ball_full_frame_indices'],
        'resumed_ball_model_frame_indices': final['ball_frame_indices'],
        'continuous_ball_model_frame_indices': continuous_final['ball_frame_indices'],
        'tiles_enabled': mid_tile, 'model_views_per_complete_frame': 5 if mid_tile else 1,
        'actual_yolo26_ball_extract_candidates': True,
        'incomplete_frame_index': 2 if mid_tile else None,
        'incomplete_view_index': 1 if mid_tile else None,
        'incomplete_frame_absent_from_committed_journal': mid_tile,
        'empty_raw_ball_frame_index': 4 if mid_tile else None,
        'empty_raw_frame_committed_on_resume': mid_tile,
        'model_view_predictions_before_exit': exited['ball_predict_calls'],
        'resumed_ball_model_view_predictions': final['ball_predict_calls'],
        'actual_ffmpeg_normalization_and_browser_encode': True, 'actual_native_pipeline_and_worker': True,
        'actual_result_builder_and_generation_publication': True, 'retained_input_prefix_audits_unchanged': True,
        'semantic_exports_and_decoded_annotated_pixels_equal': True,
        'actual_model_loads_or_forwards': 0, 'asset_admission_and_model_results': 'EXPLICIT_SYNTHETIC_BOUNDARIES',
        'forbidden_import_attempts_or_loaded_runtimes': [], 'hardware_resource_probes': False,
        'real_match_accuracy_long_match_or_live_qualification': False}
    (tmp_path / 'native_worker_process_recovery_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    PROCESS_OBSERVATIONS.append(evidence)


def test_hard_process_loss_resumes_same_worker_job_from_two_committed_ball_frames(tmp_path):
    _exercise_worker_recovery(tmp_path)


def test_hard_process_loss_during_first_ball_tile_replays_only_incomplete_suffix(tmp_path):
    _exercise_worker_recovery(tmp_path, mid_tile=True)

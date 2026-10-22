"""Actual private corrected result normalization and streamed report preparation.

Source imagery, coordinates, reviews and retained model hashes are explicit
synthetic fixtures. No detector/physical/prediction accuracy is qualified.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import tracemalloc
from types import MappingProxyType

import pytest

from tests.test_corrected_exports import fresh_forecast, export, digests, read
from tests.test_corrected_rendering import render
from tests.test_corrected_rule_rebuild import unchanged
from tests.test_corrected_shot_rebuild import (
    advance_equivalent_generation, build_shot_base, no_model_imports,
)


OBSERVATIONS = []
REPORT_FILES = {'report.json', 'readable_report.html', 'telemetry.csv',
                'corrected_preparation_manifest.json'}


def object_file(directory, name, value):
    path = directory/name
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return path


def build_complete_base(tmp_path, *, frame_count=12, omit=(), provenance_present=True,
                        provenance_change=None, with_optional=False):
    """Publish actual native-format complete observations plus original bytes.

    Original rule/shot/timeline assets are written from the real verified
    unchanged reproducer. Null metric/pose/racket evidence remains unavailable.
    The original annotation slot is an explicitly synthetic unannotated video;
    every correction must replace it with actual freshly rendered overlays.
    """
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.analysis_resume import REQUIRED_PIPELINE_FILES
    from src.api.result_storage import ResultStorage
    from src.api.results import build_result, IDENTITY_SCOPE, PARTICIPANT_SEMANTICS
    from src.court.court_geometry import TennisCourtGeometry
    from src.scoring.corrected_exports import _physical_events
    from src.scoring.point_timeline import native_match_analytics
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    from src.scoring.rule_replay_source import _source_event
    from src.utils.analytics_json_artifact import write_membership_analytics, write_match_analytics
    from src.utils.json_array_artifact import write_json_array_artifact
    from src.utils.point_timeline_artifact import publish_point_timeline_manifest

    fixture = build_shot_base(tmp_path, frame_count=frame_count,
        repeated_hits=frame_count > 12, playable_source=True)
    directory = fixture.directory/'complete-original-assets'
    directory.mkdir()
    files = {path.name: path for path in fixture.snapshot.files_directory.iterdir()
             if path.name not in {'result.json', 'summary.json'}}
    for name, path in tuple(files.items()):
        destination = directory/name
        shutil.copyfile(path, destination)
        files[name] = destination
    detections = read(files['detections.json'])
    count, fps = frame_count, 30.
    samples = [{'frame_index': index, 'timestamp_seconds': index/fps,
                'ground_position_m': None, 'speed_kmh': None,
                'position_fit_rmse_m': None} for index in range(count)]
    trajectories = [{'frame_index': index, 'timestamp_seconds': index/fps,
        'x_px': frame['ball']['position_px'][0], 'y_px': frame['ball']['position_px'][1],
        'court_x_m': None, 'court_y_m': None, 'speed_kmh': None,
        'confidence': frame['ball']['confidence'], 'state': frame['ball']['state'],
        'source': 'EXPLICIT_SYNTHETIC_IMAGE_OBSERVATION'}
        for index, frame in enumerate(detections['frames'])]
    court = read(files['court_geometry.json'])
    canonical = TennisCourtGeometry.get_canonical_keypoints().tolist()
    court.update(reprojection_error_px=.25, canonical_keypoints_m=canonical,
        court_keypoints_px=[[100*x, 100*y] for x, y in canonical])
    files['court_geometry.json'] = object_file(directory, 'court_geometry.json', court)
    files['trajectories.json'] = object_file(directory, 'trajectories.json',
        {'schema_version': '1.0', 'ball_trajectory': trajectories})
    files['player_motion.json'] = object_file(directory, 'player_motion.json',
        {'schema_version': '2.0', 'metric_measurements_available': False,
         'ground_position_source': 'UNAVAILABLE_UNQUALIFIED_SYNTHETIC_FIXTURE',
         'pose_frames': [], **{key: {'samples': samples, 'observed_distance_m': None,
            'mean_speed_kmh': None, 'max_speed_kmh': None} for key in ('player_1', 'player_2')}})
    files['racket_tracking.json'] = object_file(directory, 'racket_tracking.json',
        {'schema_version': '1.0', 'enabled': False, 'frames': []})
    people = []
    for index, frame in enumerate(detections['frames']):
        people.append({'frame_index': index, 'timestamp_seconds': index/fps,
            'people': [{'bbox': frame['player_'+str(role)]['bbox'],
                'confidence': frame['player_'+str(role)]['confidence'],
                'sourceTrackId': frame['player_'+str(role)]['source_track_id'],
                'courtRole': role, 'participantStatus': 'COURT_ROLE_CANDIDATE'} for role in (1, 2)]})
    files['image_people.json'] = object_file(directory, 'image_people.json',
        {'schema_version': '1.0', 'coordinateSystem': 'normalized_video_pixels',
         'identityScope': IDENTITY_SCOPE, 'participantSemantics': PARTICIPANT_SEMANTICS,
         'accuracyValidated': False, 'frames': people})
    files['media_metadata.json'] = object_file(directory, 'media_metadata.json',
        {'schema_version': '1.0',
         'original': {'width': 1280, 'height': 960, 'fps': fps, 'frames': count,
                      'durationSeconds': count/fps, 'sourceSha256': fixture.source_sha},
         'normalization': {'status': 'EXPLICIT_SYNTHETIC_ALREADY_CFR_FIXTURE'},
         'timeMapping': {'status': 'FRAME_MAPPING_UNAVAILABLE',
                         'validMeasurementTransitionIntervals': []}})
    inference = directory/'inference_config.yaml'
    shutil.copyfile(files['run_config.yaml'], inference)
    files['inference_config.yaml'] = inference
    for name, key in (
        ('event_candidates.json', 'event_candidates'),
        ('event_verification_trace.json', 'event_verification_trace'),
        ('player_metrics.json', 'player_metrics'), ('ball_metrics.json', 'ball_metrics'),
        ('temporal_spacing_audit.json', 'records'), ('temporal_detour_audit.json', 'records'),
        ('camera_intervals.json', 'intervals'), ('match_identity.json', 'frames'),
        ('stationary_candidate_audit.json', 'records'), ('pixel_motion_audit.json', 'records'),
        ('patch_persistence_audit.json', 'records')):
        files[name] = object_file(directory, name, {'schema_version': '1.0',
            'analysis_id': 'fixture', 'scope': 'EXPLICIT_SYNTHETIC_EMPTY_AUDIT_NOT_QUALIFIED',
            'independent_accuracy_validated': False, key: []})
    # Preserve real aligned interval and identity source records when available.
    files['camera_intervals.json'] = object_file(directory, 'camera_intervals.json',
        {'schema_version': '1.0', 'analysis_id': 'fixture',
         'frames': [row['interval'] for row in fixture.document['source_frames']]})
    files['match_identity.json'] = object_file(directory, 'match_identity.json',
        {'schema_version': '1.0', 'analysis_id': 'fixture',
         'frames': [row['identity'] for row in fixture.document['source_frames']]})
    replay_scratch = fixture.scratch/'original-native-export'
    replay_scratch.mkdir()
    with reproduce_rule_timeline(files['native_rule_replay.json'], replay_scratch,
            artifact_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'),
            source_sha256=fixture.source_sha, analysis_id='fixture', analysis_revision=0) as replay:
        owner, engine = replay.owner, replay.engine
        analytics = native_match_analytics(owner, replay.shots, engine.state, authoritative=True)
        for name, kind in (('point_analytics.json', 'points'), ('rallies.json', 'rallies')):
            files[name] = directory/name
            write_membership_analytics(files[name], kind, owner.rows(kind), lambda value: value,
                                       schema_version='2.0')
        files['match_analytics.json'] = directory/'match_analytics.json'
        write_match_analytics(files['match_analytics.json'], analytics, lambda value: value)
        physical = []
        for row in owner.rows('replay_inputs'):
            event, _ = _source_event(row['source_input'], replay.source_frames, 'fixture', fixture.source_sha)
            if row['source_input']['reviewed_evidence_ref'] is None:
                physical.append(event)
        for name, key, rows, metadata in (
            ('serve_executions.json', 'serve_executions', owner.rows('attempts'),
                {'schema_version': '2.0', 'analysis_revision': 0}),
            ('shot_play_scope.json', 'shots', owner.member_rows(),
                {'schema_version': '2.0', 'analysis_revision': 0}),
            ('scoring_events.json', 'scoring_events', owner.rows('transitions'), {'schema_version': '2.0'}),
            ('shot_events.json', 'shot_events', (shot.to_dict() for shot in replay.shots), {'schema_version': '1.0'}),
            ('line_calls.json', 'line_calls', (call.to_dict() for call in engine.native_line_calls()),
                {'schema_version': '1.0'}),
            ('match_events.json', 'events', _physical_events(event for event in physical), {'schema_version': '1.0',
                'analysis_id': 'fixture', 'analysis_revision': 0, 'events_count': len(physical)})):
            files[name] = directory/name
            write_json_array_artifact(files[name], metadata, key, rows, allow_nan=False)
        files['score_history.json'] = directory/'score_history.json'
        engine.save_history_json(files['score_history.json'])
        files['match_state.json'] = object_file(directory, 'match_state.json',
            {'schema_version': '1.0', **engine.state.to_dict(), 'authoritative_events_enabled': True,
             'semantic_role': 'INFERRED_SCORING_STATE'})
    publish_point_timeline_manifest(directory, analysis_id='fixture', analysis_revision=0,
        source_sha256=fixture.source_sha, scoring_revision=0, authoritative_events_enabled=True)
    files['point_timeline_manifest.json'] = directory/'point_timeline_manifest.json'
    files['metrics.json'] = object_file(directory, 'metrics.json', {'schema_version': '1.0',
        'total_frames': count, 'processing_fps': 3.14159, 'authoritative_events_enabled': True,
        'production_qualified': False, 'ball_backend': 'EXPLICIT_SYNTHETIC_FIXTURE',
        'tracking_strategy': 'EXPLICIT_SYNTHETIC_RETAINED_OBSERVATIONS'})
    for name in ('annotated.mp4', 'original.mp4'):
        destination = directory/name
        shutil.copyfile(files['source.mp4'], destination)
        files[name] = destination
    if with_optional:
        from src.api.media_timing import frame_mapping
        stats = directory/'normalization_frame_stats.txt'
        stats.write_text(''.join(f'{index} {index} {10**18+index*100} 1/3000 {index} 1/30\n'
                                 for index in range(count)), encoding='utf-8')
        mapping = directory/'source_frame_mapping.jsonl'
        frame_mapping(stats, mapping, fps, source_frame_count=count)
        files[stats.name], files[mapping.name] = stats, mapping
        files['pose_tracking.json'] = object_file(directory, 'pose_tracking.json',
            {'schema_version': '1.0', 'pose_frames': [],
             'scope': 'EXPLICIT_SYNTHETIC_NO_POSE_OBSERVATIONS'})
        source_credit = {'id': 'explicit-synthetic-source',
            'title': 'EXPLICIT SYNTHETIC FIXTURE; NO REAL MEDIA RIGHTS ASSERTION',
            'author': 'Synthetic <author> & fixture',
            'sourcePage': 'https://commons.wikimedia.org/wiki/File:Synthetic_fixture_only.webm',
            'license': 'CC BY 4.0', 'licenseUrl': 'https://creativecommons.org/licenses/by/4.0/',
            'derivativeLicense': 'CC BY 4.0',
            'modifications': 'EXPLICIT SYNTHETIC original test transform.',
            'sourceSha256': fixture.source_sha, 'inputSha256': fixture.source_sha}
        files['media_source.json'] = object_file(directory, 'media_source.json', source_credit)
        files['annotated_web_credit.json'] = object_file(directory, 'annotated_web_credit.json',
            {'schemaVersion': 1, 'analysisId': 'fixture',
             'inputSha256': fixture.source_sha, 'sourceSha256': fixture.source_sha,
             'annotatedSha256': '0'*64, 'modifications': 'STALE_OLD_ANNOTATION_MUST_NOT_BE_COPIED'})
    assert set(REQUIRED_PIPELINE_FILES) <= set(files)
    job = {'id': 'fixture', 'filename': 'Tennis <script>alert(1)</script> é 🎾.mp4',
           'extension': '.mp4', 'owner_id': 'synthetic-owner',
           'started_at': '2026-10-18T03:00:00+00:00',
           'completed_at': '2026-10-18T03:00:01+00:00'}
    # Build an actual bounded original result before publishing, then preserve
    # explicit synthetic captured inference hashes as immutable provenance.
    with ResultStorage(fixture.directory) as storage:
        original_result = build_result(directory, job, _storage=storage)
        provenance = copy.deepcopy(original_result['provenance'])
        provenance.update(bundleVersion='EXPLICIT_SYNTHETIC_CAPTURED_INFERENCE',
            bundleSha256='d'*64, configSha256=hashlib.sha256(inference.read_bytes()).hexdigest(),
            checkpointHashes={role: {'filename': 'explicit-synthetic-'+role+'.pt', 'sha256': digit*64}
                for role, digit in (('players', 'a'), ('ball', 'b'), ('court', 'c'))})
        provenance.update(provenance_change or {})
        if provenance_present:
            original_result['provenance'] = provenance
        else:
            original_result.pop('provenance', None)
        token = publish_analysis_generation(storage, original_result, fixture.directory,
            files={name: path for name, path in files.items() if name not in set(omit)},
            analysis_revision=0, expected_generation=fixture.snapshot.generation)
    fixture.snapshot = resolve_generation(fixture.directory)
    fixture.request.update(base_generation=token,
        base_rule_replay_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'))
    fixture.marker = (fixture.directory/'.result/current.store').read_bytes()
    fixture.manifest = digests(fixture.snapshot.files_directory)
    fixture.job, fixture.inference_provenance = job, provenance if provenance_present else None
    return fixture


@contextmanager
def complete_inputs(fixture):
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        exports = export(rebuilt, shots, forecast, fixture.scratch)
        rendered = render(rebuilt, shots, forecast, exports, fixture.scratch)
        yield rebuilt, shots, forecast, exports, rendered


def prepare(rebuilt, shots, forecast, exports, rendered, scratch, job):
    from src.scoring.corrected_preparation import prepare_corrected_result
    return prepare_corrected_result(rebuilt, shots, forecast, exports, rendered, scratch, job=job)


def expected_inventory(fixture, exports, rendered):
    from src.api.analysis_resume import REQUIRED_PIPELINE_FILES, OPTIONAL_PIPELINE_FILES
    result = (set(REQUIRED_PIPELINE_FILES) | set(exports.files) | set(rendered.files)
        | {name for name in OPTIONAL_PIPELINE_FILES if fixture.snapshot.resolve_file(name) is not None}
        | {'original'+fixture.job['extension']} | REPORT_FILES)
    if fixture.snapshot.resolve_file('media_source.json') is not None:
        result.add('annotated_web_credit.json')
    return result


def assert_prepared(prepared, fixture, exports, rendered):
    from src.api.result_storage import ResultStorage
    assert prepared.directory.parent == fixture.scratch.resolve()
    assert prepared.directory not in {fixture.scratch.resolve(), exports.directory, rendered.directory}
    assert isinstance(prepared.storage, ResultStorage)
    prepared.storage._require_open()
    assert prepared.base_generation == fixture.snapshot.generation and prepared.analysis_revision == 1
    assert set(prepared.files) == expected_inventory(fixture, exports, rendered)
    assert set(digests(prepared.directory)) == set(prepared.files)
    with pytest.raises(TypeError):
        prepared.files['injected'] = prepared.directory/'injected'
    manifest = read(prepared.files['corrected_preparation_manifest.json'])
    assert hashlib.sha256(prepared.files['corrected_preparation_manifest.json'].read_bytes()).hexdigest() == prepared.manifest_sha256
    assert manifest['analysis_id'] == 'fixture' and manifest['analysis_revision'] == 1
    assert manifest['base_generation'] == fixture.snapshot.generation
    assert manifest['source_sha256'] == fixture.source_sha
    assert manifest['status'] == 'PRIVATE_PENDING_PUBLICATION'
    assert manifest['published'] is False and manifest['independent_accuracy_validated'] is False
    assert manifest['export_manifest_sha256'] == exports.manifest_sha256
    assert manifest['render_manifest_sha256'] == rendered.manifest_sha256
    assert set(manifest['artifacts']) == set(prepared.files)-{'corrected_preparation_manifest.json'}
    for name, record in manifest['artifacts'].items():
        path = prepared.files[name]
        assert path == prepared.directory/name and path.stat().st_nlink == 1
        assert record == {'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name, path in exports.files.items():
        assert prepared.files[name].read_bytes() == path.read_bytes()
    for name, path in rendered.files.items():
        assert prepared.files[name].read_bytes() == path.read_bytes()
    return manifest


def test_actual_complete_preparation_normalizes_fresh_double_fault_and_streams_reports(tmp_path, monkeypatch):
    from src.api import results
    fixture = build_complete_base(tmp_path)
    fixture.request['reason'] = '<script>reviewed-target</script>'
    fixture.request['reviewer_id'] = 'Synthetic reviewer <owner> "quoted"'
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        before_export, before_render = digests(exports.directory), digests(rendered.directory)
        def no_current_weights(*_args, **_kwargs):
            raise AssertionError('Delayed correction tried to recompute inference provenance from current weights')
        monkeypatch.setattr(results, 'provenance', no_current_weights)
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            assert_prepared(prepared, fixture, exports, rendered)
            result = prepared.result
            assert result['summary']['id'] == 'fixture' and result['summary']['frames'] == 12
            assert result['analysisRevision'] == 1
            point = result['pointAnalysis']
            assert point['sourceSha256'] == fixture.source_sha
            assert point['summary']['totalPointsPlayed'] == 1 and point['summary']['totalRallies'] == 0
            assert point['serveByPlayer']['player_1']['doubleFaults'] == 1
            assert point['serveByPlayer']['player_1']['firstServePercentage'] is None
            assert result['pipelineMetrics']['pipelineFps'] is None
            assert result['provenance'] == fixture.inference_provenance
            assert result['ballMetrics'] is None and result['world3d'] is None
            assert result['playerMetrics'] is None
            prediction = result['nativeForecast']
            assert prediction['analysisId'] == 'fixture' and prediction['analysisRevision'] == 1
            assert prediction['sourceSha256'] == fixture.source_sha
            assert prediction['baseGeneration'] == fixture.snapshot.generation
            assert prediction['current']['sample_count'] == 1
            assert prediction['current']['serve_estimates'][0]['probability'] == .4
            assert len(prediction['history']) == 7
            correction = result['analysisCorrection']
            assert correction['analysisId'] == 'fixture' and correction['analysisRevision'] == 1
            assert correction['baseGeneration'] == fixture.snapshot.generation
            assert correction['authorizationStatus'] == 'PRIVATE_LOCAL_REVIEW_NOT_AUTHENTICATED'
            assert correction['review']['reviewerId'] == fixture.request['reviewer_id']
            assert correction['review']['reason'] == fixture.request['reason']
            assert correction['published'] is False and correction['independentAccuracyValidated'] is False
            report = read(prepared.files['report.json'])
            assert report['analysisRevision'] == 1
            assert report['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 1
            assert report['nativeForecast']['current'] == prediction['current']
            html = prepared.files['readable_report.html'].read_text(encoding='utf-8')
            assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
            assert '<script>alert(1)</script>' not in html
            assert '&lt;script&gt;reviewed-target&lt;/script&gt;' in html
            assert '<script>reviewed-target</script>' not in html
            assert '&lt;owner&gt;' in html and '&quot;quoted&quot;' in html
            assert 'Analysis revision: 1' in html and 'Statistical forecasts' in html
            assert 'uncalibrated' in html and 'data:image/jpeg;base64,' in html
            rows = list(csv.reader(io.StringIO(prepared.files['telemetry.csv'].read_text(encoding='utf-8'))))
            assert len(rows) == 13 and rows[0][:3] == ['analysis_id', 'frame', 'normalized_seconds']
            assert [int(row[1]) for row in rows[1:]] == list(range(12))
            assert {row[0] for row in rows[1:]} == {'fixture'}
            owned = prepared.storage.root()
            assert owned['summary']['id'] == 'fixture' and owned['analysisRevision'] == 1
            assert owned['nativeForecast']['current'] == prediction['current']
            frozen_files = dict(prepared.files)
        assert prepared.storage._connection is None
        assert all(path.is_file() for path in frozen_files.values())
        assert digests(exports.directory) == before_export and digests(rendered.directory) == before_render
    unchanged(fixture)
    OBSERVATIONS.append({'complete_private_preparation': {'normalized_frames': 12,
        'analysis_revision': 1, 'fresh_double_faults': 1, 'causal_point_samples': 1,
        'report_json': True, 'escaped_html_with_real_video_images': True, 'csv_rows': 13,
        'production_qualified': False, 'published': False}})


def test_prepared_history_is_owned_by_its_result_storage_and_closes_with_context(tmp_path):
    from src.api.result_storage import DiskSeries
    fixture = build_complete_base(tmp_path, frame_count=272)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        assert len(forecast.history) > 64
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            history = prepared.result['nativeForecast']['history']
            assert isinstance(history, DiskSeries) and history._storage is prepared.storage
            assert len(history) == len(forecast.history) > 64
            assert history[-1]['forecast']['sample_count'] == 1
            assert history[0]['forecast']['sample_count'] == 0
            assert read(prepared.files['report.json'])['nativeForecast']['history'][-1] == history[-1]
        with pytest.raises(RuntimeError, match='open|closed'):
            len(history)
    unchanged(fixture)


def test_optional_sealed_observations_keep_exact_bytes_and_derivative_credit_is_fresh(tmp_path):
    from src.api.analysis_resume import OPTIONAL_PIPELINE_FILES
    fixture = build_complete_base(tmp_path, with_optional=True)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            assert_prepared(prepared, fixture, exports, rendered)
            assert len(prepared.files) == 54
            for name in set(OPTIONAL_PIPELINE_FILES)-{'landing_zones.json'}:
                assert prepared.files[name].read_bytes() == fixture.snapshot.resolve_file(name).read_bytes()
                assert prepared.files[name].stat().st_nlink == 1
            credit = read(prepared.files['annotated_web_credit.json'])
            assert credit['analysisId'] == 'fixture'
            assert credit['inputSha256'] == credit['sourceSha256'] == fixture.source_sha
            assert credit['annotatedSha256'] == hashlib.sha256(prepared.files['annotated_web.mp4'].read_bytes()).hexdigest()
            assert credit['annotatedSha256'] != '0'*64
            assert 'Fresh corrected overlays' in credit['modifications']
            assert 'STALE_OLD_ANNOTATION' not in credit['modifications']
            assert prepared.files['annotated_web_credit.json'].read_bytes() != fixture.snapshot.resolve_file('annotated_web_credit.json').read_bytes()
            source = prepared.result['mediaSource']
            assert source['modifications'] == credit['modifications']
            assert read(prepared.files['report.json'])['mediaSource']['modifications'] == credit['modifications']
            html = prepared.files['readable_report.html'].read_text(encoding='utf-8')
            assert 'Fresh corrected overlays' in html
            assert 'Synthetic &lt;author&gt; &amp; fixture' in html
            assert '<author>' not in html
            assert prepared.result['exports']['sourceFrameMapping'] == '/api/analyses/fixture/download/source-timing'
            assert prepared.result['poseTracking'] is None
    unchanged(fixture)


def test_missing_retained_inference_provenance_stays_explicitly_unavailable(tmp_path, monkeypatch):
    from src.api import results
    fixture = build_complete_base(tmp_path, provenance_present=False)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        def no_current_weights(*_args, **_kwargs):
            raise AssertionError('Absent retained provenance triggered current checkpoint lookup')
        monkeypatch.setattr(results, 'provenance', no_current_weights)
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            provenance = prepared.result['provenance']
            assert provenance is None or (provenance.get('bundleSha256') is None
                and not provenance.get('checkpointHashes') and provenance.get('productionQualified') is False)
            assert any('provenance' in warning.lower() and 'unavailable' in warning.lower()
                       for warning in prepared.result['warnings'])
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['config_mismatch', 'oversized'])
def test_retained_inference_provenance_must_match_sealed_config_and_bounded_metadata(tmp_path, damage):
    change = {'configSha256': '0'*64} if damage == 'config_mismatch' else {'notes': 'x'*(70*1024)}
    fixture = build_complete_base(tmp_path, provenance_change=change)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with pytest.raises(ValueError, match='provenance|config|bound|size|limit'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                pytest.fail('Mismatched or unbounded retained inference provenance was accepted')
        assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
    unchanged(fixture)


def test_original_label_and_analysis_time_come_from_sealed_summary(tmp_path):
    fixture = build_complete_base(tmp_path)
    job = {**fixture.job, 'filename': 'UNTRUSTED_REPLACEMENT_LABEL.mp4',
           'started_at': '2099-01-01T00:00:00+00:00',
           'completed_at': '2099-01-02T00:00:00+00:00'}
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, job) as prepared:
            assert prepared.result['summary']['label'] == fixture.job['filename']
            assert prepared.result['summary']['analyzedAt'] == fixture.job['completed_at']
            assert 'UNTRUSTED_REPLACEMENT_LABEL' not in prepared.files['readable_report.html'].read_text(encoding='utf-8')
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['summary', 'forecast_history'])
def test_exposed_result_mutation_before_context_exit_invalidates_the_preparation_seal(tmp_path, damage):
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        candidate = None
        with pytest.raises(ValueError, match='result|report|changed|prepared|serialized'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
                candidate = prepared
                if damage == 'summary':
                    prepared.result['summary']['frames'] = 999
                else:
                    history = prepared.result['nativeForecast']['history']
                    row = history[0]
                    row['forecast']['sample_count'] = 999
                    history[0] = row
        assert candidate is not None and candidate.storage._connection is None
        assert not (candidate.directory/'corrected_preparation_manifest.json').exists()
    unchanged(fixture)


@pytest.mark.parametrize('missing', ['pixel_motion_audit.json', 'original.mp4'])
def test_missing_required_sealed_input_has_no_mutable_root_fallback(tmp_path, missing):
    fixture = build_complete_base(tmp_path, omit=(missing,))
    assert (fixture.directory/'complete-original-assets'/missing).is_file()
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with pytest.raises(ValueError, match='required|missing|original|observation|artifact'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                pytest.fail('Incomplete sealed base was accepted')
        assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['foreign_id', 'extension_path', 'different_scratch', 'closed_forecast'])
def test_preparation_requires_current_owned_context_and_original_job_binding(tmp_path, damage):
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        job, scratch, argument = dict(fixture.job), fixture.scratch, forecast
        if damage == 'foreign_id':
            job['id'] = 'different-analysis'
        elif damage == 'extension_path':
            job['extension'] = '/../outside.mp4'
        elif damage == 'different_scratch':
            scratch = fixture.directory
        else:
            # A retained handle after actual owner closure must never fall back
            # to already serialized forecast data from the export directory.
            argument.owner.__exit__(None, None, None)
        with pytest.raises((ValueError, RuntimeError), match='job|analysis|extension|scratch|owner|open|closed|binding|path'):
            with prepare(rebuilt, shots, argument, exports, rendered, scratch, job):
                pytest.fail('Invalid bound preparation was accepted')
        assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
    unchanged(fixture)


@pytest.mark.parametrize('name', ['annotated_web.mp4', 'corrected_render_manifest.json'])
def test_changed_render_output_rejects_before_preparation_seal(tmp_path, name):
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        path = rendered.files[name]
        original = path.read_bytes()
        try:
            path.write_bytes(b'SYNTHETIC_CHANGED_RENDER_BEFORE_PREPARATION')
            with pytest.raises((ValueError, json.JSONDecodeError), match='render|artifact|hash|manifest|changed|Expecting'):
                with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                    pytest.fail('Changed render was accepted')
            assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
        finally:
            path.write_bytes(original)
    unchanged(fixture)


def test_consumer_failure_closes_result_owner_removes_only_pending_seal_and_preserves_previous(tmp_path):
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as first:
            first_path, first_hashes = first.directory, digests(first.directory)
        failed = None
        with pytest.raises(RuntimeError, match='SYNTHETIC_PREPARED_CONSUMER_FAILURE'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as candidate:
                failed = candidate
                assert candidate.directory != first_path
                raise RuntimeError('SYNTHETIC_PREPARED_CONSUMER_FAILURE')
        assert failed is not None and failed.storage._connection is None
        assert not (failed.directory/'corrected_preparation_manifest.json').exists()
        assert digests(first_path) == first_hashes
    unchanged(fixture)


def test_actual_generation_advance_during_consumption_rejects_prepared_seal(tmp_path):
    from src.api.analysis_generation import resolve_generation
    fixture = build_complete_base(tmp_path)
    advanced, candidate = [], None
    with pytest.raises(ValueError, match='current generation|base'):
        with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
                candidate = prepared
                advanced.append(advance_equivalent_generation(fixture))
    assert candidate is not None and candidate.storage._connection is None
    assert not (candidate.directory/'corrected_preparation_manifest.json').exists()
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert digests(fixture.snapshot.files_directory) == fixture.manifest


@pytest.mark.parametrize('target', ['match_state.json', 'readable_report.html'])
def test_later_writer_cannot_rebind_changed_copied_semantics_or_an_earlier_report(tmp_path, monkeypatch, target):
    from src.scoring import corrected_preparation as module
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        original_export, original_render = digests(exports.directory), digests(rendered.directory)
        mutated = []
        if target == 'match_state.json':
            native = module.readable_report_chunks
            def chunks(result, directory):
                for index, chunk in enumerate(native(result, directory)):
                    if index == 0:
                        path = directory/target
                        document = read(path)
                        document['player_2']['points_display'] = 'FORGED999'
                        object_file(directory, target, document)
                        mutated.append(path)
                    yield chunk
            monkeypatch.setattr(module, 'readable_report_chunks', chunks)
        else:
            native = module.telemetry_csv_chunks
            def chunks(result, *args, **kwargs):
                for index, chunk in enumerate(native(result, *args, **kwargs)):
                    if index == 0:
                        candidates = list(fixture.scratch.glob('corrected-prepared-*'))
                        assert len(candidates) == 1
                        path = candidates[0]/target
                        assert path.is_file()
                        path.write_text('FORGED_LATER_CSV_CALLBACK_HTML', encoding='utf-8')
                        mutated.append(path)
                    yield chunk
            monkeypatch.setattr(module, 'telemetry_csv_chunks', chunks)
        with pytest.raises(ValueError, match='artifact|copied|changed|hash|report'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                pytest.fail('A writer callback rebound altered private bytes into the final seal')
        assert len(mutated) == 1 and mutated[0].is_file()
        assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
        assert digests(exports.directory) == original_export and digests(rendered.directory) == original_render
    unchanged(fixture)


@pytest.mark.parametrize('writer', ['html', 'csv'])
def test_mid_report_writer_failure_closes_owner_and_preserves_earlier_complete_stage(tmp_path, monkeypatch, writer):
    from src.scoring import corrected_preparation as module
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as first:
            first_path, first_hashes = first.directory, digests(first.directory)
        captured = []
        build = module.build_result
        def build_result(*args, **kwargs):
            captured.append(kwargs['_storage'])
            return build(*args, **kwargs)
        monkeypatch.setattr(module, 'build_result', build_result)
        name = 'readable_report_chunks' if writer == 'html' else 'telemetry_csv_chunks'
        native = getattr(module, name)
        def chunks(*args, **kwargs):
            for chunk in native(*args, **kwargs):
                yield chunk
                raise RuntimeError('SYNTHETIC_MID_'+writer.upper()+'_WRITER_FAILURE')
        monkeypatch.setattr(module, name, chunks)
        with pytest.raises(RuntimeError, match='SYNTHETIC_MID_'+writer.upper()+'_WRITER_FAILURE'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                pytest.fail('Partial report serialization reached a prepared result')
        assert captured and all(owner._connection is None for owner in captured)
        assert digests(first_path) == first_hashes
        assert set(fixture.scratch.rglob('corrected_preparation_manifest.json')) == {
            first_path/'corrected_preparation_manifest.json'}
    unchanged(fixture)


def test_replacing_stored_root_without_changing_exposed_result_invalidates_seal(tmp_path):
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        candidate = None
        with pytest.raises(ValueError, match='result|report|changed|serialization'):
            with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
                candidate = prepared
                stored = prepared.storage.root()
                stored['summary']['frames'] = 999
                prepared.storage.set_root(stored)
                assert prepared.result['summary']['frames'] == 12
        assert candidate is not None and candidate.storage._connection is None
        assert not (candidate.directory/'corrected_preparation_manifest.json').exists()
    unchanged(fixture)


def test_transient_changed_producer_bytes_cannot_replace_frozen_export_copy_identity(tmp_path, monkeypatch):
    from src.scoring import corrected_preparation as module
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        path = exports.files['metrics.json']
        original = path.read_bytes()
        original_export, original_render = digests(exports.directory), digests(rendered.directory)
        mutated, attempted_copy = [], []
        native_inputs, native_copy = module._base_inputs, module._copy
        def base_inputs(*args, **kwargs):
            value = native_inputs(*args, **kwargs)
            changed = read(path)
            changed['timing_scope'] = 'FORGED_TRANSIENT_PRODUCER_METRICS'
            path.write_text(json.dumps(changed), encoding='utf-8')
            mutated.append(True)
            return value
        def copying(source, destination, expected):
            try:
                if source == path:
                    attempted_copy.append(True)
                return native_copy(source, destination, expected)
            finally:
                if source == path:
                    # A changed-then-restored producer must still fail. The
                    # expected digest comes from its already admitted seal.
                    path.write_bytes(original)
        monkeypatch.setattr(module, '_base_inputs', base_inputs)
        monkeypatch.setattr(module, '_copy', copying)
        try:
            with pytest.raises(ValueError, match='producer|artifact|bytes|hash|copy|changed'):
                with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job):
                    pytest.fail('Transient producer bytes replaced a sealed export artifact')
            assert mutated == [True]
            assert not list(fixture.scratch.rglob('corrected_preparation_manifest.json'))
        finally:
            path.write_bytes(original)
        assert digests(exports.directory) == original_export and digests(rendered.directory) == original_render
    unchanged(fixture)


def test_preparation_owned_python_buffers_stay_bounded_as_real_video_frames_and_history_grow(tmp_path):
    measurements = []
    for count in (12, 272):
        parent = tmp_path/str(count)
        parent.mkdir()
        fixture = build_complete_base(parent, frame_count=count)
        # Actual source encode, rules, relinking, causal forecasts, semantic
        # exports and both video codecs finish before preparation is traced.
        with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            try:
                with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
                    assert len(prepared.result['detections']['frames']) == count
                    history = prepared.result['nativeForecast']['history']
                    assert len(history) == len(forecast.history)
                    gc.collect()
                    retained, peak = tracemalloc.get_traced_memory()
                    measurements.append({'normalized_video_frames': count,
                        'accepted_inputs': len(rebuilt.engine._inputs), 'history_rows': len(history),
                        'completed_point_samples': forecast.current['sample_count'],
                        'retained_python_bytes_inside_context': retained, 'peak_python_bytes': peak,
                        'result_owner_disk_bytes': prepared.storage._path.stat().st_size,
                        'prepared_artifact_bytes': sum(path.stat().st_size for path in prepared.files.values()),
                        'elapsed_seconds_through_owned_context': time.perf_counter()-started})
            finally:
                tracemalloc.stop()
        unchanged(fixture)
    small, large = measurements
    assert small['history_rows'] == 7 and large['history_rows'] > 64
    assert large['retained_python_bytes_inside_context']-small['retained_python_bytes_inside_context'] < 1024*1024
    assert large['peak_python_bytes']-small['peak_python_bytes'] < 3*1024*1024
    OBSERVATIONS.append({'private_preparation_capacity': measurements,
        'includes': 'Independent copies, strict point/landing validation, native normalized result, fresh forecast replay/presentation and actual JSON/HTML/CSV serialization',
        'excludes': 'Fixture/source creation, corrected rules/shots/forecast/export/render, SQLite native memory, numerical and codec native allocations',
        'completed_point_growth_claimed': False,
        'long_match_or_live_qualification': False, 'independent_accuracy_validated': False})

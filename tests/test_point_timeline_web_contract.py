"""Actual saved native rule fixtures through the optional web/report bridge.

These files were produced by exact Phase6 source-block execution with supplied
rule evidence. Their 69-byte source identity is deliberately NOT a video.
This suite tests artifact, rule-contract and report integrity, not CV accuracy.
"""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from types import ModuleType

import pytest

from src.api import point_timeline_contract as contract
from src.api.reporting import point_timeline_report_chunks, readable_report
from src.api.result_storage import DiskSeries, ResultStorage
from src.utils import point_timeline_artifact as publication


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/point_bundle_pipeline_checks_v1/physical_events_v4/temp/pytest'
FIXTURE_MANIFESTS = {
    True: '42061cfc90870af131b153014917f6335db6a5316394fb5c95ac2de114881bd3',
    False: 'ea6195fe7d84c0d7dd179a4cc0272db71a28f5b87227b15aa06ab3be939e4bca',
}
FIXTURE_PHYSICAL_EVENTS = {
    True: '907b0adbedfd89b755e05981de4d0288cf9e0d0d0ad89d77ed44a0260148af2f',
    False: '8a033a30b66e6944ef0446c11fdbd59a9457b16696dd3e47f8ce2429af244893',
}
REVIEWED_SAVED = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/point_bundle_pipeline_checks_v1/reviewed_binding_green_v1/temp/pytest/test_exact_native_phase6_revie0'


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def fixture_bundle(directory, authority=True):
    saved = SAVED / ('test_exact_native_phase6_scori0' if authority else 'test_exact_native_phase6_scori1')
    manifest = saved / publication.MANIFEST_NAME
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FIXTURE_MANIFESTS[authority]
    assert hashlib.sha256((saved/'match_events.json').read_bytes()).hexdigest() == FIXTURE_PHYSICAL_EVENTS[authority]
    directory.mkdir(exist_ok=True)
    for name in (*publication.POINT_TIMELINE_FILES, publication.MANIFEST_NAME, 'match_events.json'):
        shutil.copyfile(saved / name, directory / name)
    shutil.copyfile(saved / 'synthetic_source_identity.bin', directory / 'source.mp4')
    return directory


def reviewed_fixture_bundle(directory):
    assert hashlib.sha256((REVIEWED_SAVED/publication.MANIFEST_NAME).read_bytes()).hexdigest() == '874df2461478bcb42d2c8829ab21dd3e281e169b0ab0fc9e71fd65cbab6fcd9d'
    assert hashlib.sha256((REVIEWED_SAVED/'match_events.json').read_bytes()).hexdigest() == '2f738b0c7470cb05bc27d29788f2bd26b5f978955a8cfb0ab50a3bbe65d2a195'
    directory.mkdir(exist_ok=True)
    for name in (*publication.POINT_TIMELINE_FILES, publication.MANIFEST_NAME, 'match_events.json'):
        shutil.copyfile(REVIEWED_SAVED/name, directory/name)
    shutil.copyfile(REVIEWED_SAVED/'synthetic_source_identity.bin', directory/'source.mp4')
    return directory


def reseal(directory, authority=True):
    return publication.publish_point_timeline_manifest(directory, analysis_id='fixture', analysis_revision=0,
        source_sha256=publication.artifact_sha256(directory / 'source.mp4'), scoring_revision=0,
        authoritative_events_enabled=authority)


def load(directory, authority=True, storage=None):
    return contract.load_point_analysis(directory, 'fixture', frame_count=16, authoritative=authority, storage=storage)


@pytest.mark.parametrize('authority', [True, False])
def test_actual_phase6_native_bundle_preserves_nulls_and_denominators(tmp_path, authority):
    directory = fixture_bundle(tmp_path / 'analysis', authority)
    value = load(directory, authority)
    assert value['status'] == 'UNQUALIFIED' and value['independentAccuracyValidated'] is False
    assert value['authoritativeEventsEnabled'] is authority
    assert value['summary']['totalPointsPlayed'] == (2 if authority else None)
    if authority:
        assert len(value['points']) == 2 and len(value['rallies']) == 1
        assert value['summary']['effectivePlaySeconds'] == .970
        assert value['summary']['timedCompletedRallies'] == value['summary']['completedRalliesDurationDenominator'] == 1
        assert [point['totalStrokesIncludingServe'] for point in value['points']] == [2, 0]
        assert value['points'][1]['durationSeconds'] is None
        assert value['serveByPlayer']['player_2']['doubleFaults'] == 1
        assert value['serveByPlayer']['player_1']['firstServePercentage'] == 0
    else:
        assert value['points'] == value['rallies'] == value['serveExecutions'] == []
        assert value['summary']['effectivePlaySeconds'] is None
        assert value['serveByPlayer']['player_1']['firstServePercentage'] is None


def test_empty_physical_event_export_cannot_back_nonempty_scoring_transitions(tmp_path):
    directory = fixture_bundle(tmp_path / 'analysis')
    dump(directory / 'match_events.json', {'schema_version': '1.0', 'analysis_id': 'fixture',
        'analysis_revision': 0, 'events_count': 0, 'events': []})
    with pytest.raises(ValueError, match='[Pp]hysical'):
        load(directory)


def test_authority_off_bundle_cannot_admit_reviewed_rule_transitions(tmp_path):
    directory = fixture_bundle(tmp_path/'off', False)
    reviewed = reviewed_fixture_bundle(tmp_path/'reviewed')
    signals = json.loads((reviewed/'scoring_events.json').read_text())['scoring_events']
    signal = next(row for row in signals if row['reviewed_rule_evidence_ref'] is not None)
    # Deliberately malformed authority-off input: a real reviewed source record
    # is attached to no point, attempt or rally so parent checks alone cannot
    # detect the authority violation. No physical records are manufactured.
    signal.update(point_uid=None, attempt_uid=None, rally_uid=None)
    dump(directory/'scoring_events.json', {'schema_version': '2.0', 'scoring_events': [signal]})
    reseal(directory, False)
    with pytest.raises(ValueError, match='authority'):
        load(directory, False)


@pytest.mark.parametrize('bounded', [False, True])
def test_actual_reviewed_let_is_separate_from_physical_events_and_preserves_slot(tmp_path, bounded):
    directory = reviewed_fixture_bundle(tmp_path/'reviewed')
    physical = json.loads((directory/'match_events.json').read_text())
    transitions = json.loads((directory/'scoring_events.json').read_text())['scoring_events']
    assert physical['events_count'] == len(physical['events']) == 5
    assert len(transitions) == 6
    assert all(row['event_type'] in contract.PHYSICAL_EVENT_TYPES for row in physical['events'])
    signals = [row for row in transitions if row['reviewed_rule_evidence_ref'] is not None]
    assert len(signals) == 1 and signals[0]['physical_event_type'] == 'SERVICE_LET'
    assert all(row['event_id'] != signals[0]['event_id'] for row in physical['events'])
    def check(value):
        assert value['status'] == 'UNQUALIFIED' and value['independentAccuracyValidated'] is False
        assert value['summary']['totalPointsPlayed'] == 1
        assert value['summary']['effectivePlaySeconds'] == .810
        assert [(row['officialAttempt'], row['replayIndex'], row['status']) for row in value['serveExecutions']] == [
            (1, 0, 'SERVICE_LET'), (1, 1, 'SERVE_IN_PLAY')]
        player = value['serveByPlayer']['player_1']
        assert player['contactExecutions'] == 2 and player['serviceLets'] == 1
        assert player['firstServesAttempted'] == player['firstServesIn'] == 1
        assert player['secondServesAttempted'] == 0
    if bounded:
        with ResultStorage(tmp_path) as storage:
            check(load(directory, storage=storage))
    else:
        check(load(directory))


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('damage', ['missing_record', 'extra_record', 'duplicate_id', 'count',
    'frame', 'type', 'timestamp', 'boolean_timestamp', 'reviewed_as_physical',
    'transition_timestamp', 'transition_type', 'empty_review_reference', 'blank_review_reference',
    'numeric_review_reference', 'physical_with_review_reference', 'reviewed_id_collision',
    'reviewed_without_reference'])
def test_actual_physical_export_and_scoring_transition_binding_is_required(tmp_path, damage, bounded):
    directory = fixture_bundle(tmp_path / 'analysis')
    physical_path, trace_path = directory/'match_events.json', directory/'scoring_events.json'
    physical = json.loads(physical_path.read_text())
    traces = json.loads(trace_path.read_text())
    row = physical['events'][0]
    if damage == 'missing_record':
        physical['events'].pop(0)
        physical['events_count'] -= 1
    elif damage == 'extra_record':
        extra = dict(row, event_id=99, frame=15, timestamp_s=1.5)
        physical['events'].append(extra)
        physical['events_count'] += 1
    elif damage == 'duplicate_id':
        physical['events'].append(row)
        physical['events_count'] += 1
    elif damage == 'count': physical['events_count'] += 1
    elif damage == 'frame': row['frame'] += 1
    elif damage == 'type': row['event_type'] = 'UNKNOWN_EVENT'
    elif damage == 'timestamp': row['timestamp_s'] = math.nextafter(row['timestamp_s'], math.inf)
    elif damage == 'boolean_timestamp': row['timestamp_s'] = True
    elif damage == 'reviewed_as_physical': row['event_type'] = 'SERVICE_LET'
    else:
        trace = traces['scoring_events'][0]
        if damage == 'transition_timestamp': trace['timestamp_s'] = math.nextafter(trace['timestamp_s'], math.inf)
        elif damage == 'transition_type': trace['physical_event_type'] = 'UNKNOWN_EVENT'
        elif damage == 'empty_review_reference': trace['reviewed_rule_evidence_ref'] = ''
        elif damage == 'blank_review_reference': trace['reviewed_rule_evidence_ref'] = '   '
        elif damage == 'numeric_review_reference': trace['reviewed_rule_evidence_ref'] = 1
        elif damage == 'physical_with_review_reference': trace['reviewed_rule_evidence_ref'] = 'supplied-review-ref'
        else:
            trace['physical_event_type'] = trace['rule_event_type'] = 'SERVICE_LET'
            trace['reviewed_rule_evidence_ref'] = 'supplied-review-ref' if damage == 'reviewed_id_collision' else None
    dump(physical_path, physical)
    dump(trace_path, traces)
    reseal(directory)
    if bounded:
        with ResultStorage(tmp_path) as storage:
            with pytest.raises(ValueError):
                load(directory, storage=storage)
    else:
        with pytest.raises(ValueError):
            load(directory)


def test_bounded_parser_and_normalized_arrays_remain_owned_disk_series(tmp_path):
    directory = fixture_bundle(tmp_path / 'analysis')
    with ResultStorage(tmp_path) as storage:
        value = load(directory, storage=storage)
        assert all(isinstance(value[field], DiskSeries) for field in ('points', 'rallies', 'serveExecutions'))
        stream = io.StringIO()
        storage.dump(value, stream)
        normalized = json.loads(stream.getvalue())
        assert normalized == load(directory)
        storage.set_root({'pointAnalysis': value})
        destination = tmp_path / 'point-result.sqlite3'
        storage.publish(destination)
    with ResultStorage.open(destination) as reader:
        restored = reader.root()['pointAnalysis']
        assert isinstance(restored['points'], DiskSeries)
        assert restored['points'][1]['durationSeconds'] is None
        assert ''.join(reader.json_chunks(restored)) == stream.getvalue()


@pytest.mark.parametrize('damage', ['artifact', 'source', 'missing', 'foreign_id', 'revision',
                                  'accuracy', 'scoring_revision', 'authority', 'extra_filename'])
def test_manifest_source_and_revision_mismatch_are_rejected(tmp_path, damage):
    directory = fixture_bundle(tmp_path / 'analysis')
    if damage == 'artifact':
        with (directory / 'rallies.json').open('a') as stream:
            stream.write(' ')
    elif damage == 'source':
        (directory / 'source.mp4').write_bytes(b'other synthetic identity, not media')
    elif damage == 'missing':
        (directory / 'score_history.json').unlink()
    elif damage == 'revision':
        dump(directory / 'match_events.json', {'analysis_id': 'fixture', 'analysis_revision': 1})
    else:
        path = directory / publication.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        if damage == 'foreign_id': manifest['analysis_id'] = 'other-analysis'
        elif damage == 'accuracy': manifest['independent_accuracy_validated'] = True
        elif damage == 'scoring_revision': manifest['scoring_revision'] = 1
        elif damage == 'authority': manifest['authoritative_events_enabled'] = False
        else: manifest['artifacts']['../outside.json'] = manifest['artifacts']['rallies.json']
        dump(path, manifest)
    with pytest.raises((ValueError, OSError)):
        load(directory)


@pytest.mark.parametrize('damage', ['source_context', 'match_id', 'dangling_point', 'duplicate_event',
                                  'rally_summary', 'serve_summary', 'first_percentage', 'negative_duration'])
def test_rehashed_mixed_or_malformed_native_records_are_rejected(tmp_path, damage):
    directory = fixture_bundle(tmp_path / 'analysis')
    name = 'match_analytics.json'
    if damage in ('source_context', 'duplicate_event', 'dangling_point'): name = 'scoring_events.json'
    elif damage == 'negative_duration': name = 'rallies.json'
    path = directory / name
    value = json.loads(path.read_text())
    if damage == 'source_context': value['scoring_events'][0]['context']['binding']['source_fingerprint'] = '0'*64
    elif damage == 'match_id': value['match_id'] = 'other-analysis'
    elif damage == 'dangling_point': value['scoring_events'][0]['point_uid'] = 'point_missing'
    elif damage == 'duplicate_event': value['scoring_events'].append(value['scoring_events'][0])
    elif damage == 'rally_summary': value['overview']['effective_play_seconds'] = 0
    elif damage == 'serve_summary': value['serve_analytics']['by_player']['2']['double_faults'] = 0
    elif damage == 'first_percentage': value['serve_analytics']['by_player']['1']['first_serve_percentage'] = None
    else: value['rallies'][0]['duration_s'] = -1
    dump(path, value)
    reseal(directory)
    with pytest.raises(ValueError):
        load(directory)


@pytest.mark.parametrize('target', ['match_events.json', publication.MANIFEST_NAME, 'source.mp4', 'rallies.json'])
def test_file_change_during_parse_is_rejected_even_when_header_revision_agrees(tmp_path, monkeypatch, target):
    directory = fixture_bundle(tmp_path / 'analysis')
    original = contract._point
    changed = False
    def during_parse(row):
        nonlocal changed
        if not changed:
            changed = True
            with (directory / target).open('ab') as stream:
                stream.write(b' ')
        return original(row)
    monkeypatch.setattr(contract, '_point', during_parse)
    with pytest.raises(ValueError):
        load(directory)


def test_legacy_missing_bundle_and_partial_native_bundle_are_distinct(tmp_path):
    assert load(tmp_path) is None
    dump(tmp_path / 'serve_executions.json', {'schema_version': '2.0', 'serve_executions': []})
    with pytest.raises(ValueError, match='manifest is missing'):
        load(tmp_path)


def test_manifest_write_failure_preserves_previous_bundle_manifest(tmp_path, monkeypatch):
    directory = fixture_bundle(tmp_path / 'analysis')
    path = directory / publication.MANIFEST_NAME
    before = path.read_bytes()
    def fail(*_args):
        raise OSError('supplied publication failure')
    monkeypatch.setattr(publication.os, 'replace', fail)
    with pytest.raises(OSError, match='supplied publication failure'):
        reseal(directory)
    assert path.read_bytes() == before
    assert not tuple(directory.glob('.point_timeline_manifest.json.*.tmp'))


def test_html_reports_provisional_timing_and_df_null_without_accuracy_claim(tmp_path):
    value = load(fixture_bundle(tmp_path / 'on'))
    report = ''.join(point_timeline_report_chunks({'pointAnalysis': value}))
    assert 'unqualified evidence' in report and 'not independently qualified match facts' in report
    assert '0.970 s' in report and '1 / 1 completed rallies' in report
    assert 'Double fault by Player 2' in report and 'Unavailable s' in report
    assert '0.000 s' not in report
    off = load(fixture_bundle(tmp_path / 'off', False), False)
    report = ''.join(point_timeline_report_chunks({'pointAnalysis': off}))
    assert 'event authority is withheld' in report and 'Zero exported records do not mean zero actual play' in report
    assert 'First slots attempted' not in report and '0.000 s' not in report


def exact_result_module():
    """Compile the current whole adapter; only unused pose constants avoid NumPy.

    No normalization logic is replaced. Real optional readers and ResultStorage
    run unchanged. This is not an HTTP, media or inference integration test.
    """
    source = ROOT / 'src/api/results.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    motion = ast.parse((ROOT / 'src/tracking/player_motion_tracking.py').read_text(encoding='utf-8'))
    constants = [node for node in motion.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in ('JOINT_NAMES', 'SKELETON_EDGES') for target in node.targets)]
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)
                 or node.module != 'src.tracking.player_motion_tracking']
    tree.body = constants+tree.body
    module = ModuleType('point_web_actual_result_adapter')
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    return module


def basic_artifacts(directory, authority):
    count, fps = 16, 30
    dump(directory / 'detections.json', {'metadata': {'frames': count, 'fps': fps, 'width': 100, 'height': 100},
         'frames': [{'frame_index': i, 'court_registration_valid': False} for i in range(count)]})
    dump(directory / 'trajectories.json', {'ball_trajectory': [{'frame_index': i, 'timestamp_seconds': i/fps,
         'state': 'MISSING', 'confidence': None} for i in range(count)]})
    dump(directory / 'metrics.json', {'processing_fps': 1, 'authoritative_events_enabled': authority})
    return {'id': 'fixture', 'filename': 'SYNTHETIC CONTRACT FIXTURE, NOT VIDEO.mp4', 'started_at': '2026-10-16T00:00:00Z'}


def test_exact_current_result_adapter_and_complete_html_use_same_saved_native_evidence(tmp_path):
    directory = fixture_bundle(tmp_path / 'analysis')
    job = basic_artifacts(directory, True)
    adapter = exact_result_module()
    with ResultStorage(tmp_path) as storage:
        result = adapter.build_result(directory, job, _storage=storage)
        assert result['pointAnalysis']['summary']['totalPointsPlayed'] == 2
        assert result['pointAnalysis']['points'][1]['durationSeconds'] is None
        assert result['summary']['capabilities']['scoring'] is False and result['matchState'] is None
        assert result['analysisRevision'] == 0
        report = readable_report(result)
        assert 'Rule timeline' in report and '0.970 s' in report
        (directory / 'rallies.json').write_text('{}')
        invalid = adapter.build_result(directory, job, _storage=storage)
        assert invalid['pointAnalysis'] is None and invalid['summary']['frames'] == 16
        assert any('Point timeline unavailable' in warning for warning in invalid['warnings'])

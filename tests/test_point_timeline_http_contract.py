"""Real saved-generation HTTP/download and account isolation for native points.

The supplied Phase6 rule fixtures are synthetic, with only their analysis
identity rebound to the real uploaded job. No video decode, worker or model
inference runs. These checks do not qualify tennis measurements.
"""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.jobs import Settings
from src.api.result_document import publish_result
from src.api.result_storage import ResultStorage
from src.api.results import build_result
from src.utils.point_timeline_artifact import POINT_TIMELINE_FILES, publish_point_timeline_manifest
from tests.test_point_timeline_web_contract import fixture_bundle, basic_artifacts


def rebind_fixture_identity(directory, analysis_id):
    """Rebind source-derived IDs without changing supplied rule decisions."""
    documents = {name: json.loads((directory/name).read_text()) for name in (*POINT_TIMELINE_FILES, 'match_events.json')}
    digest = hashlib.sha256((directory/'source.mp4').read_bytes()).hexdigest()
    def uid(kind, anchor):
        payload = [analysis_id, digest, 0, kind, anchor]
        return kind+'_'+hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
    mappings = {}
    events = documents['scoring_events.json']['scoring_events']
    event_keys = {}
    for event in events:
        context = event['context']
        new = uid('event', [event['event_id'], context['source_pts'], context['source_time_base'],
                            context['source_frame_index'], context['frame_index']])
        mappings[event['uid']] = event_keys[event['event_id']] = new
    for row in documents['point_analytics.json']['points']:
        mappings[row['uid']] = uid('point', event_keys[row['first_observed_event_id']])
    for row in documents['serve_executions.json']['serve_executions']:
        mappings[row['uid']] = uid('attempt', event_keys[row['contact_event_id']])
        if row['rally_uid'] is not None:
            mappings[row['rally_uid']] = uid('rally', mappings[row['uid']])
    def transform(value):
        if isinstance(value, dict):
            return {key: transform(child) for key, child in value.items()}
        if isinstance(value, list):
            return [transform(child) for child in value]
        if isinstance(value, str):
            return analysis_id if value == 'fixture' else mappings.get(value, value)
        return value
    for name, value in documents.items():
        (directory/name).write_text(json.dumps(transform(value), allow_nan=False), encoding='utf-8')
    publish_point_timeline_manifest(directory, analysis_id=analysis_id, analysis_revision=0,
        source_sha256=digest, scoring_revision=0, authoritative_events_enabled=True)


def test_real_owned_saved_native_result_html_download_and_restart(tmp_path):
    settings = Settings(tmp_path/'storage', tmp_path/'config.yaml', max_upload_bytes=64, require_auth=True)
    demos = tmp_path/'demos.json'
    demos.write_text('{"schemaVersion":1,"sources":[]}', encoding='utf-8')
    app = create_app(settings, start_worker=False, demos_manifest=demos)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        for client, name in ((alice, 'alice'), (bob, 'bob')):
            assert client.post('/api/auth/register', json={
                'username': name, 'password': 'synthetic-test-password-only-123'}).status_code == 201
        response = alice.post('/api/analyses?filename=SYNTHETIC-contract-fixture.mp4',
                              content=b'\0\0\0\x18ftypisom'+b'x'*20)
        assert response.status_code == 202
        analysis_id = response.json()['id']
        assert analysis_id.startswith('match_')
        store = app.state.store
        job = store.claim()
        assert job['id'] == analysis_id and job['started_at']
        directory = store.output_dir(analysis_id)
        directory.parent.mkdir(parents=True, exist_ok=True)
        fixture_bundle(directory)
        basic_artifacts(directory, True)
        physical_before = json.loads((directory/'match_events.json').read_text())
        assert physical_before['events_count'] == len(physical_before['events']) == 13
        rebind_fixture_identity(directory, analysis_id)
        physical_after = json.loads((directory/'match_events.json').read_text())
        assert physical_after['analysis_id'] == analysis_id
        assert physical_after['events'] == physical_before['events']
        generations = directory/'.result'
        generations.mkdir()
        with ResultStorage(generations) as storage:
            result = build_result(directory, job, _storage=storage)
            assert result['pointAnalysis']['analysisId'] == analysis_id
            publish_result(storage, result, directory)
        store.finish(analysis_id)
        marker_before = (generations/'current.store').read_bytes()
        marker = json.loads(marker_before)
        assert (generations/marker['database']).is_file()
        persisted = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        base = '/api/analyses/'+analysis_id
        payloads = {}
        observations = []
        for suffix in ('/result', '/download/report', '/download/readable-report', '/download/csv'):
            response = alice.get(base+suffix)
            assert response.status_code == 200
            assert response.headers['cache-control'] == 'private, no-store'
            foreign, anonymous = bob.get(base+suffix), guest.get(base+suffix)
            assert foreign.status_code == 404
            assert anonymous.status_code == 401
            observations.append({'path': base+suffix, 'owner_status': response.status_code,
                'other_account_status': foreign.status_code, 'guest_status': anonymous.status_code,
                'content_type': response.headers['content-type'], 'body_sha256': hashlib.sha256(response.content).hexdigest()})
            payloads[suffix] = response
        value = payloads['/result'].json()
        evidence = value['pointAnalysis']
        assert evidence['status'] == 'UNQUALIFIED' and evidence['independentAccuracyValidated'] is False
        assert evidence['summary']['totalPointsPlayed'] == 2
        assert evidence['summary']['effectivePlaySeconds'] == .970
        assert evidence['points'][1]['durationSeconds'] is None
        assert evidence['serveByPlayer']['player_2']['doubleFaults'] == 1
        assert value['summary']['capabilities']['scoring'] is False and value['matchState'] is None
        assert persisted['pointAnalysis'] == evidence
        assert payloads['/download/report'].json()['pointAnalysis'] == evidence
        html = payloads['/download/readable-report'].text
        assert 'Rule timeline' in html and '0.970 s' in html and 'Double fault by Player 2' in html
        assert 'Unavailable s' in html and 'not independently qualified match facts' in html
        assert 'normalized_seconds' in payloads['/download/csv'].text
        # Check ownership before availability for every private analysis route.
        # The supplied byte identities are not playable source/annotated media.
        for suffix in ('', '/media/source', '/media/annotated', '/download/original',
                       '/download/annotated', '/download/source-timing'):
            assert bob.get(base+suffix).status_code == 404
            assert guest.get(base+suffix).status_code == 401
            observations.append({'path': base+suffix, 'other_account_status': 404, 'guest_status': 401})
        assert bob.get('/api/runs').json() == []
        assert alice.get('/api/health').json()['worker_available'] is False
        observed = directory/'http_observations'
        observed.mkdir()
        for suffix, name in (('/result', 'owner_result.json'), ('/download/report', 'owner_report.json'),
                             ('/download/readable-report', 'owner_report.html'), ('/download/csv', 'owner_telemetry.csv')):
            (observed/name).write_bytes(payloads[suffix].content)
        (observed/'requests.json').write_text(json.dumps({'scope': 'SYNTHETIC_SAVED_NATIVE_RULE_FIXTURE; NO_MEDIA_DECODE_WORKER_OR_MODEL',
            'analysis_id': analysis_id, 'requests': observations}, indent=2), encoding='utf-8')
        cookies = dict(alice.cookies)
    # Fresh app instance opens the actual persisted generation and auth session.
    with TestClient(create_app(settings, start_worker=False, demos_manifest=demos)) as restored:
        restored.cookies.update(cookies)
        again = restored.get(base+'/result')
        assert again.status_code == 200 and again.json()['pointAnalysis'] == evidence
    assert (generations/'current.store').read_bytes() == marker_before

"""Actual loopback HTTP/cookie/Node-wire/background-worker delivery proof.

Uses only the predeclared disposable synthetic fixture, never a real match or
browser automation. The server is launched and stopped separately by root.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
FIXTURE = ART.parent/'correction_browser_fixture_v1'
manifest = json.loads((FIXTURE/'fixture_manifest.json').read_text(encoding='utf-8'))
assert manifest['label'].startswith('SYNTHETIC SOFTWARE TEST')
assert manifest['storage_root'] == str(FIXTURE/'service-v2')
assert manifest['host'] == '127.0.0.1' and manifest['port_checked_available'] == 5198
BASE = 'http://127.0.0.1:5198'
API = '/api/analyses/'+manifest['analysis_id']
credentials = json.loads((FIXTURE/'synthetic_credentials.json').read_text(encoding='utf-8'))
names = ['src/scoring/corrected_rule_rebuild.py', 'src/api/corrections.py',
    'src/api/correction_worker.py', 'src/api/worker.py', 'src/scoring/corrected_publication.py',
    'src/api/app.py', 'frontend/dist/index.html']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pins():
    return {name: sha(ROOT/name) for name in names}

def write(name, value):
    (ART/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

before = pins()
receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
    'scope': 'ISOLATED SYNTHETIC LOOPBACK HTTP; ACTUAL NODE WIRE; REAL COOKIE OWNER; SEPARATE EXCLUSIVE WORKER; NO BROWSER SUCCESS OR MODEL ACCURACY CLAIM',
    'source_before': before, 'observations': []}
write('started.json', receipt)
started = time.perf_counter()
try:
    with httpx.Client(base_url=BASE, timeout=40) as owner, httpx.Client(base_url=BASE, timeout=40) as guest:
        login = owner.post('/api/auth/login', json={
            'username': credentials['username'], 'password': credentials['password']})
        assert login.status_code == 200, login.text
        assert login.json()['user']['id'] == manifest['owner_id']
        original = owner.get(API+'/result').json()
        assert original['analysisGeneration'] == manifest['base_generation']
        history = owner.get(API+'/corrections').json()
        assert any(row['status'] == 'FAILED' for row in history)
        assert not any(row['status'] in ('QUEUED', 'PROCESSING') for row in history)
        generations = [original]
        for mode in ('initial_and_binding', 'semantic_unknown'):
            source_response = owner.get(API+'/review-source')
            assert source_response.status_code == 200, source_response.text
            source = source_response.json()
            assert source['is_current']
            source_path = ART/(mode+'_source.json')
            source_path.write_bytes(source_response.content)
            node = subprocess.run([r'C:\Program Files\nodejs\node.exe', str(ART/'browser_wire.js'),
                str(source_path), mode], capture_output=True, text=True, timeout=20, check=True)
            payload = json.loads(node.stdout)
            assert type(payload['input_amendments'][0]['replacement_source_input']['timestamp_s']) is int
            write(mode+'_request.json', payload)
            response = owner.post(API+'/corrections', content=node.stdout,
                headers={'Content-Type': 'application/json'})
            assert response.status_code == 202, response.text
            job = response.json()
            correction_id = job['id']
            states = [{'status': job['status'], 'step': job['step'], 'attempt': job['attempt']}]
            deadline = time.monotonic()+180
            while job['status'] in ('QUEUED', 'PROCESSING'):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Live worker command still pending: '+correction_id)
                time.sleep(.5)
                status = owner.get(API+'/corrections/'+correction_id)
                assert status.status_code == 200, status.text
                job = status.json()
                observed = {key: job[key] for key in ('status', 'step', 'attempt')}
                if observed != states[-1]:
                    states.append(observed)
                    print(json.dumps({'mode': mode, **observed}), flush=True)
            write(mode+'_status.json', job)
            assert job['status'] == 'COMPLETED', job
            current = owner.get(API+'/result')
            assert current.status_code == 200
            result = current.json()
            token = result['analysisGeneration']
            assert token == job['generation'] and result['analysisRevision'] == source['base_analysis_revision']+1
            assert current.headers['cache-control'] == 'private, no-store'
            assert result['analysisCorrection']['authorizationReceipt']['owner_id'] == manifest['owner_id']
            revised = owner.get(API+'/review-source', params={'generation': token}).json()
            old_event, new_event = source['inputs'][0]['source_input'], revised['inputs'][0]['source_input']
            for name in ('timestamp_s', 'ball_position_px', 'court_position_m', 'confidence', 'evidence'):
                assert json.dumps(new_event[name], sort_keys=True) == json.dumps(old_event[name], sort_keys=True)
            assert revised['inputs'][0]['input_fingerprint'] != source['inputs'][0]['input_fingerprint']
            expected_type = 'UNKNOWN_EVENT' if mode == 'semantic_unknown' else old_event['event_type']
            assert new_event['event_type'] == expected_type
            for prior in generations:
                pinned = owner.get(API+'/result', params={'generation': prior['analysisGeneration']})
                assert pinned.status_code == 200 and pinned.json() == prior
            report = owner.get(result['exports']['reportJson'])
            assert report.status_code == 200 and report.json() == result
            html = owner.get(result['exports']['readableReport'])
            assert html.status_code == 200 and 'Owner-authorized correction review' in html.text
            csv = owner.get(result['exports']['csv'])
            assert csv.status_code == 200
            video = owner.get(result['annotatedVideoUrl'])
            assert video.status_code == 200 and video.headers['x-analysis-generation'] == token
            ranged = owner.get(result['annotatedVideoUrl'], headers={'Range': 'bytes=0-63'})
            assert ranged.status_code == 206 and ranged.content == video.content[:64]
            for url in (API+'/result', API+'/review-source', API+'/corrections/'+correction_id,
                        result['annotatedVideoUrl'], *result['exports'].values()):
                assert guest.get(url).status_code == 401, url
            assert owner.post(API+'/corrections', content=node.stdout,
                headers={'Content-Type': 'application/json'}).json()['id'] == correction_id
            assert owner.get(API).json()['status'] == 'COMPLETED'
            write(mode+'_result.json', result)
            receipt['observations'].append({'mode': mode, 'correction_id': correction_id,
                'generation': token, 'revision': result['analysisRevision'], 'states': states,
                'source_measurements_exactly_preserved': True, 'immutable_history_readable': True,
                'matching_json_html_csv_video_and_range': True, 'guest_downloads_rejected': True,
                'idempotent_command_retry': True,
                'native_forecast_sample_count': result['nativeForecast']['current']['sample_count'],
                'report_json_sha256': hashlib.sha256(report.content).hexdigest(),
                'video_sha256': hashlib.sha256(video.content).hexdigest()})
            generations.append(result)
            print(json.dumps({'mode': mode, 'status': 'VERIFIED_COMPLETE', 'generation': token}), flush=True)
        receipt['status'] = 'PASS'
except BaseException as error:
    receipt['status'] = 'FAILED'
    receipt['error'] = type(error).__name__+': '+str(error)
    raise
finally:
    receipt['elapsed_seconds'] = time.perf_counter()-started
    receipt['source_after'] = pins()
    receipt['source_stable'] = before == receipt['source_after']
    write('receipt.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'source_stable', 'elapsed_seconds')}), flush=True)
assert receipt['source_stable']

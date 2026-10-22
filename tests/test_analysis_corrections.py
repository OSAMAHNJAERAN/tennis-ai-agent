"""Real owner HTTP admission and durable acknowledgement of sealed toy evidence.

The native source fixture supplies explicit synthetic observations. The small
publication helper tests the job acknowledgement boundary only; complete fresh
correction production is exercised separately by the publisher/worker suites.
No test asserts real match accuracy or imports inference models.
"""
import builtins
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import sqlite3
from threading import Event
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
            raise AssertionError('Correction HTTP must not import inference: '+name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked)


def wire(command):
    return {key: copy.deepcopy(value) for key, value in command.items()
            if key not in {'analysis_id', 'reviewer_id'}}


def setup_api(tmp_path, *, authenticated=True):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.auth import AuthStore
    from src.api.corrections import CorrectionStore, correction_router
    from src.api.jobs import JobStore, LOCAL_OWNER, Settings
    from src.api.analysis_generation import resolve_generation
    from tests.test_corrected_rule_rebuild import build_sealed_base

    fixture_root = tmp_path/'native'
    fixture_root.mkdir()
    fixture = build_sealed_base(fixture_root)
    jobs = JobStore(tmp_path/'jobs')
    settings = Settings(jobs.root, fixture.directory/'run_config.yaml', require_auth=authenticated)
    app, wakes = FastAPI(), []
    AuthStore(jobs, settings).install(app)
    app.include_router(correction_router(jobs, settings, lambda: wakes.append(True)))
    client = TestClient(app)
    stranger, guest = TestClient(app), TestClient(app)
    if authenticated:
        response = client.post('/api/auth/register', json={
            'username': 'review_owner', 'password': 'explicit-test-password-123'})
        assert response.status_code == 201, response.text
        owner = response.json()['user']['id']
        response = stranger.post('/api/auth/register', json={
            'username': 'other_owner', 'password': 'another-test-password-456'})
        assert response.status_code == 201, response.text
    else:
        owner = LOCAL_OWNER
    jobs.create('fixture', 'native.mp4', '.mp4', 100, settings.config, owner_id=owner)
    jobs.finish('fixture')
    destination = jobs.output_dir('fixture')
    destination.parent.mkdir(parents=True)
    shutil.copytree(fixture.directory, destination)
    return SimpleNamespace(client=client, stranger=stranger, guest=guest, jobs=jobs,
        store=CorrectionStore(jobs), owner=owner, directory=destination,
        snapshot=resolve_generation(destination), payload=wire(fixture.request), wakes=wakes,
        initial_marker=(destination/'.result/current.store').read_bytes(),
        url='/api/analyses/fixture/corrections')


@pytest.fixture
def api(tmp_path):
    value = setup_api(tmp_path)
    try:
        yield value
    finally:
        for client in (value.client, value.stranger, value.guest):
            client.close()


def submit(api, payload=None):
    response = api.client.post(api.url, json=payload or api.payload)
    assert response.status_code == 202, response.text
    return response.json()


def assert_base_readable(api):
    from src.api.result_document import ResultDocument
    assert api.jobs.get('fixture')['status'] == 'COMPLETED'
    assert (api.directory/'.result/current.store').read_bytes() == api.initial_marker
    with ResultDocument(api.directory, 'fixture') as document:
        assert document.value['summary']['frames'] == 12
        assert document.value['analysisRevision'] == 0


def publish_ack_fixture(api, row=None, *, mutate=None, omit=()):
    """Real atomic CAS with narrow metadata; not a complete correction fixture."""
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    current = resolve_generation(api.directory)
    revision = row['analysis_revision'] if row else current.analysis_revision
    result = {'summary': {'id': 'fixture', 'frames': current.frames},
              'analysisRevision': revision, 'testOnlyAcknowledgement': True}
    if row:
        result['analysisCorrection'] = {
            'analysisId': 'fixture', 'analysisRevision': revision,
            'baseGeneration': row['base_generation'], 'sourceSha256': row['source_sha256'],
            'requestSha256': row['request_sha256'],
            'review': {'correctionId': row['command']['correction_id']},
            'authorizationStatus': ('EXPLICIT_LOCAL_OWNER' if row['owner_id'] == 'local' else 'OWNER_AUTHORIZED'),
            'authorizationReceipt': copy.deepcopy(row['authorization'])}
    if mutate:
        mutate(result)
    files = {path.name: path for path in current.files_directory.iterdir()
             if path.name not in {'result.json', 'summary.json', *omit}}
    with ResultStorage(api.directory) as storage:
        return publish_analysis_generation(storage, result, api.directory, files=files,
            analysis_revision=revision, expected_generation=current.generation)


def test_owner_derived_immutable_authorization_and_idempotent_http(api):
    from src.api.corrections import PUBLIC_FIELDS
    from src.scoring.corrected_rule_rebuild import _digest
    first, second = submit(api), submit(api)
    assert first == second and first['status'] == 'QUEUED' and first['attempt'] == 0
    assert set(first) == set(PUBLIC_FIELDS)
    internal = api.store.get(first['id'], internal=True)
    assert internal['command']['reviewer_id'] == api.owner
    assert internal['command']['analysis_id'] == 'fixture'
    assert internal['authorization'] == {
        'schema_version': 1, 'analysis_id': 'fixture', 'owner_id': api.owner,
        'reviewer_id': api.owner, 'correction_id': api.payload['correction_id'],
        'request_sha256': _digest(internal['command']),
        'authorized_at': internal['authorization']['authorized_at'],
        'authorization_method': 'AUTHENTICATED_OWNER'}
    response = api.client.get(api.url)
    assert response.status_code == 200 and response.json() == [first]
    assert response.headers['cache-control'] == 'private, no-store'
    assert api.client.get(api.url+'/'+first['id']).json() == first
    changed = copy.deepcopy(api.payload)
    changed['reason'] = 'A distinct reviewed change cannot reuse this ID.'
    assert api.client.post(api.url, json=changed).status_code == 409
    with api.jobs.connect() as conn:
        assert conn.execute('SELECT count(*) FROM analysis_corrections').fetchone()[0] == 1
    assert len(api.wakes) == 2
    assert_base_readable(api)


def test_actual_guest_foreign_owner_and_forged_session_identity_rejected(api):
    first = submit(api)
    paths = [api.url, api.url+'/'+first['id'],
             '/api/analyses/fixture/review-source']
    for path in paths:
        assert api.guest.get(path).status_code == 401
        assert api.stranger.get(path).status_code == 404
    assert api.guest.post(api.url, json=api.payload).status_code == 401
    assert api.stranger.post(api.url, json=api.payload).status_code == 404
    assert api.stranger.post(api.url+'/'+first['id']+'/cancel').status_code == 404
    for key in ('reviewer_id', 'analysis_id'):
        payload = copy.deepcopy(api.payload)
        payload[key] = 'caller-supplied-identity'
        assert api.client.post(api.url, json=payload).status_code == 422
    assert api.client.post(api.url, json=api.payload,
                           headers={'origin': 'https://foreign.invalid'}).status_code == 403
    assert_base_readable(api)


@pytest.mark.parametrize('damage', ['schema_bool', 'revision_bool', 'extra', 'no_amendment',
                                  'counter_bool', 'foreign_initial', 'measurement', 'too_many'])
def test_strict_native_semantic_request_rejects_invalid_updates(api, damage):
    payload = copy.deepcopy(api.payload)
    if damage == 'schema_bool': payload['schema_version'] = True
    elif damage == 'revision_bool': payload['base_analysis_revision'] = False
    elif damage == 'extra': payload['unchecked'] = 'no'
    elif damage == 'no_amendment': payload['initial_state_amendment'] = None
    elif damage == 'counter_bool': payload['initial_state_amendment']['replacement_initial_state']['fields']['points_p2'] = True
    elif damage == 'foreign_initial': payload['initial_state_amendment']['replacement_initial_state']['fields']['match_id'] = 'foreign'
    else:
        source = api.client.get('/api/analyses/fixture/review-source').json()['inputs'][0]
        replacement = copy.deepcopy(source['source_input'])
        replacement['context']['binding']['evidence_ref'] = 'EXPLICIT_NEW_REVIEW'
        amendment = {'event_id': replacement['event_id'],
            'expected_input_fingerprint': source['input_fingerprint'],
            'replacement_source_input': replacement}
        payload['initial_state_amendment'] = None
        if damage == 'measurement':
            replacement['ball_position_px'][0] += 1
            payload['input_amendments'] = [amendment]
        else:
            payload['input_amendments'] = [{**copy.deepcopy(amendment), 'event_id': index} for index in range(129)]
    assert api.client.post(api.url, json=payload).status_code == 422
    assert api.store.list('fixture', api.owner) == []
    assert_base_readable(api)


@pytest.mark.parametrize('damage', ['source', 'generation', 'revision', 'replay', 'initial_fingerprint', 'event_fingerprint', 'absent_event'])
def test_stale_source_or_reviewer_fingerprint_returns_conflict(api, damage):
    payload = copy.deepcopy(api.payload)
    key = {'source': 'source_sha256', 'generation': 'base_generation',
           'revision': 'base_analysis_revision', 'replay': 'base_rule_replay_sha256'}.get(damage)
    if key:
        payload[key] = 1 if damage == 'revision' else 'e'*64
    elif damage == 'initial_fingerprint':
        payload['initial_state_amendment']['expected_initial_state_sha256'] = 'e'*64
    else:
        source = api.client.get('/api/analyses/fixture/review-source').json()['inputs'][0]
        replacement = copy.deepcopy(source['source_input'])
        replacement['context']['binding']['evidence_ref'] = 'EXPLICIT_NEW_REVIEW'
        payload['initial_state_amendment'] = None
        payload['input_amendments'] = [{'event_id': 987 if damage == 'absent_event' else replacement['event_id'],
            'expected_input_fingerprint': 'e'*64, 'replacement_source_input': replacement}]
    assert api.client.post(api.url, json=payload).status_code == 409
    assert_base_readable(api)


@pytest.mark.parametrize('body,status', [(' '* (1024*1024+1), 413), ('{"schema_version":NaN}',422),
    ('{"schema_version":1,"schema_version":1}',422), ('['*2000+'0'+']'*2000,422)],
    ids=['oversize', 'nonfinite', 'duplicate-key', 'excessive-depth'])
def test_bounded_finite_unambiguous_http_body(api, body, status):
    assert api.client.post(api.url, content=body,
                           headers={'content-type':'application/json'}).status_code == status
    assert api.wakes == []
    assert_base_readable(api)


def test_review_source_pages_real_native_inputs_and_fingerprints(api):
    from src.scoring.corrected_rule_rebuild import _digest
    response = api.client.get('/api/analyses/fixture/review-source?offset=0&limit=1')
    assert response.status_code == 200, response.text
    first = response.json()
    assert first['schema_version'] == 1 and first['is_current'] is True
    assert first['base_generation'] == api.snapshot.generation
    assert first['source_sha256'] == api.snapshot.file_sha256('source.mp4')
    assert first['base_rule_replay_sha256'] == api.snapshot.file_sha256('native_rule_replay.json')
    assert first['initial_state_sha256'] == _digest(first['initial_state'])
    assert first['initial_score']['player_2']['points_raw'] == 0
    assert first['next_offset'] == 1 and first['total_inputs'] == 2
    assert first['inputs'][0]['input_fingerprint'] == _digest(first['inputs'][0]['source_input'])
    assert first['inputs'][0]['source_input']['context']['source_pts'] == str(10**18)
    second = api.client.get('/api/analyses/fixture/review-source?offset=1&limit=1').json()
    assert second['next_offset'] is None and second['inputs'][0]['source_input']['event_type'] == 'BOUNCE'
    assert api.client.get('/api/analyses/fixture/review-source?offset=100&limit=1').json()['inputs'] == []
    for selector in ('limit=51', 'limit=0', 'offset=-1', 'generation=../foreign'):
        assert api.client.get('/api/analyses/fixture/review-source?'+selector).status_code == 422
    assert not first['measurement_edits_supported'] and not first['independent_accuracy_validated']
    assert_base_readable(api)
    publish_ack_fixture(api)
    historical = api.client.get('/api/analyses/fixture/review-source?generation='+api.snapshot.generation).json()
    assert historical['is_current'] is False
    assert historical['inputs'] == api.client.get('/api/analyses/fixture/review-source').json()['inputs']


@pytest.mark.parametrize('browser_numbers', [False, True])
def test_valid_current_semantic_review_is_admitted_without_measurement_edits(api, browser_numbers):
    from tests.test_corrected_rule_rebuild import browser_number_encoding
    source = api.client.get('/api/analyses/fixture/review-source').json()['inputs'][0]
    replacement = copy.deepcopy(source['source_input'])
    replacement['event_type'] = 'PLAYER_1_HIT'
    replacement['context']['binding']['evidence_ref'] = 'EXPLICIT_REVIEW_CONTACT_RECLASSIFICATION'
    payload = copy.deepcopy(api.payload)
    payload['initial_state_amendment'] = None
    payload['input_amendments'] = [{'event_id': replacement['event_id'],
        'expected_input_fingerprint': source['input_fingerprint'], 'replacement_source_input': replacement}]
    if browser_numbers:
        payload = browser_number_encoding(payload)
        replacement = payload['input_amendments'][0]['replacement_source_input']
    created = submit(api, payload)
    row = api.store.claim()
    assert row['id'] == created['id']
    assert row['command']['input_amendments'][0]['replacement_source_input'] == replacement
    assert_base_readable(api)


def test_old_saved_generation_without_native_source_is_readable_and_not_reviewable(api):
    from src.api.result_document import ResultDocument
    token = publish_ack_fixture(api, omit=('native_rule_replay.json',))
    assert api.client.get('/api/analyses/fixture/review-source').status_code == 409
    with ResultDocument(api.directory, 'fixture') as document:
        assert document.generation == token and document.value['summary']['id'] == 'fixture'
    assert api.jobs.get('fixture')['status'] == 'COMPLETED'


def test_actual_local_mode_receipt_identifies_explicit_local_owner(tmp_path):
    value = setup_api(tmp_path, authenticated=False)
    try:
        created = submit(value)
        row = value.store.get(created['id'], internal=True)
        assert row['command']['reviewer_id'] == 'local'
        assert row['authorization']['authorization_method'] == 'EXPLICIT_LOCAL_OWNER'
        assert value.client.get('/api/analyses/fixture/review-source').status_code == 200
    finally:
        for client in (value.client, value.stranger, value.guest): client.close()


def test_queue_cancel_reopen_and_terminal_states_preserve_completed_analysis(api):
    from src.api.corrections import CorrectionStore
    created = submit(api)
    cancelled = api.client.post(api.url+'/'+created['id']+'/cancel').json()
    assert cancelled['status'] == 'CANCELLED' and cancelled['cancel_requested'] is True
    assert api.store.claim() is None
    reopened = CorrectionStore(api.jobs)
    assert reopened.get(created['id'], api.owner) == cancelled
    assert reopened.finish(created['id'], status='FAILED', error='late failure')['status'] == 'CANCELLED'
    assert submit(api)['status'] == 'CANCELLED'
    assert_base_readable(api)


def test_processing_cancellation_prevents_final_guard_and_recovery_requeues_at_most_three(api):
    from src.api.jobs import JobCancelled
    created = submit(api)
    row = api.store.claim()
    assert row['id'] == created['id'] and row['attempt'] == 1
    api.store.stage(row['id'], 'Fresh semantic processing', 3)
    assert api.store.get(row['id'], api.owner)['step'] == 3
    api.store.recover_interrupted(lambda _: None)
    row = api.store.claim()
    assert row['attempt'] == 2
    api.store.recover_interrupted(lambda _: None)
    row = api.store.claim()
    assert row['attempt'] == 3
    api.store.recover_interrupted(lambda _: None)
    assert api.store.get(row['id'], api.owner)['status'] == 'FAILED'
    assert api.store.claim() is None
    payload = copy.deepcopy(api.payload)
    payload['correction_id'] = 'SECOND_REVIEW_CANCELLED'
    created = submit(api, payload)
    row = api.store.claim()
    response = api.client.post(api.url+'/'+row['id']+'/cancel').json()
    assert response['status'] == 'PROCESSING' and response['cancel_requested']
    with pytest.raises(JobCancelled):
        with api.store.publication_guard(row['id']):
            pytest.fail('Cancelled processing reached publication')
    api.store.recover_interrupted(lambda _: None)
    assert api.store.get(row['id'], api.owner)['status'] == 'CANCELLED'
    assert_base_readable(api)


def test_stale_queue_and_interrupted_attempt_become_conflict_without_original_job_failure(api):
    created = submit(api)
    publish_ack_fixture(api)
    assert submit(api)['id'] == created['id']  # original command remains idempotent after advance
    assert api.store.claim() is None
    assert api.store.get(created['id'], api.owner)['status'] == 'CONFLICT'
    assert api.jobs.get('fixture')['status'] == 'COMPLETED'


@pytest.mark.parametrize('ack_failure', [None, 'exception', 'interrupt'])
def test_publication_guard_binds_actual_pointer_and_serializes_lost_ack(api, ack_failure):
    from src.api.corrections import CorrectionStore
    row = api.store.claim() if api.store.list('fixture', api.owner) else None
    assert row is None
    created = submit(api)
    row = api.store.claim()
    token = None
    def publish():
        nonlocal token
        with api.store.publication_guard(row['id']) as retained:
            assert retained['command'] == row['command']
            token = publish_ack_fixture(api, retained)
            if ack_failure == 'exception': raise OSError('Explicit lost post-CAS acknowledgement')
            if ack_failure == 'interrupt': raise KeyboardInterrupt('Explicit post-CAS interrupt')
    if ack_failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt): publish()
    else:
        publish()
    assert token and api.store.committed_generation(row) == token
    stored = CorrectionStore(api.jobs).get(created['id'], api.owner)
    assert stored['status'] == 'COMPLETED' and stored['generation'] == token
    assert api.client.post(api.url+'/'+row['id']+'/cancel').json() == stored
    assert api.store.finish(row['id'], error='late failure')['status'] == 'COMPLETED'
    assert submit(api)['id'] == created['id']
    assert api.jobs.get('fixture')['status'] == 'COMPLETED'


def test_guard_no_commit_or_error_leaves_recoverable_processing(api):
    row = api.store.get(submit(api)['id'], internal=True)
    row = api.store.claim()
    with pytest.raises(ValueError, match='without a bound'):
        with api.store.publication_guard(row['id']): pass
    with pytest.raises(OSError, match='before CAS'):
        with api.store.publication_guard(row['id']): raise OSError('before CAS')
    assert api.store.get(row['id'], api.owner)['status'] == 'PROCESSING'
    assert_base_readable(api)


def test_mutated_worker_snapshot_cannot_reauthorize_publication_guard(api):
    submit(api)
    row = api.store.claim()
    with pytest.raises(ValueError, match='without a bound'):
        with api.store.publication_guard(row['id']) as mutable:
            mutable['authorization']['reviewer_id'] = 'foreign-owner'
            publish_ack_fixture(api, mutable)
    assert api.store.committed_generation(row) is None


@pytest.mark.parametrize('damage', ['request', 'receipt', 'review', 'status', 'base'])
def test_current_foreign_or_unbound_result_cannot_acknowledge_correction(api, damage):
    submit(api)
    row = api.store.claim()
    def mutate(result):
        evidence = result['analysisCorrection']
        if damage == 'request': evidence['requestSha256'] = 'e'*64
        elif damage == 'receipt': evidence['authorizationReceipt']['reviewer_id'] = 'foreign'
        elif damage == 'review': evidence['review']['correctionId'] = 'different'
        elif damage == 'status': evidence['authorizationStatus'] = 'PRIVATE_LOCAL_REVIEW_NOT_AUTHENTICATED'
        elif damage == 'base': evidence['baseGeneration'] = 'e'*64
    with pytest.raises(ValueError, match='without a bound'):
        with api.store.publication_guard(row['id']): publish_ack_fixture(api, row, mutate=mutate)
    assert api.store.committed_generation(row) is None
    api.store.recover_interrupted(lambda _: None)
    assert api.store.get(row['id'], api.owner)['status'] == 'CONFLICT'


def test_interrupted_after_actual_cas_recovery_wins_over_later_cancel(api):
    submit(api)
    row = api.store.claim()
    token = publish_ack_fixture(api, row)
    assert api.store.cancel(row['id'], api.owner)['status'] == 'PROCESSING'
    api.store.recover_interrupted(lambda current: api.store.committed_generation(current))
    assert api.store.get(row['id'], api.owner)['status'] == 'COMPLETED'
    assert api.store.get(row['id'], api.owner)['generation'] == token


def test_concurrent_cancel_is_serialized_after_publication_acknowledgement(api):
    submit(api)
    row = api.store.claim()
    started = Event()
    def cancel():
        started.set()
        return api.store.cancel(row['id'], api.owner)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with api.store.publication_guard(row['id']):
            future = pool.submit(cancel)
            assert started.wait(3)
            assert not future.done()
            token = publish_ack_fixture(api, row)
        result = future.result(timeout=5)
    assert result['status'] == 'COMPLETED' and result['generation'] == token
    assert result['cancel_requested'] is False


def test_retained_authorization_tampering_fails_closed_on_claim(api):
    created = submit(api)
    with api.jobs.connect() as conn:
        authorization = json.loads(conn.execute('SELECT authorization_json FROM analysis_corrections WHERE id=?',
            (created['id'],)).fetchone()[0])
        authorization['reviewer_id'] = 'forged-owner'
        conn.execute('UPDATE analysis_corrections SET authorization_json=? WHERE id=?',
            (json.dumps(authorization), created['id']))
    assert api.store.claim() is None
    assert api.store.get(created['id'], api.owner)['status'] == 'FAILED'
    with pytest.raises(ValueError): api.store.get(created['id'], internal=True)
    assert_base_readable(api)


def test_version_four_migration_preserves_old_rows_and_future_database_is_never_downgraded(tmp_path):
    from src.api.jobs import JobStore
    jobs = JobStore(tmp_path/'jobs')
    jobs.create('old', 'old.mp4', '.mp4', 12, Path('unused'))
    jobs.finish('old')
    with jobs.connect() as conn:
        conn.execute('DROP TABLE analysis_corrections')
        conn.execute('PRAGMA user_version=3')
    reopened = JobStore(jobs.root)
    assert reopened.get('old')['status'] == 'COMPLETED'
    with reopened.connect() as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 4
        assert conn.execute('SELECT count(*) FROM analysis_corrections').fetchone()[0] == 0
        conn.execute('PRAGMA user_version=5')
    with pytest.raises(RuntimeError, match='newer'):
        JobStore(jobs.root)
    with sqlite3.connect(jobs.db) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 5

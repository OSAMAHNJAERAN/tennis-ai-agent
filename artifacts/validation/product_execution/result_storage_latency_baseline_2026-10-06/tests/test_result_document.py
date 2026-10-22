"""Publication gates, bounded legacy readers and interrupted HTTP ownership."""
import json
from pathlib import Path
import sqlite3

import anyio
from fastapi.testclient import TestClient
import pytest
from starlette.requests import ClientDisconnect

from src.api.jobs import JobStore, Settings
from src.api import result_document
from src.api.result_document import ResultDocument, ResultStreamingResponse, open_result, publish_result
from src.api.result_storage import DiskSeries, ResultStorage
from src.api.results import build_result
from src.api import reporting
from tests.test_bounded_result_contract import artifacts


@pytest.fixture
def owned_temporary(tmp_path, monkeypatch):
    scratch = tmp_path / 'owned_temporary'
    scratch.mkdir()
    monkeypatch.setattr(result_document.tempfile, 'gettempdir', lambda: str(scratch))
    return scratch


@pytest.fixture(autouse=True)
def temporary_isolation(owned_temporary):
    pass


@pytest.fixture
def saved(tmp_path):
    directory, job = artifacts(tmp_path)
    expected = build_result(directory, job)
    generations = directory / '.result'
    generations.mkdir()
    with ResultStorage(generations) as storage:
        value = build_result(directory, job, _storage=storage)
        publish_result(storage, value, directory)
    return directory, job, expected


def serialize(document):
    return ''.join(document.json_chunks())


def test_published_generation_reopens_lazily_and_preserves_bytes(saved, monkeypatch):
    directory, job, expected = saved
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == 'result.json':
            raise AssertionError('Published readers must use the committed database.')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    with ResultDocument(directory, job['id']) as document:
        assert isinstance(document.value['trajectories']['ballTrajectory'], DiskSeries)
        assert serialize(document) == json.dumps(expected, allow_nan=False)
        database = document.storage.path
    assert document.closed and document.storage._connection is None
    assert database.is_file()
    with ResultDocument(directory, job['id']) as restarted:
        assert serialize(restarted) == json.dumps(expected, allow_nan=False)


@pytest.mark.parametrize('damage', ['missing-marker', 'invalid-marker', 'missing-database',
                                  'size', 'same-size-hash', 'path', 'schema', 'identity', 'frames'])
def test_generation_damage_never_falls_back_to_compatibility_json(saved, damage):
    directory, job, _expected = saved
    marker = directory / '.result/current.store'
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    database = marker.parent / manifest['database']
    if damage == 'missing-marker':
        marker.unlink()
    elif damage == 'invalid-marker':
        marker.write_text('{', encoding='utf-8')
    elif damage == 'missing-database':
        database.unlink()
    elif damage == 'same-size-hash':
        with database.open('r+b') as stream:
            stream.seek(100)
            byte = stream.read(1)
            stream.seek(100)
            stream.write(bytes([byte[0] ^ 1]))
    else:
        key, value = {'size': ('databaseBytes', 1), 'path': ('database', '../escape.sqlite3'),
                      'schema': ('schemaVersion', 2), 'identity': ('analysisId', 'another'),
                      'frames': ('frames', 1)}[damage]
        manifest[key] = value
        marker.write_text(json.dumps(manifest), encoding='utf-8')
    assert (directory / 'result.json').is_file()
    with pytest.raises((OSError, ValueError, sqlite3.Error)):
        ResultDocument(directory, job['id'])


def test_wrong_analysis_is_rejected(saved):
    with pytest.raises(ValueError, match='different analysis'):
        ResultDocument(saved[0], 'other_analysis')


def test_generation_switch_retains_reader_snapshot(saved):
    directory, job, expected = saved
    first = ResultDocument(directory, job['id'])
    first_database = first.storage.path
    with ResultStorage(directory / '.result') as storage:
        second = {'summary': dict(expected['summary'], label='New generation'),
                  'records': storage.series(range(129))}
        publish_result(storage, second, directory)
    try:
        assert serialize(first) == json.dumps(expected, allow_nan=False)
        with ResultDocument(directory, job['id']) as current:
            assert current.value['summary']['label'] == 'New generation'
            assert list(current.value['records']) == list(range(129))
            assert current.storage.path != first_database
    finally:
        first.close()
    assert first_database.is_file()


@pytest.mark.parametrize('boundary', ['summary-close', 'database-publication', 'summary-replace', 'manifest-replace'])
def test_uncommitted_generation_preserves_authoritative_previous_reader(saved, monkeypatch, boundary):
    directory, job, expected = saved
    marker = directory / '.result/current.store'
    previous = marker.read_bytes()
    previous_result = (directory / 'result.json').read_bytes()
    previous_summary = (directory / 'summary.json').read_bytes()
    original_replace = Path.replace
    original_open = Path.open
    original_publish = ResultStorage.publish

    class FailOnClose:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self.handle.__enter__()

        def __exit__(self, *error):
            self.handle.__exit__(*error)
            raise OSError('Injected summary close failure')

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if boundary == 'summary-close' and path.name == 'summary.tmp' and args[0] == 'w':
            return FailOnClose(handle)
        return handle

    def replace_file(path, target):
        if boundary == 'manifest-replace' and Path(target) == marker:
            raise OSError('Injected manifest replacement failure')
        if boundary == 'summary-replace' and path.name == 'summary.tmp':
            raise OSError('Injected summary replacement failure')
        return original_replace(path, target)

    def publication(storage, target):
        if boundary == 'database-publication':
            raise OSError('Injected database publication failure')
        return original_publish(storage, target)

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(Path, 'replace', replace_file)
    monkeypatch.setattr(ResultStorage, 'publish', publication)
    with pytest.raises(OSError, match='Injected'):
        with ResultStorage(directory / '.result') as storage:
            publish_result(storage, {'summary': dict(expected['summary'], label='Uncommitted')}, directory)
    assert marker.read_bytes() == previous
    assert (directory / 'result.json').read_bytes() == previous_result
    assert (directory / 'summary.json').read_bytes() == previous_summary
    with ResultDocument(directory, job['id']) as document:
        assert serialize(document) == json.dumps(expected, allow_nan=False)


def test_legacy_json_uses_owned_scratch_and_validates_entire_file(tmp_path, monkeypatch):
    directory = tmp_path / 'legacy'
    directory.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(result_document.tempfile, 'gettempdir', lambda: str(scratch))
    expected = {'summary': {'id': 'legacy', 'frames': 129}, 'records': list(range(129))}
    source = directory / 'result.json'
    original = json.dumps(expected).encode('utf-8')
    source.write_bytes(original)
    with ResultDocument(directory, 'legacy') as document:
        assert isinstance(document.value['records'], DiskSeries)
        database = document.storage.path
        assert database.parent == scratch
        assert serialize(document) == json.dumps(expected)
    assert not database.exists() and not list(scratch.iterdir())
    assert source.read_bytes() == original


@pytest.mark.parametrize('tail', [', "late": NaN}', ', "late": 1e999}', '} garbage', ', "late": ['])
def test_invalid_legacy_tail_is_rejected_and_owned_scratch_removed(tmp_path, monkeypatch, tail):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(result_document.tempfile, 'gettempdir', lambda: str(scratch))
    value = json.dumps({'summary': {'id': 'legacy'}, 'records': list(range(129))})[:-1] + tail
    (tmp_path / 'result.json').write_text(value, encoding='utf-8')
    with pytest.raises(ValueError):
        ResultDocument(tmp_path, 'legacy')
    assert not list(scratch.iterdir())


@pytest.mark.parametrize('mode', ['complete', 'send-failure', 'iterator-failure'])
def test_streaming_response_owns_reader_through_threadpool_and_send_failure(tmp_path, monkeypatch, mode):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(result_document.tempfile, 'gettempdir', lambda: str(scratch))
    expected = {'summary': {'id': 'legacy'}, 'records': list(range(5000))}
    (tmp_path / 'result.json').write_text(json.dumps(expected), encoding='utf-8')
    document = ResultDocument(tmp_path, 'legacy')
    database = document.storage.path

    def chunks():
        yield from document.json_chunks()
        if mode == 'iterator-failure':
            raise ValueError('Injected iterator failure')

    response = ResultStreamingResponse(document, chunks(), media_type='application/json')
    bodies = []

    async def run():
        async def receive():
            return {'type': 'http.disconnect'}

        async def send(message):
            if message['type'] == 'http.response.body':
                if mode == 'send-failure':
                    raise OSError('Client connection failed')
                bodies.append(message.get('body', b''))

        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)

    if mode == 'complete':
        anyio.run(run)
        assert b''.join(bodies) == json.dumps(expected).encode('utf-8')
    else:
        with pytest.raises(ClientDisconnect if mode == 'send-failure' else ValueError):
            anyio.run(run)
    assert document.closed and document.storage._connection is None
    assert not database.exists() and not list(scratch.iterdir())


def test_corrupt_generation_returns_preheader_503_for_every_result_export(saved, tmp_path, monkeypatch):
    directory, job, _ = saved
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'bootstrap'))
    from src.api.app import create_app
    settings = Settings(tmp_path / 'store', tmp_path / 'unused.yaml')
    store = JobStore(settings.root)
    store.create(job['id'], job['filename'], '.mp4', 12, settings.config)
    store.claim()
    store.finish(job['id'])
    monkeypatch.setattr(JobStore, 'output_dir', lambda _self, _id: directory)
    (directory / '.result/current.store').unlink()
    app = create_app(settings, start_worker=False)
    with TestClient(app) as client:
        for suffix in ('result', 'download/report', 'download/csv', 'download/readable-report'):
            response = client.get(f'/api/analyses/{job["id"]}/{suffix}')
            assert response.status_code == 503
            assert response.headers['content-type'] == 'application/json'
            assert response.json()['detail'].startswith('The saved analysis is unavailable.')


@pytest.mark.parametrize('mode', ['parse', 'context', 'response'])
def test_cleanup_failure_preserves_primary_reader_error(tmp_path, monkeypatch, mode):
    (tmp_path / 'result.json').write_text('{"summary": {}, "records": [1, 2]}' if mode != 'parse'
                                        else '{"records": [1,', encoding='utf-8')

    def remove(_storage):
        raise OSError('Injected owned scratch cleanup failure')

    monkeypatch.setattr(ResultStorage, '_remove_scratch', remove)
    with pytest.raises(ValueError) as captured:
        if mode == 'parse':
            ResultDocument(tmp_path)
        elif mode == 'context':
            with open_result(tmp_path):
                raise ValueError('Primary consumer failure')
        else:
            document = ResultDocument(tmp_path)

            def chunks():
                yield '{}'
                raise ValueError('Primary stream failure')

            async def run():
                async def receive():
                    return {'type': 'http.disconnect'}

                async def send(_message):
                    pass

                await ResultStreamingResponse(document, chunks())(
                    {'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)

            anyio.run(run)
    assert any('Injected owned scratch cleanup failure' in note for note in captured.value.__notes__)


@pytest.mark.parametrize('format', ['published', 'legacy'])
def test_api_result_json_csv_html_exports_preserve_existing_content(saved, tmp_path, monkeypatch, format):
    directory, job, expected = saved
    if format == 'legacy':
        directory = tmp_path / 'legacy'
        directory.mkdir()
        (directory / 'result.json').write_text(json.dumps(expected), encoding='utf-8')
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'bootstrap'))
    from src.api.app import create_app
    settings = Settings(tmp_path / 'store', tmp_path / 'unused.yaml')
    store = JobStore(settings.root)
    store.create(job['id'], job['filename'], '.mp4', 12, settings.config)
    store.claim()
    store.finish(job['id'])
    monkeypatch.setattr(JobStore, 'output_dir', lambda _self, _id: directory)
    expected_content = {
        'result': json.dumps(expected, allow_nan=False).encode(),
        'download/report': json.dumps(expected, allow_nan=False).encode(),
        'download/csv': ''.join(reporting.telemetry_csv_chunks(expected)).encode(),
        'download/readable-report': reporting.readable_report(expected, directory).encode(),
    }
    with TestClient(create_app(settings, start_worker=False)) as client:
        for suffix, content in expected_content.items():
            response = client.get(f'/api/analyses/{job["id"]}/{suffix}')
            assert response.status_code == 200 and response.content == content
            if suffix.startswith('download/'):
                assert 'attachment;' in response.headers['content-disposition']


def test_profile_skips_unavailable_sqlite_generation(saved, tmp_path, monkeypatch):
    directory, job, _ = saved
    marker = directory / '.result/current.store'
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    database = marker.parent / manifest['database']
    with sqlite3.connect(database) as connection:
        connection.execute('DROP TABLE storage_format')
    from src.api.results import file_hash
    manifest.update(databaseBytes=database.stat().st_size, databaseSha256=file_hash(database))
    marker.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'bootstrap'))
    from src.api.app import create_app
    settings = Settings(tmp_path / 'store', tmp_path / 'unused.yaml')
    store = JobStore(settings.root)
    store.create(job['id'], job['filename'], '.mp4', 12, settings.config)
    store.claim()
    store.finish(job['id'])
    monkeypatch.setattr(JobStore, 'output_dir', lambda _self, _id: directory)
    with TestClient(create_app(settings, start_worker=False)) as client:
        player = client.post('/api/players', json={'name': 'Player'}).json()
        metadata = {'participants': [{'name': 'Player', 'profile_id': player['id']}, {}],
                    'role_assignment': 'same'}
        edit = client.patch(f'/api/matches/{job["id"]}/metadata',
                          json={'expected_revision': 0, 'metadata': metadata})
        assert edit.status_code == 200
        profile = client.get('/api/players/' + player['id'])
        assert profile.status_code == 200
        assert len(profile.json()['matches']) == 1 and profile.json()['movement'] == []

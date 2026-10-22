"""Immutable generation HTTP coherence with synthetic bytes and actual storage.

The worker, media decoder and vision models do not run. HTML evidence extraction
is replaced only at its image boundary to observe the actual selected directory.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest


UPLOAD = b'\0\0\0\x18ftypisom' + bytes(range(128))
PASSWORD = 'synthetic-generation-http-password-123'
GENERATION = re.compile(r'[0-9a-f]{64}\Z')
HTTP_OBSERVATIONS = []


@pytest.fixture
def service(tmp_path, monkeypatch):
    # app.py owns a default app at import time; keep that bootstrap local too.
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'bootstrap'))
    monkeypatch.setenv('TENNIS_REQUIRE_AUTH', '1')
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from src.api.jobs import Settings
    from src.api import reporting

    settings = Settings(tmp_path / 'storage', tmp_path / 'unused.yaml',
                        max_upload_bytes=1024, require_auth=True)
    demos = tmp_path / 'demos.json'
    demos.write_text('{"schemaVersion":1,"sources":[]}', encoding='utf-8')
    app = create_app(settings, start_worker=False, demos_manifest=demos)
    image_calls = []
    def evidence_images(directory, result):
        directory = Path(directory)
        media = (directory / 'annotated_web.mp4').read_bytes()
        image_calls.append({'directory': str(directory), 'label': result['summary']['label'],
                            'annotated_sha256': hashlib.sha256(media).hexdigest()})
        return '<p>PINNED-IMAGE-' + hashlib.sha256(media).hexdigest() + '</p>'
    monkeypatch.setattr(reporting, 'evidence_images', evidence_images)
    with ExitStack() as stack:
        alice, bob, guest = [stack.enter_context(TestClient(app)) for _ in range(3)]
        for client, username in ((alice, 'generation_alice'), (bob, 'generation_bob')):
            assert client.post('/api/auth/register', json={
                'username': username, 'password': PASSWORD}).status_code == 201
        uploaded = alice.post('/api/analyses?filename=SYNTHETIC-generation-fixture.mp4', content=UPLOAD)
        assert uploaded.status_code == 202
        analysis_id = uploaded.json()['id']
        job = app.state.store.claim()
        assert job['id'] == analysis_id
        yield SimpleNamespace(alice=alice, bob=bob, guest=guest, app=app,
            store=app.state.store, job=job, analysis_id=analysis_id, settings=settings,
            demos=demos, tmp_path=tmp_path, image_calls=image_calls,
            base='/api/analyses/' + analysis_id)


def publish(service, label, *, revision=0, timing=True, credit=None):
    from src.api.analysis_generation import publish_analysis_generation
    from src.api.result_document import ResultDocument
    from src.api.result_storage import ResultStorage
    from src.api.results import build_result
    from tests.test_bounded_result_contract import artifacts

    directory = service.store.output_dir(service.analysis_id)
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        built, _ = artifacts(directory.parent, count=65)
        built.rename(directory)
    source_bytes = b'SYNTHETIC normalized source; no video decode' + bytes(range(128))
    annotated = ('SYNTHETIC annotation ' + label + ':').encode() + bytes(range(128))
    (directory / 'source.mp4').write_bytes(source_bytes)
    (directory / 'annotated_web.mp4').write_bytes(annotated)
    mapping = ''.join(json.dumps({'frame_index': index, 'version': label}) + '\n' for index in range(65)).encode()
    timing_path = directory / 'source_frame_mapping.jsonl'
    if timing:
        timing_path.write_bytes(mapping)
    else:
        timing_path.unlink(missing_ok=True)
    if credit is not None:
        (directory / 'media_source.json').write_text(json.dumps(credit), encoding='utf-8')
    generation_storage = directory / '.result'
    generation_storage.mkdir(exist_ok=True)
    job = {**service.job, 'filename': label}
    with ResultStorage(generation_storage) as storage:
        result = build_result(directory, job, _storage=storage)
        result['analysisRevision'] = revision
        files = {path.name: path for path in directory.iterdir()
                 if path.is_file() and path.name not in {'result.json', 'summary.json'}
                 and path.suffix in {'.json', '.jsonl', '.yaml', '.mp4'}}
        original = service.store.upload_dir(service.analysis_id) / 'original.mp4'
        files['original.mp4'] = original
        token = publish_analysis_generation(storage, result, directory,
            files=files, analysis_revision=revision)
    assert GENERATION.fullmatch(token)
    service.store.finish(service.analysis_id)
    with ResultDocument(directory, service.analysis_id, generation=token) as document:
        assert document.generation == token
        asset_directory = document.asset_directory
        assert asset_directory != directory
        presented = json.loads(''.join(document.json_chunks(document.presented(service.analysis_id))))
    marker = json.loads((directory / '.result/current.store').read_text(encoding='utf-8'))
    assert marker == {'schemaVersion': 2, 'generation': token}
    sealed = (directory / '.result' / (token + '.generation')).read_bytes()
    assert hashlib.sha256(sealed).hexdigest() == token
    return SimpleNamespace(token=token, directory=directory, assets=asset_directory,
        value=presented, annotated=annotated, source=source_bytes, original=UPLOAD,
        timing=mapping if timing else None, label=label)


def pinned(service, suffix, generation):
    return service.base + suffix + '?generation=' + generation


def assert_all_urls_pin(service, value, generation):
    assert value['analysisGeneration'] == generation
    urls = [value.get('videoUrl'), value.get('annotatedVideoUrl')]
    urls.extend((value.get('exports') or {}).values())
    owned = [url for url in urls if url is not None and url.startswith(service.base + '/')]
    assert owned
    for url in owned:
        assert parse_qs(urlsplit(url).query) == {'generation': [generation]}


def damage_mutable_root(service, snapshot):
    for name in ('source.mp4', 'annotated_web.mp4', 'source_frame_mapping.jsonl',
                 'summary.json', 'result.json', 'media_source.json'):
        (snapshot.directory / name).write_bytes(b'MUTABLE ROOT MUST NOT BE SERVED')
    original = service.store.upload_dir(service.analysis_id) / 'original.mp4'
    original.write_bytes(b'MUTABLE ORIGINAL MUST NOT REPLACE PINNED INPUT')


def test_two_generations_keep_all_old_http_outputs_and_pinned_html_images(service):
    first = publish(service, 'Synthetic generation one', revision=0)
    old = service.alice.get(service.base + '/result')
    assert old.status_code == 200
    assert_all_urls_pin(service, old.json(), first.token)
    second = publish(service, 'Synthetic generation two', revision=1)
    assert first.token != second.token
    damage_mutable_root(service, second)
    latest = service.alice.get(service.base + '/result')
    assert latest.status_code == 200 and latest.json()['summary']['label'] == second.label
    assert_all_urls_pin(service, latest.json(), second.token)
    for snapshot in (first, second):
        response = service.alice.get(pinned(service, '/result', snapshot.token))
        assert response.status_code == 200
        value = response.json()
        assert value['summary']['label'] == snapshot.label
        assert_all_urls_pin(service, value, snapshot.token)
        assert service.alice.get(value['exports']['reportJson']).json() == value
        for url, expected in (
            (value['videoUrl'], snapshot.source),
            (value['annotatedVideoUrl'], snapshot.annotated),
            (value['exports']['annotatedVideo'], snapshot.annotated),
            (value['exports']['originalVideo'], snapshot.original),
            (value['exports']['sourceFrameMapping'], snapshot.timing),
        ):
            downloaded = service.alice.get(url)
            assert downloaded.status_code == 200 and downloaded.content == expected
            assert downloaded.headers['cache-control'] == 'private, no-store'
        html = service.alice.get(value['exports']['readableReport'])
        csv = service.alice.get(value['exports']['csv'])
        assert html.status_code == csv.status_code == 200
        assert snapshot.label in html.text
        assert 'PINNED-IMAGE-' + hashlib.sha256(snapshot.annotated).hexdigest() in html.text
        assert service.image_calls[-1]['directory'] == str(snapshot.assets)
        assert 'normalized_seconds' in csv.text and len(csv.text.splitlines()) == 66
        HTTP_OBSERVATIONS.append({'generation': snapshot.token, 'label': snapshot.label,
            'annotated_sha256': hashlib.sha256(snapshot.annotated).hexdigest(),
            'html_asset_directory': str(snapshot.assets)})
    # Library and match summaries must also stop reading mutable root files.
    runs = service.alice.get('/api/runs').json()
    assert next(row for row in runs if row['id'] == service.analysis_id)['label'] == second.label
    match = service.alice.get('/api/matches/by-analysis/' + service.analysis_id)
    assert match.status_code == 200
    assert match.json()['analysis_summary']['label'] == second.label


def test_pinned_media_and_download_keep_file_response_range_semantics(service):
    first = publish(service, 'Synthetic range generation')
    second = publish(service, 'Synthetic replacement generation', revision=1)
    damage_mutable_root(service, second)
    for suffix, expected in (('/media/source', first.source),
                             ('/media/annotated', first.annotated),
                             ('/download/annotated', first.annotated),
                             ('/download/original', first.original)):
        response = service.alice.get(pinned(service, suffix, first.token),
                                     headers={'Range': 'bytes=8-19'})
        assert response.status_code == 206 and response.content == expected[8:20]
        assert response.headers['content-range'] == f'bytes 8-19/{len(expected)}'
        assert response.headers['cache-control'] == 'private, no-store'


def test_legacy_cache_bust_cannot_silently_select_full_generation_files(service):
    snapshot = publish(service, 'Synthetic immutable generation')
    for suffix in ('/media/source', '/media/annotated', '/download/original',
                   '/download/annotated', '/download/source-timing'):
        for parameters in ({'v': '18d-ab'}, {'v': '18d-ab', 'generation': snapshot.token}):
            response = service.alice.get(service.base + suffix, params=parameters)
            assert response.status_code == 409
        assert service.alice.get(pinned(service, suffix, snapshot.token)).status_code == 200


def test_missing_generation_assets_never_fall_back_to_old_generation_or_root(service):
    first = publish(service, 'Synthetic generation with timing')
    second = publish(service, 'Synthetic generation without timing', revision=1, timing=False)
    (second.directory / 'source_frame_mapping.jsonl').write_bytes(b'ROOT TIMING IS NOT THIS GENERATION')
    value = service.alice.get(pinned(service, '/result', second.token)).json()
    assert value['exports'].get('sourceFrameMapping') is None
    missing = service.alice.get(pinned(service, '/download/source-timing', second.token))
    assert missing.status_code == 404
    previous = service.alice.get(pinned(service, '/download/source-timing', first.token))
    assert previous.status_code == 200 and previous.content == first.timing


def test_unknown_malformed_or_missing_generation_never_resolves_current_or_root(service):
    first = publish(service, 'Synthetic retained generation')
    second = publish(service, 'Synthetic current generation', revision=1)
    for token in ('f' * 64, 'g' * 64, first.token.upper(), first.token[:32], '../' + first.token, ''):
        for suffix in ('/result', '/media/annotated', '/download/report',
                       '/download/readable-report', '/download/csv', '/download/source-timing',
                       '/download/original'):
            response = service.alice.get(service.base + suffix, params={'generation': token})
            assert response.status_code in {400, 404, 422, 503}, (suffix, token, response.status_code)
            assert response.content not in (first.annotated, second.annotated, UPLOAD)
    (first.directory / '.result' / (first.token + '.generation')).unlink()
    for suffix in ('/result', '/media/annotated', '/download/report', '/download/original'):
        assert service.alice.get(pinned(service, suffix, first.token)).status_code in {404, 503}
    assert service.alice.get(service.base + '/result').json()['analysisGeneration'] == second.token


def test_generation_resolution_never_precedes_account_authorization(service):
    first = publish(service, 'Synthetic private generation')
    for token in (first.token, '../malformed-generation', 'f' * 64):
        for suffix in ('/result', '/media/source', '/media/annotated', '/download/original',
                       '/download/annotated', '/download/report', '/download/readable-report',
                       '/download/csv', '/download/source-timing'):
            assert service.bob.get(service.base + suffix, params={'generation': token}).status_code == 404
            assert service.guest.get(service.base + suffix, params={'generation': token}).status_code == 401
    # A valid token does not grant access through another account's own job ID.
    other = service.bob.post('/api/analyses?filename=OTHER-synthetic.mp4', content=UPLOAD).json()
    service.store.finish(other['id'])
    response = service.bob.get('/api/analyses/' + other['id'] + '/result',
                               params={'generation': first.token})
    assert response.status_code in {404, 503}


def test_original_upload_remains_available_before_completion_without_generation(service):
    response = service.alice.get(service.base + '/download/original')
    assert response.status_code == 200 and response.content == UPLOAD
    assert service.alice.get(service.base + '/media/source').status_code == 404
    assert service.alice.get(service.base + '/result').status_code == 409
    assert service.bob.get(service.base + '/download/original').status_code == 404
    assert service.guest.get(service.base + '/download/original').status_code == 401


def test_pinned_saved_generation_reopens_after_http_app_restart(service):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    first = publish(service, 'Synthetic generation before restart')
    second = publish(service, 'Synthetic current after replacement', revision=1)
    damage_mutable_root(service, second)
    cookies = dict(service.alice.cookies)
    with TestClient(create_app(service.settings, start_worker=False,
                              demos_manifest=service.demos)) as restored:
        restored.cookies.update(cookies)
        result = restored.get(pinned(service, '/result', first.token))
        assert result.status_code == 200
        assert result.json()['summary']['label'] == first.label
        assert restored.get(result.json()['annotatedVideoUrl']).content == first.annotated
        assert restored.get(result.json()['exports']['originalVideo']).content == UPLOAD


def test_curated_generation_uses_sealed_credit_and_revocation_also_blocks_pinned_urls(service):
    from tests.test_public_demos import credit
    source = credit(UPLOAD)
    source['sourceSha256'] = hashlib.sha256(
        b'SYNTHETIC normalized source; no video decode' + bytes(range(128))).hexdigest()
    snapshot = publish(service, 'Synthetic curated generation', credit=source)
    service.demos.write_text(json.dumps({'schemaVersion': 1, 'sources': [
        {**source, 'analysisId': service.analysis_id, 'generation': snapshot.token,
         'status': 'COMPLETED'}]}), encoding='utf-8')
    (snapshot.directory / 'media_source.json').write_text('{}', encoding='utf-8')
    (snapshot.directory / 'annotated_web.mp4').write_bytes(b'UNPUBLISHED ROOT PRESENTATION')
    catalog = service.guest.get('/api/demos')
    assert catalog.status_code == 200
    demo = catalog.json()['demos'][0]
    assert parse_qs(urlsplit(demo['previewUrl']).query) == {'generation': [snapshot.token]}
    assert demo['attribution']['author'] == source['author']
    assert service.guest.get(demo['previewUrl']).content == snapshot.annotated
    result_url = pinned(service, '/result', snapshot.token)
    result = service.guest.get(result_url)
    assert result.status_code == 200 and result.json()['mediaSource'] == source
    assert service.bob.get(result_url).status_code == 200
    service.demos.write_text('{"schemaVersion":1,"sources":[]}', encoding='utf-8')
    assert service.guest.get(result_url).status_code == 401
    assert service.guest.get(demo['previewUrl']).status_code == 401
    assert service.bob.get(result_url).status_code == 404
    assert service.bob.get(demo['previewUrl']).status_code == 404
    assert service.alice.get(result_url).status_code == 200


def test_curating_one_generation_never_publishes_later_private_revisions(service):
    from tests.test_public_demos import credit
    source = credit(UPLOAD)
    source['sourceSha256'] = hashlib.sha256(
        b'SYNTHETIC normalized source; no video decode' + bytes(range(128))).hexdigest()
    public = publish(service, 'Synthetic approved public generation', credit=source)
    private = publish(service, 'Synthetic later PRIVATE generation', revision=1, credit=source)
    entry = {**source, 'analysisId': service.analysis_id, 'status': 'COMPLETED'}

    # A historical analysis-wide grant cannot implicitly publish a full bundle.
    service.demos.write_text(json.dumps({'schemaVersion': 1, 'sources': [entry]}), encoding='utf-8')
    assert service.guest.get('/api/demos').json()['demos'] == []
    assert service.guest.get(service.base + '/result').status_code == 401
    assert service.bob.get(service.base + '/result').status_code == 404
    assert service.alice.get(service.base + '/result').json()['analysisGeneration'] == private.token

    entry['generation'] = public.token
    service.demos.write_text(json.dumps({'schemaVersion': 1, 'sources': [entry]}), encoding='utf-8')
    catalog = service.guest.get('/api/demos').json()['demos']
    assert len(catalog) == 1
    assert parse_qs(urlsplit(catalog[0]['previewUrl']).query) == {'generation': [public.token]}
    for client in (service.guest, service.bob):
        bare = client.get(service.base + '/result')
        assert bare.status_code == 200
        assert bare.json()['analysisGeneration'] == public.token
        assert bare.json()['summary']['label'] == public.label
        report = client.get(service.base + '/download/report')
        assert report.status_code == 200 and report.json()['analysisGeneration'] == public.token
        for suffix, expected in (('/media/annotated', public.annotated),
                                 ('/download/annotated', public.annotated),
                                 ('/download/source-timing', public.timing),
                                 ('/download/original', public.original)):
            response = client.get(service.base + suffix)
            assert response.status_code == 200 and response.content == expected
            assert response.headers['x-analysis-generation'] == public.token
        approved = client.get(pinned(service, '/result', public.token))
        assert approved.status_code == 200 and approved.json()['analysisGeneration'] == public.token

    for suffix in ('/result', '/media/source', '/media/annotated', '/download/original',
                   '/download/annotated', '/download/report', '/download/readable-report',
                   '/download/csv', '/download/source-timing'):
        url = pinned(service, suffix, private.token)
        assert service.guest.get(url).status_code == 401
        assert service.bob.get(url).status_code == 404
    assert service.alice.get(service.base + '/result').json()['analysisGeneration'] == private.token
    assert service.alice.get(pinned(service, '/result', public.token)).json()['analysisGeneration'] == public.token
    assert service.alice.get(pinned(service, '/media/annotated', private.token)).content == private.annotated

    service.demos.write_text('{"schemaVersion":1,"sources":[]}', encoding='utf-8')
    for url in (service.base + '/result', service.base + '/media/annotated',
                pinned(service, '/result', public.token), catalog[0]['previewUrl']):
        assert service.guest.get(url).status_code == 401
        assert service.bob.get(url).status_code == 404
    assert service.alice.get(pinned(service, '/result', private.token)).status_code == 200


def test_curated_bare_urls_preserve_private_owner_cache_and_cookie_variation(service):
    from tests.test_public_demos import credit
    source = credit(UPLOAD)
    source['sourceSha256'] = hashlib.sha256(
        b'SYNTHETIC normalized source; no video decode' + bytes(range(128))).hexdigest()
    public = publish(service, 'Synthetic cache approved generation', credit=source)
    private = publish(service, 'Synthetic cache PRIVATE generation', revision=1, credit=source)
    service.demos.write_text(json.dumps({'schemaVersion': 1, 'sources': [
        {**source, 'analysisId': service.analysis_id, 'generation': public.token,
         'status': 'COMPLETED'}]}), encoding='utf-8')
    for suffix in ('/result', '/media/source', '/media/annotated', '/download/original',
                   '/download/annotated', '/download/report', '/download/readable-report',
                   '/download/csv', '/download/source-timing'):
        for client, expected in ((service.alice, private), (service.guest, public)):
            response = client.get(service.base + suffix)
            assert response.status_code == 200, (suffix, response.text)
            assert 'cookie' in {value.strip().lower()
                                for value in response.headers.get('vary', '').split(',')}
            if client is service.alice:
                assert response.headers['cache-control'] == 'private, no-store'
            if suffix in {'/result', '/download/report'}:
                assert response.json()['analysisGeneration'] == expected.token
            elif suffix in {'/media/annotated', '/download/annotated'}:
                assert response.content == expected.annotated
            elif suffix == '/download/readable-report':
                assert expected.label in response.text
            elif suffix == '/download/source-timing':
                assert response.content == expected.timing
            elif suffix == '/download/csv':
                assert 'normalized_seconds' in response.text and len(response.text.splitlines()) == 66
            else:
                assert response.content == {'/media/source': expected.source,
                                            '/download/original': expected.original}[suffix]
    for client, expected in ((service.alice, private), (service.guest, public)):
        response = client.get(service.base + '/media/annotated', headers={'Range': 'bytes=8-19'})
        assert response.status_code == 206 and response.content == expected.annotated[8:20]
        assert 'cookie' in response.headers['vary'].lower()
        if client is service.alice:
            assert response.headers['cache-control'] == 'private, no-store'
    # Authenticated visitors also retain private caching, even on approved URLs.
    approved = service.bob.get(pinned(service, '/media/annotated', public.token))
    assert approved.status_code == 200 and approved.content == public.annotated
    assert approved.headers['cache-control'] == 'private, no-store'
    assert 'cookie' in approved.headers['vary'].lower()


def test_legacy_three_argument_publication_cannot_downgrade_a_full_generation(service):
    from src.api.result_document import publish_result
    from src.api.result_storage import ResultStorage
    snapshot = publish(service, 'Synthetic complete generation stays authoritative')
    pointer = snapshot.directory / '.result/current.store'
    previous_pointer = pointer.read_bytes()
    previous_exports = {name: (snapshot.directory / name).read_bytes()
                        for name in ('result.json', 'summary.json')}
    previous_report = service.alice.get(service.base + '/download/report')
    assert previous_report.status_code == 200
    with ResultStorage(snapshot.directory / '.result') as storage:
        downgraded = {'summary': {**snapshot.value['summary'], 'label': 'UNCOMMITTED LEGACY DOWNGRADE'}}
        with pytest.raises(ValueError, match='Legacy publication cannot replace'):
            publish_result(storage, downgraded, snapshot.directory)
    assert pointer.read_bytes() == previous_pointer
    assert {name: (snapshot.directory / name).read_bytes() for name in previous_exports} == previous_exports
    current_report = service.alice.get(service.base + '/download/report')
    assert current_report.status_code == 200 and current_report.content == previous_report.content
    assert current_report.json()['analysisGeneration'] == snapshot.token

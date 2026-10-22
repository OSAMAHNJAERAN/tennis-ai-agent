"""Only explicitly curated, hash-matched clips become public demonstrations."""

import csv
import hashlib
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import Settings
from src.api.media_source import validate_media_source
from src.api.reporting import readable_report, telemetry_csv
from src.api.results import build_result
from tests.test_product_api import write_artifacts


@pytest.fixture
def demo_service(tmp_path):
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'schemaVersion': 1, 'sources': []}))
    app = create_app(Settings(tmp_path / 'storage', Path('unused.yaml')), start_worker=False, demos_manifest=manifest)
    with TestClient(app) as client:
        yield client, app.state.store, manifest


def credit(input_bytes=b'original clip'):
    return {'id': 'licensed_example', 'title': 'An analyzed tennis excerpt', 'author': 'JC',
            'sourcePage': 'https://commons.wikimedia.org/wiki/File:Example.webm',
            'license': 'CC BY-SA 2.0', 'licenseUrl': 'https://creativecommons.org/licenses/by-sa/2.0/',
            'derivativeLicense': 'CC BY-SA 2.0', 'modifications': 'Audio omitted; tracking overlays added.',
            'sourceSha256': 'a' * 64, 'inputSha256': hashlib.sha256(input_bytes).hexdigest()}


def curated_job(service, completed=True, extension='.mp4'):
    _, store, manifest = service
    analysis_id = store.new_id()
    job = store.create(analysis_id, 'licensed' + extension, extension, 13, Path('unused.yaml'))
    input_bytes = b'original clip'
    upload = store.upload_dir(analysis_id)
    upload.mkdir(parents=True)
    (upload / ('original' + extension)).write_bytes(input_bytes)
    directory = store.output_dir(analysis_id)
    directory.mkdir(parents=True)
    source = credit(input_bytes)
    (directory / 'media_source.json').write_text(json.dumps(source))
    artifacts_job = write_artifacts(directory)
    artifacts_job.update(id=analysis_id, filename=job['filename'])
    result = build_result(directory, artifacts_job)
    (directory / 'result.json').write_text(json.dumps(result))
    (directory / 'annotated_web.mp4').write_bytes(b'annotated media')
    if completed:
        store.finish(analysis_id)
    entry = {**source, 'analysisId': analysis_id, 'status': 'COMPLETED'}
    manifest.write_text(json.dumps({'schemaVersion': 1, 'sources': [entry]}))
    return directory, result, entry


def write_manifest(manifest, entries):
    manifest.write_text(json.dumps({'schemaVersion': 1, 'sources': entries}))


def test_curated_completed_demo_has_exact_preview_route_and_credit(demo_service):
    client, _, _ = demo_service
    _, _, entry = curated_job(demo_service)
    response = client.get('/api/demos')
    assert response.status_code == 200
    demo = response.json()['demos'][0]
    assert demo['analysisId'] == entry['analysisId']
    assert demo['previewUrl'].startswith(f'/api/analyses/{entry["analysisId"]}/media/annotated?v=')
    assert demo['attribution']['author'] == 'JC'
    assert demo['attribution']['modifications'] == entry['modifications']
    assert demo['qualificationEvidence'] is False
    assert client.get(demo['previewUrl']).status_code == 200
    assert 'inputPath' not in demo and 'sourceUrl' not in demo


def test_media_revision_changes_after_presentation_refresh_without_rewriting_result(demo_service):
    client, _, _ = demo_service
    directory, saved, entry = curated_job(demo_service)
    endpoint = f'/api/analyses/{entry["analysisId"]}'
    # This fixture builds its result before writing fake media; explicitly make
    # a completed, media-available result for the cache regression scenario.
    saved['annotatedVideoUrl'] = endpoint + '/media/annotated'
    (directory / 'result.json').write_text(json.dumps(saved))
    original_result = (directory / 'result.json').read_bytes()
    first = client.get('/api/demos').json()['demos'][0]['previewUrl']
    displayed = client.get(endpoint + '/result').json()
    assert '?v=' in displayed['annotatedVideoUrl']
    assert client.get(endpoint + '/download/report').json() == displayed
    assert client.get(first).headers['cache-control'] == 'no-cache'
    (directory / 'annotated_web.mp4').write_bytes(b'refreshed and validated presentation')
    second = client.get('/api/demos').json()['demos'][0]['previewUrl']
    assert second != first
    refreshed = client.get(endpoint + '/result').json()
    assert refreshed['annotatedVideoUrl'] != displayed['annotatedVideoUrl']
    assert client.get(endpoint + '/download/report').json() == refreshed
    assert (directory / 'result.json').read_bytes() == original_result
    assert client.get(second).content == b'refreshed and validated presentation'


def test_private_complete_upload_is_never_automatically_a_demo(demo_service):
    client, _, manifest = demo_service
    curated_job(demo_service)
    write_manifest(manifest, [])
    assert client.get('/api/demos').json() == {'schemaVersion': 1, 'demos': []}


@pytest.mark.parametrize('extension', ['.webm', '.mkv'])
def test_trusted_curated_source_containers_are_supported_without_expanding_uploads(demo_service, extension):
    client, _, _ = demo_service
    curated_job(demo_service, extension=extension)
    assert len(client.get('/api/demos').json()['demos']) == 1
    response = client.post('/api/analyses', params={'filename': 'ordinary' + extension}, content=b'ordinary upload')
    assert response.status_code == 415


@pytest.mark.parametrize('mutation', ['pending_job', 'pending_manifest', 'missing_media', 'result_id',
                                      'result_hash', 'source_hash', 'upload_hash', 'credit_mismatch', 'missing_sidecar'])
def test_incomplete_or_mismatched_analysis_is_not_published(demo_service, mutation):
    client, store, manifest = demo_service
    directory, result, entry = curated_job(demo_service, completed=mutation != 'pending_job')
    if mutation == 'pending_manifest':
        entry['status'] = 'QUEUED'
        write_manifest(manifest, [entry])
    elif mutation == 'missing_media':
        (directory / 'annotated_web.mp4').unlink()
    elif mutation == 'result_id':
        result['summary']['id'] = store.new_id()
    elif mutation == 'result_hash':
        result['mediaSource']['inputSha256'] = 'b' * 64
    elif mutation == 'source_hash':
        sidecar = {**entry, 'sourceSha256': 'c' * 64}
        (directory / 'media_source.json').write_text(json.dumps(sidecar))
    elif mutation == 'upload_hash':
        (store.upload_dir(entry['analysisId']) / 'original.mp4').write_bytes(b'changed clip')
    elif mutation == 'credit_mismatch':
        result['mediaSource']['author'] = 'Another author'
    elif mutation == 'missing_sidecar':
        (directory / 'media_source.json').unlink()
    (directory / 'result.json').write_text(json.dumps(result))
    assert client.get('/api/demos').json()['demos'] == []


@pytest.mark.parametrize('field,value', [('id', '../private'), ('analysisId', '../../private'),
                                       ('sourcePage', 'javascript:alert(1)'),
                                       ('sourcePage', 'https://commons.wikimedia.org@private.example/video'),
                                       ('sourcePage', 'https://commons.wikimedia.org.evil.example/video'),
                                       ('sourcePage', 'https://localhost/video'),
                                       ('sourcePage', 'https://commons.wikimedia.org:9000/video'),
                                       ('sourcePage', 'http://commons.wikimedia.org/video'),
                                       ('licenseUrl', 'https://evil.example/license')])
def test_manifest_identifiers_and_links_are_whitelisted(demo_service, field, value):
    client, _, manifest = demo_service
    _, _, entry = curated_job(demo_service)
    entry[field] = value
    write_manifest(manifest, [entry])
    assert client.get('/api/demos').json()['demos'] == []


def test_missing_corrupt_and_duplicate_manifests_fail_closed(demo_service):
    client, _, manifest = demo_service
    _, _, entry = curated_job(demo_service)
    write_manifest(manifest, [entry, entry])
    assert len(client.get('/api/demos').json()['demos']) == 1
    manifest.write_text('{broken')
    assert client.get('/api/demos').json()['demos'] == []
    manifest.unlink()
    assert client.get('/api/demos').json()['demos'] == []


def test_excerpt_fields_and_report_credit_are_preserved_without_html_or_csv_injection(tmp_path):
    job = write_artifacts(tmp_path)
    source = credit()
    source.update(title='<script>unsafe title</script>', author='=InjectedAuthor',
                  modifications='Overlays added <img src=x onerror=alert(1)>')
    source['excerpt'] = {'startSeconds': 6, 'endSeconds': 16, 'startFrame': 360, 'endFrameExclusive': 960,
                         'sourceTimeBase': '1/16000', 'firstSourcePts': '96000', 'lastSourcePts': '255733'}
    (tmp_path / 'media_source.json').write_text(json.dumps(source))
    result = build_result(tmp_path, job)
    assert result['mediaSource']['excerpt'] == source['excerpt']
    html = readable_report(result)
    assert '<script>' not in html and '<img src=x' not in html
    assert '&lt;script&gt;' in html and 'Footage credit and sharing' in html
    assert 'CC BY-SA 2.0' in html and '6.000–16.000 s' in html
    assert 'Share-alike derivative license applies' in html
    assert 'https://commons.wikimedia.org/wiki/File:Example.webm' in html
    rows = list(csv.DictReader(io.StringIO(telemetry_csv(result))))
    assert len(rows) == 3 and rows[0]['media_author'] == "'=InjectedAuthor"
    assert rows[0]['media_source_sha256'] == source['sourceSha256']
    assert rows[1]['player1_speed_kmh'] == ''


def test_invalid_optional_credit_keeps_analysis_sections_available(tmp_path):
    job = write_artifacts(tmp_path)
    (tmp_path / 'media_source.json').write_text(json.dumps({**credit(), 'licenseUrl': 'javascript:alert(1)'}))
    result = build_result(tmp_path, job)
    assert result['mediaSource'] is None
    assert result['trajectories']['ballTrajectory'][0]['state'] == 'DETECTED'
    assert 'javascript:' not in readable_report(result)
    assert any('Source attribution unavailable' in warning for warning in result['warnings'])


def test_embedded_evidence_frames_include_the_source_credit(tmp_path):
    import cv2
    import numpy as np
    from src.api.reporting import evidence_images
    job = write_artifacts(tmp_path)
    source = credit()
    (tmp_path / 'media_source.json').write_text(json.dumps(source))
    writer = cv2.VideoWriter(str(tmp_path / 'annotated_web.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (100, 100))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((100, 100, 3), index * 40, dtype=np.uint8))
    writer.release()
    result = build_result(tmp_path, job)
    figures = evidence_images(tmp_path, result)
    assert figures.count('data:image/jpeg;base64,') == 3
    assert figures.count('by JC') == 3
    assert figures.count('Audio omitted; tracking overlays added.') == 3
    assert figures.count('Annotated footage and included evidence images: CC BY-SA 2.0.') == 3


@pytest.mark.parametrize('excerpt', [{'startSeconds': 1, 'endSeconds': 0}, 'full clip'])
def test_invalid_excerpt_never_publishes_partial_metadata(excerpt):
    assert validate_media_source({**credit(), 'excerpt': excerpt}) is None


def test_source_timing_download_uses_ndjson_type(demo_service):
    client, _, _ = demo_service
    directory, _, entry = curated_job(demo_service)
    content = '{"frame_index": 0}\n'
    (directory / 'source_frame_mapping.jsonl').write_bytes(content.encode('utf-8'))
    response = client.get(f'/api/analyses/{entry["analysisId"]}/download/source-timing')
    assert response.status_code == 200 and response.headers['content-type'] == 'application/x-ndjson'
    assert response.text == content

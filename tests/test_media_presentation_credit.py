"""Current derivative credits agree across responses without changing saved evidence."""

import copy
import csv
import hashlib
import io
import json
import os

import pytest

from src.api.media_revision import presentation_urls, presented_media_source
from tests.test_public_demos import curated_job, demo_service, write_manifest


MODIFICATIONS = ('Audio omitted; image ball observations and trails retained. '
                 'Header states that shot, rally and court positions are unavailable.')


@pytest.fixture
def presented_demo(demo_service):
    import cv2
    import numpy as np

    client, store, manifest = demo_service
    directory, saved, entry = curated_job(demo_service)
    source = saved['mediaSource']
    source['excerpt'] = {'startSeconds': 6, 'endSeconds': 16, 'startFrame': 360, 'endFrameExclusive': 960,
                         'sourceTimeBase': '1/16000', 'firstSourcePts': '96000', 'lastSourcePts': '255733'}
    entry['excerpt'] = copy.deepcopy(source['excerpt'])
    (directory / 'media_source.json').write_text(json.dumps(source), encoding='utf-8')
    endpoint = f'/api/analyses/{entry["analysisId"]}'
    saved['annotatedVideoUrl'] = endpoint + '/media/annotated'
    saved['exports']['annotatedVideo'] = endpoint + '/download/annotated'
    (directory / 'result.json').write_text(json.dumps(saved), encoding='utf-8')
    write_manifest(manifest, [entry])
    video = directory / 'annotated_web.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (100, 100))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((100, 100, 3), index * 40, dtype=np.uint8))
    writer.release()
    sidecar = {'schemaVersion': 1, 'analysisId': entry['analysisId'],
               'inputSha256': source['inputSha256'], 'sourceSha256': source['sourceSha256'],
               'annotatedSha256': hashlib.sha256(video.read_bytes()).hexdigest(),
               'modifications': MODIFICATIONS}
    sidecar_path = directory.parent / f'{entry["analysisId"]}.annotated_web_credit.json'
    return {'client': client, 'store': store, 'manifest': manifest, 'directory': directory,
            'saved': saved, 'source': source, 'entry': entry, 'endpoint': endpoint,
            'sidecar': sidecar, 'sidecar_path': sidecar_path}


def save_credit(example):
    example['sidecar_path'].write_text(json.dumps(example['sidecar']), encoding='utf-8')


def assert_response_credit(example, expected):
    client, endpoint = example['client'], example['endpoint']
    catalog = client.get('/api/demos').json()['demos']
    assert len(catalog) == 1
    result = client.get(endpoint + '/result').json()
    download = client.get(endpoint + '/download/report').json()
    assert result == download
    assert result['mediaSource']['modifications'] == catalog[0]['attribution']['modifications'] == expected
    html = client.get(endpoint + '/download/readable-report').text
    rows = list(csv.DictReader(io.StringIO(client.get(endpoint + '/download/csv').text)))
    assert expected in html
    assert len(rows) == 3 and {row['media_modifications'] for row in rows} == {expected}
    return catalog[0], result, html, rows


def test_hash_bound_sibling_credit_agrees_in_catalog_result_and_all_report_formats(presented_demo):
    example = presented_demo
    directory = example['directory']
    historical = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    manifest_before = example['manifest'].read_bytes()
    save_credit(example)
    assert example['sidecar_path'].parent == directory.parent
    assert not example['sidecar_path'].is_relative_to(directory)
    demo, result, html, rows = assert_response_credit(example, MODIFICATIONS)
    expected_source = {**example['source'], 'modifications': MODIFICATIONS}
    assert result['mediaSource'] == expected_source
    assert {key: result[key] for key in result if key not in {'mediaSource', 'annotatedVideoUrl', 'exports'}} == {
        key: example['saved'][key] for key in example['saved'] if key not in {'mediaSource', 'annotatedVideoUrl', 'exports'}}
    assert {key: result['exports'][key] for key in result['exports'] if key != 'annotatedVideo'} == {
        key: example['saved']['exports'][key] for key in example['saved']['exports'] if key != 'annotatedVideo'}
    for field in ('author', 'sourcePage', 'license', 'licenseUrl', 'derivativeLicense'):
        assert demo['attribution'][field] == expected_source[field]
    assert demo['excerpt'] == expected_source['excerpt']
    assert html.count('data:image/jpeg;base64,') == 3
    assert html.count(MODIFICATIONS) == 5  # Main credit, three embedded evidence images, appendix.
    assert 'Share-alike derivative license applies' in html and '6.000–16.000 s' in html
    assert rows[0]['media_author'] == expected_source['author']
    assert rows[0]['media_source_sha256'] == expected_source['sourceSha256']
    assert rows[0]['media_input_sha256'] == expected_source['inputSha256']
    assert {path: path.read_bytes() for path in historical} == historical
    assert example['manifest'].read_bytes() == manifest_before


def test_presentation_urls_does_not_mutate_the_callers_nested_source_or_exports(presented_demo):
    example = presented_demo
    save_credit(example)
    caller = example['saved']
    before = copy.deepcopy(caller)
    presented = presentation_urls(caller, example['directory'], example['entry']['analysisId'])
    assert caller == before
    assert presented is not caller and presented['mediaSource'] is not caller['mediaSource']
    assert presented['exports'] is not caller['exports']
    assert presented['mediaSource'] == {**before['mediaSource'], 'modifications': MODIFICATIONS}
    assert '?v=' in presented['annotatedVideoUrl'] and '?v=' in presented['exports']['annotatedVideo']


@pytest.mark.parametrize('invalid', [
    'missing', 'corrupt_json', 'non_object', 'too_large', 'invalid_utf8', 'directory',
    'cross_id', 'wrong_schema', 'boolean_schema', 'unknown_field',
    'input_hash', 'source_hash', 'annotated_hash', 'invalid_hash', 'missing_field',
    'blank_text', 'non_text', 'too_long_text', 'html_text', 'formula_text', 'control_text', 'bidi_text',
])
def test_absent_malformed_or_mismatched_derivative_credit_keeps_original_source(presented_demo, invalid):
    example = presented_demo
    sidecar = example['sidecar']
    path = example['sidecar_path']
    if invalid == 'missing':
        pass
    elif invalid == 'corrupt_json':
        path.write_text('{broken', encoding='utf-8')
    elif invalid == 'non_object':
        path.write_text('[]', encoding='utf-8')
    elif invalid == 'too_large':
        path.write_text(' ' * 8193, encoding='utf-8')
    elif invalid == 'invalid_utf8':
        path.write_bytes(b'\xff\xfe\xff')
    elif invalid == 'directory':
        path.mkdir()
    else:
        if invalid == 'cross_id':
            sidecar['analysisId'] = 'match_' + '0' * 32
        elif invalid == 'wrong_schema':
            sidecar['schemaVersion'] = 2
        elif invalid == 'boolean_schema':
            sidecar['schemaVersion'] = True
        elif invalid == 'unknown_field':
            sidecar['author'] = 'Another author'
        elif invalid in {'input_hash', 'source_hash', 'annotated_hash'}:
            sidecar[{'input_hash': 'inputSha256', 'source_hash': 'sourceSha256',
                     'annotated_hash': 'annotatedSha256'}[invalid]] = 'f' * 64
        elif invalid == 'invalid_hash':
            sidecar['annotatedSha256'] = '../another.mp4'
        elif invalid == 'missing_field':
            del sidecar['inputSha256']
        else:
            sidecar['modifications'] = {
                'blank_text': ' ', 'non_text': ['Overlays added'], 'too_long_text': 'a' * 1001,
                'html_text': '<img src=x onerror=alert(1)>', 'formula_text': ' =HYPERLINK("https://example.test")',
                'control_text': 'Overlays added\nNew line', 'bidi_text': 'Overlays\u202e added',
            }[invalid]
        save_credit(example)
    before = copy.deepcopy(example['source'])
    presented = presented_media_source(example['source'], example['directory'], example['entry']['analysisId'])
    assert presented is example['source'] and example['source'] == before
    _, result, html, _ = assert_response_credit(example, example['source']['modifications'])
    assert result['mediaSource'] == example['source']
    assert MODIFICATIONS not in html


@pytest.mark.parametrize('mismatch', ['missing_persisted', 'corrupt_persisted', 'source_credit', 'result_credit'])
def test_original_credit_equality_is_required_before_a_derivative_override(presented_demo, mismatch):
    example = presented_demo
    save_credit(example)
    directory = example['directory']
    if mismatch == 'missing_persisted':
        (directory / 'media_source.json').unlink()
    elif mismatch == 'corrupt_persisted':
        (directory / 'media_source.json').write_text('{broken', encoding='utf-8')
    elif mismatch == 'source_credit':
        source = {**example['source'], 'author': 'Another author'}
        (directory / 'media_source.json').write_text(json.dumps(source), encoding='utf-8')
    else:
        saved = copy.deepcopy(example['saved'])
        saved['mediaSource']['author'] = 'Another author'
        (directory / 'result.json').write_text(json.dumps(saved), encoding='utf-8')
    assert example['client'].get('/api/demos').json()['demos'] == []
    result = example['client'].get(example['endpoint'] + '/result').json()
    assert result['mediaSource']['modifications'] == example['source']['modifications']


def test_media_byte_change_invalidates_cached_derivative_credit(presented_demo):
    example = presented_demo
    save_credit(example)
    _, initial, _, _ = assert_response_credit(example, MODIFICATIONS)
    path = example['directory'] / 'annotated_web.mp4'
    previous_stat = path.stat()
    content = path.read_bytes()
    # Same size, explicit changed mtime: exercise the reused cached hash's
    # stat-based invalidation rather than relying on a different file length.
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000))
    _, current, _, _ = assert_response_credit(example, example['source']['modifications'])
    assert current['annotatedVideoUrl'] != initial['annotatedVideoUrl']


@pytest.mark.parametrize('analysis_id', ['../match_' + 'a' * 32, 'match_' + 'A' * 32, '', None])
def test_invalid_analysis_id_never_reads_a_sidecar(presented_demo, analysis_id, monkeypatch):
    def no_read(*_args, **_kwargs):
        raise AssertionError('Invalid analysis ID must be rejected before reading a path.')
    monkeypatch.setattr('src.api.media_revision._read_object', no_read)
    example = presented_demo
    assert presented_media_source(example['source'], example['directory'], analysis_id) is example['source']


def test_credit_inside_analysis_directory_is_not_used(presented_demo):
    example = presented_demo
    (example['directory'] / example['sidecar_path'].name).write_text(json.dumps(example['sidecar']), encoding='utf-8')
    assert_response_credit(example, example['source']['modifications'])

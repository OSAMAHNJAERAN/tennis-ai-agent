"""CSV values stay stable while the HTTP response consumes bounded chunks."""

import asyncio
import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from src.api import reporting
from src.api.app import create_app
from src.api.jobs import Settings


def telemetry_result():
    return {
        'summary': {'id': 'match_telemetry'},
        'mediaSource': {
            'id': 'tennis', 'title': 'Tennis, "footage"', 'author': '=Example é author',
            'sourcePage': 'https://commons.wikimedia.org/wiki/File:Tennis.webm',
            'license': 'CC BY-SA 4.0',
            'licenseUrl': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'derivativeLicense': 'CC BY-SA 4.0', 'modifications': 'Video excerpt; overlays added.',
            'sourceSha256': 'a' * 64, 'inputSha256': 'b' * 64,
        },
        'trajectories': {
            'ballTrajectory': [
                {'timestamp_seconds': 0., 'state': 'DETECTED', 'x_px': 30., 'y_px': 50.,
                 'confidence': .9, 'court_x_m': 4., 'court_y_m': 8.},
                {'timestamp_seconds': 1 / 30, 'state': 'MISSING', 'x_px': None, 'y_px': None,
                 'confidence': None, 'court_x_m': None, 'court_y_m': None},
                {'timestamp_seconds': 2 / 30, 'state': 'OCCLUDED', 'x_px': None, 'y_px': None,
                 'confidence': .2, 'court_x_m': None, 'court_y_m': None},
            ],
            'player1CourtPositions': [[4., 8.], None], 'player1Speeds': [0., None, 10.],
            'player2CourtPositions': [None, [5., 9.], [6., 9.]], 'player2Speeds': [None, 12.],
        },
    }


def expected_csv():
    # Independent rows pin the existing column order, nulls and quoting rules.
    rows = [
        ['analysis_id', 'frame', 'normalized_seconds', 'ball_state', 'ball_x_px', 'ball_y_px',
         'ball_confidence', 'ball_projected_court_x_m', 'ball_projected_court_y_m',
         'player1_ground_x_m', 'player1_ground_y_m', 'player1_speed_kmh',
         'player2_ground_x_m', 'player2_ground_y_m', 'player2_speed_kmh',
         'media_source_id', 'media_title', 'media_author', 'media_source_page',
         'media_license', 'media_license_url', 'media_derivative_license', 'media_modifications',
         'media_source_sha256', 'media_input_sha256'],
    ]
    credit = ['tennis', 'Tennis, "footage"', "'=Example é author",
              'https://commons.wikimedia.org/wiki/File:Tennis.webm', 'CC BY-SA 4.0',
              'https://creativecommons.org/licenses/by-sa/4.0/', 'CC BY-SA 4.0',
              'Video excerpt; overlays added.', 'a' * 64, 'b' * 64]
    rows.extend([
        ['match_telemetry', 0, 0., 'DETECTED', 30., 50., .9, 4., 8., 4., 8., 0., None, None, None, *credit],
        ['match_telemetry', 1, 1 / 30, 'MISSING', None, None, None, None, None,
         None, None, None, 5., 9., 12., *credit],
        ['match_telemetry', 2, 2 / 30, 'OCCLUDED', None, None, .2, None, None,
         None, None, 10., 6., 9., None, *credit],
    ])
    buffer = io.StringIO(newline='')
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


@pytest.mark.parametrize('chunk_chars', [1, 1000, 65536])
def test_streamed_csv_and_compatibility_wrapper_preserve_exact_bytes(chunk_chars):
    result = telemetry_result()
    expected = expected_csv()
    assert ''.join(reporting.telemetry_csv_chunks(result, chunk_chars=chunk_chars)) == expected
    assert reporting.telemetry_csv(result) == expected
    assert expected.endswith('\r\n')


@pytest.mark.parametrize('source', [None, {'id': 'invalid'}])
def test_absent_or_invalid_credit_remains_empty_without_changing_observations(source):
    result = telemetry_result()
    result['mediaSource'] = source
    rows = list(csv.DictReader(io.StringIO(''.join(reporting.telemetry_csv_chunks(result)))))
    assert len(rows) == 3
    assert rows[0]['ball_x_px'] == '30.0' and rows[0]['player1_speed_kmh'] == '0.0'
    assert rows[1]['ball_state'] == 'MISSING' and rows[2]['ball_state'] == 'OCCLUDED'
    assert all(row[key] == '' for row in rows for key in row if key.startswith('media_'))


class CountedFrames:
    """A long input that records how far serialization reads ahead."""

    def __init__(self, count):
        self.count = count
        self.consumed = 0

    def __iter__(self):
        for _ in range(self.count):
            self.consumed += 1
            yield {'timestamp_seconds': 0., 'state': 'MISSING'}


def test_header_is_immediate_and_long_input_is_consumed_in_bounded_chunks():
    frames = CountedFrames(100_000)
    result = {'summary': {'id': 'match_large'}, 'trajectories': {'ballTrajectory': frames}}
    chunks = reporting.telemetry_csv_chunks(result, chunk_chars=1024)
    assert frames.consumed == 0
    assert next(chunks).startswith('analysis_id,frame,normalized_seconds,ball_state,')
    assert frames.consumed == 0
    first = next(chunks)
    assert len(first) <= 1024
    assert 1 < frames.consumed < 100
    chunk_count, row_count = 1, first.count('\r\n')
    for chunk in chunks:
        chunk_count += 1
        row_count += chunk.count('\r\n')
        assert len(chunk) <= 1024
    assert chunk_count > 1000 and row_count == frames.consumed == frames.count


def test_a_single_long_csv_row_is_never_split_inside_a_quoted_cell():
    result = telemetry_result()
    chunks = list(reporting.telemetry_csv_chunks(result, chunk_chars=100))
    # Header or a single row can exceed the target. The original row remains whole.
    assert len(chunks) == 4
    assert all(chunk.endswith('\r\n') for chunk in chunks)
    assert ''.join(chunks) == expected_csv()


@pytest.fixture
def saved_csv_service(tmp_path):
    app = create_app(Settings(tmp_path, Path('unused.yaml')), start_worker=False)
    store = app.state.store
    job = store.create(store.new_id(), 'tennis.mp4', '.mp4', 12, Path('unused.yaml'))
    directory = store.output_dir(job['id'])
    directory.mkdir(parents=True)
    result = telemetry_result()
    result['summary']['id'] = job['id']
    (directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    store.finish(job['id'])
    return app, store, job['id'], result


def request_scope(path):
    return {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
            'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
            'path': path, 'raw_path': path.encode('ascii'), 'query_string': b'',
            'root_path': '', 'headers': [], 'server': ('127.0.0.1', 80),
            'client': ('127.0.0.1', 12345)}


async def empty_request():
    return {'type': 'http.request', 'body': b'', 'more_body': False}


def test_fastapi_csv_route_streams_utf8_with_download_headers_and_no_full_buffer(saved_csv_service, monkeypatch):
    app, _, analysis_id, result = saved_csv_service
    real_chunks = reporting.telemetry_csv_chunks
    events = []

    def observe_chunks(saved):
        events.append('iterator_started')
        for chunk in real_chunks(saved, chunk_chars=100):
            events.append('chunk_produced')
            yield chunk

    def prohibit_legacy_buffer(_saved):
        raise AssertionError('The HTTP route must not materialize the entire CSV.')

    monkeypatch.setattr(reporting, 'telemetry_csv_chunks', observe_chunks)
    monkeypatch.setattr(reporting, 'telemetry_csv', prohibit_legacy_buffer)
    messages = []

    async def send(message):
        messages.append(message.copy())
        events.append(message['type'])

    asyncio.run(app(request_scope(f'/api/analyses/{analysis_id}/download/csv'), empty_request, send))
    assert events[:2] == ['http.response.start', 'iterator_started']
    start, *bodies = messages
    assert start['status'] == 200
    headers = dict(start['headers'])
    assert headers[b'content-type'] == b'text/csv; charset=utf-8'
    assert headers[b'content-disposition'] == f'attachment; filename="{analysis_id}_csv.csv"'.encode()
    assert headers[b'x-content-type-options'] == b'nosniff'
    assert headers[b'content-security-policy'] == b"default-src 'none'; img-src data:; style-src 'unsafe-inline'"
    assert b'content-length' not in headers
    assert len(bodies) == 5 and bodies[-1] == {'type': 'http.response.body', 'body': b'', 'more_body': False}
    assert all(body['more_body'] for body in bodies[:-1])
    assert b''.join(body['body'] for body in bodies) == ''.join(real_chunks(result)).encode('utf-8')


def test_disconnected_csv_request_stops_before_reading_future_rows(saved_csv_service, monkeypatch):
    app, _, analysis_id, _ = saved_csv_service
    produced = []

    def observe_chunks(_saved):
        for index in range(1000):
            produced.append(index)
            yield f'row {index}\r\n'

    monkeypatch.setattr(reporting, 'telemetry_csv_chunks', observe_chunks)

    async def send(message):
        if message['type'] == 'http.response.body':
            raise OSError('Client disconnected while sending the first chunk.')

    with pytest.raises(ClientDisconnect):
        asyncio.run(app(request_scope(f'/api/analyses/{analysis_id}/download/csv'), empty_request, send))
    assert produced == [0]


@pytest.mark.parametrize('saved_contents', [None, '{broken JSON'])
def test_missing_or_corrupt_saved_result_fails_before_csv_headers(saved_csv_service, saved_contents):
    app, store, analysis_id, _ = saved_csv_service
    path = store.output_dir(analysis_id) / 'result.json'
    if saved_contents is None:
        path.unlink()
    else:
        path.write_text(saved_contents, encoding='utf-8')
    with TestClient(app) as client:
        response = client.get(f'/api/analyses/{analysis_id}/download/csv')
    assert response.status_code == 503
    assert response.headers['content-type'] == 'application/json'
    assert 'content-disposition' not in response.headers
    assert response.json()['detail'] == 'The saved analysis is unavailable. Upload the video again to retry.'


def test_csv_iteration_failure_does_not_send_a_successful_end_of_stream(saved_csv_service, monkeypatch):
    app, _, analysis_id, _ = saved_csv_service

    def fail_after_header(_saved):
        yield 'header\r\n'
        raise ValueError('Invalid persisted frame row.')

    monkeypatch.setattr(reporting, 'telemetry_csv_chunks', fail_after_header)
    messages = []

    async def send(message):
        messages.append(message.copy())

    with pytest.raises(ValueError, match='Invalid persisted frame row'):
        asyncio.run(app(request_scope(f'/api/analyses/{analysis_id}/download/csv'), empty_request, send))
    assert messages[0]['status'] == 200  # Headers have already started; the body must abort.
    assert messages[1] == {'type': 'http.response.body', 'body': b'header\r\n', 'more_body': True}
    assert not any(message.get('more_body') is False for message in messages)

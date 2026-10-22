"""Exercise worker publication with real storage and no media/model processing."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.jobs import JobStore, Settings
from src.api import worker


@pytest.fixture
def publication_job(tmp_path, monkeypatch):
    # Importing app.py creates its default app. Confine that bootstrap store to
    # this fixture too, rather than opening the workspace's product database.
    monkeypatch.setenv('TENNIS_STORAGE_ROOT', str(tmp_path / 'bootstrap'))
    from src.api.app import create_app

    settings = Settings(tmp_path / 'storage', tmp_path / 'unused.yaml')
    store = JobStore(settings.root)
    store.create(store.new_id(), 'Tennis é.mp4', '.mp4', 12, settings.config)
    job = store.claim()
    output = store.output_dir(job['id'])
    output.mkdir(parents=True)
    upload = store.upload_dir(job['id'])
    upload.mkdir(parents=True)
    (upload / 'original.mp4').write_bytes(b'isolated placeholder')
    result = {
        'contractVersion': '2.0',
        'summary': {'id': job['id'], 'label': 'Tennis é 🎾', 'frames': 3, 'fps': 30.,
                    'durationSeconds': .1, 'status': 'COMPLETED'},
        'trajectories': {'ballTrajectory': [
            {'frame_index': 0, 'timestamp_seconds': 0., 'x_px': 0., 'y_px': 12.5,
             'state': 'DETECTED', 'confidence': .9},
            {'frame_index': 1, 'timestamp_seconds': 1 / 30, 'x_px': None, 'y_px': None,
             'state': 'MISSING', 'confidence': None},
            {'frame_index': 2, 'timestamp_seconds': 2 / 30, 'x_px': None, 'y_px': None,
             'state': 'OCCLUDED', 'confidence': .2},
        ]},
        'empty': [], 'nullable': None, 'qualified': False,
        'coordinates': [0, -0., 1.25, 1e-12],
    }
    processing_calls = []

    def check_models(config):
        assert config == settings.config
        processing_calls.append('model_check_stub')
        return {}

    def execute(args, timeout=7200):
        assert args[0] in {'ffmpeg', 'ffprobe'}
        processing_calls.append(args)
        if args[0] == 'ffprobe':
            if '-count_frames' in args:
                return '{"streams":[{"nb_read_frames":"3"}]}'
            assert timeout == 60
            return json.dumps({'streams': [{'width': 100, 'height': 100,
                               'avg_frame_rate': '30/1', 'r_frame_rate': '30/1',
                               'duration': '0.1', 'nb_frames': '3', 'start_time': '0',
                               'time_base': '1/30000', 'codec_name': 'h264'}]})
        assert Path(args[-1]).parent == output
        Path(args[-1]).write_bytes(b'no real encoding or inference')
        return ''

    def frame_mapping(_stats, destination, *_args):
        assert destination == output / 'source_frame_mapping.jsonl'
        destination.write_text('{"frame_index":0}\n', encoding='utf-8')
        return {'frames': 3, 'status': 'CAPTURED_FFMPEG_FRAME_MAP',
                'validMeasurementTransitionIntervals': [{'startFrame': 1, 'endFrameExclusive': 3}]}

    class PipelineStub:
        def __init__(self, config):
            assert Path(config) == output / 'inference_config.yaml'

        def run(self, source, directory):
            assert Path(source) == output / 'source.mp4' and Path(directory) == output
            processing_calls.append('pipeline_stub')
            (output / 'annotated.mp4').write_bytes(b'no model import')

    def build_result(directory, claimed_job):
        assert directory == output and claimed_job == job
        processing_calls.append('result_builder_stub')
        return result

    pipeline_module = ModuleType('src.pipeline.phase6_pipeline')
    pipeline_module.Phase6Pipeline = PipelineStub
    monkeypatch.setitem(sys.modules, 'src.pipeline.phase6_pipeline', pipeline_module)
    # The worker's finally block must not inspect any real CUDA runtime that a
    # broader test session might already have imported.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(worker, 'check_models', check_models)
    monkeypatch.setattr(worker, 'binary', lambda name: name)
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'frame_mapping', frame_mapping)
    monkeypatch.setattr(worker, 'build_result', build_result)

    app = create_app(settings, start_worker=False)
    with TestClient(app) as client:
        yield SimpleNamespace(store=store, job=job, settings=settings, output=output,
                              result=result, client=client, processing_calls=processing_calls)


def observe_publication(fixture, monkeypatch, previous, failure=None):
    """Keep real file writes while injecting failure at one publication boundary."""
    destination = fixture.output / 'result.json'
    temporary = fixture.output / 'result.tmp'
    if previous is not None:
        destination.write_bytes(previous)
    original_open = Path.open
    original_replace = Path.replace
    original_finish = fixture.store.finish
    events = []
    streams = []
    finishes = []

    def assert_unpublished():
        assert fixture.store.get(fixture.job['id'])['status'] == 'PROCESSING'
        if previous is None:
            assert not destination.exists()
        else:
            assert destination.read_bytes() == previous

    class PublicationStream:
        def __init__(self, handle):
            self.handle = handle
            self.written = 0

        def __enter__(self):
            self.handle.__enter__()
            events.append('opened')
            assert_unpublished()
            return self

        def write(self, value):
            assert_unpublished()
            events.append('write')
            if failure == 'write' and self.written + len(value) > 80:
                part = value[:max(0, 80 - self.written)]
                self.handle.write(part)
                self.written += len(part)
                self.handle.flush()
                events.append('write_failure')
                raise OSError('Injected mid-result storage failure')
            self.written += len(value)
            return self.handle.write(value)

        def __exit__(self, exc_type, exc_value, traceback):
            handled = self.handle.__exit__(exc_type, exc_value, traceback)
            assert self.handle.closed
            events.append('closed')
            assert_unpublished()
            if failure == 'close' and exc_type is None:
                assert json.loads(temporary.read_text(encoding='utf-8')) == fixture.result
                events.append('close_failure')
                raise OSError('Injected result stream close failure')
            return handled

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == temporary and args and args[0] == 'w':
            assert kwargs.get('encoding') == 'utf-8'
            stream = PublicationStream(handle)
            streams.append(stream)
            return stream
        return handle

    def replace_file(path, target):
        if path == temporary:
            assert Path(target) == destination
            assert streams and all(stream.handle.closed for stream in streams)
            assert_unpublished()
            assert json.loads(temporary.read_text(encoding='utf-8')) == fixture.result
            events.append('replace')
            if failure == 'replace':
                events.append('replace_failure')
                raise OSError('Injected result replacement failure')
        return original_replace(path, target)

    def finish(analysis_id, error=None):
        assert analysis_id == fixture.job['id']
        assert fixture.store.get(analysis_id)['status'] == 'PROCESSING'
        finishes.append(error)
        if error is None:
            assert 'replace' in events
            assert json.loads(destination.read_text(encoding='utf-8')) == fixture.result
            assert json.loads((fixture.output / 'summary.json').read_text(encoding='utf-8')) == fixture.result['summary']
        else:
            assert_unpublished()
        return original_finish(analysis_id, error)

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(Path, 'replace', replace_file)
    monkeypatch.setattr(fixture.store, 'finish', finish)
    return SimpleNamespace(destination=destination, temporary=temporary, events=events,
                           streams=streams, finishes=finishes)


PREVIOUS = b'{"previous_completed_result": true, "untouched": "original bytes"}\n'


@pytest.mark.parametrize('previous', [None, PREVIOUS], ids=['new-result', 'replace-prior-result'])
def test_successful_worker_publication_matches_previous_json_serializer(publication_job, monkeypatch, previous):
    fixture = publication_job
    observed = observe_publication(fixture, monkeypatch, previous)
    expected = json.dumps(fixture.result, allow_nan=False).encode('utf-8')
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    saved_job = fixture.store.get(fixture.job['id'])
    assert saved_job['status'] == 'COMPLETED' and saved_job['error'] is None
    assert saved_job['completed_at'] and saved_job['step'] == 11
    assert observed.finishes == [None]
    assert observed.destination.read_bytes() == expected
    assert not observed.temporary.exists()
    assert len(observed.streams) == 1 and observed.streams[0].handle.closed
    assert observed.events.index('closed') < observed.events.index('replace')
    assert fixture.processing_calls.count('pipeline_stub') == 1
    assert fixture.processing_calls.count('result_builder_stub') == 1
    response = fixture.client.get(f'/api/analyses/{fixture.job["id"]}/result')
    assert response.status_code == 200 and response.json() == fixture.result


@pytest.mark.parametrize('previous', [None, PREVIOUS], ids=['new-result', 'preserve-prior-result'])
@pytest.mark.parametrize('failure', ['write', 'encoder', 'close', 'replace'])
def test_failed_worker_publication_never_exposes_partial_result(publication_job, monkeypatch, previous, failure):
    fixture = publication_job
    observed = observe_publication(fixture, monkeypatch, previous, failure)
    if failure == 'encoder':
        fixture.result['invalid_measurement'] = float('nan')
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    saved_job = fixture.store.get(fixture.job['id'])
    assert saved_job['status'] == 'FAILED' and saved_job['completed_at']
    assert saved_job['step'] == 0 and saved_job['error']
    assert len(observed.finishes) == 1 and observed.finishes[0] == saved_job['error']
    assert len(observed.streams) == 1 and observed.streams[0].handle.closed
    if previous is None:
        assert not observed.destination.exists()
    else:
        assert observed.destination.read_bytes() == previous
    assert not (fixture.output / 'summary.json').exists()
    assert fixture.processing_calls.count('pipeline_stub') == 1
    assert fixture.processing_calls.count('result_builder_stub') == 1
    assert sum(isinstance(call, list) and '-count_frames' in call for call in fixture.processing_calls) == 2
    log = (fixture.output / 'processing.log').read_text(encoding='utf-8')
    if failure == 'encoder':
        assert 'Out of range float values are not JSON compliant' in log
        assert 'Video or analysis data was invalid' in saved_job['error']
        assert 'replace' not in observed.events
    else:
        assert f'{failure}_failure' in observed.events
        assert 'Injected' in log and 'Injected' not in saved_job['error']
        assert 'analysis pipeline failed' in saved_job['error']
    if failure in {'write', 'encoder'}:
        with pytest.raises(json.JSONDecodeError):
            json.loads(observed.temporary.read_text(encoding='utf-8'))
    else:
        assert json.loads(observed.temporary.read_text(encoding='utf-8')) == fixture.result
    # Failed jobs retain artifacts for inspection, but none can be presented as
    # a completed analysis, even when a previous result file still exists.
    base = f'/api/analyses/{fixture.job["id"]}'
    status = fixture.client.get(base)
    assert status.status_code == 200 and status.json()['status'] == 'FAILED'
    result_response = fixture.client.get(base + '/result')
    assert result_response.status_code == 409 and result_response.json()['detail'] == saved_job['error']
    for suffix in ('download/report', 'download/csv', 'download/readable-report', 'media/source', 'media/annotated'):
        assert fixture.client.get(base + '/' + suffix).status_code == 404

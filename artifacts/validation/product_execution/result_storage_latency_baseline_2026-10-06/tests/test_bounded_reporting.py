"""Frozen-output checks for reports reading generated, disk-like series."""

from collections.abc import Sequence
import copy
import importlib.util
import json
from pathlib import Path
import tracemalloc
import weakref

import pytest

from src.api import reporting
from src.api.result_document import ResultDocument
from src.api.result_storage import DiskSeries
from tests.test_streaming_telemetry_csv import telemetry_result


@pytest.fixture(scope='module')
def original_reporting():
    path = (Path(__file__).resolve().parents[1] / 'artifacts' / 'validation' /
            'product_execution' / 'result_storage_baseline_2026-10-06' /
            'src' / 'api' / 'reporting.py')
    spec = importlib.util.spec_from_file_location('original_bounded_reporting', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedSeries(Sequence):
    """Fresh records on each read, with no slice/materialization API."""

    def __init__(self, count, factory):
        self.count = count
        self.factory = factory
        self.reads = 0
        self.iterations = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError('Report consumers must not slice a growing series.')
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        return self.factory(index)

    def __iter__(self):
        self.iterations += 1
        for index in range(self.count):
            yield self[index]


class GeneratedDiskSeries(GeneratedSeries, DiskSeries):
    """A DiskSeries-shaped producer for retention and read-ahead checks."""


def report_result(count):
    result = telemetry_result()
    result['summary'].update(label='Court <clip> & "evidence"', analyzedAt='2026-10-06',
                             frames=count, fps=59.94, durationSeconds=count / 59.94)
    result['contractVersion'] = '2.0'
    result['provenance'] = {'bundleVersion': '<bundle>'}
    result['court'] = {'reprojectionErrorPx': None}
    result['pipelineMetrics'] = {'pipelineFps': 1.25}
    result['warnings'] = ['<unverified>', 'Court fit & availability differ.']
    result['events'] = [{'event_type': '<hit> & "contact"', 'timestamp_s': 0., 'confidence': .2},
                        {'event_type': 'bounce', 'timestamp_s': None, 'confidence': None}]
    result['playerMetrics'] = {'player_1': {'coverage_pct': 45., 'total_distance_m': 0.,
                                          'avg_speed_kmh': 3.5, 'max_speed_kmh': 9.25}}
    result['playerTracking'] = {
        'player_1': {'speedIntervals': [{'startSeconds': index / 60,
                                       'endSeconds': (index + 1) / 60} for index in range(35)]},
        'player_2': {'metrics': {'coverage_pct': 0.}, 'speedIntervals': []},
    }
    result['poseTracking'] = {'coveragePct': {'player_1': None, 'player_2': 0.}}
    result['racketTracking'] = {'coveragePct': {'player_1': 2., 'player_2': None}}
    result['trajectories'] = {
        'ballTrajectory': [{'timestamp_seconds': index / 59.94,
                            'state': ('DETECTED', 'MISSING', 'OCCLUDED')[index % 3],
                            'x_px': 3.25 if index % 3 == 0 else None,
                            'y_px': 0., 'confidence': .75, 'court_x_m': None,
                            'court_y_m': None} for index in range(count)],
        'player1CourtPositions': [[8., 2.] if index % 3 == 0 else
                                 [1., 20.] if index % 3 == 1 else None for index in range(count)],
        'player2CourtPositions': [[-3., -4.] if index % 2 else [13.97, 27.77]
                                 for index in range(count)],
        'player1Speeds': [None if index % 17 == 0 else (index % 29) / 3 for index in range(count)],
        'player2Speeds': [0. if index % 11 == 0 else (index % 13) / 7 for index in range(count)],
    }
    result['imagePeople'] = {'frames': [
        {'people': None} if index % 3 == 0 else {'people': []} if index % 3 == 1 else
        {'people': [{'bbox': [1., 2., 3., 4.]}]} for index in range(count)]}
    return result


def lazy_result(result):
    lazy = copy.deepcopy(result)
    for key, values in lazy['trajectories'].items():
        lazy['trajectories'][key] = GeneratedSeries(len(values), values.__getitem__)
    frames = lazy['imagePeople']['frames']
    lazy['imagePeople']['frames'] = GeneratedDiskSeries(len(frames), frames.__getitem__)
    for player in lazy['playerTracking'].values():
        values = player['speedIntervals']
        player['speedIntervals'] = GeneratedSeries(len(values), values.__getitem__)
    for key in ('events', 'warnings'):
        values = lazy[key]
        lazy[key] = GeneratedSeries(len(values), values.__getitem__)
    return lazy


@pytest.mark.parametrize('count', [0, 1, 600, 601, 1201])
def test_report_and_csv_are_exactly_equal_for_lazy_series(original_reporting, count):
    result = report_result(count)
    expected_html = original_reporting.readable_report(result)
    expected_csv = original_reporting.telemetry_csv(result)
    lazy = lazy_result(result)
    assert ''.join(reporting.readable_report_chunks(lazy)) == expected_html
    assert reporting.readable_report(lazy) == expected_html
    assert ''.join(reporting.telemetry_csv_chunks(lazy, chunk_chars=1024)) == expected_csv
    assert reporting.telemetry_csv(lazy) == expected_csv


@pytest.mark.parametrize('samples', [
    [], [None, float('nan'), float('inf')], [0., -0., True, False, 2.],
    [1e16, 1., -1e16] * 601, [0.1, 0.2, 0.3] * 401,
    [10 ** 308, 10 ** 308, 1., None] * 601,
    [None if index % 151 == 0 else float(index % 13) for index in range(1201)],
])
def test_speed_bins_preserve_previous_sums_gaps_and_points(original_reporting, samples):
    series = GeneratedSeries(len(samples), samples.__getitem__)
    assert reporting.speed_svg(series, 60.) == original_reporting.speed_svg(samples, 60.)
    assert series.reads <= 2 * len(samples)


def test_heatmap_retains_first_occupied_cell_order_and_resident_vector_rules(original_reporting):
    points = [[8., 2.], [1., 20.], [8., 2.], None, (2., 3.), [True, False],
              [float('inf'), 2.], [-3., -4.], [13.97, 27.77], [0., 0., 0.]]
    series = GeneratedSeries(len(points), points.__getitem__)
    assert reporting.heatmap_svg(series, '#123456') == original_reporting.heatmap_svg(points, '#123456')
    assert series.reads == len(points)


@pytest.mark.parametrize('frames', [None, 'not an array', b'bytes', bytearray(b'bytes'),
                                    [], [{'people': None}], [{'people': []}],
                                    [{'people': []}, {'people': [1, 2]}, {'people': None},
                                     {'people': (1, 2)}, None]])
def test_people_summary_preserves_missing_vs_empty_observations(original_reporting, frames):
    result = {'summary': {'frames': 4}, 'imagePeople': {'frames': frames}}
    expected = original_reporting.image_people_summary(result)
    if isinstance(frames, list):
        result['imagePeople']['frames'] = GeneratedDiskSeries(len(frames), frames.__getitem__)
    assert reporting.image_people_summary(result) == expected


def test_people_counts_fold_without_a_per_frame_count_buffer():
    frames = GeneratedDiskSeries(100_000, lambda index: {'people': [] if index % 2 else [{}]})
    result = {'summary': {'frames': len(frames)}, 'imagePeople': {'frames': frames}}
    tracemalloc.start()
    try:
        summary = reporting.image_people_summary(result)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert '50,000 observed boxes across 50,000 of 100,000 frames' in summary
    assert 'available for 100,000 frames' in summary
    assert frames.reads == 100_000
    assert peak < 128 * 1024


def test_speed_svg_retains_only_bounded_generated_values():
    live = weakref.WeakSet()
    peak_live = 0

    class ObservedFloat(float):
        pass

    def value(index):
        nonlocal peak_live
        observation = ObservedFloat(index)
        live.add(observation)
        peak_live = max(peak_live, len(live))
        return observation

    series = GeneratedSeries(60_001, value)
    svg = reporting.speed_svg(series, 60.)
    assert svg.count(',') <= 600
    assert peak_live <= 8
    assert len(live) == 0
    assert series.reads == 2 * len(series)


def test_event_rows_are_read_on_demand_and_report_close_stops_future_reads():
    result = report_result(0)
    events = GeneratedSeries(100_000, lambda index: {'event_type': f'event {index}',
                                                   'timestamp_s': index / 60,
                                                   'confidence': .5})
    result['events'] = events
    chunks = reporting.readable_report_chunks(result)
    assert events.reads == 0
    assert next(chunks).startswith('<!doctype html>')
    assert events.reads == 0
    for chunk in chunks:
        if '<td>event 0</td>' in chunk:
            assert events.reads == 1
            assert len(chunk) < 256
            break
    else:
        raise AssertionError('The first event row was not emitted.')
    chunks.close()
    assert events.reads == 1


def test_warning_rows_are_read_on_demand():
    result = report_result(0)
    warnings = GeneratedSeries(100_000, lambda index: f'<warning {index}>')
    result['warnings'] = warnings
    chunks = reporting.readable_report_chunks(result)
    for chunk in chunks:
        if '<li>&lt;warning 0&gt;</li>' in chunk:
            assert warnings.reads == 1
            break
    else:
        raise AssertionError('The first warning row was not emitted.')
    chunks.close()
    assert warnings.reads == 1


def test_csv_header_and_first_chunk_do_not_prefetch_lazy_future_frames(original_reporting):
    count = 100_000
    balls = GeneratedSeries(count, lambda index: {'timestamp_seconds': index / 60,
                                                 'state': 'MISSING'})
    points = GeneratedSeries(count, lambda index: [0., float(index % 20)])
    speeds = GeneratedSeries(count, lambda index: 0.)
    result = {'summary': {'id': 'match_lazy'}, 'trajectories': {
        'ballTrajectory': balls, 'player1CourtPositions': points, 'player1Speeds': speeds}}
    chunks = reporting.telemetry_csv_chunks(result, chunk_chars=1024)
    assert next(chunks).startswith('analysis_id,frame,normalized_seconds,ball_state,')
    assert balls.reads == points.reads == speeds.reads == 0
    chunk = next(chunks)
    assert len(chunk) <= 1024
    assert 1 < balls.reads < 100
    assert balls.reads == points.reads == speeds.reads
    reads = balls.reads
    chunks.close()
    assert balls.reads == points.reads == speeds.reads == reads


@pytest.mark.parametrize('source', [None, {'id': 'invalid'}])
def test_report_without_credit_or_events_is_byte_identical(original_reporting, source):
    result = report_result(1)
    result['mediaSource'] = source
    result['events'] = []
    result['warnings'] = []
    expected = original_reporting.readable_report(result)
    assert ''.join(reporting.readable_report_chunks(lazy_result(result))) == expected


def test_report_embedded_evidence_keeps_original_position_and_bytes(original_reporting, monkeypatch, tmp_path):
    image_html = '<figure><img src="data:image/jpeg;base64,AB+/="/><figcaption>Frame 0</figcaption></figure>'
    image_calls = []

    def evidence(directory, result):
        image_calls.append((directory, result['summary']['id']))
        return image_html

    monkeypatch.setattr(original_reporting, 'evidence_images', evidence)
    monkeypatch.setattr(reporting, 'evidence_images', evidence)
    result = report_result(1)
    expected = original_reporting.readable_report(result, tmp_path)
    assert ''.join(reporting.readable_report_chunks(lazy_result(result), tmp_path)) == expected
    assert image_calls == [(tmp_path, result['summary']['id'])] * 2


@pytest.mark.parametrize('first_count,second_count', [(65, 129), (129, 65)])
def test_legacy_document_nested_people_arrays_preserve_summary_and_html(
        original_reporting, tmp_path, first_count, second_count):
    result = report_result(129)
    result['imagePeople']['frames'][0]['people'] = [{'bbox': [1., 2., 3., 4.]}] * first_count
    result['imagePeople']['frames'][1]['people'] = None
    result['imagePeople']['frames'][2]['people'] = []
    result['imagePeople']['frames'][3]['people'] = [{'bbox': [4., 3., 2., 1.]}] * second_count
    result['trajectories']['player1CourtPositions'][0] = [0.] * first_count
    result['trajectories']['player1CourtPositions'][3] = [0.] * second_count
    result['playerTracking']['player_1']['speedIntervals'] = [
        {'startSeconds': index / 60, 'endSeconds': (index + 1) / 60} for index in range(129)]
    result['events'] = [{'event_type': f'contact {index}', 'timestamp_s': index / 60,
                         'confidence': .25} for index in range(129)]
    result['warnings'] = [f'<warning {index}>' for index in range(129)]
    (tmp_path / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    expected_summary = original_reporting.image_people_summary(result)
    expected_html = original_reporting.readable_report(result)
    expected_csv = original_reporting.telemetry_csv(result)
    with ResultDocument(tmp_path, result['summary']['id']) as document:
        frames = document.value['imagePeople']['frames']
        assert isinstance(frames, DiskSeries)
        assert isinstance(frames[0]['people'], DiskSeries)
        assert len(frames[0]['people']) == first_count
        assert frames[1]['people'] is None and frames[2]['people'] == []
        assert isinstance(frames[3]['people'], DiskSeries)
        assert len(frames[3]['people']) == second_count
        assert isinstance(document.value['trajectories']['player1CourtPositions'][0], DiskSeries)
        assert isinstance(document.value['playerTracking']['player_1']['speedIntervals'], DiskSeries)
        assert isinstance(document.value['events'], DiskSeries)
        assert isinstance(document.value['warnings'], DiskSeries)
        assert reporting.image_people_summary(document.value) == expected_summary
        assert ''.join(reporting.readable_report_chunks(document.value)) == expected_html
        assert ''.join(reporting.telemetry_csv_chunks(document.value)) == expected_csv


@pytest.mark.parametrize('frames', [({'people': []},),
                                    GeneratedSeries(1, lambda _: {'people': []})])
def test_people_summary_excludes_non_json_sequence_outer_arrays(original_reporting, frames):
    result = {'summary': {'frames': 1}, 'imagePeople': {'frames': frames}}
    assert reporting.image_people_summary(result) == original_reporting.image_people_summary(result)


@pytest.mark.parametrize('people', [({'bbox': [1., 2., 3., 4.]},),
                                    GeneratedSeries(65, lambda _: {'bbox': [1., 2., 3., 4.]})])
def test_people_summary_excludes_non_json_sequence_inner_arrays(original_reporting, people):
    result = {'summary': {'frames': 1}, 'imagePeople': {'frames': [{'people': people}]}}
    assert reporting.image_people_summary(result) == original_reporting.image_people_summary(result)


"""Bounded record production and atomic failure handling for pipeline exports."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import json_array_artifact as module
from src.utils.json_array_artifact import JsonArrayArtifact, write_json_array_artifact


def test_completed_artifact_preserves_metadata_nulls_and_record_order(tmp_path):
    destination = tmp_path / 'audit.json'
    metadata = {'schema_version': '1.0', 'enabled': True, 'settings': {'value': .5}}
    records = [{'frame_index': 0, 'missing': None}, {'frame_index': 1, 'valid': False}]
    with JsonArrayArtifact(destination, metadata, allow_nan=False) as artifact:
        for record in records:
            artifact.append(record)
        assert not destination.exists()
    assert json.loads(destination.read_text()) == {**metadata, 'frames': records}
    assert not list(tmp_path.glob('.*.tmp'))


def test_disabled_artifact_preserves_empty_array(tmp_path):
    destination = tmp_path / 'disabled.json'
    write_json_array_artifact(destination, {'enabled': False}, 'frames', iter(()))
    assert json.loads(destination.read_text()) == {'enabled': False, 'frames': []}


@pytest.mark.parametrize('failure', ['producer', 'serialization', 'replace', 'write', 'sync'])
def test_failed_export_preserves_previous_completed_file_and_closes_producer(tmp_path, monkeypatch, failure):
    destination = tmp_path / 'audit.json'
    previous = b'{"enabled":false,"frames":[]}'
    destination.write_bytes(previous)
    closed = []

    def records():
        try:
            yield {'frame_index': 0}
            if failure == 'producer':
                raise RuntimeError('Producer failed')
            yield {'frame_index': 1, 'confidence': float('nan') if failure == 'serialization' else .9}
        finally:
            closed.append(True)

    def fail(*_args, **_kwargs):
        raise OSError('Simulated storage failure')

    if failure == 'replace':
        monkeypatch.setattr(module.os, 'replace', fail)
    elif failure == 'sync':
        monkeypatch.setattr(module.os, 'fsync', fail)
    elif failure == 'write':
        real_dump = module.json.dump

        def fail_record(value, stream, **kwargs):
            if isinstance(value, dict) and value.get('frame_index') == 1:
                fail()
            return real_dump(value, stream, **kwargs)
        monkeypatch.setattr(module.json, 'dump', fail_record)

    with pytest.raises((RuntimeError, ValueError, OSError)):
        write_json_array_artifact(destination, {'enabled': True}, 'frames', records(), allow_nan=False)
    assert closed == [True]
    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob('.*.tmp'))


def test_long_generator_writes_each_record_before_requesting_the_next(tmp_path, monkeypatch):
    real_dump = module.json.dump
    written = [0]

    def counted_dump(value, stream, **kwargs):
        if isinstance(value, dict) and 'frame_index' in value:
            assert value['frame_index'] == written[0]
            result = real_dump(value, stream, **kwargs)
            written[0] += 1
            return result
        return real_dump(value, stream, **kwargs)
    monkeypatch.setattr(module.json, 'dump', counted_dump)

    def records():
        for index in range(100_000):
            assert written[0] == index
            yield {'frame_index': index, 'state': 'MISSING', 'position_px': None}

    write_json_array_artifact(tmp_path / 'long.json', {}, 'frames', records())
    assert written[0] == 100_000


def test_pipeline_failure_closes_all_audit_writers(tmp_path, monkeypatch):
    from src.pipeline.phase6_pipeline import Phase6Pipeline
    pipeline = Phase6Pipeline.__new__(Phase6Pipeline)

    def fail(_input, output, _frames, _metadata, stack):
        for filename in ('camera', 'stationary', 'pixel', 'patch'):
            artifact = stack.enter_context(JsonArrayArtifact(Path(output) / f'{filename}.json', {}))
            artifact.append({'frame_index': 0})
        raise RuntimeError('Later stage failed')

    monkeypatch.setattr(pipeline, '_run_with_frame_exports', fail)
    with pytest.raises(RuntimeError, match='Later stage failed'):
        pipeline._run_with_frames('unused.mp4', tmp_path, [], None)
    assert list(tmp_path.iterdir()) == []


def test_lazy_detection_and_trajectory_records_preserve_established_values():
    from src.pipeline.phase6_pipeline import detection_export_records, trajectory_export_records
    from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
    from src.utils.bbox_utils import BBox
    points = [TemporalBallPoint(0, 0., 0., 12.345, confidence=.87654, state=BallState.DETECTED),
              TemporalBallPoint(1, 1 / 60, None, None, state=BallState.MISSING)]
    boxes = [BBox(0., 1., 20., 30., confidence=.8, track_id=5), None]
    calibrations = [SimpleNamespace(is_valid=True), SimpleNamespace(is_valid=False)]
    detections = list(detection_export_records(boxes, [None, None], points, calibrations))
    assert detections == [
        {'frame_index': 0, 'court_registration_valid': True,
         'player_1': {'bbox': [0., 1., 20., 30.], 'source_track_id': 5, 'confidence': .8},
         'player_2': None, 'ball': {'position_px': [0., 12.345], 'confidence': .87654, 'state': 'DETECTED'}},
        {'frame_index': 1, 'court_registration_valid': False, 'player_1': None,
         'player_2': None, 'ball': {'position_px': None, 'confidence': None, 'state': 'MISSING'}},
    ]
    trajectories = list(trajectory_export_records(points))
    assert trajectories == [
        {'frame_index': 0, 'timestamp_seconds': 0., 'x_px': 0., 'y_px': 12.35,
         'court_x_m': None, 'court_y_m': None, 'speed_kmh': None, 'confidence': .877, 'state': 'DETECTED'},
        {'frame_index': 1, 'timestamp_seconds': .0167, 'x_px': None, 'y_px': None,
         'court_x_m': None, 'court_y_m': None, 'speed_kmh': None, 'confidence': None, 'state': 'MISSING'},
    ]

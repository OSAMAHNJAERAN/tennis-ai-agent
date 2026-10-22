"""Dense observations must not repeatedly decode their selected frame evidence.

The real builder reads actual optional JSON files through real SQLite storage.
A delegating decoder observes read/lifetime costs at that storage boundary;
the preserved resident normalizer independently pins exact exported bytes.
"""
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import weakref

import pytest

from src.api.result_storage import ResultStorage
from src.api.results import build_result
from src.tracking.player_motion_tracking import JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06/src/api/results.py'
BASELINE_SHA256 = '30a64418e141d2ab5d847c33aa6fcc0386807ff7a04a5891b5172a904f9c1a55'
MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class ObservedFrame(dict):
    """A dict snapshot with weak-reference support; values remain unchanged."""


class ObservedStorage(ResultStorage):
    def __init__(self, directory):
        super().__init__(directory)
        self.frame_reads = self.live_frames = self.peak_live_frames = 0

    def _released(self):
        self.live_frames -= 1

    def _decode(self, value):
        decoded = super()._decode(value)
        if (isinstance(decoded, dict) and set(decoded) ==
                {'frame_index', 'court_registration_valid', 'player_1', 'player_2', 'ball'}):
            decoded = ObservedFrame(decoded)
            self.frame_reads += 1
            self.live_frames += 1
            self.peak_live_frames = max(self.peak_live_frames, self.live_frames)
            weakref.finalize(decoded, self._released)
        return decoded


@pytest.fixture(scope='module')
def frozen():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
    spec = importlib.util.spec_from_file_location('frozen_result_builder_reads_20261006', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(directory, name, value):
    (directory / name).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def artifacts(tmp_path, count, proposals_per_frame):
    directory = tmp_path / 'analysis'
    directory.mkdir()
    detections, balls, cameras, samples, poses, rackets, people = [], [], [], [], [], [], []
    for index in range(count):
        valid = index % 13 != 7
        detections.append({'frame_index': index, 'court_registration_valid': True,
                           'player_1': {'bbox': [1, 1, 10, 20], 'confidence': .8, 'source_track_id': 7},
                           'player_2': {'bbox': [40, 3, 50, 30], 'confidence': .7, 'source_track_id': 8}})
        balls.append({'frame_index': index, 'timestamp_seconds': index / 60,
                      'x_px': 30., 'y_px': 50., 'court_x_m': 4., 'court_y_m': 8.,
                      'confidence': .9, 'speed_kmh': 180, 'state': 'DETECTED', 'unknown': {'zero': -.0}})
        cameras.append({'frame_index': index, 'timestamp_seconds': index / 60,
                        'is_valid': valid, 'image_to_court': MATRIX if valid else None,
                        'anchor_to_frame_px': MATRIX if valid else None})
        samples.append({'frame_index': index, 'timestamp_s': index / 60,
                        'ground_position_m': [4. + index / 1000, 8.], 'speed_kmh': 0. if index == 0 else 10.,
                        'position_fit_rmse_m': -.0, 'velocity_window_seconds': .15})
        joints = {name: {'position_px': [4., 9.], 'confidence': .8} for name in JOINT_NAMES}
        joints['left_wrist'] = {'position_px': None, 'confidence': .1}
        poses.append({'frame_index': index, 'timestamp_seconds': index / 60, '1': joints,
                      '2': None if index % 11 == 3 else joints})
        racket = {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.],
                  'position_px': [6., 7.], 'confidence': .7, 'velocity_px_per_s': [-.0, 12.5]}
        rackets.append({'frame_index': index, 'timestamp_seconds': index / 60,
                        '1': racket, '2': None if index % 11 == 3 else racket})
        proposals = [{'bbox': [1, 1, 10, 20], 'confidence': .8, 'sourceTrackId': 7,
                      'courtRole': 1, 'participantStatus': 'COURT_ROLE_CANDIDATE'}]
        proposals += [{'bbox': [2 + offset / 100, 2, 11 + offset / 100, 21], 'confidence': .6,
                       'sourceTrackId': 20 + offset, 'courtRole': 1,
                       'participantStatus': 'COURT_ROLE_CANDIDATE'}
                      for offset in range(proposals_per_frame - 1)]
        people.append({'frame_index': index, 'timestamp_seconds': index / 60,
                       'people': None if index % 17 == 6 else [] if index % 17 == 5 else proposals})
    write_json(directory, 'detections.json', {'metadata': {'frames': count, 'fps': 60, 'width': 100, 'height': 100}, 'frames': detections})
    write_json(directory, 'trajectories.json', {'ball_trajectory': balls})
    write_json(directory, 'court_geometry.json', {'is_valid': True, 'homography_matrix': MATRIX,
                                                'reprojection_error_px': 1.})
    write_json(directory, 'camera_registration.json', {'enabled': True, 'frames': cameras})
    write_json(directory, 'player_motion.json', {'player_1': {'samples': samples}, 'player_2': {'samples': samples}})
    write_json(directory, 'pose_tracking.json', {'frames': poses})
    write_json(directory, 'racket_tracking.json', {'enabled': True, 'frames': rackets})
    write_json(directory, 'image_people.json', {'coordinateSystem': 'normalized_video_pixels',
               'identityScope': 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID', 'frames': people})
    write_json(directory, 'metrics.json', {'processing_fps': 5., 'authoritative_events_enabled': False})
    return directory, {'id': 'match_result_builder_reads', 'filename': 'Tennis é 🎾.mp4',
                       'started_at': '2026-10-06T00:00:00+00:00'}


@pytest.mark.parametrize('count', [65, 600])
def test_dense_people_and_body_observations_use_bounded_frame_snapshots(tmp_path, frozen, count):
    directory, job = artifacts(tmp_path, count, proposals_per_frame=24)
    expected = frozen.build_result(directory, job)
    expected_bytes = json.dumps(expected, allow_nan=False).encode('utf-8')
    scratch = tmp_path / 'storage'
    scratch.mkdir()
    with ObservedStorage(scratch) as storage:
        result = build_result(directory, job, _storage=storage)
        gc.collect()
        reads, peak, live = storage.frame_reads, storage.peak_live_frames, storage.live_frames
        stream = io.StringIO()
        storage.dump(result, stream)
        actual_bytes = stream.getvalue().encode('utf-8')
        assert actual_bytes == expected_bytes
        # The emitted tree owns disk references, never a growing frame cache.
        assert live == 0 and peak <= 4
        print(json.dumps({'frames': count, 'normalizedFrameDecodesDuringBuild': reads,
                          'peakLiveNormalizedFrameSnapshots': peak, 'liveAfterBuild': live,
                          'exactFrozenJsonBytes': True}), flush=True)
        # Seven other normalization passes plus one snapshot per optional
        # pose/racket/person frame. Person density must not multiply reads.
        assert reads <= 10 * count
    result = json.loads(actual_bytes)
    assert result['imagePeople']['frames'][5]['people'] == []
    assert result['imagePeople']['frames'][6]['people'] is None
    assert result['imagePeople']['frames'][7]['people'][0]['participantStatus'] == 'UNKNOWN'
    assert result['imagePeople']['frames'][0]['people'][0]['courtRole'] == 1
    assert all(person['participantStatus'] == 'UNKNOWN'
               for person in result['imagePeople']['frames'][0]['people'][1:])
    assert result['poseTracking']['frames'][0]['player_1'][9]['x_px'] is None


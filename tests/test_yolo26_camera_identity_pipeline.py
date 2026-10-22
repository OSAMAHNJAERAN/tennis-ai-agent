"""Deterministic orchestration evidence, without model or accuracy claims."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.court.camera_intervals import CameraIntervals, SUPPORTED_VIEW
from src.court.camera_registration import RegistrationResult
from src.court.court_geometry import TennisCourtGeometry
from src.detection.yolo26_adapters import PoseBBox
from src.pipeline import phase6_pipeline as module
from src.tracking.match_identity import MatchIdentity
from src.utils.frame_registration_sequence import FrameRegistrationSequence
from src.utils.json_array_artifact import JsonArrayArtifact
from src.utils.player_detection_sequence import PlayerDetectionSequence


class PlayerStub:
    model = object()

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def detect_and_track(self, frame, persist=True):
        # Recycling a numerical detector ID at a cut must not confirm A/B.
        return [PoseBBox(80, 150, 110, 210, track_id=1, ground_point_px=np.array([95., 205.])),
                PoseBBox(180, 20, 210, 65, track_id=2, ground_point_px=np.array([195., 60.]))]

    def observe(self, frame, boxes):
        return {key: None for key in boxes}


class CourtStub:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        shift = 2 if frame[0, 0, 0] > 200 else 0
        return TennisCourtGeometry.get_canonical_keypoints() * np.array([20., 8.]) + shift


class RegistrationStub:
    instances = []

    def __init__(self, calibration, points, **settings):
        self.calibration, self.points = calibration, np.array(points)
        self.instances.append(self)

    def update(self, frame, detections):
        return RegistrationResult(self.calibration, np.eye(3), {'is_valid': True})


def frames_for_cut(count=6):
    result = []
    for index in range(count):
        frame = np.full((240, 320, 3), 245 if index >= 3 else 10, np.uint8)
        frame[150:183, 80:110] = [30, 30, 220]
        frame[20:44, 180:210] = [220, 30, 30]
        result.append(frame)
    return result


def write_mapping(output, times):
    with (output / 'source_frame_mapping.jsonl').open('w') as stream:
        for index, (pts, source_index) in enumerate(times):
            stream.write(json.dumps({'frame_index': index, 'timestamp_seconds': index / 25,
                                     'source_pts': str(pts), 'source_time_base': '1/1000',
                                     'source_frame_index': source_index,
                                     'validMeasurementTransitionFromPrevious': index > 0}) + '\n')


def anchor(frame, player, track, time, segment, command):
    return {'frame_index': frame, 'anchor_id': command, 'player_id': player,
            'track_id': track, 'timestamp': time, 'segment_id': segment,
            'reason': 'Reviewed test association', 'source': 'manual_review'}


def candidate_pass(tmp_path, monkeypatch, *, times=None, settings=None, frames=None):
    monkeypatch.setattr(module, 'CourtCameraRegistration', RegistrationStub)
    RegistrationStub.instances = []
    pipeline = module.Phase6Pipeline.__new__(module.Phase6Pipeline)
    pipeline.config = {'match_configuration': {'match_id': 'integration'}, **(settings or {})}
    pipeline.player_detector, pipeline.court_detector = PlayerStub(), CourtStub()
    frames = frames if frames is not None else frames_for_cut()
    if times is not None:
        write_mapping(tmp_path, times)
    with ExitStack() as stack:
        detections = stack.enter_context(PlayerDetectionSequence(tmp_path))
        registrations = stack.enter_context(FrameRegistrationSequence(tmp_path))
        audit = stack.enter_context(JsonArrayArtifact(tmp_path / 'camera_registration.json', {}))
        candidate = pipeline._candidate_player_pass(frames, str(tmp_path), 25., detections,
                                                     registrations, audit, {}, stack)
        result = [candidate[index] for index in range(len(candidate))]
        valid = [registrations[index].calibration.is_valid for index in range(len(registrations))]
    return pipeline, result, valid


def test_cut_reanchors_current_court_resets_actual_tracker_and_requires_new_identity_anchor(tmp_path, monkeypatch):
    segment = CameraIntervals('integration').segment_id
    cut_segment = segment[:-1] + '1'
    settings = {
        'camera_intervals': {'reviews': [
            {'evidence_id': 'play', 'start_source_seconds': 100., 'end_source_seconds': 100.15,
             'label': 'match_play', 'viewpoint': SUPPORTED_VIEW, 'normal_speed_verified': True},
            {'evidence_id': 'replay', 'start_source_seconds': 100.16, 'end_source_seconds': 100.25,
             'label': 'replay', 'viewpoint': SUPPORTED_VIEW}]},
        'match_identity': {'anchors': [anchor(0, 'A', 1, 100., segment, 'a0'),
                                       anchor(0, 'B', 2, 100., segment, 'b0'),
                                       anchor(4, 'A', 2, 100.16, cut_segment, 'a4'),
                                       anchor(4, 'B', 1, 100.16, cut_segment, 'b4')]}}
    pipeline, records, valid = candidate_pass(tmp_path, monkeypatch, settings=settings,
        times=[(100000 + index * 40, index) for index in range(6)])
    assert pipeline.player_detector.resets == 2  # Initial session and the cut.
    assert pipeline.court_detector.calls == 6
    assert len(RegistrationStub.instances) == 2
    assert np.array_equal(RegistrationStub.instances[1].points, records[3][2])
    assert records[3][0]['local_reset']
    assert records[3][0]['needs_new_registration']
    assert records[3][0]['measurement_transition_allowed'] is False
    assert [item['player_id'] for item in records[0][1]['observations']] == ['A', 'B']
    assert all(item['player_id'] is None for item in records[3][1]['observations'])
    assert all(item['state'] == 'PROVISIONAL' for item in records[3][1]['observations'])
    assert [item['player_id'] for item in records[4][1]['observations']] == ['B', 'A']
    assert valid == [True, True, True, True, False, False]
    assert records[4][0]['activity_status'] == 'replay'
    assert 'reviewed_replay' in records[4][0]['unavailable_reasons']
    assert all(not item['motion_continuity'] for record in records for item in record[1]['observations'])
    audit = json.loads((tmp_path / 'match_identity.json').read_text())
    assert sum(len(record['corrections']) for record in audit['frames']) == 4
    restored = MatchIdentity.restore(audit['frames'][-1]['checkpoint'], 'integration')
    assert restored.revision == 4
    assert restored.bindings[2]['player_id'] == 'A'
    assert not list(tmp_path.glob('*.sqlite3'))


@pytest.mark.parametrize('timing', ['missing', 'duplicate', 'negative', 'reversed'])
def test_original_timing_preserved_and_bad_timing_never_creates_identity_measurements(tmp_path, monkeypatch, timing):
    times = {'missing': None, 'duplicate': [(1000, 0), (1000, 0)],
             'negative': [(-1000, 0), (-960, 1)], 'reversed': [(1000, 0), (960, 1)]}[timing]
    pipeline, records, valid = candidate_pass(tmp_path, monkeypatch, times=times,
                                             frames=frames_for_cut(2))
    assert valid == [False, False]  # No normal-play/view review was supplied.
    assert all(item['player_id'] is None for record in records for item in record[1]['observations'])
    if timing == 'missing':
        assert records[0][0]['source_pts'] is None
        assert records[0][1]['source_timestamp_seconds'] is None
    else:
        assert records[0][0]['source_pts'] == str(times[0][0])
    if timing in {'duplicate', 'reversed'}:
        assert records[1][0]['local_reset']
        assert pipeline.player_detector.resets == 2
    if timing == 'negative':
        assert records[0][1]['source_timestamp_seconds'] == -1.
        assert records[0][1]['identity_time_origin_seconds'] == -1.
        assert records[1][1]['checkpoint']['last_timestamp'] == pytest.approx(.04)


def test_mapping_alignment_failure_does_not_publish_partial_identity_audit(tmp_path, monkeypatch):
    write_mapping(tmp_path, [(1000, 0)])
    with pytest.raises(ValueError, match='ended before'):
        candidate_pass(tmp_path, monkeypatch, frames=frames_for_cut(2))
    assert not (tmp_path / 'match_identity.json').exists()
    assert not (tmp_path / 'camera_intervals.json').exists()
    # Existing raw detection/registration storage retains diagnostic databases
    # on failure; the new identity store still cleans up its owned frames.
    assert not list(tmp_path.glob('identity_frames_*.sqlite3'))


def test_identity_gap_resets_detector_before_accepting_this_frames_tracks(tmp_path, monkeypatch):
    original = PlayerStub.detect_and_track

    def tracked_after_reset(self, frame, persist=True):
        boxes = original(self, frame, persist)
        for box in boxes:
            box.track_id += self.resets * 100
        return boxes

    monkeypatch.setattr(PlayerStub, 'detect_and_track', tracked_after_reset)
    pipeline, records, _ = candidate_pass(tmp_path, monkeypatch,
        times=[(1000, 0), (1040, 1)], frames=frames_for_cut(2),
        settings={'match_identity': {'max_gap_seconds': .02}})
    assert pipeline.player_detector.resets == 2
    assert records[1][0]['local_reset'] is False
    assert records[1][1]['local_reset'] is True
    assert records[0][1]['observations'][0]['track_id'] == 101
    assert records[1][1]['observations'][0]['track_id'] == 201


def test_ball_association_windows_cannot_bridge_ineligible_source_transitions():
    from src.tracking.temporal_ball_tracker import TemporalBallPoint
    calls = []

    class Tracker:
        def track_video_candidates(self, records, fps, frame_size, *, output=None):
            calls.append(list(records))
            output = [] if output is None else output
            output.extend(TemporalBallPoint(index, index / fps, None, None) for index in range(len(records)))
            return output

    candidate_frames = [{0: {'measurement_transition_allowed': value, 'timestamp_seconds': index / 25}}
                        for index, value in enumerate((False, True, False, True, False))]
    points = module.track_candidate_intervals(Tracker(), [[index] for index in range(5)],
                                             candidate_frames, 25., (320, 240))
    assert calls == [[[0], [1]], [[2], [3]], [[4]]]
    assert [point.frame_index for point in points] == list(range(5))
    assert [point.timestamp_seconds for point in points] == [index / 25 for index in range(5)]


@pytest.mark.parametrize('family', ['yolo26', 'legacy'])
@pytest.mark.parametrize('strategy', ['kalman', 'model_top1'])
def test_pipeline_exports_candidate_unknown_identity_or_preserves_legacy_numpy_ground_points(tmp_path, monkeypatch, family, strategy):
    from src.detection import yolo26_bundle
    video = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (320, 240))
    assert writer.isOpened()
    for frame in frames_for_cut():
        writer.write(frame)
    writer.release()
    player, court = PlayerStub(), CourtStub()
    bundle = SimpleNamespace(player_detector=player, court_detector=court,
                             ball_detector=SimpleNamespace(extract_candidates=lambda frame, imgsz: []))
    monkeypatch.setattr(yolo26_bundle, 'build_yolo26_bundle', lambda config: bundle)
    monkeypatch.setattr(module, 'CourtCameraRegistration', RegistrationStub)
    config = yaml.safe_load(Path('configs/production/yolo26.yaml').read_text())
    config['temporal_tracking']['strategy'] = strategy
    if family == 'legacy':
        config.pop('model_system')
        config['ball_detection']['backend'] = 'yolo11'
        config['pose_estimation']['enabled'] = False
        config['player_motion']['enabled'] = False
        monkeypatch.setattr(module, 'CourtKeypointDetector', lambda **kwargs: court)
        monkeypatch.setattr(module, 'PlayerDetector', lambda **kwargs: player)
        monkeypatch.setattr(module, 'YOLO11BallDetector', lambda **kwargs: bundle.ball_detector)
    config_path = tmp_path / 'pipeline.yaml'
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / 'output'
    result = module.Phase6Pipeline(str(config_path)).run(str(video), str(output))
    assert result['status'] == 'success'
    detection = json.loads((output / 'detections.json').read_text())
    if family == 'legacy':
        assert not (output / 'match_identity.json').exists()
        motion = json.loads((output / 'player_motion.json').read_text())
        assert motion['player_1']['samples'][0]['ground_position_m'] == pytest.approx([4.75, 25.625])
        assert motion['player_2']['samples'][0]['ground_position_m'] == pytest.approx([9.75, 7.5])
        return
    assert detection['metadata']['metric_measurements_available'] is False
    assert len(detection['frames']) == 6
    assert all(not frame['court_registration_valid'] for frame in detection['frames'])
    assert all(item['player_id'] is None for frame in detection['frames'] for item in frame['match_identity'])
    metrics = json.loads((output / 'player_metrics.json').read_text())
    assert metrics['player_1']['total_distance_m'] is None
    assert metrics['player_2']['mean_speed_kmh'] is None
    assert json.loads((output / 'match_identity.json').read_text())['pipeline_resume_supported'] is False
    assert not list(output.glob('*.sqlite3'))


@pytest.mark.parametrize('opened', [False, True])
def test_candidate_render_failure_releases_handles_and_never_publishes_complete_audits(tmp_path, monkeypatch, opened):
    from src.detection import yolo26_bundle
    player = PlayerStub()
    bundle = SimpleNamespace(player_detector=player, court_detector=CourtStub(),
                             ball_detector=SimpleNamespace(extract_candidates=lambda frame, imgsz: []))
    monkeypatch.setattr(yolo26_bundle, 'build_yolo26_bundle', lambda config: bundle)
    monkeypatch.setattr(module, 'CourtCameraRegistration', RegistrationStub)
    config_path = tmp_path / 'pipeline.yaml'
    config_path.write_text(yaml.safe_dump(yaml.safe_load(Path('configs/production/yolo26.yaml').read_text())))
    pipeline = module.Phase6Pipeline(str(config_path))

    class Frames(list):
        def close(self):
            self.closed = True

    frames = Frames(frames_for_cut())

    class Capture:
        released = False
        index = 0

        def isOpened(self):
            return opened

        def read(self):
            self.index += 1
            return (True, frames[0].copy()) if self.index <= 2 else (False, None)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda path: capture)
    output = tmp_path / 'output'
    with pytest.raises(RuntimeError, match='frame 2|Unable to open candidate render'):
        pipeline._run_with_frames('injected.mp4', str(output), frames, SimpleNamespace(fps=25.))
    assert capture.released
    assert frames.closed
    assert not (output / 'match_identity.json').exists()
    assert not (output / 'camera_intervals.json').exists()
    assert not (output / 'metrics.json').exists()
    assert (output / 'annotated.mp4').exists()  # Preserve the failed partial output.

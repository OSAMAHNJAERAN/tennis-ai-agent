"""Exercise real orchestration/export with deterministic detector substitutes."""

import json

import cv2
import numpy as np
import pytest
import yaml
from pathlib import Path

from src.court.court_geometry import TennisCourtGeometry
from src.pipeline import phase6_pipeline as module
from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import BallObservation


@pytest.mark.parametrize("size", [(320, 240), (1920, 1080)])
@pytest.mark.parametrize("valid_court", [True, False])
@pytest.mark.parametrize("backend", ["yolo11", "wasb", "wasb_ensemble", "wasb_spaced", "wasb_returning"])
def test_phase6_passes_native_dimensions_and_exports_aligned_video(tmp_path, monkeypatch, size, valid_court, backend):
    ensemble = backend == 'wasb_ensemble'
    spaced = backend == 'wasb_spaced'
    returning = backend == 'wasb_returning'
    fps = 60 if spaced else 25
    backend = 'wasb' if ensemble or spaced or returning else backend
    width, height = size
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    texture = np.random.default_rng(7).integers(20, 220, (height, width, 3), dtype=np.uint8) if returning else None
    for index in range(6):
        writer.write(texture if returning and index != 2 else np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    class CourtDetector:
        def __init__(self, **kwargs):
            pass

        def predict(self, frame):
            if not valid_court:
                return np.zeros((14, 2))
            return TennisCourtGeometry.get_canonical_keypoints() * np.array([width / 12, height / 26])

    class PlayerDetector:
        def __init__(self, **kwargs):
            pass

        def detect_and_track(self, frame, persist=True):
            return [
                BBox(width * .4, height * .65, width * .5, height * .85, track_id=1, confidence=.9),
                BBox(width * .45, height * .05, width * .5, height * .2, track_id=2, confidence=.9),
            ]

    class BallDetector:
        def __init__(self, **kwargs):
            pass

        def extract_candidates(self, frame, imgsz):
            return []

        def predict_triplet(self, frames):
            assert 1 <= len(frames) <= 3
            return [[] for _ in frames]

        def predict_stream(self, frames):
            for _ in frames:
                yield [BallObservation(width * .3, height * .5, .9)] if ensemble or spaced else []

    monkeypatch.setattr(module, "CourtKeypointDetector", CourtDetector)
    monkeypatch.setattr(module, "PlayerDetector", PlayerDetector)
    monkeypatch.setattr(module, "YOLO11BallDetector", BallDetector)
    monkeypatch.setattr(module, "WASBBallDetector", BallDetector)
    monkeypatch.setattr(module, "WASBEnsembleDetector", BallDetector)
    monkeypatch.setattr(module, "PoseFeatureExtractor", lambda **kwargs: None)

    class MotionTracker:
        def __init__(self, **kwargs):
            pass

        def observe(self, frame, boxes):
            return {identity: None for identity in boxes}

    class RacketTracker(MotionTracker):
        def observe(self, frame, boxes, timestamp):
            return {identity: {"bbox_xyxy": None, "position_px": None, "state": "MISSING"} for identity in boxes}

    monkeypatch.setattr(module, "PlayerMotionTracking", MotionTracker)
    monkeypatch.setattr(module, "RacketTracking", RacketTracker)
    PlayerDetector.model = object()
    config = yaml.safe_load(Path("configs/phase6_analytics/pipeline.yaml").read_text())
    config["ball_detection"]["backend"] = backend
    config["ball_detection"]["source_directory"] = "unused-fake-model"
    if ensemble:
        config['ball_detection'].update({'ensemble_checkpoint': 'unused-second-model', 'ensemble_weight': .75,
                                         'temporal_step': 1, 'stationary_candidates': {'enabled': True}})
        config['court_detection']['camera_registration'] = {'enabled': True}
        config['ball_detection']['pixel_motion'] = {'enabled': True, 'minimum_score': 12.}
    if spaced:
        config['ball_detection'].update({'temporal_step': 1, 'spatial_crops': {'enabled': True},
                                        'temporal_spacing': {'enabled': True}, 'temporal_detours': {'enabled': True},
                                        'patch_persistence': {'enabled': True}})
    config["temporal_tracking"]["strategy"] = "model_top1" if backend == "wasb" else "kalman"
    config["event_detection"]["authoritative_enabled"] = False
    if returning:
        config['court_detection']['camera_registration'] = {
            'enabled': True, 'returning_view': {'enabled': True}}
    config["player_motion"] = {"enabled": backend == "wasb"}
    config["racket_tracking"] = {"enabled": backend == "wasb"}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    pipeline = module.Phase6Pipeline(str(config_path))
    original_track = pipeline.temporal_tracker.track_video_candidates
    calls = []

    def track(candidates, fps, frame_size):
        calls.append((fps, frame_size))
        return original_track(candidates, fps=fps, frame_size=frame_size)

    monkeypatch.setattr(pipeline.temporal_tracker, "track_video_candidates", track)
    output = tmp_path / "output"
    result = pipeline.run(str(path), str(output))
    assert calls == ([(float(fps), size)] if backend == "yolo11" else [])
    assert result["status"] == "success"
    analytics = json.loads((output / 'match_analytics.json').read_text())
    assert analytics['overview']['total_points_played'] is None
    assert analytics['overview']['average_rally_length'] is None
    assert analytics['overview']['longest_rally_length'] is None
    assert analytics['score_summary'] is None
    state = json.loads((output / 'match_state.json').read_text())
    assert state['authoritative_events_enabled'] is False
    assert state['semantic_role'] == 'CONFIGURED_INITIAL_STATE_NOT_OBSERVED_MATCH_SCORE'
    detections = json.loads((output / 'detections.json').read_text())
    assert detections['metadata']['width'] == width
    assert detections['metadata']['height'] == height
    if valid_court:
        assert detections['frames'][0]['player_1']['source_track_id'] == 1
        assert detections['frames'][0]['player_2']['source_track_id'] == 2
        assert detections['frames'][0]['player_1']['confidence'] == .9
    camera = json.loads((output / 'camera_registration.json').read_text())
    assert camera['enabled'] is (ensemble or returning)
    if returning:
        expected = [True, True, False, False, False, True] if valid_court else [False] * 6
        assert [frame['court_registration_valid'] for frame in detections['frames']] == expected
        assert [frame['is_valid'] for frame in camera['frames']] == expected
        assert camera['failure_behavior'] == 'WITHHOLD_UNTIL_ORIGINAL_ANCHOR_VIEW_CONFIRMED'
        for index in (2, 3, 4):
            assert detections['frames'][index]['player_1'] is None
            assert camera['frames'][index]['image_to_court'] is None
        if valid_court:
            assert camera['frames'][5]['recovered_this_frame']
            assert detections['frames'][5]['player_1'] is not None
    if ensemble:
        assert len(camera['frames']) == 6
        # Featureless synthetic frames cannot register after the initial frame.
        assert detections['frames'][1]['court_registration_valid'] is False
        assert detections['frames'][1]['player_1'] is None
    trajectory = json.loads((output / "trajectories.json").read_text())["ball_trajectory"]
    assert len(trajectory) == 6
    if ensemble:
        assert [point['state'] for point in trajectory] == ['DETECTED'] * 3 + ['MISSING'] * 3
    elif spaced:
        assert [point['state'] for point in trajectory] == ['DETECTED'] * 6
    else:
        assert all(point["state"] == "MISSING" for point in trajectory)
    assert trajectory[-1]["timestamp_seconds"] == pytest.approx(round(5/fps,4))
    court = json.loads((output / "court_geometry.json").read_text())
    assert court["is_valid"] is valid_court
    assert "reprojection_error_m" in court
    assert pipeline.line_call_engine.refiner.fps == fps
    ball_metrics = json.loads((output / "ball_metrics.json").read_text())
    assert ball_metrics["summary"]["available"] is False
    assert ball_metrics["summary"]["maximum_speed_kmh"] is None
    poses = json.loads((output / "player_motion.json").read_text())
    assert poses['schema_version'] == '2.0'
    assert len(poses['player_1']['samples']) == 6
    assert len(poses['player_2']['samples']) == 6
    rackets = json.loads((output / "racket_tracking.json").read_text())
    assert len(poses["pose_frames"]) == (6 if backend == "wasb" else 0)
    assert len(rackets["frames"]) == (6 if backend == "wasb" else 0)
    stationary = json.loads((output / 'stationary_candidate_audit.json').read_text())
    assert stationary['enabled'] is ensemble
    assert len(stationary['frames']) == (6 if ensemble else 0)
    pixel_audit = json.loads((output / 'pixel_motion_audit.json').read_text())
    assert pixel_audit['enabled'] is ensemble
    assert len(pixel_audit['frames']) == (6 if ensemble else 0)
    if ensemble:
        assert pixel_audit['frames'][0]['candidates'][0]['history_available'] is False
        assert pixel_audit['frames'][3]['candidates'][0]['accepted'] is False
    if not valid_court:
        assert court["homography_matrix"] is None
        players = json.loads((output / "player_metrics.json").read_text())
        for name in ("player_1", "player_2"):
            assert players[name]["total_distance_m"] is None
            assert players[name]["mean_speed_kmh"] is None
        assert json.loads((output / "line_calls.json").read_text())["line_calls"] == []
    spacing_audit = json.loads((output / 'temporal_spacing_audit.json').read_text())
    assert spacing_audit['enabled'] is spaced
    assert spacing_audit['native_stride'] == (2 if spaced else 1)
    assert spacing_audit['frames'] == 6
    assert spacing_audit['native_fps'] == fps
    detours = json.loads((output / 'temporal_detour_audit.json').read_text())
    assert detours['enabled'] is spaced
    assert detours['rejected_frames'] == []
    cap = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(fps)
    cap.release()

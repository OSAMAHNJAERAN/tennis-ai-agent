"""Selected model boundary, coordinate transforms and shared pose evidence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.detection.yolo26_adapters import YOLO26BallDetector, YOLO26CourtPoseDetector, YOLO26PlayerPoseDetector, tile_regions
from src.detection.yolo26_bundle import validate_yolo26_config


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.values


def result(boxes, keypoints=None, ids=None):
    class Boxes(SimpleNamespace):
        def __len__(self):
            return len(self.conf.values)
    box_data = Boxes(xyxy=Tensor(boxes), conf=Tensor([.9] * len(boxes)), id=Tensor(ids) if ids is not None else None)
    return SimpleNamespace(boxes=box_data, keypoints=SimpleNamespace(data=Tensor(keypoints)) if keypoints is not None else None)


class Model:
    def __init__(self, response):
        self.response, self.calls = response, []
    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.response]
    def track(self, frame, **kwargs):
        return self.predict(frame, **kwargs)


def config():
    return yaml.safe_load(Path("configs/production/yolo26.yaml").read_text())


def test_config_candidate_is_valid_and_missing_specialists_fail_closed():
    data = config()
    assert validate_yolo26_config(data)["status"].startswith("UNQUALIFIED")
    with pytest.raises(FileNotFoundError, match="No legacy fallback"):
        validate_yolo26_config(data, check_assets=True)


def test_production_rejects_even_hashed_pilot_weights(tmp_path):
    import json
    from src.detection.yolo26_bundle import sha256_file
    data = config()
    manifest = validate_yolo26_config(data)
    for role, record in manifest["models"].items():
        artifact = tmp_path / Path(record["path"]).name
        artifact.write_bytes(b"isolated pilot checkpoint")
        record.update(path=str(artifact), sha256=sha256_file(artifact), training_provenance="pilot-only receipt")
    data["ball_detection"]["model"] = manifest["models"]["ball"]["path"]
    data["player_detection"]["model"] = manifest["models"]["players"]["path"]
    data["court_detection"]["model_path"] = manifest["models"]["court"]["path"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    data["model_system"]["artifact_manifest"] = str(path)
    with pytest.raises(ValueError, match="independent tennis qualification"):
        validate_yolo26_config(data, check_assets=True)


@pytest.mark.parametrize("change", [
    lambda c: c["model_system"].update(vision_model_count=4),
    lambda c: c["ball_detection"].update(backend="wasb"),
    lambda c: c["pose_estimation"].update(model_path="yolo11n-pose.pt"),
    lambda c: c.update(racket_tracking={"enabled": True}),
    lambda c: c["ball_detection"].update(ensemble_checkpoint="another.pt"),
    lambda c: c["model_system"].update(nms=True),
])
def test_forbidden_scope_changes_fail(change):
    data = config()
    change(data)
    with pytest.raises(ValueError):
        validate_yolo26_config(data)


def test_pose_boxes_keep_track_id_and_ankles_without_second_inference():
    points = np.zeros((1, 17, 3))
    points[0, 15] = [20, 80, .9]
    points[0, 16] = [30, 82, .8]
    model = Model(result([[10, 10, 40, 90]], points, [7]))
    detector = YOLO26PlayerPoseDetector("injected.pt", tracker="pinned.yaml", _model=model)
    boxes = detector.detect_and_track(np.zeros((100, 100, 3), np.uint8))
    assert boxes[0].track_id == 7
    assert boxes[0].ground_point_px == (25., 81.)
    assert boxes[0].ground_point_confidence == .8
    assert boxes[0].ground_point_quality == "ANKLES_VISIBLE_GROUND_CONTACT_UNVERIFIED"
    joints = detector.observe(None, {1: boxes[0], 2: None})
    assert joints[1]["left_ankle"]["position_px"] == [20., 80.]
    assert joints[2] is None
    assert len(model.calls) == 1
    assert model.calls[0]["end2end"] is True and model.calls[0]["nms"] is False
    assert model.calls[0]["tracker"] == "pinned.yaml"


def test_missing_ankle_retains_declared_box_estimate():
    points = np.zeros((1, 17, 3))
    model = Model(result([[10, 10, 40, 90]], points))
    detector = YOLO26PlayerPoseDetector("injected.pt", tracker="pinned.yaml", _model=model)
    box = detector.detect(np.zeros((100, 100, 3), np.uint8))[0]
    assert box.ground_point_px is None
    assert box.track_id is None
    assert box.ground_point_source == "BBOX_BOTTOM_CENTER_ESTIMATE"


def test_tiles_cover_native_edges_once():
    tiles = tile_regions(1920, 1080, 960, .2)
    assert (960, 120, 1920, 1080) in tiles
    assert len(tiles) == len(set(tiles))
    assert all(0 <= left < right <= 1920 and 0 <= top < bottom <= 1080 for left, top, right, bottom in tiles)


def test_tile_translation_and_dedup_preserve_original_pixel_center():
    class TileModel:
        def __init__(self):
            self.calls = 0
        def predict(self, frame, **kwargs):
            self.calls += 1
            # Full image observes [110,20]; left tile misses; right tile
            # observes the same object at [10,20] in local coordinates.
            coords = [[108, 18, 112, 22]] if self.calls == 1 else [] if self.calls == 2 else [[8, 18, 12, 22]]
            return [result(coords)]
    detector = YOLO26BallDetector("injected.pt", _model=TileModel(), tiles={"enabled": True, "size_px": 100, "overlap": 0.})
    candidates = detector.extract_candidates(np.zeros((100, 200, 3), np.uint8))
    assert len(candidates) == 1
    assert (candidates[0].x_px, candidates[0].y_px) == (110, 20)
    assert candidates[0].bbox.x1 == 108


def test_court_visibility_masks_absent_landmarks():
    points = np.zeros((1, 14, 3))
    points[0, :, :2] = np.arange(28).reshape(14, 2)
    points[0, :, 2] = .9
    points[0, 3, 2] = .1
    detector = YOLO26CourtPoseDetector("injected.pt", _model=Model(result([[1, 1, 50, 50]], points)))
    observations = detector.predict(np.zeros((100, 100, 3), np.uint8))
    assert observations.shape == (14, 2)
    assert np.isnan(observations[3]).all()
    assert detector.last_observation["visible_keypoints"] == 13


def test_pipeline_constructs_only_the_bundle_and_reuses_player_pose(monkeypatch, tmp_path):
    from src.pipeline import phase6_pipeline
    from src.detection import yolo26_bundle
    model = object()
    player = SimpleNamespace(model=model)
    bundle = SimpleNamespace(court_detector=object(), player_detector=player, ball_detector=object())
    calls = []
    monkeypatch.setattr(yolo26_bundle, "build_yolo26_bundle", lambda config: calls.append(config) or bundle)
    def forbidden(**kwargs):
        pytest.fail("The YOLO26 branch constructed an additional legacy vision model")
    for name in ("CourtKeypointDetector", "PlayerDetector", "YOLO11BallDetector", "WASBBallDetector", "PlayerMotionTracking"):
        monkeypatch.setattr(phase6_pipeline, name, forbidden)
    data = config()
    data["event_detection"]["authoritative_enabled"] = True
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data))
    pipeline = phase6_pipeline.Phase6Pipeline(str(path))
    assert len(calls) == 1
    assert pipeline.motion_tracker is player
    assert pipeline.pose_extractor.model is model

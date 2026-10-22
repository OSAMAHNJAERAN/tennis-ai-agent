"""Prepared helper contract tests; not executed during helper preparation."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import struct

import pytest


HELPER = Path(__file__).with_name("helper.py")
spec = importlib.util.spec_from_file_location("prepared_pose_racket_measurement", HELPER)
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


def test_dense_complete_native_observations_survive_fixture_round_trip():
    for index in range(6):
        original = measurement.generated_frame(index)
        restored = json.loads(json.dumps(original, allow_nan=False))
        first_pose, first_racket = measurement.validate_frame(original, index)
        second_pose, second_racket = measurement.validate_frame(restored, index)
        assert measurement.record_bytes(index, index / measurement.FPS, first_pose, first_racket) == \
            measurement.record_bytes(index, index / measurement.FPS, second_pose, second_racket)
        assert all(len(joints) == 17 for joints in second_pose.values())
        assert list(second_pose) == ([2, 1] if index % 2 else [1, 2])
        assert all(type(identity) is int for identity in second_pose)
        assert struct.pack("!d", second_pose[1]["nose"]["position_px"][0]) == struct.pack("!d", -0.0)
        assert second_pose[1]["left_eye"]["position_px"][0] == math.nextafter(221.0, math.inf)


def test_independent_fidelity_detects_type_order_and_ieee_bit_changes():
    encode = measurement.native_node
    assert encode(-0.0) != encode(0.0)
    assert encode(1.0) != encode(math.nextafter(1.0, math.inf))
    assert encode({1: True}) != encode({"1": True})
    assert encode(True) != encode(1)
    assert encode([1, 2]) != encode((1, 2))
    assert encode({1: None, 2: None}) != encode({2: None, 1: None})


def test_missing_rackets_retain_all_unknown_fields_without_imputation():
    detected = missing = 0
    for index in range(10):
        _, rackets = measurement.validate_frame(measurement.generated_frame(index), index)
        for value in rackets.values():
            assert tuple(value) == measurement.RACKET_KEYS
            if value["state"] == "MISSING":
                missing += 1
                assert all(value[key] is None for key in measurement.RACKET_KEYS[1:])
            else:
                detected += 1
                assert value["contact_event"] is None
                assert len(value["bbox_xyxy"]) == 4
    assert missing > 0 and detected > 0


def test_indexed_reads_consume_all_identities_joints_and_rackets():
    poses, rackets = [], []
    for index in range(6):
        pose, racket = measurement.validate_frame(measurement.generated_frame(index), index)
        poses.append(pose)
        rackets.append(racket)
    result = measurement.render_like_reads(poses, rackets, 6)
    assert result["indexedFrames"] == 6
    assert result["densePoseIdentities"] == 12
    assert result["poseJoints"] == 204
    assert result["detectedRackets"] + result["missingRackets"] == 12
    assert result["identityColorSum"] == 6 * (17 + 31)


@pytest.mark.parametrize("rows,declared", [(1, 2), (3, 2)])
def test_stream_input_rejects_incomplete_or_extra_frames(tmp_path, rows, declared):
    path = tmp_path / "input.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(rows):
            stream.write(json.dumps(measurement.generated_frame(index)) + "\n")
    with pytest.raises(ValueError):
        for _ in measurement.iter_fixture(path, declared):
            pass


def test_selected_sanitizer_has_no_import_or_whole_pipeline_execution():
    frozen = measurement.sanitizer_node(measurement.FROZEN)
    current = measurement.sanitizer_node(measurement.PIPELINE)
    assert frozen.name == "sanitize_json_value"
    assert ast.dump(frozen, include_attributes=False) == ast.dump(current, include_attributes=False)
    assert not frozen.decorator_list
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(frozen))


def test_measurement_metadata_keeps_scientific_fields_unknown():
    metadata = measurement.metadata()
    motion = metadata["playerMotion"]
    assert motion["pose_accuracy_validated"] is False
    assert motion["foot_contact_validated"] is False
    for key in ("player_1", "player_2"):
        assert motion[key]["samples"] == []
        assert motion[key]["observed_distance_m"] is None
        assert motion[key]["mean_speed_kmh"] is None
    assert metadata["racketTracking"]["accuracy_validated"] is False

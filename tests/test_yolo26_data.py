import pytest

from scripts.data.prepare_yolo26_data import encode_instances, grouped_splits, prepare


def test_grouped_split_is_deterministic_and_match_session_is_indivisible():
    sources = [{"source_id": f"video{i}", "match_id": f"match{i}", "session_id": f"session{i}", "original_source_video_id": f"original{i}"} for i in range(5)]
    sources += [{"source_id": "second_camera0", "match_id": "match0", "session_id": "other_session0", "original_source_video_id": "camera0"}]
    assignments, counts = grouped_splits(sources)
    reversed_assignments, reversed_counts = grouped_splits(list(reversed(sources)))
    assert assignments == reversed_assignments
    assert counts == reversed_counts == {"groups": 5, "train": 3, "val": 1, "test": 1}
    assert assignments["second_camera0"] == assignments["video0"]


def test_insufficient_group_diversity_remains_a_pilot():
    with pytest.raises(ValueError, match="at least five independent"):
        grouped_splits([{"source_id": "video", "match_id": "match", "session_id": "session"}])


def test_native_box_conversion_preserves_center_and_size():
    label = encode_instances([{"bbox_xyxy_px": [100, 200, 110, 210], "visibility": "visible"}], "ball", 1920, 1080)
    values = list(map(float, label.split()))
    assert values[1] * 1920 == pytest.approx(105)
    assert values[2] * 1080 == pytest.approx(205)
    assert values[3] * 1920 == pytest.approx(10)
    assert values[4] * 1080 == pytest.approx(10)


def test_hidden_ball_is_not_turned_into_observed_ground_truth():
    with pytest.raises(ValueError, match="visible evidence-supported"):
        encode_instances([{"bbox_xyxy_px": [100, 200, 110, 210], "visibility": "occluded"}], "ball", 1920, 1080)


def test_unknown_court_landmark_preserves_visibility_zero():
    points = [None] * 14
    points[0] = [100, 200, 2]
    label = encode_instances([{"bbox_xyxy_px": [0, 0, 1920, 1080], "keypoints_px": points}], "court", 1920, 1080)
    values = list(map(float, label.split()))
    assert len(values) == 47
    assert values[5:8] == pytest.approx([100 / 1920, 200 / 1080, 2])
    assert values[8:] == [0.] * 39


def test_agent_visual_review_is_recorded_and_limited_to_pilot(tmp_path):
    import cv2
    import json
    import numpy as np
    image = tmp_path / "native.png"
    cv2.imwrite(str(image), np.zeros((20, 30, 3), dtype=np.uint8))
    sources = [{"source_id": f"v{i}", "match_id": f"m{i}", "session_id": f"s{i}",
                "original_source_video_id": f"original{i}", "owner": "author", "rights_evidence": "reviewed publisher grant",
                "sha256": "a" * 64, "fps": 25, "resolution": [30, 20], "surface": "hard", "training_allowed": True,
                "view": "elevated_end_court_singles", "camera_reviewed": True} for i in range(5)]
    annotation = {"role": "ball", "source_id": "v0", "frame_index": 0, "timestamp_seconds": 0,
                  "image": str(image), "human_reviewed": False, "manual_reviewed": True,
                  "reviewer_type": "AGENT_VISUAL_REVIEW", "annotation_provenance": "AGENT_VISUAL_MANUAL",
                  "instances": [{"bbox_xyxy_px": [5, 5, 7, 7], "visibility": "visible"}]}
    document = {"sources": sources, "annotations": [annotation]}
    manifest = json.loads(prepare(document, tmp_path / "pilot", pilot=True).read_text())
    assert manifest["qualification_evidence"] is False
    assert manifest["samples"][0]["independent_human_reviewed"] is False
    with pytest.raises(ValueError, match="Agent-only"):
        prepare(document, tmp_path / "frozen", pilot=False)
    document["sources"][0]["resolution"] = [15, 10]
    with pytest.raises(ValueError, match="original source resolution"):
        prepare(document, tmp_path / "resized", pilot=True)


def test_pilot_cannot_split_neighboring_sources_from_one_session(tmp_path):
    document = {"sources": [
        {"source_id": "rally1", "match_id": "match", "session_id": "session", "original_source_video_id": "v1",
         "owner": "author", "rights_evidence": "reviewed", "sha256": "a" * 64, "fps": 25,
         "resolution": [30, 20], "surface": "hard", "training_allowed": True,
         "view": "elevated_end_court_singles", "camera_reviewed": True, "development_split": "train"},
        {"source_id": "rally2", "match_id": "match", "session_id": "session", "original_source_video_id": "v2",
         "owner": "author", "rights_evidence": "reviewed", "sha256": "b" * 64, "fps": 25,
         "resolution": [30, 20], "surface": "hard", "training_allowed": True,
         "view": "elevated_end_court_singles", "camera_reviewed": True, "development_split": "val"},
    ], "annotations": [{"role": "ball"}]}
    with pytest.raises(ValueError, match="Development split leakage"):
        prepare(document, tmp_path / "pilot", pilot=True)

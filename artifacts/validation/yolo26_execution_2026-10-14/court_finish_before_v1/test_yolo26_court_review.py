"""Behavioral guards for court development context and visibility review."""

from copy import deepcopy

import pytest

from scripts.data.prepare_yolo26_court_review import context_components, point_defects, validate_review


def fixture_review():
    samples = []
    rows = []
    points = [[10 + index, 20 + index] for index in range(14)]
    for number, video, event, split in [(0, "compile", "Miami", "train"), (1, "compile", "Rome", "train"),
                                        (2, "RomeClip", "Rome", "train"), (3, "LondonClip", "London", "val")]:
        ident = str(number)
        samples.append({"id": ident, "source_video_id": video, "width": 1280, "height": 720,
                        "publisher_keypoints_xy_px": deepcopy(points), "out_of_image_keypoint_indices": []})
        rows.append({"id": ident, "visual_event_family": event, "development_split": split,
                     "decision": "ACCEPT", "reviewer_type": "AGENT_VISUAL_REVIEW", "review_note": "Visible intersections checked",
                     "scope": "elevated_end_court_singles", "semantic_order_reviewed": True,
                     "reviewed_keypoints_px": [point + [2] for point in points]})
    by_video, _ = context_components(samples, rows)
    for row, sample in zip(rows, samples):
        row["context_group"] = by_video[sample["source_video_id"]]
    return {"samples": samples}, {"samples": rows}


def test_compilation_event_union_is_transitive_and_order_independent():
    inventory, decisions = fixture_review()
    assignments, groups = context_components(inventory["samples"], decisions["samples"])
    reverse_assignments, reverse_groups = context_components(inventory["samples"], list(reversed(decisions["samples"])))
    assert assignments == reverse_assignments
    assert groups == reverse_groups
    assert assignments["compile"] == assignments["RomeClip"]
    assert assignments["compile"] != assignments["LondonClip"]


def test_full_event_group_cannot_cross_training_and_development():
    inventory, decisions = fixture_review()
    decisions["samples"][2]["development_split"] = "val"
    with pytest.raises(ValueError, match="context crosses"):
        validate_review(inventory, decisions)


def test_invented_independence_context_is_rejected():
    inventory, decisions = fixture_review()
    decisions["samples"][2]["context_group"] = "Rome_independent"
    with pytest.raises(ValueError, match="transitive"):
        validate_review(inventory, decisions)


def test_hidden_point_can_be_null_but_never_fake_visible_or_occluded():
    inventory, decisions = fixture_review()
    decisions["samples"][0]["reviewed_keypoints_px"][12] = None
    accepted, _ = validate_review(inventory, decisions)
    assert accepted[0]["reviewed_keypoints_px"][12] is None
    decisions["samples"][0]["reviewed_keypoints_px"][12] = [22, 32, 1]
    with pytest.raises(ValueError, match="never inferred"):
        validate_review(inventory, decisions)


def test_publisher_outside_points_are_not_clamped_into_visible_truth():
    assert point_defects([[1280, 10]] + [[100, 100]] * 13, 1280, 720) == [0]
    inventory, decisions = fixture_review()
    inventory["samples"][0]["out_of_image_keypoint_indices"] = [3]
    with pytest.raises(ValueError, match="out-of-image"):
        validate_review(inventory, decisions)


def test_missing_or_duplicated_image_review_cannot_freeze():
    inventory, decisions = fixture_review()
    decisions["samples"][1] = deepcopy(decisions["samples"][0])
    with pytest.raises(ValueError, match="exactly one"):
        validate_review(inventory, decisions)


def test_coordinate_correction_cannot_replace_original_without_provenance():
    inventory, decisions = fixture_review()
    decisions["samples"][0]["reviewed_keypoints_px"][3] = [300, 400, 2]
    with pytest.raises(ValueError, match="separate provenance"):
        validate_review(inventory, decisions)


def test_rejecting_bad_frame_keeps_other_video_context_links():
    inventory, decisions = fixture_review()
    decisions["samples"][1].update(decision="REJECT", rejection_reason="Shifted court labels")
    accepted, groups = validate_review(inventory, decisions)
    assert len(accepted) == 3
    assert len(groups) == 2


def test_too_few_visible_points_does_not_become_full_court_truth():
    inventory, decisions = fixture_review()
    decisions["samples"][0]["reviewed_keypoints_px"][:5] = [None] * 5
    with pytest.raises(ValueError, match="at least ten"):
        validate_review(inventory, decisions)

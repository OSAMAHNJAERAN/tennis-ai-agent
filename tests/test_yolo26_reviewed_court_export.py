"""Lightweight guards for the native court development export contract."""

import json

import pytest

from scripts.data.export_yolo26_reviewed_court import encode_court_label, selected_rows, sha, validate_approval


def court_fixture():
    sample = {"width": 1280, "height": 720}
    points = [[100 + 60 * index, 100 + 20 * index, 2] for index in range(14)]
    points[12] = None
    row = {"bbox_xyxy_px": [90, 90, 1000, 600], "reviewed_keypoints_px": points}
    return sample, row


def test_unknown_landmark_roundtrip_preserves_fixed_slot_and_native_coordinates():
    sample, row = court_fixture()
    values = list(map(float, encode_court_label(sample, row).split()))
    assert len(values) == 47
    assert values[5 + 12 * 3:5 + 13 * 3] == [0, 0, 0]
    for index, point in enumerate(row["reviewed_keypoints_px"]):
        if point is None:
            continue
        normalized = values[5 + index * 3:8 + index * 3]
        assert normalized[0] * sample["width"] == pytest.approx(point[0], abs=1e-6)
        assert normalized[1] * sample["height"] == pytest.approx(point[1], abs=1e-6)
        assert normalized[2] == 2


@pytest.mark.parametrize("point", [[100, 100, 1], [1100, 300, 2], [float("nan"), 100, 2]])
def test_hidden_guess_nonfinite_or_box_exceeding_point_cannot_export(point):
    sample, row = court_fixture()
    row["reviewed_keypoints_px"][0] = point
    with pytest.raises(ValueError, match="Visible landmark"):
        encode_court_label(sample, row)


def test_missing_semantic_slot_cannot_export():
    sample, row = court_fixture()
    row["reviewed_keypoints_px"].pop()
    with pytest.raises(ValueError, match="fourteen"):
        encode_court_label(sample, row)


def test_selection_caps_are_deterministic_and_surplus_stays_unused_development():
    rows = [{"id": f"train_{index}", "development_split": "train", "context_group": "train_context"}
            for index in range(107)]
    rows += [{"id": f"val_{index}", "development_split": "val", "context_group": "val_context"}
             for index in range(23)]
    selected, surplus = selected_rows(rows)
    reverse_selected, reverse_surplus = selected_rows(list(reversed(rows)))
    assert selected == reverse_selected
    assert surplus == reverse_surplus
    assert sum(row["development_split"] == "train" for row in selected) == 100
    assert sum(row["development_split"] == "val" for row in selected) == 20
    assert len(surplus) == 10
    assert {row["id"] for row in selected}.isdisjoint(surplus)
    assert {row["id"] for row in selected} | set(surplus) == {row["id"] for row in rows}
    assert all(row["development_split"] != "test" for row in selected)


def approval_fixture(tmp_path):
    frozen_path, decisions_path = tmp_path / "frozen.json", tmp_path / "decisions.json"
    frozen = {"selected": [{"id": "train_image"}, {"id": "development_image"}]}
    frozen_path.write_text(json.dumps(frozen))
    decisions_path.write_text("native image review")
    approval = {"status": "ROOT_AGENT_SECOND_PASS_APPROVED_FOR_DEVELOPMENT_EXPORT",
                "frozen_manifest_sha256": sha(frozen_path), "decisions_sha256": sha(decisions_path),
                "approved_selected_ids": ["train_image", "development_image"],
                "independent_human_reviewed": False, "review_note": "Root checked concrete native decisions and lineage"}
    return frozen_path, decisions_path, frozen, approval


def test_export_approval_must_bind_exact_native_decisions(tmp_path):
    frozen_path, decisions_path, frozen, approval = approval_fixture(tmp_path)
    validate_approval(approval, frozen, frozen_path, decisions_path)
    decisions_path.write_text("altered native review")
    with pytest.raises(ValueError, match="exact frozen manifest and decisions"):
        validate_approval(approval, frozen, frozen_path, decisions_path)


def test_partial_or_reordered_selected_image_approval_cannot_export(tmp_path):
    frozen_path, decisions_path, frozen, approval = approval_fixture(tmp_path)
    approval["approved_selected_ids"].reverse()
    with pytest.raises(ValueError, match="ordered selected"):
        validate_approval(approval, frozen, frozen_path, decisions_path)


def test_root_agent_review_cannot_become_human_qualification(tmp_path):
    frozen_path, decisions_path, frozen, approval = approval_fixture(tmp_path)
    approval["independent_human_reviewed"] = True
    with pytest.raises(ValueError, match="independent human"):
        validate_approval(approval, frozen, frozen_path, decisions_path)


def test_review_status_alone_does_not_authorize_export(tmp_path):
    frozen_path, decisions_path, frozen, approval = approval_fixture(tmp_path)
    approval["status"] = "REVIEWED"
    with pytest.raises(ValueError, match="concrete root"):
        validate_approval(approval, frozen, frozen_path, decisions_path)


def test_root_verification_receipt_approves_entire_exact_frozen_document(tmp_path):
    frozen_path, decisions_path, frozen, approval = approval_fixture(tmp_path)
    approval.update(status="PASS_ROOT_SOURCE_AND_REVIEW_BINDINGS_PRIVATE_TRAINING_EXPORT_APPROVED",
                    selected=2, independent_human_review=False,
                    root_visual_scope_review="Root checked native overview and source bindings")
    validate_approval(approval, frozen, frozen_path, decisions_path)
    approval["selected"] = 1
    with pytest.raises(ValueError, match="selected count"):
        validate_approval(approval, frozen, frozen_path, decisions_path)

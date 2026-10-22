"""Pure export checks; these tests never inspect or export publisher images.

The small synthetic fixtures exercise lossless box encoding and the explicit
root review gate. Passing them is not evidence of ball-detection accuracy.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import sys
import zlib

import pytest

import scripts.data.export_yolo26_reviewed_ball_wave as exporter
from scripts.data.export_yolo26_reviewed_ball_wave import (
    copy_and_verify,
    inverse_yolo_line,
    validate_crop_reviews,
    validate_destination,
    verify_root_approval,
    yolo_line,
)


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 640, 640],
        [0, 0, 4, 8],
        [636, 632, 640, 640],
        [17.25, 84.125, 26.875, 91.75],
        [0.0001, 639.0001, 0.1251, 639.1251],
        [315.9999, 199.9999, 316.0001, 200.0001],
    ],
)
def test_yolo_box_round_trip_preserves_native_subpixel_extent(box):
    original = deepcopy(box)

    line = yolo_line(box)
    values = line.split()

    assert values[0] == "0"
    assert len(values) == 5
    assert all(len(value.split(".")[-1]) == 10 for value in values[1:])
    assert inverse_yolo_line(line) == pytest.approx(box, abs=1e-6)
    assert box == original


def test_yolo_box_uses_crop_coordinates_and_explicit_alternate_size():
    line = yolo_line([16, 24, 32, 40], size=64)

    assert [float(value) for value in line.split()] == [0, 0.375, 0.5, 0.25, 0.25]
    assert inverse_yolo_line(line, size=64) == pytest.approx([16, 24, 32, 40], abs=1e-6)


@pytest.mark.parametrize(
    "box",
    [
        [],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [True, 2, 3, 4],
        [1, 2, False, 4],
        ["1", 2, 3, 4],
        [None, 2, 3, 4],
        [1, 2, math.nan, 4],
        [1, 2, math.inf, 4],
        [1, -math.inf, 3, 4],
        [-0.001, 2, 3, 4],
        [1, -0.001, 3, 4],
        [1, 2, 640.001, 4],
        [1, 2, 3, 640.001],
        [1, 2, 1, 4],
        [1, 2, 3, 2],
        [3, 2, 1, 4],
        [1, 4, 3, 2],
    ],
)
def test_invalid_manual_box_is_rejected_without_clipping(box):
    with pytest.raises((TypeError, ValueError)):
        yolo_line(box)


@pytest.mark.parametrize("size", [0, -1, True, False, math.nan, math.inf, "640", None])
def test_encode_and_inverse_both_reject_invalid_crop_size(size):
    with pytest.raises((TypeError, ValueError)):
        yolo_line([1, 2, 3, 4], size=size)
    with pytest.raises((TypeError, ValueError)):
        inverse_yolo_line("0 0.5 0.5 0.1 0.1", size=size)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "0 0.5 0.5 0.1",
        "0 0.5 0.5 0.1 0.1 0.2",
        "1 0.5 0.5 0.1 0.1",
        "-1 0.5 0.5 0.1 0.1",
        "0 nan 0.5 0.1 0.1",
        "0 0.5 inf 0.1 0.1",
        "0 0.5 0.5 -inf 0.1",
        "0 0.5 0.5 0 0.1",
        "0 0.5 0.5 0.1 -0.1",
        "0 -0.1 0.5 0.1 0.1",
        "0 0.5 1.1 0.1 0.1",
        "0 0.01 0.5 0.1 0.1",
        "0 0.99 0.5 0.1 0.1",
        "0 0.5 0.01 0.1 0.1",
        "0 0.5 0.99 0.1 0.1",
        "0 0.5 0.5 1.001 0.1",
    ],
)
def test_inverse_rejects_invalid_or_out_of_bounds_label(line):
    with pytest.raises((TypeError, ValueError)):
        inverse_yolo_line(line)


@pytest.fixture
def approved_export():
    bindings = {"native_crop_manifest": "a" * 64, "head_review": "b" * 64, "tail_review": "c" * 64}
    positive_ids = ["positive_0001", "positive_0004", "positive_0018"]
    negative_ids = ["negative_0000", "negative_0002"]
    approval = {
        "status": "ROOT_APPROVED_PRIVATE_SUPERVISED_BALL_DEVELOPMENT_EXPORT",
        "reviewer_type": "ROOT_AGENT_VISUAL_REVIEW",
        "qualification_evidence": False,
        "independent_human_review": False,
        "test_count": 0,
        "approved_positive_proposal_ids": positive_ids.copy(),
        "approved_negative_proposal_ids": negative_ids.copy(),
        "input_bindings": bindings.copy(),
        "review_note": "Synthetic fixture: exact reviewed bytes, private development only.",
    }
    return approval, bindings, positive_ids, negative_ids


def test_exact_root_approval_accepts_only_bound_ordered_selection(approved_export):
    original = deepcopy(approved_export)

    assert verify_root_approval(*approved_export) is None

    assert approved_export == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "ROOT_REVIEW_PENDING"),
        ("reviewer_type", "AGENT"),
        ("qualification_evidence", True),
        ("qualification_evidence", 0),
        ("independent_human_review", True),
        ("independent_human_review", 0),
        ("test_count", 1),
        ("test_count", False),
        ("test_count", 0.0),
        ("review_note", ""),
        ("review_note", "   "),
        ("review_note", None),
    ],
)
def test_approval_rejects_pending_qualification_or_untruthful_review_flags(approved_export, field, value):
    approval, bindings, positive_ids, negative_ids = deepcopy(approved_export)
    approval[field] = value

    with pytest.raises(ValueError):
        verify_root_approval(approval, bindings, positive_ids, negative_ids)


@pytest.mark.parametrize(
    "field",
    [
        "status", "reviewer_type", "qualification_evidence", "independent_human_review",
        "test_count", "approved_positive_proposal_ids", "approved_negative_proposal_ids",
        "input_bindings", "review_note",
    ],
)
def test_approval_requires_explicit_review_and_scope_fields(approved_export, field):
    approval, bindings, positive_ids, negative_ids = deepcopy(approved_export)
    del approval[field]

    with pytest.raises(ValueError):
        verify_root_approval(approval, bindings, positive_ids, negative_ids)


@pytest.mark.parametrize("field", ["input_bindings", "approved_positive_proposal_ids", "approved_negative_proposal_ids"])
@pytest.mark.parametrize("change", ["removed", "added", "altered", "reordered"])
def test_approval_does_not_survive_input_or_ordered_selection_changes(approved_export, field, change):
    approval, bindings, positive_ids, negative_ids = deepcopy(approved_export)
    if field == "input_bindings":
        if change == "removed":
            del approval[field]["head_review"]
        elif change == "added":
            approval[field]["unreviewed_input"] = "d" * 64
        elif change == "altered":
            approval[field]["tail_review"] = "e" * 64
        else:
            # Merely reordering a mapping is harmless; change the bound payload.
            approval[field] = dict(reversed(list(approval[field].items())))
            approval[field]["native_crop_manifest"] = "f" * 64
    else:
        ids = approval[field]
        if change == "removed":
            ids.pop()
        elif change == "added":
            ids.append("unreviewed_9999")
        elif change == "altered":
            ids[0] = "unreviewed_9998"
        else:
            ids.reverse()

    with pytest.raises(ValueError):
        verify_root_approval(approval, bindings, positive_ids, negative_ids)


@pytest.fixture
def reviewed_crops():
    crops, reviews = [], []
    rejected_indices = {2, 8, 12, 21, 30, 39, 49, 57, 68, 83}
    for index in range(85):
        ident = f"positive_{index:04d}"
        crop = {
            "proposal_id": ident,
            "path": f"synthetic/native640/{ident}.png",
            "sha256": hashlib.sha256(f"crop {index}".encode()).hexdigest(),
            "pixel_sha256": hashlib.sha256(f"pixels {index}".encode()).hexdigest(),
            "shape": [640, 640, 3],
            "source_crop_xyxy_native_px": [200, 100, 840, 740],
            "bbox_xyxy_crop_px": [16, 32, 24, 40],
            "bbox_xyxy_native_px": [216, 132, 224, 140],
            "board_path": f"synthetic/boards/board_{index // 4:02d}.png",
            "board_sha256": hashlib.sha256(f"board {index // 4}".encode()).hexdigest(),
            "derived_split": "train" if index < 60 else "development",
        }
        reviews.append({
            "accepted_crop_manifest_index": index,
            "proposal_id": ident,
            "crop_path": crop["path"],
            "crop_sha256": crop["sha256"],
            "board_path": crop["board_path"],
            "board_sha256": crop["board_sha256"],
            "source_crop_xyxy_native_px": crop["source_crop_xyxy_native_px"].copy(),
            "preserved_main_bbox_xyxy_crop_px": crop["bbox_xyxy_crop_px"].copy(),
            "preserved_main_bbox_xyxy_native_px": crop["bbox_xyxy_native_px"].copy(),
            "decision": "REJECT_CROP_UNANNOTATED_OR_AMBIGUOUS" if index in rejected_indices else "ACCEPT_CROP_SINGLE_REVIEWED_BALL",
            "full_current_crop_actually_viewed": True,
            "all_crop_pixels_scanned": True,
            "review_scope_native_xyxy": [0, 0, 640, 640],
            "reviewer": "AGENT /root" if index < 44 else "AGENT /root/tail",
            "reviewer_type": "AGENT",
            "root_agent_actually_viewed_this_crop": index < 44,
            "independent_human_review": False,
            "qualification_evidence": False,
            "main_bbox_changed": False,
            "additional_ball_boxes_created": 0,
            "supervised_export_approved": False,
        })
        crops.append(crop)
    return crops, {"records": reviews[:44]}, {"records": reviews[44:]}


def test_complete_reviews_preserve_main_boxes_and_rejected_extra_ball_crops(reviewed_crops):
    original = deepcopy(reviewed_crops)

    accepted = validate_crop_reviews(*reviewed_crops)

    assert len(accepted) == 75
    assert all(set(row) == {"crop", "review"} for row in accepted)
    assert all(row["review"]["decision"] == "ACCEPT_CROP_SINGLE_REVIEWED_BALL" for row in accepted)
    assert [row["crop"]["proposal_id"] for row in accepted] == [
        row["proposal_id"] for part in reviewed_crops[1:] for row in part["records"]
        if row["decision"] == "ACCEPT_CROP_SINGLE_REVIEWED_BALL"
    ]
    assert reviewed_crops == original


@pytest.mark.parametrize("part_index,record_index", [(1, 0), (1, 43), (2, 0), (2, 40)])
@pytest.mark.parametrize(
    "field,value",
    [
        ("crop_path", "synthetic/altered.png"),
        ("crop_sha256", "0" * 64),
        ("board_path", "synthetic/altered_board.png"),
        ("board_sha256", "1" * 64),
        ("full_current_crop_actually_viewed", False),
        ("all_crop_pixels_scanned", False),
        ("review_scope_native_xyxy", [0, 0, 639, 640]),
        ("decision", "REVIEW_PENDING"),
        ("decision", "ACCEPT_INFERRED_BALL"),
        ("independent_human_review", True),
        ("qualification_evidence", True),
        ("main_bbox_changed", True),
        ("additional_ball_boxes_created", 1),
        ("additional_ball_boxes_created", False),
        ("supervised_export_approved", True),
        ("preserved_main_bbox_xyxy_crop_px", [16, 32, 25, 40]),
        ("preserved_main_bbox_xyxy_native_px", [216, 132, 225, 140]),
        ("source_crop_xyxy_native_px", [201, 100, 841, 740]),
        ("accepted_crop_manifest_index", 999),
    ],
)
def test_review_must_bind_complete_unchanged_crop_and_actual_scope(reviewed_crops, part_index, record_index, field, value):
    inputs = deepcopy(reviewed_crops)
    inputs[part_index]["records"][record_index][field] = value

    with pytest.raises(ValueError):
        validate_crop_reviews(*inputs)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "cross_partition_duplicate"])
def test_every_prepared_crop_requires_one_review_without_extra_or_duplicate_ids(reviewed_crops, change):
    crops, head, tail = deepcopy(reviewed_crops)
    if change == "missing":
        tail["records"].pop()
    elif change == "extra":
        extra = deepcopy(head["records"][0])
        extra["proposal_id"] = "unreviewed_9999"
        tail["records"].append(extra)
    elif change == "duplicate":
        head["records"][1] = deepcopy(head["records"][0])
    else:
        tail["records"][0] = deepcopy(head["records"][0])

    with pytest.raises(ValueError):
        validate_crop_reviews(crops, head, tail)


@pytest.mark.parametrize(
    "field",
    [
        "accepted_crop_manifest_index", "crop_path", "crop_sha256", "board_path", "board_sha256",
        "full_current_crop_actually_viewed", "all_crop_pixels_scanned", "review_scope_native_xyxy",
        "qualification_evidence", "independent_human_review", "main_bbox_changed",
        "additional_ball_boxes_created", "supervised_export_approved",
        "preserved_main_bbox_xyxy_crop_px", "preserved_main_bbox_xyxy_native_px",
        "source_crop_xyxy_native_px",
    ],
)
def test_crop_review_requires_explicit_complete_scope_and_unchanged_box_fields(reviewed_crops, field):
    crops, head, tail = deepcopy(reviewed_crops)
    del tail["records"][0][field]

    with pytest.raises(ValueError):
        validate_crop_reviews(crops, head, tail)


@pytest.mark.parametrize("index", [0, 2, 44, 84])
def test_final_test_crop_cannot_enter_development_export_even_when_rejected(reviewed_crops, index):
    crops, head, tail = deepcopy(reviewed_crops)
    crops[index]["derived_split"] = "test"

    with pytest.raises(ValueError):
        validate_crop_reviews(crops, head, tail)


def test_byte_copy_preserves_exact_native_file_without_image_library(tmp_path):
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    payload = b"\x89PNG\r\n\x1a\nsynthetic exact retained bytes\x00\xff"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert copy_and_verify(source, destination, expected) == expected
    assert source.read_bytes() == destination.read_bytes() == payload


def test_copy_refuses_changed_source_before_creating_destination(tmp_path):
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    source.write_bytes(b"changed synthetic bytes")

    with pytest.raises(ValueError):
        copy_and_verify(source, destination, hashlib.sha256(b"reviewed original").hexdigest())

    assert not destination.exists()


def test_copy_does_not_overwrite_preexisting_destination(tmp_path):
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    source.write_bytes(b"new synthetic source")
    destination.write_bytes(b"retained output")

    with pytest.raises(FileExistsError):
        copy_and_verify(source, destination, hashlib.sha256(source.read_bytes()).hexdigest())

    assert destination.read_bytes() == b"retained output"


def test_export_module_import_does_not_load_image_video_or_model_runtimes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import scripts.data.export_yolo26_reviewed_ball_wave; "
            "print(json.dumps(sorted(set(sys.modules).intersection({'PIL', 'cv2', 'numpy', 'torch', 'ultralytics'}))))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("relative", ["dataset_v1", "development/dataset_v1"])
def test_destination_validation_accepts_only_fresh_descendant_without_creating_it(tmp_path, relative):
    allowed_root = tmp_path / "allowed_data"
    output = allowed_root / relative

    assert validate_destination(output, allowed_root) == output.resolve()
    assert not allowed_root.exists()


@pytest.mark.parametrize("relative", [".", "..", "../allowed_data_extra/escape", "dataset/../../outside"])
def test_destination_cannot_be_root_or_escape_by_traversal_or_prefix(tmp_path, relative):
    allowed_root = tmp_path / "allowed_data"
    output = allowed_root / relative

    with pytest.raises(ValueError):
        validate_destination(output, allowed_root)

    assert not allowed_root.exists()


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_existing_dataset_destination_is_never_reused_or_overwritten(tmp_path, existing_kind):
    allowed_root = tmp_path / "allowed_data"
    allowed_root.mkdir()
    output = allowed_root / "retained"
    if existing_kind == "directory":
        output.mkdir()
    else:
        output.write_bytes(b"retained prior output")

    with pytest.raises(FileExistsError):
        validate_destination(output, allowed_root)

    assert output.exists()
    if existing_kind == "file":
        assert output.read_bytes() == b"retained prior output"


@pytest.mark.parametrize(
    "status,approval",
    [
        ("PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL", None),
        ("PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL", {"path": "unused"}),
        ("ROOT_APPROVED_READY_TO_EXPORT", None),
    ],
)
def test_unapproved_execution_fails_before_any_destination_or_resource_operation(tmp_path, monkeypatch, status, approval):
    output = tmp_path / "never_created"
    def forbidden(*args, **kwargs):
        raise AssertionError("Unapproved data must not reach admission or output operations")
    monkeypatch.setattr(exporter, "validate_destination", forbidden)
    monkeypatch.setattr(exporter, "guarded_resources", forbidden)

    with pytest.raises(ValueError):
        exporter.execute_export({"status": status, "root_final_approval": approval, "output": str(output)})

    assert not output.exists()


def test_default_cli_preflight_never_calls_export_or_creates_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "never_created"
    calls = []
    def pure_preflight(path, approval, approval_sha):
        calls.append((path, approval, approval_sha))
        return {
            "status": "PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL", "output": str(path),
            "counts": {"test": 0}, "input_bindings": {}, "approval_template": {},
        }
    def forbidden(*args, **kwargs):
        raise AssertionError("Default CLI cannot execute export")
    monkeypatch.setattr(exporter, "preflight", pure_preflight)
    monkeypatch.setattr(exporter, "execute_export", forbidden)

    assert exporter.main(["--output", str(output)]) == 0

    assert calls == [(output, None, None)]
    assert json.loads(capsys.readouterr().out)["status"] == "PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL"
    assert not output.exists()


def test_execute_cli_requires_saved_root_approval_and_does_not_write(tmp_path, monkeypatch):
    output = tmp_path / "never_created"
    monkeypatch.setattr(exporter, "preflight", lambda *args: {"status": "PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing approval must never execute export")
    monkeypatch.setattr(exporter, "execute_export", forbidden)

    with pytest.raises(ValueError):
        exporter.main(["--output", str(output), "--execute"])

    assert not output.exists()


@pytest.mark.parametrize("alteration", ["source_changed", "approval_changed", "sample_changed", "count_changed"])
def test_approved_execution_revalidates_current_bytes_and_selection_before_creating_output(tmp_path, monkeypatch, alteration):
    output = tmp_path / "never_created"
    source = tmp_path / "retained_review.json"
    approval = tmp_path / "retained_approval.json"
    source.write_bytes(b'{"synthetic_review": true}')
    approval.write_bytes(b'{"synthetic_approval": true}')
    proposal = {
        "status": "ROOT_APPROVED_READY_TO_EXPORT",
        "output": str(output),
        "input_bindings": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
        "root_final_approval": {"path": str(approval), "sha256": hashlib.sha256(approval.read_bytes()).hexdigest()},
        "proposed_samples": [{"sample_id": "synthetic_0000", "label_text": "0 0.5 0.5 0.1 0.1\n"}],
        "counts": {"positive": 1, "test": 0},
    }
    independently_revalidated = deepcopy(proposal)
    if alteration == "source_changed":
        source.write_bytes(b'{"synthetic_review": "changed"}')
    elif alteration == "approval_changed":
        approval.write_bytes(b'{"synthetic_approval": "changed"}')
    elif alteration == "sample_changed":
        proposal["proposed_samples"][0]["label_text"] = "0 0.9 0.9 0.1 0.1\n"
    else:
        proposal["counts"]["positive"] = 2
    destination_validator = exporter.validate_destination
    monkeypatch.setattr(exporter, "validate_destination", lambda path: destination_validator(path, tmp_path))
    monkeypatch.setattr(exporter, "guarded_resources", lambda: {"synthetic_admission": True})
    monkeypatch.setattr(exporter, "preflight", lambda *args: deepcopy(independently_revalidated))

    with pytest.raises(ValueError):
        exporter.execute_export(proposal)

    assert not output.exists()


def _fixture_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _fixture_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def existing_synthetic_export(tmp_path, monkeypatch):
    """Seventy-nine byte fixtures, never real publisher images or an export run."""
    allowed_root = tmp_path / "allowed_data"
    output = allowed_root / "synthetic_dataset_v1"
    source_directory = tmp_path / "synthetic_original_crops"
    source_directory.mkdir()
    for split in ("train", "val"):
        (output / "ball/images" / split).mkdir(parents=True)
        (output / "ball/labels" / split).mkdir(parents=True)
    ihdr = struct.pack(">IIBBBBB", 640, 640, 8, 2, 0, 0, 0)
    # A structurally valid33-byte RGB8 header is sufficient for a metadata-only
    # verifier. The suffix deliberately is not image data and is never decoded.
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
    header += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    assert len(header) == 33
    rows = []
    for index in range(79):
        positive = index < 75
        identifier = f"synthetic_positive_{index:04d}" if positive else f"synthetic_negative_{index - 75:04d}"
        split = "train" if index < 55 or 75 <= index < 78 else "val"
        source_path = source_directory / f"{identifier}.png"
        image_path = output / "ball/images" / split / f"{identifier}.png"
        label_path = output / "ball/labels" / split / f"{identifier}.txt"
        payload = header + b"synthetic undecoded payload" + index.to_bytes(2, "big")
        source_path.write_bytes(payload)
        image_path.write_bytes(payload)
        line = "0 0.0312500000 0.0562500000 0.0125000000 0.0125000000\n" if positive else ""
        label_path.write_bytes(line.encode())
        rows.append({
            "sample_id": identifier, "source_id": f"synthetic_source_{split}",
            "source_type": "NATIVE_PTS_VIDEO_FRAME_EXACT_640_CROP",
            "development_split": split, "derived_split": "train" if split == "train" else "development",
            "image_source_path": str(source_path), "image_source_sha256": _fixture_sha(source_path),
            "image_path": str(image_path), "label_path": str(label_path), "label_text": line,
            "label_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "box_xyxy_crop_px": [16, 32, 24, 40] if positive else None,
            "box_xyxy_native_px": [216, 132, 224, 140] if positive else None,
            "crop_xyxy_native_px": [200, 100, 840, 740], "crop_size_px": [640, 640], "resized": False,
            "negative_visible_absence_only": not positive,
            "source_frame_index": index + 9, "source_pts": (index + 9) * 512,
            "source_time_base": "1/12800", "derived_component_id": f"synthetic_{split}_component",
            "match_id": None, "session_id": None, "actual_match_identity_verified": False,
            "actual_session_identity_verified": False, "source_independence_verified": False,
            "training_allowed": True, "allowed_use": "PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT",
            "raw_broadcast_redistribution_cleared": False,
            "review_evidence": {"synthetic_fixture_only": True},
            "independent_human_review": False, "qualification_evidence": False,
        })
    review_path = tmp_path / "synthetic_review.json"
    approval_path = tmp_path / "synthetic_approval.json"
    _fixture_json(review_path, {"synthetic_fixture_only": True, "reviewed_ids": [row["sample_id"] for row in rows]})
    bindings = {str(review_path): _fixture_sha(review_path)}
    approval_document = {
        "status": "ROOT_APPROVED_PRIVATE_SUPERVISED_BALL_DEVELOPMENT_EXPORT",
        "reviewer_type": "ROOT_AGENT_VISUAL_REVIEW", "qualification_evidence": False,
        "independent_human_review": False, "test_count": 0,
        "input_bindings": bindings,
        "approved_positive_proposal_ids": [row["sample_id"] for row in rows[:75]],
        "approved_negative_proposal_ids": [row["sample_id"] for row in rows[75:]],
        "review_note": "Synthetic byte fixtures, no actual video or annotation.",
    }
    _fixture_json(approval_path, approval_document)
    approval_binding = {"path": str(approval_path), "sha256": _fixture_sha(approval_path)}
    counts = {"images": 79, "positive_boxes": 75, "negative_tiles": 4, "train": 58, "val": 21,
              "test": 0, "positive_split_counts": {"train": 55, "development": 20}}
    plan = {
        "schema_version": "tennis.private-reviewed-ball-development-export.v1",
        "status": "ROOT_APPROVED_READY_TO_EXPORT", "created_at_utc": "2026-10-15T00:00:00Z",
        "output": str(output), "input_bindings": bindings, "proposed_samples": rows,
        "counts": counts, "labels_exported": 0,
        "components": [{"component_id": f"synthetic_{split}_component", "split": split} for split in ("train", "val")],
        "derived_source_provenance": [{"synthetic_fixture_only": True}],
        "context_lineage_includes_rejected_and_low_view": True,
        "match_ids_fabricated": False, "session_ids_fabricated": False,
        "independent_human_review": False, "source_independence_verified": False,
        "qualification_evidence": False, "final_test_used": False,
        "root_final_approval": approval_binding, "approval_template": approval_document,
    }
    frozen_plan = deepcopy(plan)
    manifest = deepcopy(plan)
    manifest.update({
        "status": "EXPORTED_PRIVATE_AGENT_REVIEWED_BALL_DEVELOPMENT_NOT_QUALIFICATION",
        "labels_exported": 79, "sources": [deepcopy(row) | {"image_copied_byte_identically": True} for row in rows],
        "model_used": False, "gpu_used": False, "training_performed": False,
    })
    manifest_path = output / "source_manifest.json"
    yaml_path = output / "ball/data.yaml"
    receipt_path = output / "export_receipt.json"
    # This text is independently specified rather than invoking export code.
    yaml_path.write_text(
        "path: " + json.dumps(str(output / "ball")) + "\ntrain: images/train\nval: images/val\nnames:\n  0: tennis_ball\n",
        encoding="utf-8",
    )
    _fixture_json(manifest_path, manifest)
    receipt = {
        "status": "PASS_BYTE_IDENTICAL_NATIVE640_AND_EXACT_YOLO_INVERSE_PRIVATE_DEVELOPMENT_EXPORT",
        "counts": counts, "source_manifest_sha256": _fixture_sha(manifest_path),
        "data_yaml_sha256": _fixture_sha(yaml_path), "root_final_approval": approval_binding,
        "input_files_changed": [], "qualification_evidence": False,
        "training_performed": False, "gpu_used": False, "model_used": False,
    }
    _fixture_json(receipt_path, receipt)
    original_destination_validator = exporter.validate_destination
    monkeypatch.setattr(
        exporter, "validate_destination",
        lambda path, *, require_fresh=True: original_destination_validator(path, allowed_root, require_fresh=require_fresh),
    )
    preflight_calls = []
    def constrained_preflight(path, approval, approval_sha, *, allow_existing=False):
        preflight_calls.append((Path(path), approval, approval_sha, allow_existing))
        if Path(path).resolve() != output.resolve() or allow_existing is not True:
            raise ValueError("Synthetic verifier must reconstruct only this existing dataset")
        if approval != frozen_plan["root_final_approval"]["path"] or approval_sha != frozen_plan["root_final_approval"]["sha256"]:
            raise ValueError("Synthetic root approval binding changed")
        if _fixture_sha(Path(approval)) != approval_sha:
            raise ValueError("Synthetic saved approval bytes changed")
        for input_path, expected_hash in frozen_plan["input_bindings"].items():
            if _fixture_sha(Path(input_path)) != expected_hash:
                raise ValueError("Synthetic reviewed input bytes changed")
        return deepcopy(frozen_plan)
    monkeypatch.setattr(exporter, "preflight", constrained_preflight)
    return {
        "output": output, "fixture_root": tmp_path, "plan": plan, "manifest": manifest,
        "receipt": receipt, "manifest_path": manifest_path, "receipt_path": receipt_path,
        "yaml_path": yaml_path, "review_path": review_path, "approval_path": approval_path,
        "manifest_sha256": _fixture_sha(manifest_path), "preflight_calls": preflight_calls,
    }


def _fixture_tree_hashes(root):
    return {str(path.relative_to(root)): _fixture_sha(path) for path in root.rglob("*") if path.is_file()}


def _resave_synthetic_manifest(fixture):
    _fixture_json(fixture["manifest_path"], fixture["manifest"])
    fixture["receipt"]["source_manifest_sha256"] = _fixture_sha(fixture["manifest_path"])
    _fixture_json(fixture["receipt_path"], fixture["receipt"])


def test_existing_79_sample_export_is_verified_without_writes_or_image_decode(existing_synthetic_export, monkeypatch):
    fixture = existing_synthetic_export
    before = _fixture_tree_hashes(fixture["fixture_root"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only verification must not export, copy, or admit resources")
    monkeypatch.setattr(exporter, "copy_and_verify", forbidden)
    monkeypatch.setattr(exporter, "execute_export", forbidden)
    monkeypatch.setattr(exporter, "guarded_resources", forbidden)

    verified = exporter.verify_export(fixture["output"], fixture["manifest_sha256"])

    assert verified["status"] == "PASS_EXISTING_PRIVATE_DEVELOPMENT_EXPORT_INTEGRITY_NOT_QUALIFICATION"
    assert verified["counts"] == fixture["plan"]["counts"]
    assert len(verified["samples"]) == 79
    assert sum(row["negative_visible_absence_only"] for row in verified["samples"]) == 4
    assert verified["manifest_sha256"] == fixture["manifest_sha256"]
    assert verified["receipt_sha256"] == _fixture_sha(fixture["receipt_path"])
    assert verified["data_yaml_sha256"] == _fixture_sha(fixture["yaml_path"])
    assert verified["qualification_evidence"] is verified["independent_human_review"] is False
    assert _fixture_tree_hashes(fixture["fixture_root"]) == before
    assert fixture["preflight_calls"] == [(
        fixture["output"], fixture["plan"]["root_final_approval"]["path"],
        fixture["plan"]["root_final_approval"]["sha256"], True,
    )]


def test_existing_export_rejects_wrong_external_manifest_pin_before_reconstruction(existing_synthetic_export):
    fixture = existing_synthetic_export

    with pytest.raises(ValueError, match="externally pinned"):
        exporter.verify_export(fixture["output"], "0" * 64)

    assert fixture["preflight_calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "other.schema"),
        ("status", "ROOT_APPROVED_READY_TO_EXPORT"),
        ("labels_exported", 78),
        ("labels_exported", 79.0),
        ("model_used", True),
        ("gpu_used", True),
        ("training_performed", True),
        ("model_used", 0),
        ("qualification_evidence", True),
        ("independent_human_review", True),
        ("source_independence_verified", True),
        ("final_test_used", True),
        ("input_bindings", {}),
        ("proposed_samples", []),
        ("sources", []),
        ("counts", {"images": 79, "train": 79, "test": 0}),
        ("output", "outside_dataset"),
        ("root_final_approval", None),
        ("root_final_approval", {"path": "changed", "sha256": "0" * 64}),
        ("root_final_approval", {"path": "changed", "sha256": "0" * 64, "extra": True}),
    ],
)
def test_existing_export_rejects_changed_manifest_even_with_updated_receipt_hash(existing_synthetic_export, field, value):
    fixture = existing_synthetic_export
    fixture["manifest"][field] = value
    _resave_synthetic_manifest(fixture)

    with pytest.raises(ValueError):
        exporter.verify_export(fixture["output"])


@pytest.mark.parametrize("field", ["qualification_evidence", "independent_human_review", "model_used", "sources", "root_final_approval"])
def test_existing_export_requires_explicit_manifest_declarations(existing_synthetic_export, field):
    fixture = existing_synthetic_export
    del fixture["manifest"][field]
    _resave_synthetic_manifest(fixture)

    with pytest.raises(ValueError):
        exporter.verify_export(fixture["output"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "PENDING"), ("counts", {"images": 78}),
        ("source_manifest_sha256", "0" * 64), ("data_yaml_sha256", "0" * 64),
        ("root_final_approval", None), ("input_files_changed", ["changed_source"]),
        ("qualification_evidence", True), ("training_performed", True),
        ("gpu_used", True), ("model_used", True), ("model_used", 0),
    ],
)
def test_existing_export_requires_consistent_terminal_receipt(existing_synthetic_export, field, value):
    fixture = existing_synthetic_export
    fixture["receipt"][field] = value
    _fixture_json(fixture["receipt_path"], fixture["receipt"])

    with pytest.raises(ValueError):
        exporter.verify_export(fixture["output"])


@pytest.mark.parametrize("which", ["source_image", "copied_image", "positive_label", "absence_label", "root_approval", "review_input"])
def test_existing_export_rejects_changed_retained_bytes(existing_synthetic_export, which):
    fixture = existing_synthetic_export
    row = fixture["plan"]["proposed_samples"][0]
    if which == "source_image":
        path = Path(row["image_source_path"])
    elif which == "copied_image":
        path = Path(row["image_path"])
    elif which == "positive_label":
        path = Path(row["label_path"])
    elif which == "absence_label":
        path = Path(fixture["plan"]["proposed_samples"][-1]["label_path"])
    else:
        path = fixture["approval_path"] if which == "root_approval" else fixture["review_path"]
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ValueError):
        exporter.verify_export(fixture["output"], fixture["manifest_sha256"])


@pytest.mark.parametrize("subdirectory", ["images/train", "images/val", "labels/train", "labels/val", "images/test", "labels/test"])
def test_existing_export_rejects_extra_files_and_final_test_directories(existing_synthetic_export, subdirectory):
    fixture = existing_synthetic_export
    directory = fixture["output"] / "ball" / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    if not subdirectory.endswith("test"):
        (directory / "unreviewed_extra.txt").write_bytes(b"unreviewed")

    with pytest.raises(ValueError):
        exporter.verify_export(fixture["output"], fixture["manifest_sha256"])


@pytest.mark.parametrize("yaml_change", ["train: images/test", "path: elsewhere", "names:\n  0: other_ball", "\n"])
def test_existing_export_yaml_cannot_redirect_or_change_even_if_receipt_rehashed(existing_synthetic_export, yaml_change):
    fixture = existing_synthetic_export
    fixture["yaml_path"].write_text(yaml_change, encoding="utf-8")
    fixture["receipt"]["data_yaml_sha256"] = _fixture_sha(fixture["yaml_path"])
    _fixture_json(fixture["receipt_path"], fixture["receipt"])

    with pytest.raises(ValueError, match="YAML"):
        exporter.verify_export(fixture["output"], fixture["manifest_sha256"])

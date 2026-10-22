"""Court repair corruption/leakage/approval and native roundtrip regression checks.

Small synthetic PNGs below test the exporter contract, never vision accuracy or
independent annotation. The actual175-source preflight is recorded separately.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from scripts.data import export_yolo26_court_repair_v2 as subject


def points():
    xy = [[400, 180], [880, 180], [200, 550], [1080, 550],
          [460, 180], [310, 550], [820, 180], [970, 550],
          [440, 240], [840, 240], [370, 420], [910, 420], [640, 240], [640, 420]]
    return [[*p, 2] for p in xy]


def png(path, color=0, width=1280, height=720):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    content = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress((b"\x00" + bytes([color, 0, 0]) * width) * height)) + chunk(b"IEND", b""))
    path.write_bytes(content)
    return content


def test_null_slot_preserves_semantics_without_exposing_publisher_estimate():
    pts = points()
    pts[2] = None
    box = subject.publisher_extent([p[:2] for p in points()])
    text = subject.encode_label(box, pts)
    values = [float(x) for x in text.split()]
    assert len(values) == 47
    assert values[11:14] == [0, 0, 0]
    assert values[14:17] == pytest.approx([1080 / 1280, 550 / 720, 2])
    # The bbox may use a publisher hidden-corner estimate, but its null slot does not.
    assert box == [200, 180, 1080, 550]
    subject.validate_label(text, box, pts)


@pytest.mark.parametrize("slot", range(14))
def test_every_semantic_slot_survives_native_roundtrip(slot):
    pts = points()
    pts[slot] = None
    text = subject.encode_label([200, 180, 1080, 550], pts)
    subject.validate_label(text, [200, 180, 1080, 550], pts)
    assert text.split()[5 + 3 * slot:8 + 3 * slot] == ["0", "0", "0"]


@pytest.mark.parametrize("bad", [[True, 5], [5, False], [float("nan"), 4], [5, float("inf")],
                                [-1, 4], [1280, 4], [1, 720], [1], None, "1,4"])
def test_invalid_or_outside_corner_never_gets_clamped(bad):
    proposals = [p[:2] for p in points()]
    proposals[3] = bad
    with pytest.raises(ValueError):
        subject.publisher_extent(proposals)


def test_box_uses_all_four_corners_even_when_one_x_is_extreme():
    proposals = [p[:2] for p in points()]
    proposals[0] = [1, 90]
    proposals[1] = [1200, 100]
    proposals[2] = [30, 700]
    proposals[3] = [1240, 690]
    assert subject.publisher_extent(proposals) == [1, 90, 1240, 700]


@pytest.mark.parametrize("box", [[200, 180, 200, 550], [-1, 180, 1080, 550],
                                [200, 180, 1280, 550], [200, 180, 1080, 720],
                                [True, 180, 1080, 550], [200, float("nan"), 1080, 550]])
def test_invalid_box_cannot_enter_yolo_label(box):
    with pytest.raises(ValueError):
        subject.encode_label(box, points())


@pytest.mark.parametrize("bad", [[440, 240, 1], [440, 240, True], [440, 240],
                                [440, 240, 0], [1, 240, 2], [440, float("nan"), 2]])
def test_only_visible_finite_in_extent_triples_or_null_allowed(bad):
    pts = points()
    pts[8] = bad
    with pytest.raises(ValueError):
        subject.encode_label([200, 180, 1080, 550], pts)


@pytest.mark.parametrize("mutation", ["extra_line", "extra_field", "wrong_class", "nan", "wrong_visibility", "invented_null", "reordered_slot"])
def test_label_corruption_is_detected_at_native_roundtrip(mutation):
    pts = points()
    pts[2] = None
    text = subject.encode_label([200, 180, 1080, 550], pts)
    tokens = text.split()
    if mutation == "extra_line":
        text += text
    elif mutation == "extra_field":
        text = text.strip() + " 0\n"
    else:
        if mutation == "wrong_class": tokens[0] = "1"
        if mutation == "nan": tokens[1] = "nan"
        if mutation == "wrong_visibility": tokens[7] = "1"
        if mutation == "invented_null": tokens[11:14] = ["0.2", "0.3", "2"]
        if mutation == "reordered_slot": tokens[5:8], tokens[8:11] = tokens[8:11], tokens[5:8]
        text = " ".join(tokens) + "\n"
    with pytest.raises(ValueError):
        subject.validate_label(text, [200, 180, 1080, 550], pts)


def test_png_geometry_checks_native_dimensions_and_all_chunk_crcs(tmp_path):
    path = tmp_path / "source.png"
    content = png(path)
    assert subject.png_geometry(path) == (1280, 720)
    corrupted = bytearray(content)
    corrupted[45] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="CRC"):
        subject.png_geometry(path)


@pytest.mark.parametrize("kind", ["resized", "truncated", "trailing", "non_png"])
def test_png_rejects_rescale_truncation_and_unregistered_trailing_bytes(tmp_path, kind):
    path = tmp_path / "source.png"
    content = png(path, width=640 if kind == "resized" else 1280)
    if kind == "truncated": path.write_bytes(content[:-8])
    if kind == "trailing": path.write_bytes(content + b"extra")
    if kind == "non_png": path.write_bytes(b"jpg")
    with pytest.raises(ValueError):
        subject.png_geometry(path)


def split_fixture():
    rows = [{"id": "sourceA0001_1", "split": "train", "source_video_id": "sourceA0001", "context_group": "a",
             "visual_event_family": "Rome", "source_image": {"sha256": "a"}},
            {"id": "sourceB0001_1", "split": "val", "source_video_id": "sourceB0001", "context_group": "b",
             "visual_event_family": "MonteCarlo", "source_image": {"sha256": "b"}}]
    groups = {"a": {"development_split": "train", "publisher_video_ids": ["sourceA0001"], "visual_event_families": ["Rome"]},
              "b": {"development_split": "val", "publisher_video_ids": ["sourceB0001"], "visual_event_families": ["MonteCarlo"]}}
    return rows, groups


@pytest.mark.parametrize("mutation", ["test", "video_bridge", "component_split", "duplicate_image", "duplicate_id", "lost_family", "prior_dev_to_train"])
def test_cross_split_bridges_rejected_links_and_dev_constraints_are_not_bypassed(mutation):
    rows, groups = split_fixture()
    if mutation == "test": rows[1]["split"] = "test"
    if mutation == "video_bridge": rows[1]["source_video_id"] = "sourceA0001"; groups["b"]["publisher_video_ids"] = ["sourceA0001"]
    if mutation == "component_split": groups["a"]["development_split"] = "val"
    if mutation == "duplicate_image": rows[1]["source_image"]["sha256"] = "a"
    if mutation == "duplicate_id": rows[1]["id"] = rows[0]["id"]
    if mutation == "lost_family": groups["a"]["visual_event_families"] = ["Paris"]
    if mutation == "prior_dev_to_train": groups["a"]["visual_event_families"].append("LondonFinals")
    with pytest.raises(ValueError):
        subject.validate_split_integrity(rows, groups, {"train": 1, "val": 1, "test": 0})


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    output = tmp_path / "export"
    records = []
    rows, components = split_fixture()
    for n, item in enumerate(rows):
        path = tmp_path / f"source{n}.png"
        png(path, color=n)
        records.append(item | {"origin": "SYNTHETIC_SERIALIZER_TEST_ONLY", "source_frame_ordinal": 1,
            "source_image": {"path": path.name, "sha256": subject.digest(path), "bytes": path.stat().st_size},
            "surface": "hard", "prior_context_group": None, "keypoints_px": points(),
            "bbox_xyxy_px": [200, 180, 1080, 550], "bbox_provenance": "TEST_ONLY",
            "review_note": "Synthetic contract fixture; no annotation/vision evidence", "reviewer_type": "AGENT_VISUAL_REVIEW",
            "label_text": subject.encode_label([200, 180, 1080, 550], points()),
            "publisher_annotation_file": "test", "publisher_annotation_sha256": "test"})
    plan = {"output": output.as_posix(), "records": records, "counts": {"train": 1, "val": 1, "test": 0},
            "keypoint_slots": {"visible": 28, "unknown_null": 0}, "components": components,
            "input_bindings": {}, "rights": {"test_only": True}, "native_image_bytes": sum(r["source_image"]["bytes"] for r in records),
            "yaml_text": subject.dataset_yaml(output)}
    def mock_plan(requested, *, root, require_fresh=True):
        assert Path(requested) == output
        if require_fresh and output.exists(): raise FileExistsError("Fresh required")
        return deepcopy(plan)
    monkeypatch.setattr(subject, "prepare_plan", mock_plan)
    monkeypatch.setattr(subject, "guarded_copy_resources", lambda *args: {})
    approval = subject.approval_template(plan)
    approval.update(status=subject.APPROVED, reviewer="ROOT_AGENT", root_review_note="Synthetic API contract test only")
    approval_path = tmp_path / "approval.json"
    subject.write_json(approval_path, approval)
    return output, approval_path, plan, tmp_path


@pytest.mark.parametrize("mutation", ["pending", "plan_hash", "id_order", "output", "counts", "human", "qualification", "training", "missing_note", "reviewer"])
def test_exact_separate_root_approval_required_before_output_mutation(export_fixture, mutation):
    output, approval_path, plan, root = export_fixture
    approval = subject.read_json(approval_path)
    if mutation == "pending": approval["status"] = "PENDING_ROOT_REVIEW_NOT_APPROVED"
    if mutation == "plan_hash": approval["plan_sha256"] = "0" * 64
    if mutation == "id_order": approval["approved_ordered_ids"].reverse()
    if mutation == "output": approval["output"] += "elsewhere"
    if mutation == "counts": approval["counts"]["train"] += 1
    if mutation == "human": approval["independent_human_review"] = True
    if mutation == "qualification": approval["qualification_evidence"] = True
    if mutation == "training": approval["training_authorized_by_this_export_approval"] = True
    if mutation == "missing_note": approval["root_review_note"] = " "
    if mutation == "reviewer": approval["reviewer"] = "OTHER_AGENT"
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        subject.export_dataset(output, approval_path, subject.digest(approval_path), root=root)
    assert not output.exists()


def test_actual_byte_copy_reconstructs_and_keeps_immutable_plan_source_and_null_contract(export_fixture):
    output, approval, plan, root = export_fixture
    result = subject.export_dataset(output, approval, subject.digest(approval), root=root)
    assert result["files_verified"] == 8
    assert result["pixels_decoded"] is False
    for row in plan["records"]:
        assert (output / f"court/images/{row['split']}/{row['id']}.png").read_bytes() == (root / row["source_image"]["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        subject.export_dataset(output, approval, subject.digest(approval), root=root)


@pytest.mark.parametrize("mutation", ["extra_file", "image", "label", "yaml_flip", "manifest_gt", "saved_plan", "receipt", "approval"])
def test_postexport_corruption_cannot_pass_verification(export_fixture, mutation):
    output, approval, plan, root = export_fixture
    subject.export_dataset(output, approval, subject.digest(approval), root=root)
    row = plan["records"][0]
    if mutation == "extra_file": (output / "unexpected.cache").write_bytes(b"cache")
    if mutation == "image": (output / f"court/images/{row['split']}/{row['id']}.png").write_bytes(b"bad")
    if mutation == "label": (output / f"court/labels/{row['split']}/{row['id']}.txt").write_text(row["label_text"] + row["label_text"])
    if mutation == "yaml_flip": (output / "court/data.yaml").write_text(plan["yaml_text"].replace("[1, 0,", "[0, 1,"))
    if mutation in {"manifest_gt", "saved_plan", "receipt", "approval"}:
        path = {"manifest_gt": output / "source_manifest.json", "saved_plan": output / "frozen_export_plan.json",
                "receipt": output / "export_receipt.json", "approval": approval}[mutation]
        document = subject.read_json(path)
        if mutation == "manifest_gt": document["qualification_evidence"] = True
        elif mutation == "saved_plan": document["records"][0]["keypoints_px"][0][0] += 10
        elif mutation == "receipt": document["counts"]["train"] = 999
        else: document["root_review_note"] += "changed"
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        subject.verify_export(output, root=root)


def test_wrong_approval_hash_rejected_before_any_output(export_fixture):
    output, approval, _, root = export_fixture
    with pytest.raises(ValueError, match="SHA256"):
        subject.export_dataset(output, approval, "0" * 64, root=root)
    assert not output.exists()


def test_output_and_approval_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError, match="workspace"):
        subject.workspace_path(tmp_path / ".." / "outside", tmp_path)
    with pytest.raises(ValueError):
        subject.workspace_path(tmp_path, tmp_path)

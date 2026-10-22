"""Private repair admission checks, using reviewed metadata and tiny RGB fixtures.

No model, source video decoder or third-party image library is loaded. The one
actual metadata test reads pinned review artifacts but creates no dataset.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

from scripts.data import export_yolo26_author_ball_repair_v2 as repair


def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _png(width=7, height=5, method=0, *, extra_decoded=b"", interlace=0, split_idat=False):
    """Independent forward PNG encoder, referencing only original RGB samples."""
    rows = [bytes((x * 67 + y * 43 + x * y * 11) % 256 for x in range(width * 3)) for y in range(height)]
    encoded = bytearray()
    for y, row in enumerate(rows):
        encoded.append(method)
        for x, original in enumerate(row):
            left = row[x - 3] if x >= 3 else 0
            above = rows[y - 1][x] if y else 0
            corner = rows[y - 1][x - 3] if y and x >= 3 else 0
            if method == 0:
                predictor = 0
            elif method == 1:
                predictor = left
            elif method == 2:
                predictor = above
            elif method == 3:
                predictor = (left + above) // 2
            elif method == 4:
                prediction = left + above - corner
                distances = [abs(prediction - candidate) for candidate in (left, above, corner)]
                predictor = (left, above, corner)[distances.index(min(distances))]
            else:
                predictor = 0
            encoded.append((original - predictor) % 256)
    compressed = zlib.compress(bytes(encoded) + extra_decoded)
    idats = (_chunk(b"IDAT", compressed[:4]) + _chunk(b"IDAT", compressed[4:])) if split_idat else _chunk(b"IDAT", compressed)
    raw = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, interlace)) + idats + _chunk(b"IEND", b"")
    return raw, b"".join(rows)


@pytest.mark.parametrize("method", range(5))
@pytest.mark.parametrize("split_idat", [False, True])
def test_png_inverse_matches_independent_rgb_fixture_for_every_filter(tmp_path, method, split_idat):
    raw, expected = _png(method=method, split_idat=split_idat)
    path = tmp_path / "native.png"
    path.write_bytes(raw)
    actual, shape = repair.read_rgb8_png(path)
    assert shape == (7, 5)
    assert actual == expected


@pytest.mark.parametrize("damage", ["signature", "crc", "truncated_chunk", "trailing", "too_many_pixels", "too_many_file_bytes", "interlaced", "extra_decoded", "unknown_filter", "no_iend", "zlib_truncated", "zlib_concatenated", "idat_interrupted", "unknown_critical"])
def test_png_reader_rejects_corruption_and_bounded_decode_violations(tmp_path, damage):
    raw, _ = _png(interlace=1 if damage == "interlaced" else 0,
                  extra_decoded=b"X" if damage == "extra_decoded" else b"",
                  method=5 if damage == "unknown_filter" else 0)
    options = {}
    if damage == "signature":
        raw = b"NO" + raw[2:]
    elif damage == "crc":
        raw = raw[:29] + bytes([raw[29] ^ 1]) + raw[30:]
    elif damage == "truncated_chunk":
        raw = raw[:35]
    elif damage == "trailing":
        raw += b"hidden"
    elif damage == "too_many_pixels":
        options["max_pixels"] = 34
    elif damage == "too_many_file_bytes":
        options["max_file_bytes"] = len(raw) - 1
    elif damage == "no_iend":
        raw = raw[:-12]
    elif damage in {"zlib_truncated", "zlib_concatenated", "idat_interrupted", "unknown_critical"}:
        length = struct.unpack_from(">I", raw, 33)[0]
        compressed = raw[41:41 + length]
        if damage == "zlib_truncated":
            idats = _chunk(b"IDAT", compressed[:-1])
        elif damage == "zlib_concatenated":
            idats = _chunk(b"IDAT", compressed + zlib.compress(b"hidden"))
        elif damage == "idat_interrupted":
            idats = _chunk(b"IDAT", compressed[:4]) + _chunk(b"tEXt", b"A\x00B") + _chunk(b"IDAT", compressed[4:])
        else:
            idats = _chunk(b"FAKE", b"payload") + _chunk(b"IDAT", compressed)
        raw = raw[:33] + idats + _chunk(b"IEND", b"")
    path = tmp_path / "corrupt.png"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        repair.read_rgb8_png(path, **options)


def test_exact_roi_rejects_one_changed_pixel_and_shifted_or_resized_copy():
    _, native = _png(width=7, height=5)
    crop = b"".join(native[((y * 7 + 2) * 3):((y * 7 + 5) * 3)] for y in range(1, 4))
    assert repair.compare_roi_pixels(native, (7, 5), crop, (2, 1), (3, 3)) == hashlib.sha256(crop).hexdigest()
    changed = bytes([crop[0] ^ 1]) + crop[1:]
    for suspect, origin, size in ((changed, (2, 1), (3, 3)), (crop, (1, 1), (3, 3)), (crop, (2, 1), (9, 1)), (crop, (6, 1), (3, 3))):
        with pytest.raises(ValueError):
            repair.compare_roi_pixels(native, (7, 5), suspect, origin, size)


def _geometry():
    return {"native_width": 1920, "native_height": 1080, "actual_crop_origin_native_px": [529, 414],
            "crop_size_px": [640, 640], "native_pixels_resized": False,
            "reviewed_label_clamped_or_changed": False, "coordinate_roundtrip_exact": True,
            "reviewed_box_xyxy_crop_px": [317.25, 317.5, 324.75, 324.125],
            "reviewed_box_xyxy_native_px": [846.25, 731.5, 853.75, 738.125],
            "source_frame_index": 1, "source_pts": "9032", "source_time_base": "1/59940",
            "source_timestamp_fraction": "2258/14985", "source_native_image": {"frame_index": 1, "source_pts": "9032"}}


def test_fractional_reviewed_extent_serializes_without_rounding_or_clamping():
    sample = _geometry()
    original = deepcopy(sample)
    box = repair.validate_geometry(sample)
    line = repair.parent.yolo_line(box)
    restored = repair.parent.inverse_yolo_line(line)
    assert max(abs(a - b) for a, b in zip(box, restored, strict=True)) < 1e-6
    assert sample == original


@pytest.mark.parametrize("field,value", [("native_width", True), ("actual_crop_origin_native_px", [529.0, 414]),
    ("actual_crop_origin_native_px", [1500, 414]), ("native_pixels_resized", True),
    ("reviewed_label_clamped_or_changed", True), ("coordinate_roundtrip_exact", False),
    ("reviewed_box_xyxy_crop_px", [317.25, -1, 324.75, 324.125]),
    ("reviewed_box_xyxy_native_px", [846, 731, 854, 739]), ("source_pts", "9033"),
    ("source_time_base", "1/30"), ("source_frame_index", True), ("source_timestamp_fraction", "1/25")])
def test_geometry_rejects_loss_of_native_pixels_reviewed_box_or_actual_pts(field, value):
    sample = _geometry()
    sample[field] = value
    with pytest.raises(ValueError):
        repair.validate_geometry(sample)


def _contexts():
    return [
        {"source_ids": ["beijing_1", "beijing_2"], "aliases": ["beijing"], "derived_split": "train"},
        {"source_ids": ["miami_1", "washington_1"], "aliases": ["miami", "washington"], "derived_split": "development"},
        {"source_ids": ["rome_1", "dubai_1"], "aliases": ["rome", "dubai"], "derived_split": "development"},
        {"source_ids": ["stockholm_1"], "aliases": ["stockholm"], "derived_split": "development"},
        {"source_ids": ["paris_1"], "aliases": ["paris"], "derived_split": "quarantine"},
        {"source_ids": ["stpetersburg_1"], "aliases": ["stpetersburg"], "derived_split": "quarantine"},
    ]


def _aliases():
    return [{"publisher_original_alias": name, "logical_video_alias": f"tennis_{i}"} for i, name in enumerate((
        "America_Best_10_1", "America_Best_Move_1", "Federer_Top_40_1", "Federer_Top_40_2", "Federer_Top_40_3", "France_Review_Red_Court_1"))]


def _source_rows():
    return [{"id": "usa_accept", "alias": "America_Best_10_1"}, {"id": "usa_reject", "alias": "America_Best_10_1"},
            {"id": "france_reject", "alias": "France_Review_Red_Court_1"}]


def test_whole_compilations_rejected_rows_and_unseen_related_aliases_stay_in_graph():
    old, aliases, rows = _contexts(), _aliases(), _source_rows()
    before = deepcopy((old, aliases, rows))
    graph = repair.reconstruct_contexts(old, aliases, rows)
    usa = next(c for c in graph["components"] if "author:America_Best_10_1" in c["members"])
    for name in ("America_Best_Move_1", "Federer_Top_40_1", "Federer_Top_40_2", "Federer_Top_40_3"):
        assert "author:" + name in usa["members"]
    assert "reviewed-frame:usa_reject" in usa["members"]
    france = next(c for c in graph["components"] if "author:France_Review_Red_Court_1" in c["members"])
    assert "event:rolandgarros" in france["members"]
    assert "legacy:paris_1" not in france["members"]
    assert graph["test_count"] == 0
    assert all(c["independent_original_match_or_camera_session_verified"] is False for c in graph["components"])
    assert (old, aliases, rows) == before


@pytest.mark.parametrize("new_node", ["author:America_Best_10_1", "reviewed-frame:usa_reject", "author:Federer_Top_40_3", "reviewed-frame:france_reject"])
@pytest.mark.parametrize("old_node", ["event:miami", "event:washington", "event:dubai", "event:rome", "event:stockholm", "event:paris", "event:stpetersburg"])
def test_evidenced_bridge_including_rejected_or_unseen_source_cannot_move_dev_or_quarantine_to_train(new_node, old_node):
    with pytest.raises(ValueError, match="development/quarantine exposure"):
        repair.reconstruct_contexts(_contexts(), _aliases(), _source_rows(), [(new_node, old_node)])


def test_transitive_graph_cannot_preserve_conflicting_legacy_split_claims():
    old = _contexts()
    old[0]["aliases"].append("miami")
    with pytest.raises(ValueError, match="split boundaries"):
        repair.reconstruct_contexts(old, _aliases(), _source_rows())


def test_all_old_metadata_and_empty_label_bytes_are_preserved():
    old = {"sample_id": "absence", "image_path": "old.png", "label_path": "old.txt", "label_text": "",
           "label_sha256": hashlib.sha256(b"").hexdigest(), "development_split": "val", "exposure_note": "ALL_REJECTED_SOURCE_LINKS_RETAINED"}
    new = deepcopy(old)
    new.update(image_path="new.png", label_path="new.txt", copy_source_path="old.png", copy_label_source_path="old.txt", inherited_from_old79=True)
    repair.validate_old_preservation([old], [new])
    for key, value in (("label_text", "\n"), ("development_split", "train"), ("exposure_note", "discarded"), ("copy_label_source_path", "another.txt")):
        suspect = deepcopy(new)
        suspect[key] = value
        with pytest.raises(ValueError, match="was altered"):
            repair.validate_old_preservation([old], [suspect])


def _approval():
    return {"status": repair.APPROVAL_STATUS, "reviewer_type": "ROOT_AGENT", "proposal_sha256": "p",
            "input_bindings": {"source": "h"}, "approved_ordered_sample_ids": ["a", "b"], "test_count": 0,
            "allowed_use": "PRIVATE_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT", "review_note": "Reviewed concrete source and label plan.",
            **{key: False for key in ("qualification_evidence", "independent_human_review", "raw_media_public_redistribution_cleared", "weights_public_release_cleared", "commercial_clearance")}}


@pytest.mark.parametrize("field,value", [("proposal_sha256", "another"), ("input_bindings", {"source": "another"}),
    ("approved_ordered_sample_ids", ["b", "a"]), ("reviewer_type", "HUMAN"), ("qualification_evidence", True),
    ("independent_human_review", True), ("raw_media_public_redistribution_cleared", True), ("weights_public_release_cleared", True),
    ("commercial_clearance", True), ("test_count", 1), ("test_count", False), ("review_note", " "), ("allowed_use", "COMMERCIAL")])
def test_approval_is_bound_to_exact_reviewed_inputs_order_and_private_scope(field, value):
    approval = _approval()
    repair.verify_approval(approval, "p", {"source": "h"}, ["a", "b"])
    approval[field] = value
    with pytest.raises(ValueError):
        repair.verify_approval(approval, "p", {"source": "h"}, ["a", "b"])


def test_hash_check_detects_changed_inputs_and_outside_workspace_binding(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("frozen", encoding="utf-8")
    bindings = {str(path): repair.parent.digest(path)}
    repair.check_bindings(bindings)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        repair.check_bindings(bindings)
    with pytest.raises(ValueError, match="inside workspace"):
        repair.bind({}, repair.ROOT.parent / "outside.txt", "a" * 64)


def test_execute_rejects_unapproved_plan_without_creating_destination(tmp_path):
    target = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Unapproved"):
        repair.execute_export({"status": "PROPOSED101_REQUIRES_ROOT_REVIEW_BEFORE_EXPORT", "contract": {"output": str(target)}})
    assert not target.exists()


def _actual_source_review_records():
    native = repair.read_json(repair.NATIVE)["samples"]
    crops = repair.read_json(repair.CROPS)["samples"]
    names = ("first8_final_agent_visual_review.json", "france_tail12_agent_visual_review.json", "usa_tail12_root_visual_review.json")
    reviews = [(repair.AUDIT / name, repair.read_json(repair.AUDIT / name)) for name in names]
    return native, reviews, crops


@pytest.mark.parametrize("damage", ["duplicate_review", "missing_rejected_review", "missing_rejected_proposal", "unviewed_patch", "independent_claim", "wrong_pts", "wrong_event", "changed_original_box", "changed_crop_box", "rejected_crop", "crop_source_receipt", "crop_source_image"])
def test_source_to_crop_reconstruction_rejects_lost_adjudication_or_changed_evidence(damage):
    native, reviews, crops = _actual_source_review_records()
    first = reviews[0][1]["rows"][0]
    if damage == "duplicate_review":
        reviews[0][1]["rows"].append(deepcopy(first))
    elif damage == "missing_rejected_review":
        reviews[1][1]["rows"].pop(2)
    elif damage == "missing_rejected_proposal":
        reviews[0][1]["rows"][6]["reviewed_proposals"].pop()
    elif damage == "unviewed_patch":
        first["actually_viewed_every_unmarked_native96_temporal_patch_locally"] = False
    elif damage == "independent_claim":
        first["independent_human_review"] = True
    elif damage == "wrong_pts":
        first["source_pts"] = "9033"
    elif damage == "wrong_event":
        first["visual_event_family"] = "Miami"
    elif damage == "changed_original_box":
        first["reviewed_proposals"][0]["publisher_proposal_preserved"]["center_native_px"][0] += 1
    elif damage == "changed_crop_box":
        crops[0]["reviewed_box_xyxy_native_px"][0] += 1
    elif damage == "rejected_crop":
        crops[0]["id"] = reviews[1][1]["rows"][2]["id"]
    elif damage == "crop_source_receipt":
        crops[0]["source_review_receipt"]["sha256"] = "0" * 64
    elif damage == "crop_source_image":
        crops[0]["source_native_image"]["source_pts"] = "9033"
    with pytest.raises(ValueError):
        repair.validate_source_reviews(native, reviews, crops)


@pytest.mark.parametrize("france", [False, True])
@pytest.mark.parametrize("damage", ["unviewed", "extra_target", "unresolved", "changed_crop", "changed_box", "qualification", "rejected"])
def test_complete_crop_review_cannot_admit_unresolved_background_or_changed_assets(france, damage):
    path = repair.AUDIT / ("france_fullcrop9_agent_review.json" if france else "usa_fullcrop13_root_review.json")
    row = repair.read_json(path)["rows"][0]
    sample = next(crop for crop in repair.read_json(repair.CROPS)["samples"] if crop["id"] == row["id"])
    repair.validate_closure(sample, row, france=france)
    if damage == "unviewed":
        row["actually_viewed_complete_unmarked640_crop_locally" if france else "actually_viewed_entire_unmarked_native640_crop"] = False
    elif damage == "extra_target":
        row["additional_visible_ball_target_count" if france else "additional_visible_ball_targets_observed"] = 1
    elif damage == "unresolved":
        row["unresolved_spare_ball_or_equipment_target_observed" if france else "unresolved_background_ball_proposal_observed"] = True
    elif damage == "changed_crop":
        row["unmarked_crop"]["sha256"] = "0" * 64
    elif damage == "changed_box":
        row["reviewed_box_xyxy_crop_px"][0] += 1
    elif damage == "qualification":
        row["independent_human_review"] = True
    elif damage == "rejected":
        row["crop_closure_decision" if france else "review_decision"] = "REJECT_UNRESOLVED"
    with pytest.raises(ValueError):
        repair.validate_closure(sample, row, france=france)


def _native_probe_records(alias):
    native = repair.read_json(repair.NATIVE)
    sample = next(row for row in repair.read_json(repair.CROPS)["samples"] if row["alias"] == alias)
    acquisition = repair.read_json(repair.AUDIT / "acquisition_receipt.json")
    source = next(row for row in acquisition["files"] if Path(row["path"]).stem == alias)
    path = repair.AUDIT / "native_probe" / (alias + ".json")
    expected = next(sha for p, sha in native["input_bindings"].items() if Path(p).name == path.name and "native_probe" in Path(p).parts)
    return sample, source, acquisition, path, expected, repair.read_json(path)


@pytest.mark.parametrize("alias,nominal,average,count,stream_duration,container_duration", [
    ("America_Best_10_1", "2997/50", "9597060/160111", 1441, "24.040691", "24.174692"),
    ("France_Review_Red_Court_1", "25/1", "1881000/75211", 1881, "75.211000", "75.211000")])
def test_actual_probe_rates_and_durations_stay_separate_from_normal_speed_claim(alias, nominal, average, count, stream_duration, container_duration):
    records = _native_probe_records(alias)
    original = deepcopy(records)
    metadata = repair.native_source_metadata(*records)
    assert metadata["fps_rational"] == nominal
    assert metadata["source_stream_r_frame_rate"] == nominal
    assert metadata["source_stream_avg_frame_rate"] == average
    assert nominal != average
    assert metadata["normal_speed_verified"] is False
    assert metadata["constant_frame_cadence_verified"] is False
    assert metadata["source_native_probe"]["native_frame_pts_count"] == count
    assert metadata["source_native_probe"]["stream_metadata"]["duration"] == stream_duration
    assert metadata["source_native_probe"]["container_metadata"]["duration"] == container_duration
    assert metadata["source_acquisition_receipt_created_utc"] == "2026-10-16T01:41:53.710990+00:00"
    assert metadata["source_publisher_repository"] == "Zoutianjian/Ball-Yolo-A-Sport-Ball-Dataset"
    assert metadata["source_grant_attribution"][0]["sha256"] == "69789292b396d6fd622e0c482c45d8253d7c20c90085103609aea86dcc08a745"
    assert records == original


@pytest.mark.parametrize("damage", ["wrong_probe_source", "wrong_video_source", "shape", "time_base", "frame_pts", "nonmonotonic_pts", "frame_count", "video_byte_size", "wrong_commit_url", "nominal_zero", "average_zero", "no_acquisition_timezone"])
def test_source_probe_metadata_consistency_refuses_another_source_or_cadence_assumption(damage):
    sample, source, acquisition, path, expected, probe = _native_probe_records("America_Best_10_1")
    if damage == "wrong_probe_source":
        path = path.with_name("France_Review_Red_Court_1.json")
    elif damage == "wrong_video_source":
        source["path"] = "France_Review_Red_Court_1.mp4"
    elif damage == "shape":
        probe["streams"][0]["width"] = 1280
    elif damage == "time_base":
        probe["streams"][0]["time_base"] = "1/25"
    elif damage == "frame_pts":
        sample["source_pts"] = str(int(sample["source_pts"]) + 1)
    elif damage == "nonmonotonic_pts":
        probe["frames"][1]["pts"] = probe["frames"][0]["pts"]
    elif damage == "frame_count":
        probe["streams"][0]["nb_frames"] = "1440"
    elif damage == "video_byte_size":
        probe["format"]["size"] = "12345"
    elif damage == "wrong_commit_url":
        source["url"] = source["url"].replace(acquisition["pinned_commit"], "other")
    elif damage == "nominal_zero":
        probe["streams"][0]["r_frame_rate"] = "0/1"
    elif damage == "average_zero":
        probe["streams"][0]["avg_frame_rate"] = "0/1"
    elif damage == "no_acquisition_timezone":
        acquisition["created_utc"] = "2026-10-16T01:41:53.710990"
    with pytest.raises(ValueError):
        repair.native_source_metadata(sample, source, acquisition, path, expected, probe)


def test_import_does_not_load_image_numerical_or_model_runtimes():
    source = "import json,sys; from scripts.data import export_yolo26_author_ball_repair_v2; print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {'PIL','numpy','torch','ultralytics','cv2'})))"
    result = subprocess.run([sys.executable, "-c", source], cwd=repair.ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def test_actual_pinned_metadata_reconstructs_101_plan_without_export_or_runtime_imports():
    existed_before = repair.OUTPUT.exists()
    plan = repair.prepare_plan(allow_existing=True)
    contract = plan["contract"]
    assert plan["status"] == "PROPOSED101_REQUIRES_ROOT_REVIEW_BEFORE_EXPORT"
    assert plan["root_final_approval"] is None
    assert contract["counts"] == repair.EXPECTED_COUNTS
    assert contract["source_review_counts"] == {"reviewed_frames": 32, "reviewed_publisher_proposals": 34,
        "accepted_positive_frames": 22, "rejected_frames": 10, "rejected_publisher_proposals": 12}
    assert len(contract["all32_new_source_review_rows_including_rejected"]) == 32
    assert sum(len(row["boxes"]) for row in contract["all34_publisher_proposals_preserved"]) == 34
    assert len(contract["all_author_aliases_preserved"]) == 16
    assert len(contract["proposed_samples"]) == 101
    new = [row for row in contract["proposed_samples"] if not row["inherited_from_old79"]]
    assert len(new) == 22 and all(row["development_split"] == "train" for row in new)
    assert all(row["fps_rational"] and row["source_stream_avg_frame_rate"] for row in new)
    assert all(row["normal_speed_verified"] is False and row["constant_frame_cadence_verified"] is False for row in new)
    assert Path(contract["output"]).exists() is existed_before
    assert not any(name in sys.modules for name in ("PIL", "numpy", "torch", "ultralytics", "cv2"))

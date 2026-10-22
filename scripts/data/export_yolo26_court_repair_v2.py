"""Preflight, approve, export and reconstruct the native 175-image court repair.

Only standard-library modules are imported. No pixels are decoded, no model is
loaded, and neither preflight nor verification writes files. Export requires a
separate hash-bound root approval and a fresh workspace directory. Historical
120-image exports and all 229 reviewed sources remain immutable evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import zlib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/yolo26/court_reviewed_development_v2"
OLD_EXPORT = "data/yolo26/court_reviewed_development_v1"
WAVE = "data/yolo26/court_source_wave64_development_v1/grouped_review_manifest.json"
CONSOLIDATOR = "scripts/data/consolidate_yolo26_court_source_wave.py"
OLD_DECISIONS = "data/yolo26/court_review_decisions_v1.json"
OLD_FROZEN = "data/yolo26/court_development_frozen_v1/grouped_development_manifest.json"
OLD_INVENTORY = "data/yolo26/court_review_expansion_v1/cache_inventory.json"
OLD_PROTOCOL = "data/yolo26/court_review_expansion_v1/predeclared_group_protocol.json"
OLD_APPROVAL = "artifacts/validation/yolo26_execution_2026-10-14/court_root_export_approval.json"
PINS = {
    WAVE: "f860f8b97df674661a4dfa26c4936e4352ce13ec4beafe515227e47722c054a9",
    CONSOLIDATOR: "9b1e8d5035ebc70be84a576c274b48bbd444e9ce56ccd0ca676fa05d09f57803",
    OLD_DECISIONS: "b085770df3330b89a8371806d2376ea9106db9f2f82d4e5436c78f5af7a634cb",
    OLD_FROZEN: "35ae2e741c6e8abdd6eb9b0ed516316be2ec5b1f73ef257b61867ea9e5cd851e",
    OLD_INVENTORY: "02cc85fa195c4ea5e0c1990f71a0d7f2954c561940f01dde0112c17077beeeae",
    OLD_PROTOCOL: "bc37510cf64fa71b4027fda9f88b089e7687ec1a53af940419afe7070202f17d",
    OLD_APPROVAL: "26b77be89c57f07ee60a5a3b5303409e23c39a651d0d26627ff6620d77374869",
    OLD_EXPORT + "/source_manifest.json": "0fadd340acc2fdfa5db9083474f620b396eea1d1eaf8fe025fdf050e43c3c90f",
}
POINT_NAMES = ["far_left_doubles", "far_right_doubles", "near_left_doubles", "near_right_doubles",
               "far_left_singles", "near_left_singles", "far_right_singles", "near_right_singles",
               "far_service_left", "far_service_right", "near_service_left", "near_service_right",
               "far_service_center", "near_service_center"]
FLIP_IDX = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]
APPROVED = "ROOT_APPROVED_PRIVATE_NATIVE_COURT175_DEVELOPMENT_EXPORT"
COUNTS = {"train": 149, "val": 26, "test": 0}
BBOX_PROVENANCE = ("FOUR_PUBLISHER_DOUBLES_CORNERS_0_1_2_3_ENVELOPE_NATIVE_TRAINING_EXTENT_ESTIMATE; "
                   "OCCLUDED_EXTENT_IS_NOT_VISIBLE_KEYPOINT_OR_MEASUREMENT_GROUND_TRUTH")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def workspace_path(path, root=ROOT):
    root = Path(root).resolve()
    path = Path(path)
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Require a contained workspace path")
    return result


def _relative(path, root=ROOT):
    return workspace_path(path, root).relative_to(Path(root).resolve()).as_posix()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def native_point(point):
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or not all(_number(value) for value in point)
            or not 0 <= point[0] < 1280 or not 0 <= point[1] < 720):
        raise ValueError("Require finite in-image native xy without clamping")
    return list(point)


def publisher_extent(points):
    if not isinstance(points, list) or len(points) != 14:
        raise ValueError("Require exactly fourteen publisher slots")
    corners = [native_point(points[slot]) for slot in (0, 1, 2, 3)]
    return [min(p[0] for p in corners), min(p[1] for p in corners),
            max(p[0] for p in corners), max(p[1] for p in corners)]


def encode_label(box, points):
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_number(v) for v in box)
            or not 0 <= box[0] < box[2] < 1280 or not 0 <= box[1] < box[3] < 720):
        raise ValueError("Court extent must be finite positive and inside native image")
    if not isinstance(points, list) or len(points) != 14:
        raise ValueError("Require fourteen nested visible/null semantic slots")
    x1, y1, x2, y2 = box
    values = [0, (x1 + x2) / 2560, (y1 + y2) / 1440, (x2 - x1) / 1280, (y2 - y1) / 720]
    for point in points:
        if point is None:
            values.extend([0, 0, 0])
        else:
            if not isinstance(point, list) or len(point) != 3 or type(point[2]) is not int or point[2] != 2:
                raise ValueError("Only actually visible coordinate triples or null are allowed")
            x, y = native_point(point[:2])
            if not x1 <= x <= x2 or not y1 <= y <= y2:
                raise ValueError("Visible semantic point leaves reviewed extent")
            values.extend([x / 1280, y / 720, 2])
    text = " ".join(format(value, ".12g") for value in values) + "\n"
    validate_label(text, box, points)
    return text


def validate_label(text, box, points):
    if not isinstance(text, str) or len(text.splitlines()) != 1 or not text.endswith("\n"):
        raise ValueError("Require exactly one complete court label line")
    tokens = text.split()
    if len(tokens) != 47 or tokens[0] != "0":
        raise ValueError("Require class0 and exactly47 court14 fields")
    values = list(map(float, tokens))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Label contains nonfinite coordinates")
    cx, cy, width, height = values[1:5]
    restored = [(cx - width / 2) * 1280, (cy - height / 2) * 720,
                (cx + width / 2) * 1280, (cy + height / 2) * 720]
    if any(abs(a - b) > 1e-6 for a, b in zip(restored, box, strict=True)):
        raise ValueError("Bounding extent native roundtrip failed")
    if len(points) != 14:
        raise ValueError("Require fourteen ordered semantic slots")
    for slot, point in enumerate(points):
        triple = values[5 + 3 * slot:8 + 3 * slot]
        if point is None:
            if triple != [0, 0, 0]:
                raise ValueError("Unknown point gained invented coordinates or visibility")
        elif (triple[2] != 2 or abs(triple[0] * 1280 - point[0]) > 1e-6
              or abs(triple[1] * 720 - point[1]) > 1e-6):
            raise ValueError("Visible semantic point native roundtrip failed")


def png_geometry(path):
    """Check bounded PNG chunks and every CRC; do not inflate/decode pixels."""
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Native PNG signature changed")
        geometry, first = None, True
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated native PNG chunk")
            size, kind = struct.unpack(">I4s", header)
            if size > Path(path).stat().st_size or first and (kind != b"IHDR" or size != 13):
                raise ValueError("Invalid native PNG chunk/header")
            checksum, remaining = zlib.crc32(kind), size
            payload = b""
            while remaining:
                chunk = stream.read(min(remaining, 512 * 1024))
                if not chunk:
                    raise ValueError("Truncated native PNG payload")
                checksum = zlib.crc32(chunk, checksum)
                if kind == b"IHDR":
                    payload += chunk
                remaining -= len(chunk)
            crc = stream.read(4)
            if len(crc) != 4 or struct.unpack(">I", crc)[0] != checksum & 0xFFFFFFFF:
                raise ValueError("Native PNG chunk CRC changed")
            if kind == b"IHDR":
                if not first:
                    raise ValueError("Repeated native PNG IHDR")
                geometry = struct.unpack(">II", payload[:8])
                if geometry != (1280, 720):
                    raise ValueError("Native PNG geometry must remain1280x720")
            first = False
            if kind == b"IEND":
                if size != 0 or stream.read(1):
                    raise ValueError("Invalid PNG end or trailing bytes")
                return geometry


def dataset_yaml(output):
    return (f"path: {json.dumps((Path(output) / 'court').resolve().as_posix())}\n"
            "train: images/train\nval: images/val\nnames:\n  0: court\nkpt_shape: [14, 3]\n"
            f"flip_idx: {json.dumps(FLIP_IDX)}\n")


def guarded_copy_resources(output, native_bytes):
    """Admit bounded byte copies with the established metadata128MiB floor."""
    if not hasattr(ctypes, "windll"):
        raise RuntimeError("Actual export RAM admission requires the declared Windows host")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    status = MemoryStatus()
    status.size = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot measure available host RAM")
    parent = Path(output)
    while not parent.exists():
        parent = parent.parent
    free_disk = shutil.disk_usage(parent).free
    if status.available < 128 * 1024**2:
        raise RuntimeError("Available RAM below unchanged metadata128MiB byte-copy floor")
    if free_disk < native_bytes + 1024**3:
        raise RuntimeError("Insufficient disk for native copies plus1GiB headroom")
    return {"available_ram_bytes": status.available, "free_disk_bytes": free_disk,
            "ram_floor_bytes": 128 * 1024**2, "disk_headroom_bytes": 1024**3}


def _unique(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate identity in reviewed records: " + key)
    return result


def validate_split_integrity(records, components, counts):
    expected = Counter(counts)
    expected.pop("test", None)
    if Counter(row["split"] for row in records) != expected:
        raise ValueError("Selected source split counts differ from exact frozen plan")
    _unique(records, "id")
    videos, groups, hashes = {}, {}, {}
    for row in records:
        if row["split"] not in {"train", "val"}:
            raise ValueError("Only train/DEV; no TEST samples")
        group = components[row["context_group"]]
        if row["split"] != group["development_split"] or row["source_video_id"] not in group["publisher_video_ids"]:
            raise ValueError("Source differs from complete transitive context assignment")
        if row["visual_event_family"] not in group["visual_event_families"]:
            raise ValueError("Dropped visual event family link")
        if any(family in {"MonteCarlo", "LondonFinals"} for family in group["visual_event_families"]) and row["split"] != "val":
            raise ValueError("Previous development event entered training")
        for mapping, key in ((videos, row["source_video_id"]), (groups, row["context_group"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Source video or complete context crossed splits")
            mapping[key] = row["split"]
        if row["source_image"]["sha256"] in hashes:
            raise ValueError("Exact duplicate native images cannot add independent examples")
        hashes[row["source_image"]["sha256"]] = row["id"]


def _load_consolidator(root):
    spec = importlib.util.spec_from_file_location("court175_frozen_consolidator", root / CONSOLIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_plan(output=None, *, root=ROOT, require_fresh=True):
    """Reconstruct all reviewed selections and old label bytes, without writes."""
    root = Path(root).resolve()
    output = workspace_path(output or root / DEFAULT_OUTPUT.relative_to(ROOT), root)
    if require_fresh and output.exists():
        raise FileExistsError("Fresh court repair output required; preserve prior attempts")
    bindings = {}
    def bind(path, expected=None):
        path = workspace_path(path, root)
        relative = path.relative_to(root).as_posix()
        observed = digest(path)
        if expected is not None and observed != expected or relative in bindings and bindings[relative] != observed:
            raise ValueError("Frozen source/review input changed: " + relative)
        bindings[relative] = observed
        return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}
    for path, expected in PINS.items():
        bind(path, expected)
    bind(Path(__file__))
    wave = read_json(root / WAVE)
    rebuilt = _load_consolidator(root).prepare_plan(root)
    if rebuilt != wave:
        raise ValueError("Frozen64 manifest differs from fresh full review/context reconstruction")
    for path, expected in wave["input_bindings"].items():
        bind(path, expected)
    old, frozen, inventory, previous, protocol = [read_json(root / p) for p in
        (OLD_DECISIONS, OLD_FROZEN, OLD_INVENTORY, OLD_EXPORT + "/source_manifest.json", OLD_PROTOCOL)]
    if any(previous.get(key) is not False for key in ("qualification_evidence", "source_independence_verified", "final_test_used")):
        raise ValueError("Historical export must remain unqualified development only")
    if frozen["decisions_sha256"] != PINS[OLD_DECISIONS] or frozen["inventory_sha256"] != PINS[OLD_INVENTORY]:
        raise ValueError("Historical frozen review lineage changed")
    approval = read_json(root / OLD_APPROVAL)
    if approval["status"] != "PASS_ROOT_SOURCE_AND_REVIEW_BINDINGS_PRIVATE_TRAINING_EXPORT_APPROVED" or approval["frozen_manifest_sha256"] != PINS[OLD_FROZEN]:
        raise ValueError("Historical root approval changed")
    for path in (OLD_EXPORT + "/export_receipt.json", OLD_EXPORT + "/training_plan.json",
                 OLD_EXPORT + "/post_export_verification.json", OLD_EXPORT + "/court/data.yaml"):
        bind(path)
    if previous["lineage"]["root_approval_sha256"] != PINS[OLD_APPROVAL]:
        raise ValueError("Historical export approval mismatch")
    details = read_json(root / old["native_detail_receipt"])
    bind(old["native_detail_receipt"], old["native_detail_receipt_sha256"])
    for item in old["manual_review_documents"]:
        bind(item["path"], item["sha256"])
    for board in details["boards"]:
        bind(board["image"], board["sha256"])
    inv = _unique(inventory["samples"], "id")
    reviewed = _unique(old["samples"], "id")
    if len(inv) != 165 or set(inv) != set(reviewed) or len(frozen["selected"]) != 120:
        raise ValueError("All165 previous accepted/rejected source links must remain")
    for row in inventory["samples"]:
        bind(row["source_image"], row["sha256"])
    for item in protocol.get("annotation_file_sha256", {}):
        bind("artifacts/validation/vision_upgrade/court_heatmap_training_source/" + item,
             protocol["annotation_file_sha256"][item])
    selected = []
    for split, limit in (("train", 100), ("val", 20)):
        candidates = sorted((r for r in old["samples"] if r["decision"] == "ACCEPT" and r["development_split"] == split),
            key=lambda r: hashlib.sha256(f"court-development-v1|{r['id']}".encode()).hexdigest())
        selected.extend(candidates[:limit])
    if selected != frozen["selected"]:
        raise ValueError("Historical selected120 differs from immutable capped review")
    if len(frozen["unused_accepted_development_ids"]) != 25 or len(frozen["rejected_ids"]) != 20:
        raise ValueError("Historical unused25/rejected20 development membership changed")
    previous_records = _unique(previous["samples"], "id")
    if set(previous_records) != {r["id"] for r in selected}:
        raise ValueError("Historical export membership changed")
    assignment = {video: (group_id, group["development_split"]) for group_id, group in wave["components"].items()
                  for video in group["publisher_video_ids"]}
    records = []
    for row in selected:
        sample, old_record = inv[row["id"]], previous_records[row["id"]]
        if row != reviewed[row["id"]] or assignment[row["source_video_id"]][1] != row["development_split"]:
            raise ValueError("Historical selected review changed or split migrated")
        points, box = row["reviewed_keypoints_px"], row["bbox_xyxy_px"]
        text = encode_label(box, points)
        if old_record["keypoints_px"] != points or old_record["bbox_xyxy_px"] != box:
            raise ValueError("Historical exported native coordinates changed")
        bind(old_record["image"], sample["sha256"])
        bind(old_record["label"], old_record["label_sha256"])
        if (root / old_record["label"]).read_bytes() != text.encode():
            raise ValueError("Historical label bytes no longer reconstruct exactly")
        source = bind(sample["source_image"], sample["sha256"])
        records.append({"id": row["id"], "origin": "PRESERVED_OLD120", "source_video_id": row["source_video_id"],
            "source_frame_ordinal": sample["source_frame_ordinal"], "source_image": source,
            "visual_event_family": row["visual_event_family"], "surface": row["surface"],
            "context_group": assignment[row["source_video_id"]][0], "prior_context_group": row["context_group"],
            "split": row["development_split"], "keypoints_px": points, "bbox_xyxy_px": box,
            "bbox_provenance": row["bbox_provenance"], "review_note": row["review_note"],
            "reviewer_type": "AGENT_VISUAL_REVIEW", "label_text": text,
            "publisher_annotation_file": sample["publisher_annotation_file"],
            "publisher_annotation_sha256": sample["publisher_annotation_sha256"]})
    aids = read_json(root / "artifacts/validation/yolo26_execution_2026-10-15/court_wave64_native_review_aids_v1/review_aids_manifest.json")
    new_inv = _unique(aids["samples"], "id")
    for row in wave["sources"]:
        if row["decision"] != "ACCEPT":
            continue
        points = [None if point is None else [*point, 2] for point in row["reviewed_keypoints_px"]]
        box = publisher_extent(row["publisher_keypoints_xy_px"])
        source = bind(row["source_image"]["path"], row["source_image"]["sha256"])
        sample = new_inv[row["id"]]
        records.append({"id": row["id"], "origin": "REVIEWED_NEW55_OF64", "source_video_id": row["source_video_id"],
            "source_frame_ordinal": row["source_frame_ordinal"], "source_image": source,
            "visual_event_family": row["visual_event_family"], "surface": row["surface"],
            "context_group": row["context_group"], "prior_context_group": None, "split": row["development_split"],
            "keypoints_px": points, "bbox_xyxy_px": box, "bbox_provenance": BBOX_PROVENANCE,
            "review_note": row["review_note"], "reviewer_type": "AGENT_VISUAL_REVIEW", "label_text": encode_label(box, points),
            "publisher_annotation_file": sample["publisher_annotation_file"],
            "publisher_annotation_sha256": sample["publisher_annotation_sha256"]})
    validate_split_integrity(records, wave["components"], COUNTS)
    for row in records:
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}_[0-9]+", row["id"]):
            raise ValueError("Invalid publisher source image identifier")
        png_geometry(root / row["source_image"]["path"])
    visible = sum(point is not None for r in records for point in r["keypoints_px"])
    if len(records) != 175 or visible != 2421 or 175 * 14 - visible != 29:
        raise ValueError("Combined175 visible2421/null29 lineage differs")
    return {"schema_version": "2.0", "status": "PREFLIGHT_NATIVE_COURT175_AWAITING_SEPARATE_ROOT_APPROVAL",
        "output": output.as_posix(), "counts": COUNTS, "keypoint_slots": {"visible": visible, "unknown_null": 29},
        "native_image_bytes": sum(row["source_image"]["bytes"] for row in records),
        "input_bindings": dict(sorted(bindings.items())), "records": records, "components": wave["components"],
        "prior_review_rows_retained": 165, "new_review_rows_retained": 64,
        "rejected_links_retained": True, "unused_previous_development_ids_excluded": frozen["unused_accepted_development_ids"],
        "source_timing_fps_match_and_camera_session": "UNKNOWN_NOT_INFERRED_FROM_PUBLISHER_FRAME_ORDINAL",
        "landmark_schema": "tennis_court_14_v1", "point_names": POINT_NAMES, "flip_idx": FLIP_IDX,
        "kpt_shape": [14, 3], "rights": protocol["rights"], "usage_scope": "PRIVATE_LOCAL_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT_ONLY",
        "raw_public_redistribution_cleared": False, "qualification_evidence": False,
        "independent_human_review": False, "source_independence_verified": False, "final_test_used": False,
        "training_performed": False, "pixels_decoded": False, "old_export_mutated": False,
        "bbox_definition_new55": BBOX_PROVENANCE, "yaml_text": dataset_yaml(output)}


def approval_template(plan):
    return {"status": "PENDING_ROOT_REVIEW_NOT_APPROVED", "plan_sha256": canonical_hash(plan),
        "output": plan["output"], "approved_ordered_ids": [r["id"] for r in plan["records"]],
        "counts": plan["counts"], "keypoint_slots": plan["keypoint_slots"], "independent_human_review": False,
        "qualification_evidence": False, "training_authorized_by_this_export_approval": False,
        "root_review_note": "", "reviewer": ""}


def validate_approval(approval, plan):
    required = approval_template(plan)
    required["status"] = APPROVED
    for key, value in required.items():
        if key not in {"root_review_note", "reviewer"} and approval.get(key) != value:
            raise ValueError("Missing or mismatched exact root approval field: " + key)
    if approval.get("reviewer") != "ROOT_AGENT" or not isinstance(approval.get("root_review_note"), str) or not approval["root_review_note"].strip():
        raise ValueError("Concrete root reviewer and review note required")


def _write_text(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _expected_manifest(plan, approval_binding):
    samples = []
    for row in plan["records"]:
        record = deepcopy(row)
        text = record.pop("label_text")
        record.update({"role": "court", "source_id": "court_image_" + row["id"],
            "image": f"court/images/{row['split']}/{row['id']}.png",
            "label": f"court/labels/{row['split']}/{row['id']}.txt",
            "label_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_pts": None, "source_time_base": None, "fps": None, "timestamp_seconds": None,
            "match_id": None, "camera_session_id": None, "context_id_is_verified_match_identity": False,
            "independent_human_review": False, "qualification_evidence": False})
        samples.append(record)
    return {"schema_version": "2.0", "status": "EXACT_NATIVE_COURT175_PRIVATE_DEVELOPMENT_NOT_QUALIFICATION",
        "scope": "elevated_end_court_singles", "counts": plan["counts"], "keypoint_slots": plan["keypoint_slots"],
        "samples": samples, "plan_sha256": canonical_hash(plan), "approval": approval_binding,
        "input_bindings": plan["input_bindings"], "components": plan["components"], "rights": plan["rights"],
        "raw_public_redistribution_cleared": False, "source_independence_verified": False,
        "independent_human_review": False, "qualification_evidence": False, "final_test_used": False,
        "training_performed": False, "native_images_decoded_during_export": False,
        "point_names": POINT_NAMES, "flip_idx": FLIP_IDX, "kpt_shape": [14, 3],
        "all_prior165_and_new64_context_links_retained": True, "prior_unused25_are_development_not_test": True}


def export_dataset(output, approval_path, approval_sha256, *, root=ROOT):
    plan = prepare_plan(output, root=root)
    root = Path(root).resolve()
    approval_path = workspace_path(approval_path, root)
    if not approval_sha256 or digest(approval_path) != approval_sha256:
        raise ValueError("Require separately frozen approval SHA256")
    validate_approval(read_json(approval_path), plan)
    approval_binding = {"path": _relative(approval_path, root), "sha256": approval_sha256}
    output = Path(plan["output"])
    if approval_path.is_relative_to(output):
        raise ValueError("Approval must exist separately before fresh export")
    guarded_copy_resources(output, plan["native_image_bytes"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_export_plan.json", plan)
    for row in plan["records"]:
        guarded_copy_resources(output, row["source_image"]["bytes"])
        image = output / "court" / "images" / row["split"] / (row["id"] + ".png")
        label = output / "court" / "labels" / row["split"] / (row["id"] + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        source = workspace_path(row["source_image"]["path"], root)
        with source.open("rb") as src, image.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if digest(image) != row["source_image"]["sha256"]:
            raise ValueError("Native image changed while copying; partial output preserved")
        _write_text(label, row["label_text"])
    _write_text(output / "court/data.yaml", plan["yaml_text"])
    manifest = _expected_manifest(plan, approval_binding)
    write_json(output / "source_manifest.json", manifest)
    # Rebuild/re-hash inputs before declaring success, preserving all partials.
    if prepare_plan(output, root=root, require_fresh=False) != plan:
        raise ValueError("Frozen input changed during export; no completion receipt")
    if digest(approval_path) != approval_sha256:
        raise ValueError("Root approval changed during export; no completion receipt")
    write_json(output / "export_receipt.json", {"status": "COMPLETE_EXACT_NATIVE_COURT175_NOT_QUALIFICATION",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "counts": plan["counts"],
        "keypoint_slots": plan["keypoint_slots"], "native_image_bytes": plan["native_image_bytes"],
        "source_manifest_sha256": digest(output / "source_manifest.json"), "plan_sha256": canonical_hash(plan),
        "approval_sha256": approval_sha256, "qualification_evidence": False, "training_performed": False,
        "gpu_used": False, "pixels_decoded": False, "all_input_hashes_unchanged": True})
    return verify_export(output, root=root)


def verify_export(output=DEFAULT_OUTPUT, *, root=ROOT, expected_manifest_sha256=None):
    """Read-only reconstruction, byte checks and exhaustive file membership."""
    root = Path(root).resolve()
    output = workspace_path(output, root)
    plan = prepare_plan(output, root=root, require_fresh=False)
    if read_json(output / "frozen_export_plan.json") != plan:
        raise ValueError("Saved export plan differs from current frozen review evidence")
    manifest_path = output / "source_manifest.json"
    if expected_manifest_sha256 is not None and digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Expected exported manifest hash changed")
    manifest = read_json(manifest_path)
    approval = manifest["approval"]
    approval_path = workspace_path(approval["path"], root)
    if digest(approval_path) != approval["sha256"]:
        raise ValueError("Export root approval bytes changed")
    validate_approval(read_json(approval_path), plan)
    if manifest != _expected_manifest(plan, approval):
        raise ValueError("Exported manifest differs from reconstructed approved native records")
    expected = {"source_manifest.json", "frozen_export_plan.json", "court/data.yaml", "export_receipt.json"}
    for row in plan["records"]:
        image = f"court/images/{row['split']}/{row['id']}.png"
        label = f"court/labels/{row['split']}/{row['id']}.txt"
        expected.update((image, label))
        if digest(output / image) != row["source_image"]["sha256"] or png_geometry(output / image) != (1280, 720):
            raise ValueError("Copied native PNG differs from immutable reviewed source")
        if (output / label).read_bytes() != row["label_text"].encode():
            raise ValueError("Exported court label bytes differ from reviewed reconstruction")
        validate_label((output / label).read_text(encoding="utf-8"), row["bbox_xyxy_px"], row["keypoints_px"])
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Missing, extra or linked file in immutable court175 export")
    if (output / "court/data.yaml").read_bytes() != plan["yaml_text"].encode():
        raise ValueError("Dataset YAML differs from exact court14 semantics/native paths")
    receipt = read_json(output / "export_receipt.json")
    for key, value in {"status": "COMPLETE_EXACT_NATIVE_COURT175_NOT_QUALIFICATION", "counts": plan["counts"],
            "keypoint_slots": plan["keypoint_slots"], "native_image_bytes": plan["native_image_bytes"],
            "source_manifest_sha256": digest(manifest_path), "plan_sha256": canonical_hash(plan),
            "approval_sha256": approval["sha256"], "qualification_evidence": False, "training_performed": False,
            "gpu_used": False, "pixels_decoded": False, "all_input_hashes_unchanged": True}.items():
        if receipt.get(key) != value:
            raise ValueError("Export completion receipt differs: " + key)
    return {"status": "PASS_RECONSTRUCTED_NATIVE_COURT175_EXPORT_NOT_QUALIFICATION", "counts": plan["counts"],
        "keypoint_slots": plan["keypoint_slots"], "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path), "input_hashes_verified": len(plan["input_bindings"]),
        "image_bytes_copied_exact": plan["native_image_bytes"], "files_verified": len(expected),
        "label_fields_per_instance": 47, "native_roundtrip_tolerance_px": 1e-6,
        "qualification_evidence": False, "training_performed": False, "pixels_decoded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "export", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--save-plan", type=Path)
    parser.add_argument("--save-approval-template", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.output)
        if args.save_plan:
            write_json(workspace_path(args.save_plan), result)
        if args.save_approval_template:
            write_json(workspace_path(args.save_approval_template), approval_template(result))
        print(json.dumps({k: result[k] for k in ("status", "counts", "keypoint_slots", "native_image_bytes")} |
                         {"plan_sha256": canonical_hash(result), "input_hashes_verified": len(result["input_bindings"])}))
    elif args.command == "export":
        if args.approval is None or args.approval_sha256 is None:
            parser.error("Export requires separate --approval and --approval-sha256")
        print(json.dumps(export_dataset(args.output, args.approval, args.approval_sha256)))
    else:
        print(json.dumps(verify_export(args.output)))


if __name__ == "__main__":
    main()

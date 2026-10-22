"""Preflight and explicitly approved private native640 ball-development export.

Import and preflight use only metadata and byte hashes: no image decoder, model,
CUDA, video runtime or source-frame resampling.  Preflight never writes files.
Execution requires a separately saved, exact-hash-bound ROOT approval.  The
original reviews, 600-item sampler and rejected crop decisions stay immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

from scripts.data.prepare_yolo26_reviewed_ball_wave import (
    ARTIFACTS, MANIFEST, PINNED as ORIGINAL_PINS, ROOT, SAMPLER,
    digest, guarded_resources, prepare_audit, read_json, translate_box, write_json,
)


PREPARATION = ARTIFACTS / "ball_reviewed_crop_preparation_v1"
CROP_MANIFEST = PREPARATION / "native_crop_review_boards.json"
DERIVED_MANIFEST = PREPARATION / "derived_development_manifest.json"
HEAD_REVIEW = PREPARATION / "crop_full_review_head_root_v1.json"
TAIL_REVIEW = PREPARATION / "crop_full_review_tail_v1.json"
PREPARER_SOURCE = ROOT / "scripts/data/prepare_yolo26_reviewed_ball_wave.py"
PINNED = {
    CROP_MANIFEST: "537d16626181ab74a7f332350d36bfd36c2efad9ee463aa8f3730924f04164cb",
    DERIVED_MANIFEST: "b94a4e0d6cfc3504817b9223abc5e43a2252fd5d826ea95bf2dd14e129b41110",
    HEAD_REVIEW: "770c8e9486aeede27fcad5de271fdab60153c851fa7abda0bb6c428ee228e14b",
    TAIL_REVIEW: "82c5ec2443d91619ab61e240f19011021195a1599bbf4e01dbfd211de0cbb2ff",
    PREPARER_SOURCE: "d9d7a94acdb457fcfd43d18d72e839d17f95df6d2cae19adab1f1779786a16ff",
}
ACCEPT = "ACCEPT_CROP_SINGLE_REVIEWED_BALL"
REJECT = "REJECT_CROP_UNANNOTATED_OR_AMBIGUOUS"
APPROVAL_STATUS = "ROOT_APPROVED_PRIVATE_SUPERVISED_BALL_DEVELOPMENT_EXPORT"


def _size(size):
    if type(size) is not int or size <= 0:
        raise ValueError("Image size must be a positive integer, not a boolean")
    return size


def _box(box, size):
    size = _size(size)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("Visible box needs four finite native coordinates")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
        raise ValueError("Visible box coordinates must be finite numbers, not booleans")
    if not (0 <= box[0] < box[2] <= size and 0 <= box[1] < box[3] <= size):
        raise ValueError("Visible box must fit inside crop without clipping")
    return list(box)


def yolo_line(box, size=640):
    box = _box(box, size)
    values = [(box[0] + box[2]) / (2 * size), (box[1] + box[3]) / (2 * size),
              (box[2] - box[0]) / size, (box[3] - box[1]) / size]
    line = "0 " + " ".join(f"{value:.10f}" for value in values) + "\n"
    restored = inverse_yolo_line(line, size)
    if max(abs(left - right) for left, right in zip(restored, box, strict=True)) > 1e-6:
        raise ValueError("YOLO serialization loses reviewed native coordinate precision")
    return line


def inverse_yolo_line(line, size=640):
    size = _size(size)
    if not isinstance(line, str):
        raise ValueError("YOLO label must be a text line")
    tokens = line.split()
    if len(tokens) != 5 or tokens[0] != "0":
        raise ValueError("Exactly one class0 tennis-ball box is required")
    try:
        x, y, width, height = map(float, tokens[1:])
    except ValueError as error:
        raise ValueError("YOLO coordinates must be numeric") from error
    if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("YOLO box must have finite positive extent")
    box = [(x - width / 2) * size, (y - height / 2) * size,
           (x + width / 2) * size, (y + height / 2) * size]
    return _box(box, size)


def validate_crop_reviews(crops, head_review, tail_review):
    """Require complete, ordered closure of all85 rendered positive crops."""
    if len(crops) != 85 or len({crop["proposal_id"] for crop in crops}) != 85:
        raise ValueError("The immutable reviewed-positive crop set must contain85 unique IDs")
    heads, tails = head_review.get("records", []), tail_review.get("records", [])
    if len(heads) != 44 or len(tails) != 41:
        raise ValueError("Root/head44 and tail41 complete-crop reviews are both required")
    accepted = []
    for index, (crop, review) in enumerate(zip(crops, heads + tails, strict=True)):
        if crop["proposal_id"] != review.get("proposal_id"):
            raise ValueError("Missing, duplicate, reordered or extra complete-crop review ID")
        if type(review.get("accepted_crop_manifest_index")) is not int or review["accepted_crop_manifest_index"] != index:
            raise ValueError("Complete-crop review index differs from frozen85crop order")
        for review_key, crop_key in (("crop_path", "path"), ("crop_sha256", "sha256"),
                ("board_path", "board_path"), ("board_sha256", "board_sha256"),
                ("source_crop_xyxy_native_px", "source_crop_xyxy_native_px"),
                ("preserved_main_bbox_xyxy_native_px", "bbox_xyxy_native_px"),
                ("preserved_main_bbox_xyxy_crop_px", "bbox_xyxy_crop_px")):
            if review.get(review_key) != crop.get(crop_key):
                raise ValueError(f"Complete-crop review altered {review_key}")
        if crop.get("derived_split") not in {"train", "development"}:
            raise ValueError("Only corrected train/development crops are eligible; TEST is forbidden")
        for key in ("full_current_crop_actually_viewed", "all_crop_pixels_scanned"):
            if review.get(key) is not True:
                raise ValueError("Every pixel of the complete640crop needs actual visual review")
        if review.get("review_scope_native_xyxy") != [0, 0, 640, 640]:
            raise ValueError("Incomplete or resized review scope cannot close a native640crop")
        for key in ("independent_human_review", "qualification_evidence", "main_bbox_changed", "supervised_export_approved"):
            if review.get(key) is not False:
                raise ValueError("Review must preserve boxes and agent-only unapproved development status")
        if type(review.get("additional_ball_boxes_created")) is not int or review["additional_ball_boxes_created"] != 0:
            raise ValueError("No additional guessed ball boxes may enter this export")
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Actual visual reviewer attribution is required")
        if review.get("reviewer_type") not in {"AGENT", "ROOT_AGENT_VISUAL_REVIEW"}:
            raise ValueError("Unknown crop visual-review attribution")
        if type(review.get("root_agent_actually_viewed_this_crop")) is not bool:
            raise ValueError("Preserve root-versus-tail actual visual-review attribution")
        if review.get("decision") not in {ACCEPT, REJECT}:
            raise ValueError("Unknown or pending complete-crop review decision")
        _box(crop["bbox_xyxy_crop_px"], 640)
        if translate_box(crop["bbox_xyxy_crop_px"], crop["source_crop_xyxy_native_px"][:2], inverse=True) != crop["bbox_xyxy_native_px"]:
            raise ValueError("Reviewed native/crop visible box inverse differs")
        if review["decision"] == ACCEPT:
            accepted.append({"crop": deepcopy(crop), "review": deepcopy(review)})
    if len(accepted) != 75:
        raise ValueError("Exactly75 closed positive crops are expected before root final approval")
    return accepted


def verify_root_approval(approval, binding_hashes, expected_positive_ids, expected_negative_ids):
    if approval.get("status") != APPROVAL_STATUS or approval.get("reviewer_type") != "ROOT_AGENT_VISUAL_REVIEW":
        raise ValueError("Export requires the separate final ROOT visual-review approval")
    for key in ("qualification_evidence", "independent_human_review"):
        if approval.get(key) is not False:
            raise ValueError("ROOT agent approval cannot assert independent human qualification")
    if type(approval.get("test_count")) is not int or approval["test_count"] != 0:
        raise ValueError("Development export must contain no final TEST data")
    if approval.get("input_bindings") != binding_hashes:
        raise ValueError("Final ROOT approval must bind every exact proposed input artifact hash")
    if approval.get("approved_positive_proposal_ids") != expected_positive_ids:
        raise ValueError("Final ROOT approval must cover the exact ordered75 positive IDs")
    if approval.get("approved_negative_proposal_ids") != expected_negative_ids:
        raise ValueError("Final ROOT approval must cover only the exact4 accepted absence tiles")
    if not isinstance(approval.get("review_note"), str) or not approval["review_note"].strip():
        raise ValueError("Final ROOT approval needs a concrete review note")


def validate_destination(output, allowed_root=None, *, require_fresh=True):
    output = Path(output).resolve()
    allowed_root = (ROOT / "data/yolo26" if allowed_root is None else Path(allowed_root)).resolve()
    if output == allowed_root or not output.is_relative_to(allowed_root):
        raise ValueError("Fresh ball export must stay within the named workspace data/yolo26 root")
    if require_fresh and output.exists():
        raise FileExistsError("Refusing to overwrite or reuse any existing dataset destination")
    return output


def copy_and_verify(source, destination, expected_sha256):
    """Copy bytes exclusively, failing before any source/hash substitution."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing copied image")
    if digest(source) != expected_sha256:
        raise ValueError("Source image changed before byte-identical copy")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
    copied = digest(destination)
    if copied != expected_sha256 or digest(source) != expected_sha256:
        raise ValueError("Copied image or source changed during copy")
    return copied


def _native640_header(path):
    with Path(path).open("rb") as source:
        header = source.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Require retained native PNG image bytes")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    if (width, height, depth, color, compression, filtering, interlace) != (640, 640, 8, 2, 0, 0, 0):
        raise ValueError("Require unchanged RGB8 native640 PNG without resizing")


def preflight(output, approval_path=None, approval_sha256=None, *, allow_existing=False):
    # Existing output is admitted only for this read-only reconstruction.  The
    # mutating execute_export entry point always requires a fresh destination.
    output = validate_destination(output, require_fresh=not allow_existing)
    bindings = {str(path): expected for path, expected in ORIGINAL_PINS.items()}
    bindings.update({str(path): expected for path, expected in PINNED.items()})
    bindings[str(Path(__file__).resolve())] = digest(__file__)
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Pinned export input changed: {path}")
    original = prepare_audit()  # No PIL, cv2, torch or output writes.
    derived, rendered, head, tail = map(read_json, (DERIVED_MANIFEST, CROP_MANIFEST, HEAD_REVIEW, TAIL_REVIEW))
    preparer_binding = {"path": str(PREPARER_SOURCE), "sha256": PINNED[PREPARER_SOURCE]}
    if derived["implementation"] != preparer_binding or rendered["implementation"] != preparer_binding:
        raise ValueError("Retained crop/source/group preparation implementation differs")
    for key in ("source_bindings", "protocol", "counts", "components", "derived_source_provenance",
                "positive_observations", "effective_root_negative_observations", "derived_all600_proposal_provenance"):
        if original[key] != derived[key]:
            raise ValueError(f"Derived review graph/source/PTS lineage changed: {key}")
    if rendered["derived_manifest"] != {"path": str(DERIVED_MANIFEST), "sha256": PINNED[DERIVED_MANIFEST]}:
        raise ValueError("Native crop preparation binds another derived graph")
    for review in (head, tail):
        if review["crop_preparation_manifest"] != {"path": str(CROP_MANIFEST), "sha256": PINNED[CROP_MANIFEST]}:
            raise ValueError("Complete crop review binds another native crop preparation")
        if review["derived_development_manifest"] != rendered["derived_manifest"]:
            raise ValueError("Complete crop review binds another source/group decision")
        if review.get("independent_human_review") is not False or review.get("supervised_export_approved") is not False:
            raise ValueError("Original complete-crop reviews must retain agent-only unapproved status")
    by_id = {row["proposal_id"]: row for row in derived["positive_observations"]}
    crops = deepcopy(rendered["crops"])
    if [crop["proposal_id"] for crop in crops] != [row["proposal_id"] for row in derived["positive_observations"] if row["decision"] == "ACCEPT_VISIBLE_BOX"]:
        raise ValueError("Rendered85crops differ from original variable manual boxes")
    for board in rendered["boards"]:
        if digest(board["path"]) != board["sha256"]:
            raise ValueError("Complete visual-review board bytes changed")
    board_by_id = {}
    for board in rendered["boards"]:
        for cell in board["cells"]:
            if cell["proposal_id"] in board_by_id:
                raise ValueError("Native crop appears on more than one review board")
            board_by_id[cell["proposal_id"]] = (board, cell)
    for crop in crops:
        row = by_id[crop["proposal_id"]]
        if digest(crop["path"]) != crop["sha256"]:
            raise ValueError("Reviewed native crop bytes changed")
        _native640_header(crop["path"])
        if crop["bbox_xyxy_native_px"] != row["bbox_xyxy_native_px"] or crop["source_crop_xyxy_native_px"] != row["proposed_native_crop"]["xyxy_native_px"]:
            raise ValueError("Rendered crop changed original visible box or exact jittered inverse")
        if crop["native_frame"] != row["native_frame"] or crop.get("resized") is not False:
            raise ValueError("Rendered crop native image/source geometry differs")
        board, cell = board_by_id[crop["proposal_id"]]
        if cell["crop_sha256"] != crop["sha256"] or cell["crop_path"] != crop["path"] or cell["source_crop_xyxy_native_px"] != crop["source_crop_xyxy_native_px"] or cell["scale"] != 1:
            raise ValueError("Board cell does not bind exact native crop pixels")
        crop.update({"board_path": board["path"], "board_sha256": board["sha256"],
                     "derived_split": row["derived_split"]})
    accepted = validate_crop_reviews(crops, head, tail)
    negatives = [row for row in derived["effective_root_negative_observations"] if row["effective_decision"] == "ACCEPT_VISIBLE_ABSENCE"]
    if len(negatives) != 4:
        raise ValueError("Only the four already root-reviewed negative tiles are eligible")
    sampler = read_json(SAMPLER)
    source_by_id = {row["source_id"]: row for row in sampler["sources"]}
    proposed = []

    def source_row(ident, source_id, frame, pts, time_base, split, image, crop_box, local_box, evidence):
        if split not in {"train", "development"}:
            raise ValueError("Quarantined, unsupported or TEST data cannot enter export")
        source = source_by_id[source_id]
        if source["source_pts"][frame] != pts or source["source_time_base"] != time_base:
            raise ValueError("Export source frame/PTS differs from immutable native timeline")
        subdir = "train" if split == "train" else "val"
        line = yolo_line(local_box) if local_box is not None else ""
        native_box = translate_box(local_box, crop_box[:2], inverse=True) if local_box is not None else None
        return {"sample_id": ident, "source_id": source_id, "source_type": "NATIVE_PTS_VIDEO_FRAME_EXACT_640_CROP",
            "development_split": subdir, "derived_split": split,
            "image_source_path": image["path"], "image_source_sha256": image["sha256"],
            "image_path": str(output / "ball/images" / subdir / (ident + ".png")),
            "label_path": str(output / "ball/labels" / subdir / (ident + ".txt")),
            "label_text": line, "label_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "box_xyxy_crop_px": local_box, "box_xyxy_native_px": native_box,
            "crop_xyxy_native_px": crop_box, "crop_size_px": [640, 640], "resized": False,
            "negative_visible_absence_only": local_box is None, "source_frame_index": frame,
            "source_pts": pts, "source_time_base": time_base,
            "source_timestamp_fraction_seconds": str(pts * Fraction(time_base)),
            "fps_rational": source["average_frame_rate"], "native_resolution": [1920, 1080],
            "source_video_path": source["video_path"], "source_video_sha256": source["video_sha256"],
            "source_url": source["video_url"], "source_timeline_sha256": source["timeline_sha256"],
            "publisher_csv_path": source["publisher_csv_path"], "publisher_csv_sha256": source["publisher_csv_sha256"],
            "frozen_event_alias": source["conservative_event_group"],
            "derived_component_id": next(c["component_id"] for c in derived["components"] if source_id in c["source_ids"]),
            "match_id": None, "session_id": None, "actual_match_identity_verified": False,
            "actual_session_identity_verified": False, "source_independence_verified": False,
            "training_allowed": True, "allowed_use": "PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT",
            "raw_broadcast_redistribution_cleared": False, "publisher_license_evidence": source["dataset_card_audit"],
            "review_evidence": evidence, "independent_human_review": False, "qualification_evidence": False}

    for accepted_row in accepted:
        crop, review = accepted_row["crop"], accepted_row["review"]
        row = by_id[crop["proposal_id"]]
        proposed.append(source_row(row["proposal_id"], row["source_id"], row["frame_index"], row["source_pts"], row["source_time_base"], row["derived_split"],
            {"path": crop["path"], "sha256": crop["sha256"]}, crop["source_crop_xyxy_native_px"], crop["bbox_xyxy_crop_px"],
            {"original_manual_point_box_observation": row["original_observation"], "complete_crop_review": review}))
    for row in negatives:
        observation = row["original_observation"]
        image = observation["actually_viewed_full_native_tile"]
        if digest(image["path"]) != image["sha256"]:
            raise ValueError("Root accepted absence crop changed")
        _native640_header(image["path"])
        origin = observation["tile_origin_xy_px"]
        proposed.append(source_row(row["proposal_id"], row["source_id"], row["frame_index"], row["source_pts"], row["source_time_base"], row["derived_split"], image,
            [origin[0], origin[1], origin[0] + 640, origin[1] + 640], None, deepcopy(row)))
    positive_ids = [row["crop"]["proposal_id"] for row in accepted]
    negative_ids = [row["proposal_id"] for row in negatives]
    counts = {"images": len(proposed), "positive_boxes": len(accepted), "negative_tiles": len(negatives),
        "train": sum(row["development_split"] == "train" for row in proposed),
        "val": sum(row["development_split"] == "val" for row in proposed), "test": 0,
        "positive_split_counts": dict(Counter(row["derived_split"] for row in proposed if not row["negative_visible_absence_only"]))}
    if counts != {"images": 79, "positive_boxes": 75, "negative_tiles": 4, "train": 58, "val": 21, "test": 0,
                  "positive_split_counts": {"train": 55, "development": 20}}:
        raise ValueError(f"Reviewed export counts unexpectedly differ: {counts}")
    splits_by_component = defaultdict(set)
    for row in proposed:
        splits_by_component[row["derived_component_id"]].add(row["development_split"])
    if any(len(values) != 1 for values in splits_by_component.values()):
        raise ValueError("A conservative transitive source/event component crosses train/val")
    approval = None
    if approval_path is not None:
        if not isinstance(approval_sha256, str) or digest(approval_path) != approval_sha256:
            raise ValueError("Require exact independently recorded final ROOT approval SHA256")
        approval = read_json(approval_path)
        verify_root_approval(approval, bindings, positive_ids, negative_ids)
    elif approval_sha256 is not None:
        raise ValueError("Approval digest without approval file is invalid")
    result = {"schema_version": "tennis.private-reviewed-ball-development-export.v1",
        "status": "ROOT_APPROVED_READY_TO_EXPORT" if approval is not None else "PREFLIGHT_ONLY_AWAITING_FINAL_ROOT_APPROVAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
        "input_bindings": bindings, "proposed_samples": proposed, "counts": counts,
        "components": deepcopy(derived["components"]), "derived_source_provenance": deepcopy(derived["derived_source_provenance"]),
        "context_lineage_includes_rejected_and_low_view": True, "labels_exported": 0,
        "match_ids_fabricated": False, "session_ids_fabricated": False,
        "independent_human_review": False, "source_independence_verified": False,
        "qualification_evidence": False, "final_test_used": False,
        "root_final_approval": {"path": str(Path(approval_path).resolve()), "sha256": approval_sha256} if approval is not None else None,
        "approval_template": {"status": APPROVAL_STATUS, "reviewer_type": "ROOT_AGENT_VISUAL_REVIEW",
            "qualification_evidence": False, "independent_human_review": False, "test_count": 0,
            "input_bindings": bindings, "approved_positive_proposal_ids": positive_ids,
            "approved_negative_proposal_ids": negative_ids, "review_note": ""}}
    return result


def prepare_plan(output=None, approval_path=None, approval_sha256=None):
    """Public pure pre-export plan for training/diagnostic protocol freezing."""
    output = ROOT / "data/yolo26/ball_reviewed_development_v1" if output is None else output
    return preflight(output, approval_path, approval_sha256)


def _data_yaml(output):
    return "path: " + json.dumps(str(Path(output) / "ball")) + "\ntrain: images/train\nval: images/val\nnames:\n  0: tennis_ball\n"


def verify_export(output, expected_manifest_sha256=None):
    """Reconstruct approved provenance and verify every existing image/label.

    This is a read-only metadata/byte verifier; it never creates a directory or
    invokes an image decoder.  Source/model qualification is not inferred from
    byte integrity or agent visual-development annotation.
    """
    output = validate_destination(output, require_fresh=False)
    manifest_path = output / "source_manifest.json"
    receipt_path = output / "export_receipt.json"
    manifest_hash = digest(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Exported manifest differs from the externally pinned dataset version")
    manifest, receipt = read_json(manifest_path), read_json(receipt_path)
    if manifest.get("schema_version") != "tennis.private-reviewed-ball-development-export.v1" or manifest.get("status") != "EXPORTED_PRIVATE_AGENT_REVIEWED_BALL_DEVELOPMENT_NOT_QUALIFICATION":
        raise ValueError("Existing dataset lacks the exact approved development-export contract")
    approval = manifest.get("root_final_approval")
    if not isinstance(approval, dict) or set(approval) != {"path", "sha256"}:
        raise ValueError("Existing export lacks a separately saved hash-bound ROOT final approval")
    expected = preflight(output, approval["path"], approval["sha256"], allow_existing=True)
    for key, value in expected.items():
        if key not in {"status", "created_at_utc", "labels_exported"} and manifest.get(key) != value:
            raise ValueError(f"Existing export metadata differs from its fully reconstructed reviewed plan: {key}")
    if type(manifest.get("labels_exported")) is not int or manifest["labels_exported"] != 79:
        raise ValueError("Existing export label count differs from reviewed79samples")
    for key in ("model_used", "gpu_used", "training_performed"):
        if manifest.get(key) is not False:
            raise ValueError("Dataset export metadata cannot claim training or model execution")
    expected_sources = [deepcopy(row) | {"image_copied_byte_identically": True} for row in expected["proposed_samples"]]
    if manifest.get("sources") != expected_sources:
        raise ValueError("Exported source/image/PTS/box/reviewer inventory differs from reviewed plan")
    seen = set()
    for row in expected_sources:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate exported sample ID")
        seen.add(row["sample_id"])
        image_path, label_path = Path(row["image_path"]), Path(row["label_path"])
        if digest(image_path) != row["image_source_sha256"] or digest(row["image_source_path"]) != row["image_source_sha256"]:
            raise ValueError("Exported/source native crop bytes differ")
        _native640_header(image_path)
        label_bytes = label_path.read_bytes()
        if hashlib.sha256(label_bytes).hexdigest() != row["label_sha256"] or label_bytes != row["label_text"].encode():
            raise ValueError("Exported label bytes differ from exact reviewed YOLO serialization")
        if row["negative_visible_absence_only"]:
            if label_bytes or row["box_xyxy_crop_px"] is not None or row["box_xyxy_native_px"] is not None:
                raise ValueError("Only approved absence tiles may have empty labels")
        else:
            decoded = inverse_yolo_line(label_bytes.decode())
            if max(abs(left - right) for left, right in zip(decoded, row["box_xyxy_crop_px"], strict=True)) > 1e-6:
                raise ValueError("Existing label no longer reproduces exact reviewed crop coordinates")
            restored = translate_box(decoded, row["crop_xyxy_native_px"][:2], inverse=True)
            if max(abs(left - right) for left, right in zip(restored, row["box_xyxy_native_px"], strict=True)) > 1e-6:
                raise ValueError("Existing label inverse no longer reproduces reviewed native coordinates")
    for split in ("train", "val"):
        rows = [row for row in expected_sources if row["development_split"] == split]
        if {path.name for path in (output / "ball/images" / split).iterdir()} != {row["sample_id"] + ".png" for row in rows}:
            raise ValueError("Unexpected, missing or unreviewed image in exported split")
        if {path.name for path in (output / "ball/labels" / split).iterdir()} != {row["sample_id"] + ".txt" for row in rows}:
            raise ValueError("Unexpected, missing or unreviewed label in exported split")
    if (output / "ball/images/test").exists() or (output / "ball/labels/test").exists():
        raise ValueError("Final TEST data is forbidden in reviewed-development export")
    yaml_path = output / "ball/data.yaml"
    if yaml_path.read_text(encoding="utf-8") != _data_yaml(output):
        raise ValueError("Dataset YAML points outside the exact reviewed train/val export")
    if receipt.get("status") != "PASS_BYTE_IDENTICAL_NATIVE640_AND_EXACT_YOLO_INVERSE_PRIVATE_DEVELOPMENT_EXPORT" or receipt.get("counts") != expected["counts"]:
        raise ValueError("Missing or inconsistent terminal export receipt")
    if receipt.get("source_manifest_sha256") != manifest_hash or receipt.get("data_yaml_sha256") != digest(yaml_path) or receipt.get("root_final_approval") != approval or receipt.get("input_files_changed") != []:
        raise ValueError("Terminal export receipt does not bind current manifest/YAML/approval/source hashes")
    for key in ("qualification_evidence", "training_performed", "gpu_used", "model_used"):
        if receipt.get(key) is not False:
            raise ValueError("Export receipt cannot claim model execution or independent qualification")
    return {"status": "PASS_EXISTING_PRIVATE_DEVELOPMENT_EXPORT_INTEGRITY_NOT_QUALIFICATION",
            "manifest": manifest, "counts": expected["counts"], "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash, "samples": expected_sources,
            "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path),
            "data_yaml_path": str(yaml_path), "data_yaml_sha256": digest(yaml_path),
            "qualification_evidence": False, "independent_human_review": False}


def execute_export(proposal):
    """Mutate only after approved preflight; never called by default CLI."""
    if proposal.get("status") != "ROOT_APPROVED_READY_TO_EXPORT" or not proposal.get("root_final_approval"):
        raise ValueError("Unapproved preflight cannot write supervised data")
    output = validate_destination(proposal["output"])
    admission = guarded_resources()
    for path, expected in proposal["input_bindings"].items():
        if digest(path) != expected:
            raise ValueError("Approved input changed before export")
    approval = proposal["root_final_approval"]
    if digest(approval["path"]) != approval["sha256"]:
        raise ValueError("Final approval changed before export")
    # Recreate the whole proposed manifest so caller mutations cannot bypass its
    # source, review, count, component or hash contracts.
    verified = preflight(output, approval["path"], approval["sha256"])
    if proposal["proposed_samples"] != verified["proposed_samples"] or proposal["counts"] != verified["counts"]:
        raise ValueError("Approved proposed sample set changed before execution")
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "ball/images" / split).mkdir(parents=True)
        (output / "ball/labels" / split).mkdir(parents=True)
    records = []
    for row in verified["proposed_samples"]:
        guarded_resources()
        copy_and_verify(row["image_source_path"], row["image_path"], row["image_source_sha256"])
        with Path(row["label_path"]).open("x", encoding="utf-8", newline="\n") as label:
            label.write(row["label_text"])
        if digest(row["label_path"]) != row["label_sha256"]:
            raise ValueError("Written YOLO label hash differs")
        records.append(deepcopy(row) | {"image_copied_byte_identically": True})
    yaml = _data_yaml(output)
    (output / "ball/data.yaml").write_text(yaml, encoding="utf-8")
    manifest = deepcopy(verified)
    manifest.update({"status": "EXPORTED_PRIVATE_AGENT_REVIEWED_BALL_DEVELOPMENT_NOT_QUALIFICATION",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(), "labels_exported": 79,
        "sources": records, "export_launch_resources": admission, "model_used": False,
        "gpu_used": False, "training_performed": False})
    # Include the original642 media/CSV/card/PTS/review bindings, not only the
    # compact top-level hashes approved transitively through the derived file.
    checked_inputs = dict(read_json(DERIVED_MANIFEST)["source_bindings"])
    checked_inputs.update(verified["input_bindings"])
    changed = [path for path, expected in checked_inputs.items() if digest(path) != expected]
    if changed:
        raise ValueError(f"Input changed during export: {changed}")
    write_json(output / "source_manifest.json", manifest)
    receipt = {"status": "PASS_BYTE_IDENTICAL_NATIVE640_AND_EXACT_YOLO_INVERSE_PRIVATE_DEVELOPMENT_EXPORT",
        "counts": verified["counts"], "source_manifest_sha256": digest(output / "source_manifest.json"),
        "data_yaml_sha256": digest(output / "ball/data.yaml"), "input_files_changed": changed,
        "input_file_bindings_rechecked_after_copy": len(checked_inputs),
        "root_final_approval": approval, "qualification_evidence": False, "training_performed": False,
        "gpu_used": False, "model_used": False}
    write_json(output / "export_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/yolo26/ball_reviewed_development_v1")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--execute", action="store_true", help="Write only after exact final ROOT approval; otherwise read-only preflight")
    args = parser.parse_args(argv)
    proposal = preflight(args.output, args.approval, args.approval_sha256)
    if args.execute:
        if not args.approval:
            raise ValueError("--execute requires a separately saved final ROOT approval")
        print(json.dumps(execute_export(proposal)))
    else:
        print(json.dumps({key: proposal[key] for key in ("status", "output", "counts", "input_bindings", "approval_template")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

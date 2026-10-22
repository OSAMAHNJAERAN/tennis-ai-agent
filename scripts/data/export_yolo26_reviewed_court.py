"""Freeze and export agent-reviewed court development data without decoding images.

The original capped protocol remains immutable. Export requires a hash-bound
root second-pass approval, preserves native image bytes and never invents FPS,
match identity, hidden keypoints, an independent test split or qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.data.prepare_yolo26_court_review import (
    FLIP_IDX, POINT_NAMES, ROOT, context_components, load, relative, save, sha,
    validate_review,
)


def validate_native_evidence(review, decisions):
    inventory_path = review / "cache_inventory.json"
    if decisions.get("inventory_sha256") != sha(inventory_path):
        raise ValueError("Review must bind the immutable native inventory")
    receipt_path = ROOT / decisions["native_detail_receipt"]
    if sha(receipt_path) != decisions.get("native_detail_receipt_sha256"):
        raise ValueError("Native detail receipt changed")
    receipt = load(receipt_path)
    if receipt.get("inventory_sha256") != sha(inventory_path):
        raise ValueError("Native detail boards refer to another inventory")
    manual_documents = {item["path"]: item["sha256"] for item in decisions["manual_review_documents"]}
    if any(sha(ROOT / path) != digest for path, digest in manual_documents.items()):
        raise ValueError("Recorded manual image observations changed")
    board_by_id = {}
    for board in receipt["boards"]:
        if sha(ROOT / board["image"]) != board["sha256"]:
            raise ValueError("Reviewed native detail board bytes changed")
        for ident in board["sample_ids"]:
            if ident in board_by_id:
                raise ValueError("Duplicate native detail image assignment")
            indices = sorted(mapping["point_index"] for mapping in board["coordinate_maps"]
                             if mapping["id"] == ident and mapping["scale"] == 1)
            if indices != list(range(14)):
                raise ValueError("Every semantic slot requires a native 1:1 crop")
            board_by_id[ident] = board
    for row in decisions["samples"]:
        if manual_documents.get(row.get("manual_review_document")) != row.get("manual_review_document_sha256"):
            raise ValueError("Image decisions must bind their actual manual observations")
        board = board_by_id[row["id"]]
        if row.get("native_detail_board") != board["image"] or row.get("native_detail_board_sha256") != board["sha256"]:
            raise ValueError("Image review does not bind its native detail board")
        if row.get("native_patch_reviewed_indices") != list(range(14)):
            raise ValueError("Record actual review of all fourteen native patches")
        if row["decision"] == "ACCEPT":
            expected = ["UNKNOWN_NULL" if point is None else "VISIBLE_2"
                        for point in row["reviewed_keypoints_px"]]
            if row.get("keypoint_visibility_decisions") != expected:
                raise ValueError("Explicit per-slot visibility and coordinates disagree")
            if any(point is None for point in row["reviewed_keypoints_px"]) and not row.get("null_visibility_note"):
                raise ValueError("Unknown landmark slots need a concrete occlusion/uncertainty note")


def selected_rows(accepted):
    selected, surplus = [], []
    for split, cap in (("train", 100), ("val", 20)):
        candidates = sorted((row for row in accepted if row["development_split"] == split),
                            key=lambda row: hashlib.sha256(f"court-development-v1|{row['id']}".encode()).hexdigest())
        selected.extend(candidates[:cap])
        surplus.extend(row["id"] for row in candidates[cap:])
    return selected, surplus


def freeze_reviewed(review: Path, decisions_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Create a fresh frozen court development version")
    inventory, decisions = load(review / "cache_inventory.json"), load(decisions_path)
    validate_native_evidence(review, decisions)
    accepted, context_splits = validate_review(inventory, decisions)
    selected, surplus = selected_rows(accepted)
    _, contexts = context_components(inventory["samples"], decisions["samples"])
    counts = dict(Counter(row["development_split"] for row in selected)) | {"test": 0}
    manifest = {
        "schema_version": "1.1", "status": "FROZEN_AGENT_REVIEWED_DEVELOPMENT_ONLY_AWAITING_ROOT_REVIEW",
        "qualification_evidence": False, "independent_human_reviewed": False,
        "source_independence_verified": False, "final_test_used": False,
        "protocol_sha256": sha(review / "predeclared_group_protocol.json"),
        "inventory_sha256": sha(review / "cache_inventory.json"), "decisions_sha256": sha(decisions_path),
        "native_detail_receipt_sha256": decisions["native_detail_receipt_sha256"],
        "context_splits": context_splits, "context_groups": contexts,
        "context_lineage_includes_rejected_samples": True,
        "selection_rule": "PREDECLARED_HASH_ORDER_CAP_100_TRAIN_20_VAL_WITHIN_INDIVISIBLE_CONTEXT_ASSIGNMENTS",
        "selected": selected, "unused_accepted_development_ids": surplus,
        "counts": counts, "all_accepted_counts": dict(Counter(row["development_split"] for row in accepted)),
        "rejected_ids": [row["id"] for row in decisions["samples"] if row["decision"] == "REJECT"],
        "all_accepted_null_keypoints": sum(point is None for row in accepted for point in row["reviewed_keypoints_px"]),
        "selected_null_keypoints": sum(point is None for row in selected for point in row["reviewed_keypoints_px"]),
        "root_second_pass_before_export": True, "yolo_exported": False,
    }
    output.mkdir(parents=True)
    save(output / "grouped_development_manifest.json", manifest)
    return {"output": str(output), "counts": counts, "all_accepted_counts": manifest["all_accepted_counts"],
            "rejected": len(manifest["rejected_ids"]), "surplus_unused_development": len(surplus),
            "context_groups": len(contexts), "manifest_sha256": sha(output / "grouped_development_manifest.json")}


def encode_court_label(sample, row):
    """One image-normalized court with fourteen fixed slots; unknown is 0,0,0."""
    width, height = sample["width"], sample["height"]
    x1, y1, x2, y2 = row["bbox_xyxy_px"]
    if not 0 <= x1 < x2 < width or not 0 <= y1 < y2 < height:
        raise ValueError("Reviewed court extent must lie inside native image")
    values = [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
              (x2 - x1) / width, (y2 - y1) / height]
    points = row["reviewed_keypoints_px"]
    if len(points) != 14:
        raise ValueError("Exactly fourteen fixed semantic slots required")
    for point in points:
        if point is None:
            values.extend((0, 0, 0))
            continue
        x, y, visibility = point
        if visibility != 2 or not (x1 <= x <= x2 and y1 <= y <= y2) or not all(math.isfinite(v) for v in point):
            raise ValueError("Visible landmark must be finite and inside reviewed court extent")
        values.extend((x / width, y / height, 2))
    return " ".join(format(value, ".12g") for value in values) + "\n"


def validate_approval(approval, frozen, frozen_path, decisions_path):
    status = approval.get("status")
    allowed = {"ROOT_AGENT_SECOND_PASS_APPROVED_FOR_DEVELOPMENT_EXPORT",
               "PASS_ROOT_SOURCE_AND_REVIEW_BINDINGS_PRIVATE_TRAINING_EXPORT_APPROVED"}
    if status not in allowed:
        raise ValueError("Export requires concrete root second-pass approval")
    if approval.get("frozen_manifest_sha256") != sha(frozen_path) or approval.get("decisions_sha256") != sha(decisions_path):
        raise ValueError("Approval must bind exact frozen manifest and decisions")
    if status == "ROOT_AGENT_SECOND_PASS_APPROVED_FOR_DEVELOPMENT_EXPORT":
        if approval.get("approved_selected_ids") != [row["id"] for row in frozen["selected"]]:
            raise ValueError("Approval must cover exact ordered selected image IDs")
        human_review = approval.get("independent_human_reviewed")
        note = approval.get("review_note")
    else:
        # This root receipt approves the entire exact hash-bound frozen document.
        if approval.get("selected") != len(frozen["selected"]):
            raise ValueError("Root receipt selected count differs from frozen image IDs")
        human_review = approval.get("independent_human_review")
        note = approval.get("root_visual_scope_review")
    if human_review is not False or not note:
        raise ValueError("Agent second pass cannot claim independent human qualification")


def export_reviewed(review: Path, decisions_path: Path, frozen_path: Path, approval_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Never overwrite a court export")
    inventory, decisions, frozen, approval = (load(path) for path in
                                             (review / "cache_inventory.json", decisions_path, frozen_path, approval_path))
    validate_native_evidence(review, decisions)
    accepted, context_splits = validate_review(inventory, decisions)
    selected, surplus = selected_rows(accepted)
    if frozen.get("selected") != selected or frozen.get("context_splits") != context_splits or frozen.get("unused_accepted_development_ids") != surplus:
        raise ValueError("Frozen selection differs from actual reviewed capped groups")
    if (frozen.get("decisions_sha256") != sha(decisions_path)
            or frozen.get("inventory_sha256") != sha(review / "cache_inventory.json")
            or frozen.get("protocol_sha256") != sha(review / "predeclared_group_protocol.json")):
        raise ValueError("Frozen review lineage changed")
    validate_approval(approval, frozen, frozen_path, decisions_path)
    by_id = {sample["id"]: sample for sample in inventory["samples"]}
    # Validate all native bytes and every prospective label before any mutation.
    labels = {}
    for row in selected:
        sample = by_id[row["id"]]
        if sha(ROOT / sample["source_image"]) != sample["sha256"]:
            raise ValueError("Native cached image bytes changed before export")
        labels[row["id"]] = encode_court_label(sample, row)
    output.mkdir(parents=True)
    sources, records = [], []
    protocol = load(review / "predeclared_group_protocol.json")
    for row in selected:
        sample = by_id[row["id"]]
        split = row["development_split"]
        image_path = output / "court" / "images" / split / (row["id"] + ".png")
        label_path = output / "court" / "labels" / split / (row["id"] + ".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / sample["source_image"], image_path)
        with label_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(labels[row["id"]])
        if sha(image_path) != sample["sha256"]:
            raise ValueError("Copied image bytes differ from native source")
        source_id = "court_image_" + row["id"]
        sources.append({
            "source_id": source_id, "source_type": "PUBLISHER_STILL_IMAGE", "source_path": sample["source_image"],
            "sha256": sample["sha256"], "owner": "yastrebksv/TennisCourtDetector dataset author",
            "rights_evidence": protocol["rights"], "training_allowed": True,
            "match_id": None, "session_id": row["context_group"], "context_group": row["context_group"],
            "context_id_is_verified_match_identity": False, "original_source_video_id": sample["source_video_id"],
            "publisher_frame_ordinal": sample["source_frame_ordinal"], "fps": None,
            "source_pts": None, "source_time_base": None, "resolution": [sample["width"], sample["height"]],
            "surface": row["surface"], "visual_event_family": row["visual_event_family"],
            "view": "elevated_end_court_singles", "camera_reviewed": True, "development_split": split,
            "original_video_bytes_acquired": False,
        })
        records.append({
            "role": "court", "source_id": source_id, "id": row["id"], "split": split,
            "image": relative(image_path), "label": relative(label_path), "sha256": sample["sha256"],
            "label_sha256": sha(label_path), "frame_index": sample["source_frame_ordinal"], "timestamp_seconds": None,
            "source_pts": None, "source_time_base": None, "reviewer_type": "AGENT_VISUAL_REVIEW",
            "annotation_provenance": "PUBLISHER_COORDINATES_AGENT_NATIVE_VISIBILITY_REVIEWED",
            "independent_human_reviewed": False, "landmark_schema": "tennis_court_14_v1",
            "bbox_xyxy_px": row["bbox_xyxy_px"], "bbox_provenance": row["bbox_provenance"],
            "keypoints_px": row["reviewed_keypoints_px"], "near_duplicate_group": row["context_group"],
            "publisher_annotation_file": sample["publisher_annotation_file"],
            "publisher_annotation_sha256": sample["publisher_annotation_sha256"],
            "native_detail_board_sha256": row["native_detail_board_sha256"], "review_note": row["review_note"],
        })
    manifest = {
        "schema_version": "1.1", "status": "AUTHORIZED_AGENT_REVIEWED_DEVELOPMENT_COURT_NOT_QUALIFICATION",
        "scope": "elevated_end_court_singles", "frozen": False, "development_decisions_frozen": True,
        "qualification_evidence": False, "source_independence_verified": False, "final_test_used": False,
        "counts": frozen["counts"], "sources": sources, "samples": records,
        "label_review": {"training_reviewed": True, "independent_test_reviewed": False},
        "split_policy": {"context_splits": context_splits, "context_groups": frozen["context_groups"],
                         "rejected_context_links_retained": True, "test": 0, "unused_ids_are_development": True},
        "lineage": {"frozen_manifest_sha256": sha(frozen_path), "decisions_sha256": sha(decisions_path),
                    "root_approval_sha256": sha(approval_path), "inventory_sha256": sha(review / "cache_inventory.json"),
                    "native_detail_receipt_sha256": decisions["native_detail_receipt_sha256"]},
        "rights": protocol["rights"], "native_images_decoded_during_export": False,
    }
    save(output / "source_manifest.json", manifest)
    yaml_text = (f"path: {json.dumps(str((output / 'court').resolve()).replace(chr(92), '/'))}\n"
                 "train: images/train\nval: images/val\nnames:\n  0: court\nkpt_shape: [14, 3]\n"
                 f"flip_idx: {json.dumps(FLIP_IDX)}\n")
    (output / "court" / "data.yaml").write_text(yaml_text, encoding="utf-8")
    save(output / "training_plan.json", {
        "initialization": "OFFICIAL_YOLO26S_POSE_PRETRAINED_REBUILT_14_POINT_COURT_HEAD",
        "reuse_prior_six_image_court_checkpoint": False, "training_executed": False,
        "kpt_shape": [14, 3], "point_names": POINT_NAMES, "flip_idx": FLIP_IDX,
        "augmentation": {"fliplr": 0, "flipud": 0, "mosaic": 0, "mixup": 0},
        "data_yaml": relative(output / "court" / "data.yaml"),
        "static_image_timing_is_unknown": True,
    })
    save(output / "export_receipt.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "counts": frozen["counts"],
        "source_manifest_sha256": sha(output / "source_manifest.json"), "root_approval_sha256": sha(approval_path),
        "copied_native_bytes_exact": True, "training_executed": False, "gpu_used": False,
        "qualification_evidence": False, "all_selected_null_keypoints": frozen["selected_null_keypoints"],
    })
    return {"output": str(output), "counts": frozen["counts"], "manifest_sha256": sha(output / "source_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "export"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_reviewed(args.review, args.decisions, args.output)
    else:
        if args.frozen is None or args.approval is None:
            parser.error("Export requires --frozen and --approval")
        result = export_reviewed(args.review, args.decisions, args.frozen, args.approval, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

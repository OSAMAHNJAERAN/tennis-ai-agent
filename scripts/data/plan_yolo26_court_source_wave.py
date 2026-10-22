"""Freeze a bounded metadata-only court wave from previously uncached videos.

This module uses only the standard library. It never opens an image, imports a
model library, downloads an archive member, exports a label, or assigns a split.
Publisher video identifiers are exclusion/grouping handles, not verified matches.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SEED = 709
MAX_IMAGES = 64
IMAGE_LIMIT = 5_000_000
RANGE_LIMIT = 80_000_000
POINT_NAMES = [
    "far_left_doubles", "far_right_doubles", "near_left_doubles", "near_right_doubles",
    "far_left_singles", "near_left_singles", "far_right_singles", "near_right_singles",
    "far_service_left", "far_service_right", "near_service_left", "near_service_right",
    "far_service_center", "near_service_center",
]
FORBIDDEN_IMPORT_PREFIXES = ("PIL", "cv2", "numpy", "torch", "ultralytics", "psutil")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stable_rank(kind: str, identifier: str) -> str:
    return hashlib.sha256(f"{SEED}\n{kind}\n{identifier}".encode("ascii")).hexdigest()


def parse_id(identifier: str) -> tuple[str, int]:
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}_[0-9]+", identifier):
        raise ValueError("Invalid publisher video/frame identifier")
    video, ordinal = identifier.rsplit("_", 1)
    return video, int(ordinal)


def inspect_points(points) -> dict:
    if not isinstance(points, list) or len(points) != 14:
        raise ValueError("Retain exactly fourteen publisher semantic slots")
    nulls, outside = [], []
    for index, point in enumerate(points):
        if point is None:
            nulls.append(index)
            continue
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Publisher slot must be null or x,y")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in point):
            raise ValueError("Publisher coordinates must be finite numbers")
        if not (0 <= point[0] < 1280 and 0 <= point[1] < 720):
            outside.append(index)
    return {"publisher_null_keypoint_indices": nulls, "out_of_image_keypoint_indices": outside}


def select_rows(annotation_documents, existing_samples, count=MAX_IMAGES):
    """Select at most one image per wholly new publisher video, across both splits."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_IMAGES:
        raise ValueError("Wave size must be between one and sixty-four")
    old_ids = set()
    old_images = set()
    for sample in existing_samples:
        video, _ = parse_id(sample["id"])
        if sample["source_video_id"] != video or sample["id"] in old_images:
            raise ValueError("Existing inventory video mapping or uniqueness changed")
        old_ids.add(video)
        old_images.add(sample["id"])
    grouped = defaultdict(list)
    all_rows, members = {}, defaultdict(set)
    for name, rows in annotation_documents.items():
        for row in rows:
            video, _ = parse_id(row["id"])
            if row["id"] in all_rows:
                raise ValueError("Duplicate image identifier across publisher annotations")
            defects = inspect_points(row["kps"])
            enriched = {"id": row["id"], "source_video_id": video,
                        "publisher_annotation_file": name,
                        "publisher_keypoints_xy_px": row["kps"], **defects}
            all_rows[row["id"]] = enriched
            members[name].add(video)
            grouped[video].append(enriched)
    if not old_images <= set(all_rows):
        raise ValueError("Existing image no longer appears in the complete publisher annotations")
    for sample in existing_samples:
        if sample["publisher_keypoints_xy_px"] != all_rows[sample["id"]]["publisher_keypoints_xy_px"]:
            raise ValueError("Existing inventory proposal differs from pinned publisher bytes")
    candidates = []
    for video in sorted(set(grouped) - old_ids):
        rows = sorted(grouped[video], key=lambda row: (stable_rank("image", row["id"]), row["id"]))
        candidates.append({"source_video_id": video, "video_rank_sha256": stable_rank("video", video),
                           "publisher_image_count": len(rows), "chosen_id": rows[0]["id"]})
    candidates.sort(key=lambda row: (row["video_rank_sha256"], row["source_video_id"]))
    selected = [all_rows[item["chosen_id"]] for item in candidates[:count]]
    if len(selected) != min(count, len(candidates)):
        raise ValueError("Deterministic selection count mismatch")
    if set(row["source_video_id"] for row in selected) & old_ids:
        raise ValueError("Selected a cached publisher video")
    publisher_sets = list(members.values())
    return selected, candidates, {
        "publisher_images": len(all_rows), "publisher_video_ids": len(grouped),
        "publisher_annotation_counts": {name: len(rows) for name, rows in annotation_documents.items()},
        "publisher_cross_split_video_ids": len(set.intersection(*publisher_sets)) if publisher_sets else 0,
        "existing_images": len(old_images), "excluded_existing_publisher_video_ids": len(old_ids),
        "new_candidate_publisher_video_ids": len(candidates), "selected_images": len(selected),
        "selected_publisher_video_ids": len({row["source_video_id"] for row in selected}),
        "selected_with_out_of_image_points": sum(bool(row["out_of_image_keypoint_indices"]) for row in selected),
        "selected_null_keypoint_slots": sum(len(row["publisher_null_keypoint_indices"]) for row in selected),
    }


def bounded_member(entry: dict, identifier: str, archive_bytes: int) -> dict:
    name = f"data/images/{identifier}.png"
    if entry["name"] != name or entry["compression"] != 8:
        raise ValueError("Unsupported image archive member")
    for field in ("bytes", "compressed_bytes", "header_offset", "crc32"):
        if isinstance(entry[field], bool) or not isinstance(entry[field], int) or entry[field] < 0:
            raise ValueError("Invalid archive member integer")
    if not 1 <= entry["bytes"] <= IMAGE_LIMIT or not 1 <= entry["compressed_bytes"] <= IMAGE_LIMIT:
        raise ValueError("Member exceeds the reviewed fixed image limit")
    if not 0 <= entry["crc32"] <= 0xFFFFFFFF:
        raise ValueError("Invalid publisher CRC32")
    range_length = 30 + len(name.encode("utf-8")) + 1024 + entry["compressed_bytes"]
    if entry["header_offset"] + range_length > archive_bytes:
        raise ValueError("Member range leaves the pinned archive")
    return {**entry, "planned_range_start": entry["header_offset"],
            "planned_range_bytes_upper_bound": range_length,
            "range_header_allowance_bytes": 30 + len(name.encode("utf-8")) + 1024,
            "range_note": "Bound includes up to 1024 bytes ZIP local-header extras; downloader must verify name, method, sizes, CRC, PNG geometry, and SHA256"}


def verified_input(path: Path, expected: str | None = None) -> dict:
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Pinned input changed: {path}")
    return {"path": str(path.resolve().relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size, "sha256": actual}


def freeze(source: Path, inventory: Path, decisions: Path, output: Path, count=MAX_IMAGES):
    if output.exists():
        raise FileExistsError("Use a fresh immutable court source-wave output")
    annotation_manifest = load(source / "annotation_manifest.json")
    archive_index = load(source / "archive_index.json")
    probe = load(source / "range_probe.json")
    old_inventory = load(inventory)
    old_review = load(decisions)
    old_protocol_path = inventory.parent / "predeclared_group_protocol.json"
    old_protocol = load(old_protocol_path)
    if annotation_manifest.get("complete") is not True:
        raise ValueError("Complete verified publisher annotation manifest required")
    if old_review.get("inventory_sha256") != digest(inventory):
        raise ValueError("Existing review is not bound to the current inventory")
    if old_inventory.get("protocol_sha256") != digest(old_protocol_path):
        raise ValueError("Existing inventory is not bound to its original source protocol")
    if archive_index["archive_bytes"] != annotation_manifest["archive_bytes"]:
        raise ValueError("Pinned archive sizes disagree")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", probe["range"])
    if not match or int(match[3]) != archive_index["archive_bytes"]:
        raise ValueError("Pinned range probe size disagrees")
    inputs = [verified_input(source / "annotation_manifest.json", old_protocol["source_manifest_sha256"]),
              verified_input(source / "archive_index.json", old_protocol["archive_index_sha256"]),
              verified_input(source / "range_probe.json", annotation_manifest["source_probe_sha256"]),
              verified_input(inventory), verified_input(decisions), verified_input(old_protocol_path)]
    documents = {}
    for item in annotation_manifest["files"]:
        if item["name"] not in ("data_train.json", "data_val.json") or item["name"] in documents:
            raise ValueError("Require exactly the two original publisher annotation files")
        path = source / item["name"]
        inputs.append(verified_input(path, item["sha256"]))
        if old_protocol["annotation_file_sha256"].get(item["name"]) != item["sha256"]:
            raise ValueError("Publisher annotation differs from the previous reviewed source protocol")
        rows = load(path)
        if len(rows) != item["rows"]:
            raise ValueError("Publisher annotation row count changed")
        documents[item["name"]] = rows
    if set(documents) != {"data_train.json", "data_val.json"}:
        raise ValueError("Do not omit either original publisher split")
    selected, candidates, counts = select_rows(documents, old_inventory["samples"], count)
    entries = {entry["name"]: entry for entry in archive_index["entries"]}
    if len(entries) != len(archive_index["entries"]):
        raise ValueError("Repeated archive member name")
    samples = []
    for number, row in enumerate(selected, 1):
        entry = entries[f"data/images/{row['id']}.png"]
        member = bounded_member(entry, row["id"], archive_index["archive_bytes"])
        samples.append({"number": number, **row,
                        "publisher_annotation_sha256": next(item["sha256"] for item in annotation_manifest["files"]
                                                             if item["name"] == row["publisher_annotation_file"]),
                        "source_frame_ordinal": parse_id(row["id"])[1], "source_pts": None,
                        "source_time_base": None, "source_fps": None, "clip_duration_seconds": None,
                        "expected_native_width": 1280, "expected_native_height": 720,
                        "native_geometry_verified": False, "image_sha256": None,
                        "publisher_visibility": "NOT_SUPPLIED", "reviewed_keypoints_px": None,
                        "review_status": "PENDING_ACTUAL_NATIVE_IMAGE_AND_INTERSECTION_REVIEW",
                        "development_split": None, "context_group": None,
                        "visual_event_family": None, "match_id": None, "camera_session_id": None,
                        "training_allowed_before_review": False, "qualification_evidence": False,
                        "archive_member": member})
    total_ranges = sum(row["archive_member"]["planned_range_bytes_upper_bound"] for row in samples)
    if total_ranges > RANGE_LIMIT:
        raise ValueError("Selection exceeds the predeclared eighty-megabyte range ceiling")
    old_links = [{"id": row["id"], "source_video_id": parse_id(row["id"])[0],
                  "visual_event_family": row["visual_event_family"], "context_group": row["context_group"],
                  "development_split": row["development_split"], "decision": row["decision"]}
                 for row in old_review["samples"]]
    if len(old_links) != len(old_inventory["samples"]) or {row["id"] for row in old_links} != {row["id"] for row in old_inventory["samples"]}:
        raise ValueError("Retain all accepted and rejected existing context links")
    plan = {
        "schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_METADATA_PLAN_AWAITING_ROOT_RANGE_ACQUISITION_RELEASE",
        "selection_seed": SEED, "selection_algorithm": "sha256(str(seed)+'\\n'+kind+'\\n'+canonical_id), ascending digest then ID; video first, then one image within video",
        "selection_unit": "PUBLISHER_VIDEO_ID_NOT_VERIFIED_MATCH", "maximum_images": MAX_IMAGES,
        "all_original_publisher_splits_combined_before_exclusion": True,
        "all_selected_splits_unassigned": True, "final_test_images": 0, "qualification_evidence": False,
        "source_inputs": inputs, "counts": counts, "new_video_candidate_order": candidates,
        "selected_annotation_file_counts": dict(Counter(row["publisher_annotation_file"] for row in samples)),
        "rights": {"author": "yastrebksv", "author_readme": "https://github.com/yastrebksv/TennisCourtDetector#dataset",
                   "training_instructions": "https://github.com/yastrebksv/TennisCourtDetector#how-to-train",
                   "dataset_url": "https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view",
                   "basis": "AUTHOR_EXPLICIT_DATASET_RELEASE_WITH_TRAINING_INSTRUCTIONS",
                   "allowed_scope": "PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT_ONLY",
                   "explicit_raw_dataset_license": None, "broadcast_redistribution_cleared": False,
                   "code_license_is_not_data_permission": True},
        "archive": {"url": probe["url"], "params": probe["params"],
                    "archive_bytes": archive_index["archive_bytes"], "whole_archive_checksum_verified": False,
                    "planned_member_compressed_bytes": sum(row["archive_member"]["compressed_bytes"] for row in samples),
                    "planned_png_bytes": sum(row["archive_member"]["bytes"] for row in samples),
                    "planned_range_bytes_upper_bound": total_ranges, "fixed_range_budget_bytes": RANGE_LIMIT,
                    "max_inflight_members": 1, "images_downloaded_by_this_planner": 0,
                    "network_bytes_received_by_this_planner": 0},
        "landmark_schema": {"name": "tennis_court_14_v1", "point_names": POINT_NAMES,
                            "publisher_coordinates_are_proposals": True,
                            "null_and_out_of_image_slots_preserved": True,
                            "source_fps_pts_and_visibility_must_not_be_invented": True},
        "context_review_constraints": {
            "development_only_visual_event_families": ["LondonFinals", "MonteCarlo"],
            "all_existing_context_groups": old_review["context_groups"],
            "all_existing_accepted_and_rejected_video_event_links": old_links,
            "rule": "Union every observed event/court link from complete compilation publisher videos before any split; retain all prior links including rejected frames. Any component touching LondonFinals or MonteCarlo stays DEV; no new TEST. Unknown contexts stay unassigned. Distinct publisher IDs do not establish independent matches.",
            "source_diversity_limit": "New video identifiers only; new events, independent matches and camera sessions require actual image/context review.",
        },
        "next_actions": ["Root inspect this fixed selection and bounded range budget before image transfer",
                         "If released, acquire only these selected member ranges and verify CRC, size, SHA256 and PNG geometry",
                         "Actual native review of all fourteen intersections, visibility, primary singles view and event/compilation context",
                         "Conservative union with all prior event links; root second pass before split freeze or supervised export"],
        "samples": samples,
    }
    if any(digest(ROOT / item["path"]) != item["sha256"] for item in inputs):
        raise ValueError("Metadata inputs changed during planning")
    output.mkdir(parents=True)
    with (output / "source_wave_plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    return {"plan": str(output / "source_wave_plan.json"), "sha256": digest(output / "source_wave_plan.json"),
            "counts": counts, "planned_range_bytes_upper_bound": total_ranges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/validation/vision_upgrade/court_heatmap_training_source")
    parser.add_argument("--inventory", type=Path, default=ROOT / "data/yolo26/court_review_expansion_v1/cache_inventory.json")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/yolo26/court_review_decisions_v1.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=MAX_IMAGES)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.inventory, args.decisions, args.output, args.count)), flush=True)


if __name__ == "__main__":
    main()

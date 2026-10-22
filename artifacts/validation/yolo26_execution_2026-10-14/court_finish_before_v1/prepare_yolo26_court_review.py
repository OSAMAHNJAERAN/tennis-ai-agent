"""Audit cached author court images and freeze reviewed development groups.

No network, model, GPU, inference, or training is performed. Audit outputs are
fresh immutable versions. Publisher coordinates are proposals until a recorded
agent review checks their semantic intersections and visibility in the image.
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

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
POINT_NAMES = [
    "far_left_doubles", "far_right_doubles", "near_left_doubles", "near_right_doubles",
    "far_left_singles", "near_left_singles", "far_right_singles", "near_right_singles",
    "far_service_left", "far_service_right", "near_service_left", "near_service_right",
    "far_service_center", "near_service_center",
]
FLIP_IDX = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]
LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7),
         (8, 9), (10, 11), (12, 13)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, document):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def font(size=18):
    # A supplied system font affects boards only, never coordinates or hashes.
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.is_file() else ImageFont.load_default(size=size)


def point_defects(points, width, height):
    if len(points) != 14 or any(len(point) != 2 for point in points):
        raise ValueError("Exactly fourteen x,y semantic slots required")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("Nonfinite publisher coordinates")
    return [index for index, (x, y) in enumerate(points)
            if not (0 <= x < width and 0 <= y < height)]


def annotate(image, points, *, labels=True):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for a, b in LINES:
        draw.line([tuple(points[a]), tuple(points[b])], fill=(240, 185, 0), width=2)
    for index, (x, y) in enumerate(points):
        if not 0 <= x < image.width or not 0 <= y < image.height:
            continue
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 75, 90), outline="white")
        if labels:
            xx, yy = min(image.width - 32, max(0, x + 5)), min(image.height - 24, max(0, y - 22))
            draw.text((xx, yy), str(index), fill="white", stroke_width=2, stroke_fill="black", font=font(20))
    return image


def audit(cache: Path, source: Path, output: Path):
    cache, source, output = cache.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Create a fresh court review version")
    document = load(cache / "manifest.json")
    annotation_manifest = load(source / "annotation_manifest.json")
    if document.get("complete") is not True or annotation_manifest.get("complete") is not True:
        raise ValueError("Require complete immutable acquisition manifests")
    if sha(source / "annotation_manifest.json") != document["source_manifest_sha256"]:
        raise ValueError("Cached annotation manifest hash changed")
    if sha(source / "archive_index.json") != document["archive_index_sha256"]:
        raise ValueError("Cached archive index hash changed")
    annotations, source_members, publisher_groups = {}, {}, defaultdict(set)
    annotation_hashes = {}
    for item in annotation_manifest["files"]:
        path = source / item["name"]
        if sha(path) != item["sha256"]:
            raise ValueError("Publisher annotation bytes changed")
        annotation_hashes[item["name"]] = item["sha256"]
        for row in load(path):
            if row["id"] in annotations:
                raise ValueError("Publisher repeats an image ID across annotation files")
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}_[0-9]+", row["id"]):
                raise ValueError("Invalid publisher video/frame ID")
            annotations[row["id"]] = row
            source_members[row["id"]] = item["name"]
            publisher_groups[item["name"]].add(row["id"].rsplit("_", 1)[0])
    acquired = {Path(item["path"]).stem: item for item in document["files"]}
    if len(acquired) != len(document["files"]):
        raise ValueError("Duplicate cached image ID")
    samples, seen_hashes = [], set()
    for row in sorted(document["samples"], key=lambda x: (x["source_video_id"], int(x["id"].rsplit("_", 1)[1]))):
        original = annotations[row["id"]]
        if original["kps"] != row["kps"]:
            raise ValueError("Cached sample labels differ from publisher bytes")
        if row["source_video_id"] != row["id"].rsplit("_", 1)[0]:
            raise ValueError("Publisher source video disagrees with frame ID")
        item = acquired[row["id"]]
        path = cache / item["path"]
        if len(path.read_bytes()) != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Cached image bytes changed")
        if item["sha256"] in seen_hashes:
            raise ValueError("Exact duplicate images require rejection before grouping")
        seen_hashes.add(item["sha256"])
        with Image.open(path) as image:
            if image.size != (1280, 720):
                raise ValueError("Cached native geometry changed")
            out_of_image = point_defects(row["kps"], *image.size)
        samples.append({"number": len(samples) + 1, "id": row["id"], "source_video_id": row["source_video_id"],
                        "source_image": relative(path), "sha256": item["sha256"], "width": 1280, "height": 720,
                        "publisher_annotation_file": source_members[row["id"]],
                        "publisher_annotation_sha256": annotation_hashes[source_members[row["id"]]],
                        "publisher_keypoints_xy_px": row["kps"], "publisher_visibility": "NOT_SUPPLIED",
                        "out_of_image_keypoint_indices": out_of_image, "previous_partition": row["partition"],
                        "data_role": "DEVELOPMENT_PREVIOUSLY_INSPECTED", "review_status": "PENDING_AGENT_REVIEW",
                        "source_frame_ordinal": int(row["id"].rsplit("_", 1)[1]),
                        "source_pts": None, "source_time_base": None, "source_fps": None,
                        "coordinate_map": {"kind": "NATIVE_IDENTITY", "image_to_source_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}})
    if set(acquired) != {item["id"] for item in samples}:
        raise ValueError("Cached files/samples disagree")
    output.mkdir(parents=True)
    protocol = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PREDECLARED_DEVELOPMENT_REVIEW_PROTOCOL_NO_EXPORT", "qualification_evidence": False,
                "scope": "elevated_end_court_singles", "target_accepted_images": {"train": 100, "val": 20},
                "all_existing_cache_is_development": True, "final_test_images": 0,
                "cache_manifest_sha256": sha(cache / "manifest.json"), "annotation_file_sha256": annotation_hashes,
                "source_manifest_sha256": sha(source / "annotation_manifest.json"), "archive_index_sha256": sha(source / "archive_index.json"),
                "rights": {"author": "yastrebksv", "author_readme": "https://github.com/yastrebksv/TennisCourtDetector#dataset",
                           "training_instructions": "https://github.com/yastrebksv/TennisCourtDetector#how-to-train",
                           "dataset_url": "https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view",
                           "basis": "AUTHOR_EXPLICIT_DATASET_RELEASE_WITH_TRAINING_INSTRUCTIONS",
                           "allowed_scope": "PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT_ONLY",
                           "explicit_raw_dataset_license": None, "broadcast_redistribution_cleared": False,
                           "code_license_is_not_data_permission": True},
                "review_rules": ["Visually identify the same event/court/session across every cached video ID before split assignment",
                                 "Conservatively merge similar unresolved event/court contexts; do not invent independent matches",
                                 "Reject non-primary-view, doubles, closeups, shifted/wrong semantic landmarks and unresolved out-of-image points",
                                 "Publisher coordinates have no visibility; assign 2 only when the actual intersection is visibly supported",
                                 "Unknown/hidden landmarks stay null; no imagined visibility or clamped coordinates",
                                 "Agent visual review is development annotation, never independent human ground truth",
                                 "Root second-pass review is required before YOLO export"],
                "selection_rule": "After context grouping and label review, deterministic hash ordering within complete train/val context assignments; cap100/20; reject surplus as UNUSED_DEV",
                "landmark_schema": "tennis_court_14_v1", "point_names": POINT_NAMES, "flip_idx": FLIP_IDX,
                "augmentation": {"fliplr": 0, "flipud": 0, "mosaic": 0, "mixup": 0},
                "sources": [{"source_video_id": group, "development_split": None, "context_group": None,
                             "group_status": "PENDING_FULL_EVENT_CONTEXT_REVIEW"}
                            for group in sorted({item["source_video_id"] for item in samples})]}
    save(output / "predeclared_group_protocol.json", protocol)
    save(output / "cache_inventory.json", {"samples": samples, "protocol_sha256": sha(output / "predeclared_group_protocol.json"),
                                          "counts": {"images": len(samples), "cached_publisher_video_ids": len(protocol["sources"]),
                                                     "publisher_total_images": len(annotations),
                                                     "publisher_video_ids": len(set().union(*publisher_groups.values())),
                                                     "publisher_train_val_shared_video_ids": len(set.intersection(*publisher_groups.values())),
                                                     "publisher_out_of_image_samples": sum(bool(point_defects(row["kps"], 1280, 720)) for row in annotations.values()),
                                                     "publisher_out_of_image_keypoints": sum(len(point_defects(row["kps"], 1280, 720)) for row in annotations.values()),
                                                     "cached_out_of_image_samples": sum(bool(item["out_of_image_keypoint_indices"]) for item in samples),
                                                     "cached_out_of_image_keypoints": sum(len(item["out_of_image_keypoint_indices"]) for item in samples)}})
    (output / "native_overlays").mkdir()
    (output / "boards").mkdir()
    boards = []
    # Board thumbnails retain every native pixel's affine map and all fourteen slots.
    for start in range(0, len(samples), 12):
        selected = samples[start:start + 12]
        board = Image.new("RGB", (1440, 1248), "#171a20")
        draw = ImageDraw.Draw(board)
        for offset, sample in enumerate(selected):
            with Image.open(ROOT / sample["source_image"]) as original:
                native = annotate(original, sample["publisher_keypoints_xy_px"])
                native_path = output / "native_overlays" / f"{sample['number']:03}_{sample['id']}.jpg"
                native.save(native_path, quality=96)
                thumbnail = native.resize((480, 270), Image.Resampling.LANCZOS)
            x, y = offset % 3 * 480, offset // 3 * 312
            board.paste(thumbnail, (x, y))
            note = f"{sample['number']:03} {sample['id']}"
            draw.text((x + 8, y + 273), note, font=font(17), fill="white")
            bounds = sample["out_of_image_keypoint_indices"]
            draw.text((x + 8, y + 295), f"DEV; publisher visibility UNKNOWN; outside={bounds}", font=font(12), fill="#ffc060")
            sample["native_review_overlay"] = relative(native_path)
            sample["board_thumbnail_map"] = {"scale_xy": [.375, .375], "offset_xy": [x, y]}
        path = output / "boards" / f"board_{start // 12 + 1:02}.jpg"
        board.save(path, quality=95)
        boards.append({"image": relative(path), "sha256": sha(path), "sample_ids": [item["id"] for item in selected]})
    # The immutable inventory is untouched; this distinct receipt adds display maps.
    save(output / "review_boards.json", {"inventory_sha256": sha(output / "cache_inventory.json"), "boards": boards, "samples": samples})
    return {"output": str(output), "images": len(samples), "boards": len(boards), "publisher_video_ids": len(protocol["sources"])}


def context_components(samples, rows):
    """Union full compilation videos with every observed event/court family."""
    by_id = {sample["id"]: sample for sample in samples}
    components = []
    for row in rows:
        event = row.get("visual_event_family")
        if not event:
            raise ValueError("Every cached frame needs a visually observed or conservative UNKNOWN event family")
        identifiers = {("video", by_id[row["id"]]["source_video_id"]), ("event", event)}
        touching = [group for group in components if group & identifiers]
        for group in touching:
            identifiers |= group
            components.remove(group)
        components.append(identifiers)
    by_video, groups = {}, {}
    for group in components:
        group_id = "context_" + hashlib.sha256(json.dumps(sorted(group)).encode()).hexdigest()[:12]
        groups[group_id] = {"publisher_video_ids": sorted(value for kind, value in group if kind == "video"),
                            "visual_event_families": sorted(value for kind, value in group if kind == "event")}
        by_video.update({value: group_id for kind, value in group if kind == "video"})
    return by_video, groups


def validate_review(inventory, decisions):
    """Require exhaustive image decisions and complete indivisible context groups."""
    samples = {sample["id"]: sample for sample in inventory["samples"]}
    rows = decisions.get("samples", [])
    if len(rows) != len(samples) or {row["id"] for row in rows} != set(samples):
        raise ValueError("Every cached image needs exactly one explicit review decision")
    computed_contexts, _ = context_components(inventory["samples"], rows)
    video_contexts, context_splits, accepted = {}, {}, []
    for row in rows:
        sample = samples[row["id"]]
        if row.get("reviewer_type") != "AGENT_VISUAL_REVIEW" or not row.get("review_note"):
            raise ValueError("Record actual agent review and a concrete image observation")
        context = row.get("context_group")
        split = row.get("development_split")
        if not context or split not in ("train", "val", "exclude"):
            raise ValueError("Every image requires an explicit event context and development assignment")
        if context != computed_contexts[sample["source_video_id"]]:
            raise ValueError("Declared context does not include all transitive video/event links")
        previous = video_contexts.setdefault(sample["source_video_id"], context)
        if previous != context:
            raise ValueError("A full publisher video must stay in a single conservative context group")
        if split != "exclude":
            previous_split = context_splits.setdefault(context, split)
            if previous_split != split:
                raise ValueError("Full event/court context crosses train/DEV")
        if row.get("decision") == "REJECT":
            if not row.get("rejection_reason"):
                raise ValueError("Rejected image requires a reason")
            continue
        if row.get("decision") != "ACCEPT" or split == "exclude":
            raise ValueError("Accepted review requires train or held-out DEV")
        if row.get("scope") != "elevated_end_court_singles":
            raise ValueError("Accepted image needs visually reviewed primary singles view")
        if sample["out_of_image_keypoint_indices"]:
            raise ValueError("Unresolved out-of-image publisher points are not accepted")
        points = row.get("reviewed_keypoints_px")
        if points is None or len(points) != 14:
            raise ValueError("Retain all fourteen semantic slots")
        visible = 0
        for index, point in enumerate(points):
            if point is None:
                continue
            if len(point) != 3 or point[2] != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError("This review accepts visible points or null, never inferred occlusion")
            if not (0 <= point[0] < sample["width"] and 0 <= point[1] < sample["height"]):
                raise ValueError("Reviewed landmark is outside the native image")
            if point[:2] != sample["publisher_keypoints_xy_px"][index] and not row.get("coordinate_correction_note"):
                raise ValueError("Manual coordinate changes need separate provenance")
            visible += 1
        if visible < 10 or row.get("semantic_order_reviewed") is not True:
            raise ValueError("Require semantic review and at least ten observed distributed landmarks")
        accepted.append(row)
    if not {"train", "val"} <= set(context_splits.values()):
        raise ValueError("Require reviewed training and separate development context groups")
    return accepted, context_splits


def render_details(review: Path, output: Path):
    """Show original native pixels around every supplied landmark, without lines."""
    if output.exists():
        raise FileExistsError("Native detail review must use a fresh output")
    inventory = load(review / "cache_inventory.json")
    output.mkdir(parents=True)
    boards = []
    for start in range(0, len(inventory["samples"]), 5):
        selected = inventory["samples"][start:start + 5]
        board = Image.new("RGB", (1008, len(selected) * 228), "#171a20")
        draw = ImageDraw.Draw(board)
        mappings = []
        for offset, sample in enumerate(selected):
            ybase = offset * 228
            with Image.open(ROOT / sample["source_image"]) as original:
                if sha(ROOT / sample["source_image"]) != sample["sha256"]:
                    raise ValueError("Native source image changed")
                original = original.convert("RGB")
                board.paste(original.resize((336, 189), Image.Resampling.LANCZOS), (0, ybase))
                for index, (x, y) in enumerate(sample["publisher_keypoints_xy_px"]):
                    x0, y0 = int(x) - 48, int(y) - 48
                    patch = original.crop((x0, y0, x0 + 96, y0 + 96))
                    pd = ImageDraw.Draw(patch)
                    pd.ellipse((40, 40, 56, 56), outline="red", width=1)
                    pd.text((2, 1), str(index), fill="white", stroke_width=1, stroke_fill="black", font=font(14))
                    bx, by = 336 + index % 7 * 96, ybase + index // 7 * 96
                    board.paste(patch, (bx, by))
                    mappings.append({"id": sample["id"], "point_index": index, "native_crop_xyxy": [x0, y0, x0 + 96, y0 + 96],
                                     "board_crop_xyxy": [bx, by, bx + 96, by + 96], "scale": 1,
                                     "point_inside_image": index not in sample["out_of_image_keypoint_indices"]})
            draw.text((8, ybase + 195), f"{sample['number']:03} {sample['id']} | circles are publisher proposals; crops 1:1 native px", font=font(15), fill="white")
            draw.text((8, ybase + 213), "Check actual line intersection and occlusion. A circle is not a verified visible label.", font=font(12), fill="#ffc060")
        path = output / f"native_detail_{start // 5 + 1:02}.jpg"
        board.save(path, quality=97)
        boards.append({"image": relative(path), "sha256": sha(path), "sample_ids": [sample["id"] for sample in selected], "coordinate_maps": mappings})
    save(output / "native_detail_receipt.json", {"inventory_sha256": sha(review / "cache_inventory.json"), "boards": boards,
                                                "model_predictions_used": False, "independent_human_review": False})
    return {"output": str(output), "boards": len(boards)}


def freeze(review: Path, decisions_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Never overwrite frozen development decisions")
    inventory, decisions = load(review / "cache_inventory.json"), load(decisions_path)
    if decisions.get("inventory_sha256") != sha(review / "cache_inventory.json"):
        raise ValueError("Review decisions are not bound to the native cache inventory")
    accepted, context_splits = validate_review(inventory, decisions)
    selected, surplus = [], []
    for split, cap in (("train", 100), ("val", 20)):
        candidates = sorted([row for row in accepted if row["development_split"] == split],
                            key=lambda row: hashlib.sha256(f"court-development-v1|{row['id']}".encode()).hexdigest())
        selected.extend(candidates[:cap])
        surplus.extend(row["id"] for row in candidates[cap:])
    output.mkdir(parents=True)
    manifest = {"schema_version": "1.0", "status": "FROZEN_AGENT_REVIEWED_DEVELOPMENT_ONLY_AWAITING_ROOT_REVIEW",
                "qualification_evidence": False, "independent_human_reviewed": False, "final_test_used": False,
                "protocol_sha256": sha(review / "predeclared_group_protocol.json"),
                "inventory_sha256": sha(review / "cache_inventory.json"), "decisions_sha256": sha(decisions_path),
                "context_splits": context_splits, "selected": selected, "unused_accepted_development_ids": surplus,
                "counts": dict(Counter(row["development_split"] for row in selected)),
                "root_second_pass_before_export": True, "yolo_exported": False}
    save(output / "grouped_development_manifest.json", manifest)
    return {"output": str(output), "counts": manifest["counts"], "context_groups": len(context_splits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--cache", type=Path, default=ROOT / "data/external/court_heatmap_pilot")
    audit_parser.add_argument("--source", type=Path, default=ROOT / "artifacts/validation/vision_upgrade/court_heatmap_training_source")
    audit_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--review", type=Path, required=True)
    freeze_parser.add_argument("--decisions", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    detail_parser = commands.add_parser("details")
    detail_parser.add_argument("--review", type=Path, required=True)
    detail_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.cache, args.source, args.output)
    elif args.command == "details":
        result = render_details(args.review, args.output)
    else:
        result = freeze(args.review, args.decisions, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

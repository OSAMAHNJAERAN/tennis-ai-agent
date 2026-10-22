"""Convert reviewed native-image tennis labels and freeze grouped 60/20/20 splits.

This tool does not create annotations from model predictions or authorize a
source. Input sources must already contain their own documented rights review.
Image bytes remain in their original resolution; output is YOLO label text.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2

from scripts.train.train_yolo26_tennis import SHAPES, validate_labels
from src.detection.yolo26_bundle import REPO_ROOT, sha256_file


def grouped_splits(sources, *, seed=709):
    """Union matching match/session/original video IDs before assignment."""
    groups = []
    for source in sources:
        identifiers = {(key, str(source[key])) for key in ("match_id", "session_id", "original_source_video_id") if source.get(key)}
        touching = [group for group in groups if group["identifiers"] & identifiers]
        combined = {"identifiers": set(identifiers), "sources": [source["source_id"]]}
        for group in touching:
            combined["identifiers"] |= group["identifiers"]
            combined["sources"] += group["sources"]
            groups.remove(group)
        groups.append(combined)
    if len(groups) < 5:
        raise ValueError("A 60/20/20 freeze requires at least five independent groups; smaller authorized samples belong to a pilot")
    groups.sort(key=lambda group: hashlib.sha256(f"{seed}|{'|'.join(sorted(group['sources']))}".encode()).hexdigest())
    val_count = max(1, round(len(groups) * .2))
    test_count = max(1, round(len(groups) * .2))
    train_count = len(groups) - val_count - test_count
    assignments = {}
    for index, group in enumerate(groups):
        split = "train" if index < train_count else "val" if index < train_count + val_count else "test"
        assignments.update({source: split for source in group["sources"]})
    return assignments, {"groups": len(groups), "train": train_count, "val": val_count, "test": test_count}


def encode_instances(instances, role, width, height):
    lines = []
    for instance in instances:
        x1, y1, x2, y2 = instance["bbox_xyxy_px"]
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Native-image box is outside image bounds")
        values = [0, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
        if SHAPES[role]:
            points = instance["keypoints_px"]
            if len(points) != SHAPES[role][0]:
                raise ValueError("Landmark ordering/shape is incompatible with selected head")
            for point in points:
                # Missing point is (0,0,0), with no invented hidden location.
                if point is None:
                    values.extend([0, 0, 0])
                    continue
                x, y, visibility = point
                if not 0 <= x <= width or not 0 <= y <= height or visibility not in (1, 2):
                    raise ValueError("Observed/occluded keypoint must have explicit valid coordinates and visibility")
                values.extend([x / width, y / height, visibility])
        elif instance.get("visibility") != "visible":
            raise ValueError("Only visible evidence-supported ball boxes can be observations")
        lines.append(" ".join(format(value, ".9g") for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def prepare(document, output, *, pilot=False):
    sources = document.get("sources", [])
    annotations = document.get("annotations", [])
    if not sources or not annotations:
        raise ValueError("No reviewed authorized sources/annotations supplied")
    for source in sources:
        required = ("source_id", "match_id", "session_id", "original_source_video_id", "owner", "rights_evidence", "sha256", "fps", "resolution", "surface")
        if any(not source.get(key) for key in required) or source.get("training_allowed") is not True:
            raise ValueError("Missing source provenance or explicit training-rights review")
        if source.get("view") != "elevated_end_court_singles" or source.get("camera_reviewed") is not True:
            raise ValueError("Each source needs reviewed elevated end-court singles framing")
        if source.get("source_path"):
            source_path = Path(source["source_path"])
            source_path = source_path if source_path.is_absolute() else REPO_ROOT / source_path
            if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
                raise ValueError("Registered original source bytes are missing or have a mismatched checksum")
    ids = [source["source_id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated source IDs")
    source_by_id = {source["source_id"]: source for source in sources}
    if pilot:
        assignments = {source["source_id"]: source.get("development_split", "train") for source in sources}
        if any(split not in ("train", "val") for split in assignments.values()):
            raise ValueError("A pilot may contain only training and held-out development sources")
        seen_groups = {}
        for source in sources:
            for key in ("match_id", "session_id", "original_source_video_id"):
                group = (key, str(source[key]))
                if group in seen_groups and seen_groups[group] != assignments[source["source_id"]]:
                    raise ValueError(f"Development split leakage for {key}: {source[key]}")
                seen_groups[group] = assignments[source["source_id"]]
        group_counts = None
    else:
        assignments, group_counts = grouped_splits(sources)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    records = []
    for index, annotation in enumerate(annotations):
        role, source_id = annotation["role"], annotation["source_id"]
        manual_pilot_review = (pilot and annotation.get("manual_reviewed") is True
                               and annotation.get("reviewer_type") == "AGENT_VISUAL_REVIEW")
        if not pilot and annotation.get("reviewer_type") == "AGENT_VISUAL_REVIEW":
            raise ValueError("Agent-only visual labels cannot freeze independent qualification data")
        if role not in SHAPES or source_id not in assignments or not (annotation.get("human_reviewed") is True or manual_pilot_review):
            raise ValueError("Use known roles/sources and explicit visual review; agent review is pilot-only")
        if role == "court" and annotation.get("landmark_schema") != "tennis_court_14_v1":
            raise ValueError("Court annotation must identify the established 14-point order")
        allowed_provenance = ("HUMAN_MANUAL", "PUBLISHER_LABEL_HUMAN_REVIEWED") + (("AGENT_VISUAL_MANUAL",) if pilot else ())
        if annotation.get("annotation_provenance") not in allowed_provenance:
            raise ValueError("Model predictions cannot be accepted as test ground truth")
        image = Path(annotation["image"])
        image = image if image.is_absolute() else REPO_ROOT / image
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Cannot read annotated native image {image}")
        if list(source_by_id[source_id]["resolution"]) != [frame.shape[1], frame.shape[0]]:
            raise ValueError("Annotated image must retain the declared original source resolution")
        digest = sha256_file(image)
        if (role, digest) in seen:
            raise ValueError("Remove duplicate image bytes before converting labels")
        seen.add((role, digest))
        split = assignments[source_id]
        stem = f"{source_id}_{role}_{annotation['frame_index']:09d}"
        destination = output / role / "images" / split / (stem + image.suffix.lower())
        label = output / role / "labels" / split / (stem + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or label.exists():
            raise ValueError("Output sample exists; create a new immutable dataset version")
        label_text = encode_instances(annotation["instances"], role, frame.shape[1], frame.shape[0])
        label.write_text(label_text, encoding="utf-8")
        validate_labels(label, role)
        shutil.copyfile(image, destination)
        records.append({key: annotation[key] for key in ("role", "source_id", "frame_index", "timestamp_seconds", "annotation_provenance")}
                       | {"image": str(destination), "label": str(label), "split": split, "sha256": digest,
                          "label_sha256": sha256_file(label), "near_duplicate_group": annotation.get("near_duplicate_group"),
                          "reviewer_type": annotation.get("reviewer_type", "NOT_SEPARATELY_RECORDED"),
                          "independent_human_reviewed": annotation.get("independent_human_reviewed") is True,
                          "source_pts": annotation.get("source_pts"), "source_time_base": annotation.get("source_time_base"),
                          "publisher_center_xy_px": annotation.get("publisher_center_xy_px"),
                          "review_method": annotation.get("review_method"), "visibility": annotation.get("visibility"),
                          "blur_severity": annotation.get("blur_severity")})
    manifest = {"schema_version": "1.0", "scope": "elevated_end_court_singles", "frozen": not pilot,
                "status": "AUTHORIZED_DEVELOPMENT_PILOT_NOT_QUALIFICATION" if pilot else "FROZEN_LABELS_REVIEWED_REQUIRED",
                "sources": sources, "samples": records, "split_policy": {"target_ratios": {"train": .6, "val": .2, "test": .2}, "group_counts": group_counts, "seed": 709},
                "label_review": {"training_reviewed": True, "independent_test_reviewed": bool(document.get("independent_test_reviewed"))},
                "source_independence_verified": document.get("source_independence_verified") is True,
                "original_annotation_document_sha256": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
                "qualification_evidence": False}
    path = output / "source_manifest.json"
    if path.exists():
        raise ValueError("Never overwrite a dataset manifest; choose a new output version")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    path = prepare(json.loads(args.annotations.read_text(encoding="utf-8")), args.output, pilot=args.pilot)
    print(json.dumps({"manifest": str(path), "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()

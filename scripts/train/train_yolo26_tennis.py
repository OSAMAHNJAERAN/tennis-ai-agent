"""Rights/split-gated training for the three selected YOLO26 specializations.

Run via ``python -m scripts.train.train_yolo26_tennis --help``. A preflight is
the default; ``--execute`` explicitly begins local training after data checks.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from src.detection.yolo26_bundle import PINNED_ULTRALYTICS, REPO_ROOT, sha256_file


SHAPES = {"ball": None, "players": [17, 3], "court": [14, 3]}
NAMES = {"ball": "tennis-ball", "players": "person", "court": "court"}
WEIGHTS = {"ball": "yolo26m.pt", "players": "yolo26s-pose.pt", "court": "yolo26s-pose.pt"}
IMAGE_SIZES = {"ball": 1280, "players": 960, "court": 1280}
COURT_DEVELOPMENT_MANIFEST_SHA256 = "0fadd340acc2fdfa5db9083474f620b396eea1d1eaf8fe025fdf050e43c3c90f"
COURT_DEVELOPMENT_WEIGHTS_SHA256 = "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646"
COURT_DEVELOPMENT_LINEAGE = {
    "frozen_manifest_sha256": ("data/yolo26/court_development_frozen_v1/grouped_development_manifest.json", "35ae2e741c6e8abdd6eb9b0ed516316be2ec5b1f73ef257b61867ea9e5cd851e"),
    "decisions_sha256": ("data/yolo26/court_review_decisions_v1.json", "b085770df3330b89a8371806d2376ea9106db9f2f82d4e5436c78f5af7a634cb"),
    "root_approval_sha256": ("artifacts/validation/yolo26_execution_2026-10-14/court_root_export_approval.json", "26b77be89c57f07ee60a5a3b5303409e23c39a651d0d26627ff6620d77374869"),
    "inventory_sha256": ("data/yolo26/court_review_expansion_v1/cache_inventory.json", "02cc85fa195c4ea5e0c1990f71a0d7f2954c561940f01dde0112c17077beeeae"),
    "native_detail_receipt_sha256": ("data/yolo26/court_native_detail_v1/native_detail_receipt.json", "e811e52595f3d02f2545ba891d768425c74c8f3c138d80f05886108300648e42"),
}
COURT_DEVELOPMENT_EXPORT_FILES = {
    "post_export_verification.json": "acfa90fc0242100e8b4a3c4b839fd4bb32a47e37354487734e5b3031e575a93f",
    "training_plan.json": "9613645693680f7abb742ca2541dd6c22a2215845339c5196fdcb688df4d1323",
    "court/data.yaml": "f4eccecb94211ea3ef6428921220479b68f9723b28902a02a44f3eb852db1134",
    "export_receipt.json": "40f006648197b288b52df64d7fd9a7cfa3ee603eac1899f1c7db727711996b3c",
}
COURT_DEVELOPMENT_EXPORT = REPO_ROOT / "data/yolo26/court_reviewed_development_v1"


def check_resource_guard(resources, *, gpu_requested, role=None, image_size=1280, development=False):
    """Bounded pilot floor, not a promise that the chosen architecture fits."""
    if resources["free_disk_bytes"] < 1024 ** 3:
        raise ValueError("Training resource guard: at least 1 GiB free disk required")
    if resources["free_ram_bytes"] < 512 * 1024 ** 2:
        raise ValueError("Training resource guard: at least 512 MiB available RAM required")
    if gpu_requested and (resources["free_vram_bytes"] is None or resources["free_vram_bytes"] < 2 * 1024 ** 3):
        raise ValueError("Training resource guard: at least 2 GiB free VRAM required; no unrelated processes are stopped")
    if development and role == "court" and image_size >= 1280 and resources["free_ram_bytes"] < 1536 * 1024 ** 2:
        raise ValueError("Reviewed static court 1280 guard: at least 1.5 GiB available RAM required after CUDA initialization")
    if gpu_requested and role == "ball" and image_size >= 1280:
        if resources["free_ram_bytes"] < 1536 * 1024 ** 2:
            raise ValueError("YOLO26m 1280 ball guard: at least 1.5 GiB available RAM required")
        if resources["free_vram_bytes"] < 5_000_000_000:
            raise ValueError("YOLO26m 1280 ball guard: at least 5 GB free VRAM required by the measured batch-1 AMP-off pilot")
    elif gpu_requested and role == "ball" and resources["free_ram_bytes"] < 1024 ** 3:
        raise ValueError("Native ball crop guard: at least 1 GiB available RAM required for model/optimizer headroom")


def validate_labels(path, role):
    """Validate normalized YOLO box+keypoint labels, including negative images."""
    count = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()], dtype=float)
        required = 5 + (int(SHAPES[role][0]) * 3 if SHAPES[role] else 0)
        if len(values) != required or not np.isfinite(values).all() or values[0] != 0:
            raise ValueError(f"Invalid {role} row {path}:{line_number}: expected class 0 and {required} finite fields")
        cx, cy, width, height = values[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Out-of-range or empty box {path}:{line_number}")
        if cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"Box exceeds image {path}:{line_number}")
        if SHAPES[role]:
            keypoints = values[5:].reshape(-1, 3)
            if not np.isin(keypoints[:, 2], [0, 1, 2]).all() or (keypoints[:, :2] < 0).any() or (keypoints[:, :2] > 1).any():
                raise ValueError(f"Invalid normalized keypoint/visibility {path}:{line_number}")
        count += 1
    if role == "court" and count > 1:
        raise ValueError("Label one court instance per image")
    return count


def audit_court_development_manifest(manifest_path, role="court"):
    """The single reviewed static export is permitted only for private development.

    Publisher still-image ordinals are not timestamps or verified match IDs.
    This exception never relaxes the temporal or qualification audit below.
    All checks are file/CPU checks; images are hashed without decoding them.
    """
    if role != "court":
        raise ValueError("Static development mode is restricted to the reviewed court export")
    path = Path(manifest_path).resolve()
    if sha256_file(path) != COURT_DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError("Require the exact pinned reviewed court development manifest SHA256")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "1.1"
            or manifest.get("status") != "AUTHORIZED_AGENT_REVIEWED_DEVELOPMENT_COURT_NOT_QUALIFICATION"
            or manifest.get("scope") != "elevated_end_court_singles"
            or manifest.get("frozen") is not False
            or manifest.get("development_decisions_frozen") is not True
            or any(manifest.get(key) is not False for key in ("qualification_evidence", "source_independence_verified", "final_test_used"))
            or manifest.get("label_review") != {"training_reviewed": True, "independent_test_reviewed": False}):
        raise ValueError("Require frozen development review without independent human, final-test or qualification claims")
    counts = {"train": 100, "val": 20, "test": 0}
    if manifest.get("counts") != counts or any(row.get("split") not in ("train", "val") or row.get("role") != "court" for row in manifest.get("samples", [])):
        raise ValueError("Reviewed static court development must contain 100 train/20 DEV and no TEST or other role")
    rights = manifest.get("rights", {})
    if (rights.get("basis") != "AUTHOR_EXPLICIT_DATASET_RELEASE_WITH_TRAINING_INSTRUCTIONS"
            or rights.get("allowed_scope") != "PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT_ONLY"
            or rights.get("broadcast_redistribution_cleared") is not False
            or rights.get("explicit_raw_dataset_license") is not None
            or rights.get("code_license_is_not_data_permission") is not True
            or any(not rights.get(key) for key in ("author", "author_readme", "training_instructions", "dataset_url"))):
        raise ValueError("Require explicit private academic training release and unresolved raw redistribution rights")
    lineage, lineage_paths = {}, {}
    for key, (relative, expected) in COURT_DEVELOPMENT_LINEAGE.items():
        lineage_path = REPO_ROOT / relative
        if manifest.get("lineage", {}).get(key) != expected or sha256_file(lineage_path) != expected:
            raise ValueError(f"Reviewed static court {key} binding changed")
        lineage[key] = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage_paths[key] = lineage_path
    frozen, decisions = lineage["frozen_manifest_sha256"], lineage["decisions_sha256"]
    from scripts.data.export_yolo26_reviewed_court import encode_court_label, selected_rows, validate_approval, validate_native_evidence
    from scripts.data.prepare_yolo26_court_review import context_components, validate_review
    validate_approval(lineage["root_approval_sha256"], frozen, lineage_paths["frozen_manifest_sha256"], lineage_paths["decisions_sha256"])
    review = lineage_paths["inventory_sha256"].parent
    if sha256_file(review / "predeclared_group_protocol.json") != frozen.get("protocol_sha256"):
        raise ValueError("Frozen predeclared development protocol changed")
    validate_native_evidence(review, decisions)
    inventory = lineage["inventory_sha256"]
    accepted, context_splits = validate_review(inventory, decisions)
    selected, surplus = selected_rows(accepted)
    _, contexts = context_components(inventory["samples"], decisions["samples"])
    if (frozen.get("selected") != selected or frozen.get("unused_accepted_development_ids") != surplus
            or frozen.get("context_splits") != context_splits or frozen.get("context_groups") != contexts
            or manifest.get("split_policy") != {"context_splits": context_splits, "context_groups": contexts,
                                               "rejected_context_links_retained": True, "test": 0, "unused_ids_are_development": True}):
        raise ValueError("Transitive original video/event context or frozen development split changed")
    selected_by_id = {row["id"]: row for row in selected}
    inventory_by_id = {row["id"]: row for row in inventory["samples"]}
    sources = manifest.get("sources", [])
    samples = manifest.get("samples", [])
    source_by_id = {source["source_id"]: source for source in sources}
    if len(sources) != 120 or len(source_by_id) != 120 or [row.get("id") for row in samples] != [row["id"] for row in selected]:
        raise ValueError("Require exact frozen ordered 120-image selection with unique sources")
    seen_hashes, seen_paths, seen_groups = set(), set(), {}
    records, actual_counts, visible, null = [], dict.fromkeys(counts, 0), 0, 0
    for sample in samples:
        ident, split = sample["id"], sample["split"]
        row, native = selected_by_id[ident], inventory_by_id[ident]
        source = source_by_id.get(sample.get("source_id"), {})
        if (source.get("source_type") != "PUBLISHER_STILL_IMAGE" or source.get("training_allowed") is not True
                or source.get("rights_evidence") != rights or not source.get("owner")
                or source.get("view") != manifest["scope"] or source.get("camera_reviewed") is not True):
            raise ValueError("Every court source requires reviewed publisher still-image provenance and private training rights")
        if (any(source.get(key) is not None for key in ("match_id", "fps", "source_pts", "source_time_base"))
                or any(sample.get(key) is not None for key in ("timestamp_seconds", "source_pts", "source_time_base"))
                or source.get("context_id_is_verified_match_identity") is not False
                or source.get("original_video_bytes_acquired") is not False):
            raise ValueError("Static image FPS/PTS/original match identity must remain unknown; context IDs are not verified sessions")
        context = row["context_group"]
        if (source.get("original_source_video_id") != native["source_video_id"]
                or source.get("publisher_frame_ordinal") != native["source_frame_ordinal"]
                or sample.get("frame_index") != native["source_frame_ordinal"]
                or source.get("source_path") != native["source_image"]
                or source.get("resolution") != [native["width"], native["height"]]
                or source.get("surface") != row["surface"] or source.get("visual_event_family") != row["visual_event_family"]
                or source.get("context_group") != context or source.get("session_id") != context
                or source.get("development_split") != split or row["development_split"] != split
                or sample.get("near_duplicate_group") != context or context_splits.get(context) != split):
            raise ValueError("Publisher ordinal/native source or transitive video/context split integrity changed")
        for kind, value in (("video", source["original_source_video_id"]), ("context", context)):
            if seen_groups.setdefault((kind, value), split) != split:
                raise ValueError("Original video or transitive event context crosses train/DEV")
        if (sample.get("reviewer_type") != "AGENT_VISUAL_REVIEW" or sample.get("independent_human_reviewed") is not False
                or sample.get("annotation_provenance") != "PUBLISHER_COORDINATES_AGENT_NATIVE_VISIBILITY_REVIEWED"
                or sample.get("landmark_schema") != "tennis_court_14_v1"
                or sample.get("keypoints_px") != row["reviewed_keypoints_px"]
                or sample.get("bbox_xyxy_px") != row["bbox_xyxy_px"]
                or sample.get("native_detail_board_sha256") != row["native_detail_board_sha256"]
                or sample.get("publisher_annotation_sha256") != native["publisher_annotation_sha256"]):
            raise ValueError("Frozen court14 visible/null mask, label review or native annotation binding changed")
        image = REPO_ROOT / sample["image"]
        label = REPO_ROOT / sample["label"]
        expected_image = COURT_DEVELOPMENT_EXPORT / "court" / "images" / split / (ident + ".png")
        expected_label = COURT_DEVELOPMENT_EXPORT / "court" / "labels" / split / (ident + ".txt")
        if image.resolve() != expected_image.resolve() or label.resolve() != expected_label.resolve() or str(label.resolve()) in seen_paths:
            raise ValueError("Require exact exported images/labels layout without reused label paths")
        seen_paths.add(str(label.resolve()))
        digest = sha256_file(image)
        if (digest != sample.get("sha256") or digest != source.get("sha256") or digest != native["sha256"]
                or sha256_file(REPO_ROOT / source["source_path"]) != digest or digest in seen_hashes
                or sha256_file(label) != sample.get("label_sha256")):
            raise ValueError("Native image/source/label checksum mismatch or duplicate image bytes")
        seen_hashes.add(digest)
        if label.read_text(encoding="utf-8") != encode_court_label(native, row) or validate_labels(label, "court") != 1:
            raise ValueError("Court label must retain exactly one class 0 instance, fourteen visible/null slots and reviewed native coordinates")
        visible += sum(point is not None for point in row["reviewed_keypoints_px"])
        null += sum(point is None for point in row["reviewed_keypoints_px"])
        actual_counts[split] += 1
        records.append({"image": str(image.resolve()), "label": str(label.resolve()), "split": split, "instances": 1})
    if actual_counts != counts or (visible, null) != (1657, 23):
        raise ValueError("Reviewed court development counts or fixed visible/null denominator changed")
    export_hashes = {}
    for relative, expected in COURT_DEVELOPMENT_EXPORT_FILES.items():
        if sha256_file(COURT_DEVELOPMENT_EXPORT / relative) != expected:
            raise ValueError(f"Pinned court export evidence changed: {relative}")
        export_hashes[relative] = expected
    verification = json.loads((COURT_DEVELOPMENT_EXPORT / "post_export_verification.json").read_text(encoding="utf-8"))
    if (verification.get("source_manifest_sha256") != COURT_DEVELOPMENT_MANIFEST_SHA256
            or verification.get("counts") != counts or verification.get("keypoint_slots") != {"visible": visible, "unknown_null": null}
            or verification.get("video_id_overlap") != [] or verification.get("context_overlap") != []):
        raise ValueError("Post-export verification does not bind this development selection")
    return manifest, records, {"manifest_sha256": sha256_file(path), "split_image_counts": counts,
                               "labeled_instances": 120, "source_count": 120, "pilot": False, "development": True,
                               "final_test_used": False, "qualification_evidence": False,
                               "held_out_development_sources_available": True, "independent_validation_available": False,
                               "independent_human_reviewed": False, "static_image_timing_is_unknown": True,
                               "visible_keypoint_slots": visible, "unknown_null_keypoint_slots": null,
                               "lineage_sha256": dict(manifest["lineage"]), "export_evidence_sha256": export_hashes,
                               "usage_scope": rights["allowed_scope"], "raw_redistribution_cleared": False}


def audit_manifest(manifest_path, role, *, pilot=False, development=False):
    """Reject missing rights, cross-match leakage and altered/missing labels."""
    if development:
        if pilot:
            raise ValueError("Development and pilot audits are separate modes")
        return audit_court_development_manifest(manifest_path, role)
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("scope") != "elevated_end_court_singles" or (not pilot and manifest.get("frozen") is not True):
        raise ValueError("Require an elevated_end_court_singles manifest, frozen for controlled full training")
    sources = manifest.get("sources", [])
    samples = [sample for sample in manifest.get("samples", []) if sample.get("role") == role
               and (not pilot or sample.get("split") != "test")]
    if not sources or not samples:
        raise ValueError(f"No authorized labeled {role} sources/samples are registered; training remains blocked")
    source_by_id = {}
    for source in sources:
        required = ("source_id", "match_id", "session_id", "rights_evidence", "sha256", "owner", "surface", "fps", "resolution")
        if any(not source.get(key) for key in required) or source.get("training_allowed") is not True:
            raise ValueError("Every source needs explicit training rights and match/session/camera provenance")
        if source.get("view") != "elevated_end_court_singles":
            raise ValueError("Source is outside the selected camera/singles scope")
        if source["source_id"] in source_by_id:
            raise ValueError("Duplicate source_id")
        source_by_id[source["source_id"]] = source
    seen_groups, seen_hashes, seen_labels = {}, {}, set()
    counts = {"train": 0, "val": 0, "test": 0}
    records = []
    for sample in samples:
        split = sample.get("split")
        if split not in counts or sample.get("source_id") not in source_by_id:
            raise ValueError("Every sample needs a declared split and registered source")
        source = source_by_id[sample["source_id"]]
        for kind in ("match_id", "session_id", "source_id", "original_source_video_id"):
            if not source.get(kind):
                continue
            group = (kind, source[kind])
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError(f"Split leakage for {kind}: {source[kind]}")
            seen_groups[group] = split
        if sample.get("near_duplicate_group"):
            group = ("near_duplicate", sample["near_duplicate_group"])
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Near-duplicate frames cross splits")
            seen_groups[group] = split
        image = Path(sample["image"])
        image = image if image.is_absolute() else REPO_ROOT / image
        label = Path(sample["label"])
        label = label if label.is_absolute() else REPO_ROOT / label
        if not image.is_file() or not label.is_file():
            raise ValueError(f"Missing image/label: {image}")
        digest = sha256_file(image)
        if digest != sample.get("sha256") or sha256_file(label) != sample.get("label_sha256"):
            raise ValueError(f"Image/label checksum mismatch: {image}")
        if digest in seen_hashes:
            raise ValueError("Duplicate image bytes must be removed before freezing splits")
        seen_hashes[digest] = split
        if str(label.resolve()) in seen_labels:
            raise ValueError("A label path was reused for multiple images")
        seen_labels.add(str(label.resolve()))
        if "images" not in image.parts or label != Path(str(image).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt"):
            raise ValueError("Use matching images/... and labels/... layout required by Ultralytics")
        instances = validate_labels(label, role)
        records.append({"image": str(image), "label": str(label), "split": split, "instances": instances})
        counts[split] += 1
    if pilot and (not counts["train"] or not manifest.get("label_review", {}).get("training_reviewed")):
        raise ValueError("Pilot requires reviewed, authorized training labels; held-out final labels are excluded")
    if not pilot and (not counts["train"] or not counts["val"] or not counts["test"]):
        raise ValueError("Reserve independent train, validation and frozen final-test sessions")
    if not pilot and not manifest.get("label_review", {}).get("independent_test_reviewed"):
        raise ValueError("Independent test label review must be recorded")
    return manifest, records, {"manifest_sha256": sha256_file(path), "split_image_counts": counts,
                               "labeled_instances": sum(record["instances"] for record in records), "source_count": len(sources),
                               "pilot": pilot, "final_test_used": False,
                               "held_out_development_sources_available": bool(counts["val"]),
                               "independent_validation_available": bool(counts["val"]) and manifest.get("source_independence_verified") is True}


def prepare_dataset(records, role, output, *, pilot=False, development=False):
    if development and (role != "court" or pilot or any(record["split"] not in ("train", "val") for record in records)
                        or not all(any(record["split"] == split for record in records) for split in ("train", "val"))):
        raise ValueError("Court development requires separate train/DEV records and forbids TEST")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in (("train", "val") if pilot or development else ("train", "val", "test")):
        (output / f"{split}.txt").write_text("\n".join(record["image"] for record in records if record["split"] == split) + "\n", encoding="utf-8")
    data = {"path": str(output), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": {0: NAMES[role]}}
    if pilot or development:
        data.pop("test")
        if pilot and not any(record["split"] == "val" for record in records):
            # A same-sample loss/label overfit check is permissible for a pilot.
            # It is explicitly excluded from accuracy/checkpoint qualification.
            data["val"] = "train.txt"
    if SHAPES[role]:
        data["kpt_shape"] = SHAPES[role]
        # Reflection is disabled below; retain correct semantics if a later
        # limited, documented experiment enables it.
        data["flip_idx"] = ([1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13] if role == "court"
                            else [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def build_training_model(role, weights, output):
    """Adapt the actual task head, transferring only shape-compatible tensors."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import yaml_model_load

    weights = Path(weights).resolve()
    if not weights.is_file() or weights.name != WEIGHTS[role]:
        raise ValueError(f"Start from the local official {WEIGHTS[role]} checkpoint")
    source = YOLO(str(weights))
    expected_task = "detect" if role == "ball" else "pose"
    expected_scale = "m" if role == "ball" else "s"
    if source.task != expected_task or source.model.yaml.get("scale") != expected_scale or "yolo26" not in source.model.yaml.get("yaml_file", ""):
        raise ValueError("Initial checkpoint is not the selected official YOLO26 task/scale")
    architecture_name = "yolo26m.yaml" if role == "ball" else "yolo26s-pose.yaml"
    architecture = yaml_model_load(architecture_name)
    architecture["nc"] = 1
    if SHAPES[role]:
        architecture["kpt_shape"] = SHAPES[role]
    architecture_path = Path(output) / ("yolo26m-tennis-ball.yaml" if role == "ball" else f"yolo26s-{role}-pose.yaml")
    architecture_path.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
    model = YOLO(str(architecture_path), task=expected_task)
    source_state, target_state = source.model.state_dict(), model.model.state_dict()
    transferred = sorted(key for key, tensor in source_state.items() if key in target_state and tensor.shape == target_state[key].shape)
    initialized = sorted(set(target_state) - set(transferred))
    model.load(str(weights))
    # The trainer sets this attribute from dataset metadata. Set it here as
    # well so a pre-training forward/predict smoke can decode the custom head.
    if SHAPES[role]:
        model.model.kpt_shape = SHAPES[role]
    model.model.names = {0: NAMES[role]}
    return model, {"official_weights": str(weights), "weights_sha256": sha256_file(weights),
                   "architecture": str(architecture_path), "transferred_tensor_keys": transferred,
                   "initialized_tensor_keys": initialized, "kpt_shape": SHAPES[role],
                   "court_semantics": "HUMAN_WEIGHTS_ARE_NOT_TRAINED_COURT_LANDMARKS" if role == "court" else None}


def validate_development_extension(evidence_path, checkpoint, metadata_path, requested_epochs):
    """Bind a root-reviewed extension to actual retained DEV metrics and weights."""
    path, checkpoint, metadata_path = Path(evidence_path).resolve(), Path(checkpoint).resolve(), Path(metadata_path).resolve()
    evidence = json.loads(path.read_text(encoding="utf-8"))
    concrete_review = all(isinstance(evidence.get(key), str) and len(evidence[key].strip()) >= 20
                          for key in ("observed_useful_learning", "review_note"))
    if (evidence.get("schema") != "tennis.yolo26.court-development-extension.v1"
            or evidence.get("status") != "ROOT_REVIEWED_USEFUL_LEARNING_EXTENSION_APPROVED"
            or evidence.get("reviewer_type") != "ROOT_AGENT_DEVELOPMENT_METRICS_REVIEW"
            or not concrete_review
            or evidence.get("qualification_evidence") is not False or evidence.get("final_test_used") is not False
            or evidence.get("requested_total_epochs") != requested_epochs or not 21 <= requested_epochs <= 100):
        raise ValueError("Extension requires root review of observed useful DEV learning, exact total <=100, and no qualification claim")
    for key, expected_path in (("prior_run_metadata", metadata_path), ("checkpoint", checkpoint)):
        binding = evidence.get(key, {})
        if Path(binding.get("path", "")).resolve() != expected_path or binding.get("sha256") != sha256_file(expected_path):
            raise ValueError(f"Development extension {key} must bind the retained original bytes")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("role") != "court" or metadata.get("mode") != "REVIEWED_STATIC_COURT_DEVELOPMENT"
            or metadata.get("data_audit", {}).get("manifest_sha256") != COURT_DEVELOPMENT_MANIFEST_SHA256):
        raise ValueError("Extension must continue the same reviewed static court development run")
    binding = evidence.get("metrics_csv", {})
    metrics = Path(binding.get("path", "")).resolve()
    if metrics != checkpoint.parents[1] / "results.csv" or binding.get("sha256") != sha256_file(metrics):
        raise ValueError("Extension must bind the original run's actual results.csv bytes")
    with metrics.open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    epochs = [int(row["epoch"]) for row in rows]
    selected_epochs = evidence.get("observed_epoch_ids", [])
    if (not rows or len(epochs) != len(set(epochs)) or len(selected_epochs) < 2
            or selected_epochs != sorted(set(selected_epochs)) or not set(selected_epochs) <= set(epochs)
            or selected_epochs[-1] >= requested_epochs or selected_epochs[-1] < 20):
        raise ValueError("Extension requires at least two actual ordered observed epoch IDs including epoch20 or later")
    checkpoint_epoch = evidence.get("checkpoint_epoch")
    if checkpoint_epoch not in epochs or checkpoint_epoch != selected_epochs[-1]:
        raise ValueError("Extension checkpoint must correspond to the last reviewed actual epoch")
    observed = {int(row["epoch"]): row for row in rows}
    improvements = evidence.get("improved_loss_columns", [])
    if not improvements:
        raise ValueError("Extension must identify observed useful loss changes in the retained CSV")
    for column in improvements:
        if not (column.startswith(("train/", "val/")) and column.endswith("_loss")):
            raise ValueError("Extension improvement columns must name actual training or DEV losses")
        values = np.asarray([float(observed[epoch][column]) for epoch in selected_epochs])
        if not np.isfinite(values).all() or (values < 0).any() or values[-1] >= values[0]:
            raise ValueError("Extension claimed loss improvement is not present in the retained observed epochs")
    return {"path": str(path), "sha256": sha256_file(path), "review": evidence}


def _development_resume_trainer(settings):
    """Keep optimizer state while applying the reviewed total and fresh output.

    Pinned 8.4.36 deliberately drops an epochs override in check_resume. Apply
    the already-audited settings immediately after that method, before trainer
    initialization constructs the scheduler or output paths. Completed stripped
    best/last checkpoints are rejected separately; use the periodic checkpoint.
    """
    from ultralytics.models.yolo.pose import PoseTrainer

    class ReviewedCourtResumeTrainer(PoseTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            for key, value in settings.items():
                setattr(self.args, key, value)
            # The resumed checkpoint retains save_dir, which otherwise wins
            # over project/name and would overwrite the original run receipts.
            self.args.save_dir = str(Path(settings["project"]) / settings["name"])

    return ReviewedCourtResumeTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=SHAPES, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="At most two epochs, or four for a reviewed pilot; not accuracy evidence")
    parser.add_argument("--pilot", action="store_true", help="Reviewed authorized development data only; never consume final-test labels")
    parser.add_argument("--development", action="store_true", help="Exact reviewed static court export only; private development, never qualification")
    parser.add_argument("--development-extension-evidence", type=Path, help="Hash-bound root review of retained useful learning before extending beyond epoch20")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, choices=[640, 960, 1280], help="640 is only permitted for a documented native ball-crop development pilot")
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true", help="Reserved until a same-model AMP check is implemented; pinned runtime's checker loads an extra YOLO26n")
    parser.add_argument("--seed", type=int, default=709)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--player-finetune-reason")
    args = parser.parse_args(argv)
    args.epochs = args.epochs if args.epochs is not None else (20 if args.development else 100)
    args.manifest = args.manifest or (COURT_DEVELOPMENT_EXPORT / "source_manifest.json" if args.development else REPO_ROOT / "data/yolo26/source_manifest.json")
    if args.amp:
        parser.error("Pinned runtime AMP check loads an extra YOLO26n; keep AMP disabled until a same-model check is verified")
    if not 1 <= args.epochs <= 150 or args.batch < 1:
        parser.error("Use 1-150 epochs and a measured positive batch size")
    if args.development:
        if args.role != "court" or args.pilot or args.smoke:
            parser.error("--development is a separate court-only mode; do not combine it with --pilot or --smoke")
        if args.batch != 1 or args.image_size not in (None, 1280) or not 15 <= args.patience <= 20:
            parser.error("Reviewed court development fixes batch1/image1280 and early-stop patience15-20")
        if args.epochs > 100 or (args.epochs > 20 and (not args.development_extension_evidence or not args.resume)):
            parser.error("Initial development is at most20 epochs; extension to at most100 requires --resume and --development-extension-evidence")
        if args.development_extension_evidence and (args.epochs <= 20 or not args.resume):
            parser.error("Extension evidence requires --resume and a reviewed total above20 epochs")
        if args.resume and args.weights:
            parser.error("Development resume uses only its exact checkpoint; omit --weights")
    elif args.development_extension_evidence or args.patience != 20:
        parser.error("Development extension evidence and configurable patience apply only to --development")
    if args.smoke and args.epochs > (4 if args.pilot else 2):
        parser.error("--smoke is limited to two epochs, or four for a reviewed development pilot")
    if args.pilot and (not args.smoke or args.epochs > 4):
        parser.error("--pilot requires --smoke and at most four epochs")
    if args.role == "players" and args.execute and not args.player_finetune_reason:
        parser.error("Fine-tune players only after an evaluation establishes need; record --player-finetune-reason")
    manifest, records, data_audit = audit_manifest(args.manifest, args.role, pilot=args.pilot, development=args.development)
    image_size = args.image_size or IMAGE_SIZES[args.role]
    if image_size != IMAGE_SIZES[args.role] and not (
        args.role == "ball" and args.pilot and image_size == 640
        and manifest.get("training_transform", {}).get("kind") == "NATIVE_640_JITTERED_DEVELOPMENT_CROPS"
    ):
        parser.error("An image-size change requires the reviewed native640 ball-crop pilot manifest")
    installed_ultralytics = importlib.metadata.version("ultralytics")
    if installed_ultralytics != PINNED_ULTRALYTICS:
        raise ValueError(f"Pinned runtime is Ultralytics {PINNED_ULTRALYTICS}")
    resume_metadata, extension = None, None
    if args.resume:
        checkpoint = args.resume.resolve()
        metadata = checkpoint.parents[2] / "run_metadata.json"
        if not checkpoint.is_file() or not metadata.is_file():
            raise ValueError("Resume checkpoint/original run metadata is missing")
        resume_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if resume_metadata["data_audit"]["manifest_sha256"] != data_audit["manifest_sha256"] or resume_metadata["role"] != args.role:
            raise ValueError("Resume must retain the original role and frozen data manifest")
        if args.development:
            if resume_metadata.get("mode") != "REVIEWED_STATIC_COURT_DEVELOPMENT":
                raise ValueError("Development resume cannot continue a pilot or qualification run")
            if args.development_extension_evidence:
                extension = validate_development_extension(args.development_extension_evidence, checkpoint, metadata, args.epochs)
            elif resume_metadata["training_arguments"]["epochs"] != args.epochs:
                raise ValueError("Interrupted development resume must retain its original total epochs")
    if args.development and not args.resume:
        args.weights = args.weights or REPO_ROOT / "models/yolo26/yolo26s-pose.pt"
        if (args.weights.resolve() != REPO_ROOT / "models/yolo26/yolo26s-pose.pt"
                or sha256_file(args.weights) != COURT_DEVELOPMENT_WEIGHTS_SHA256):
            raise ValueError("Reviewed court development starts only from the hash-pinned local official YOLO26s-pose checkpoint")
    output = args.output.resolve()
    if args.development and output.exists() and any(output.iterdir()):
        raise ValueError("Preserve development evidence; choose a fresh empty output directory, including for resume")
    output.mkdir(parents=True, exist_ok=True)
    if args.execute and not args.resume and any(output.rglob("*.pt")):
        raise ValueError("Training output already contains checkpoints; choose a new immutable run directory")
    data = prepare_dataset(records, args.role, output, pilot=args.pilot, development=args.development)
    revision = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    settings = dict(data=str(data), epochs=args.epochs, imgsz=image_size, batch=args.batch,
                    device=args.device, seed=args.seed, deterministic=True, patience=args.patience, save_period=5,
                    amp=args.amp, workers=0, cache=False, optimizer="AdamW", project=str(output), name="training",
                    mosaic=0., close_mosaic=0, mixup=0., copy_paste=0., scale=.1, translate=.03,
                    fliplr=0., flipud=0., degrees=0., perspective=0., hsv_h=.015, hsv_s=.25, hsv_v=.25)
    if args.pilot:
        # Default nbs=64 and the minimum 100 warmup batches made the six-image
        # pilot perform only two updates. A bounded learning check needs an
        # actual update for each batch, without silently consuming final data.
        settings.update(nbs=1, warmup_epochs=0., save_period=1, lr0=.0005)
    if args.development:
        settings.update(nbs=1, warmup_epochs=0., save_period=5, lr0=.0005, auto_augment=None,
                        augmentations=[], cutmix=0., erasing=0., bgr=0., multi_scale=0., rect=False,
                        translate=0., shear=0., scale=.05, hsv_h=.01, hsv_s=.15, hsv_v=.15)
    runtime_probe_performed = not args.development or args.execute
    runtime = {"torch": importlib.metadata.version("torch"), "ultralytics": installed_ultralytics,
               "cuda": None, "gpu": None, "free_vram_bytes": None, "free_ram_bytes": None}
    if runtime_probe_performed:
        import torch
        import ultralytics
        import psutil
        cuda_available = torch.cuda.is_available()
        runtime.update(torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if cuda_available else None,
                       free_vram_bytes=torch.cuda.mem_get_info()[0] if cuda_available else None,
                       free_ram_bytes=psutil.virtual_memory().available)
    evidence = {"status": "PREFLIGHT_PASSED_NOT_TRAINED", "role": args.role, "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": revision, "training_script_sha256": sha256_file(__file__),
                "python": sys.version, "python_executable": sys.executable, **runtime,
                "runtime_probe_performed": runtime_probe_performed, "resource_guard_pending": True,
                "free_disk_bytes": shutil.disk_usage(output).free, "data_audit": data_audit,
                "training_arguments": settings, "player_finetune_reason": args.player_finetune_reason,
                "selection_policy": "PILOT_CHECKPOINT_NEVER_PROMOTED" if args.pilot else "INDEPENDENT_VALIDATION_TASK_METRICS_REQUIRED_BEFORE_PROMOTION",
                "validation_policy": ("HELD_OUT_DEVELOPMENT_LABEL_CHECK_NOT_QUALIFICATION" if args.pilot and data_audit["held_out_development_sources_available"]
                                      else "TRAINING_LABEL_OVERFIT_CHECK_NOT_ACCURACY_EVIDENCE" if args.pilot
                                      else "GROUPED_DEVELOPMENT_VALIDATION_SOURCE_INDEPENDENCE_RECORDED_SEPARATELY"),
                "test_policy": "FINAL_TEST_IS_NOT_RUN_BY_TRAINING_SCRIPT"}
    evidence["dataset_yaml_sha256"] = sha256_file(data)
    evidence["training_arguments_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if args.development:
        evidence.update(mode="REVIEWED_STATIC_COURT_DEVELOPMENT", qualification_evidence=False, final_test_used=False,
                        independent_human_reviewed=False, selection_policy="DEVELOPMENT_CHECKPOINT_NEVER_PROMOTED",
                        validation_policy="AGENT_REVIEWED_STATIC_DEV_LOSS_AND_METRICS_NOT_INDEPENDENT_QUALIFICATION",
                        initialization="OFFICIAL_YOLO26S_POSE_REBUILT_NC1_COURT14_HEAD", extension_review=extension,
                        initial_epoch_protocol=20, maximum_total_epochs=100,
                        best_checkpoint_policy="ULTRALYTICS_DEV_FITNESS_FOR_ENGINEERING_REVIEW_ONLY",
                        periodic_checkpoint_policy="ULTRALYTICS_SAVE_PERIOD5_PLUS_UNSTRIPPED_LAST_EPOCH_COPY_FOR_REVIEWED_RESUME")
        if args.weights:
            evidence["initial_weights"] = {"path": str(args.weights.resolve()), "sha256": sha256_file(args.weights)}
    if manifest.get("training_transform"):
        evidence["training_transform"] = manifest["training_transform"]
        evidence["validation_policy"] = "JITTERED_NATIVE_DEVELOPMENT_CROP_CHECK_NOT_FULL_FRAME_ACCURACY_OR_QUALIFICATION"
    evidence_path = output / "run_metadata.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if args.execute:
        if not args.weights and not args.resume:
            parser.error("--execute requires local --weights or an exact --resume checkpoint")
        try:
            check_resource_guard(evidence, gpu_requested=args.device != "cpu", role=args.role, image_size=image_size, development=args.development)
        except ValueError as exc:
            evidence.update(status="RESOURCE_GUARD_BLOCKED_NOT_TRAINED", failure_type=type(exc).__name__,
                            failure_reason=str(exc), failed_utc=datetime.now(timezone.utc).isoformat())
            evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            raise
        evidence["resource_guard_pending"] = False
        if args.resume:
            from ultralytics import YOLO
            checkpoint = args.resume.resolve()
            model = YOLO(str(checkpoint))
            if args.development:
                if not model.ckpt or int(model.ckpt.get("epoch", -1)) < 0 or model.ckpt.get("optimizer") is None:
                    raise ValueError("Resume needs an unstripped periodic/interrupted checkpoint with epoch and optimizer state")
                if extension and int(model.ckpt["epoch"]) + 1 != extension["review"]["checkpoint_epoch"]:
                    raise ValueError("Actual checkpoint epoch differs from the root-reviewed retained DEV metrics")
            evidence["resume_checkpoint_sha256"] = sha256_file(checkpoint)
            evidence["original_run_metadata"] = resume_metadata
            training_args = {"resume": str(checkpoint), "device": args.device}
            if args.development:
                training_args = dict(settings, resume=str(checkpoint), trainer=_development_resume_trainer(settings))
        else:
            model, transfer = build_training_model(args.role, args.weights, output)
            evidence["transfer"] = transfer
            training_args = settings
        def record_optimizer_updates(trainer):
            steps = sorted({float(state["step"]) for state in trainer.optimizer.state.values()
                            if "step" in state})
            evidence.setdefault("optimizer_update_receipts", []).append({
                "epoch": int(trainer.epoch) + 1, "optimizer_step_values": steps,
                "batches_per_epoch": len(trainer.train_loader),
                "accumulation": int(trainer.accumulate),
                "warmup_epochs": float(trainer.args.warmup_epochs),
            })
            evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        model.add_callback("on_train_epoch_end", record_optimizer_updates)
        if args.development:
            def preserve_final_optimizer_checkpoint(trainer):
                epoch = int(trainer.epoch) + 1
                if (epoch == int(trainer.epochs) or trainer.stop) and Path(trainer.last).is_file():
                    checkpoint_copy = Path(trainer.wdir) / f"development_epoch{epoch}.pt"
                    shutil.copyfile(trainer.last, checkpoint_copy)
                    evidence["unstripped_resume_checkpoint"] = {"path": str(checkpoint_copy), "epoch": epoch,
                                                                 "sha256": sha256_file(checkpoint_copy)}
                    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            model.add_callback("on_model_save", preserve_final_optimizer_checkpoint)
        evidence["status"] = "TRAINING_STARTED_UNQUALIFIED"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        training_started = time.perf_counter()
        if torch.cuda.is_available() and args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        try:
            model.train(**training_args)
        except Exception as exc:
            evidence["status"] = "TRAINING_FAILED_UNQUALIFIED"
            evidence["failure_type"] = type(exc).__name__
            evidence["failure_reason"] = str(exc)
            evidence["failed_utc"] = datetime.now(timezone.utc).isoformat()
            evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            raise
        evidence["status"] = "TRAINED_UNQUALIFIED"
        evidence["completed_utc"] = datetime.now(timezone.utc).isoformat()
        evidence["training_elapsed_seconds"] = time.perf_counter() - training_started
        if torch.cuda.is_available() and args.device != "cpu":
            evidence["peak_torch_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            evidence["peak_torch_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        for checkpoint in sorted(output.rglob("*.pt")):
            evidence.setdefault("checkpoints", []).append({"path": str(checkpoint), "sha256": sha256_file(checkpoint)})
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "role": args.role, "metadata": str(evidence_path), "data_audit": data_audit}))


if __name__ == "__main__":
    main()

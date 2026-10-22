"""Fixed-protocol pixel diagnostic for the twenty reviewed court DEV stills.

Default execution validates completed-run provenance using only the standard
library. ``--execute`` additionally runs the one retained court checkpoint.
Scoring never searches thresholds, chooses a box by ground truth, consumes a
TEST image, infers a hidden landmark or promotes a development checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import struct
import time


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/yolo26/court_reviewed_development_v1/source_manifest.json"
RUN = ROOT / "outputs/yolo26_court_reviewed_development_run1"
CHECKPOINT = RUN / "training/weights/best.pt"
PROTOCOL = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/court_development_run1/post20_diagnostic_protocol.json"
MANIFEST_SHA256 = "0fadd340acc2fdfa5db9083474f620b396eea1d1eaf8fe025fdf050e43c3c90f"
PROTOCOL_SHA256 = "edd562e603c6ed7abd517b384daca3226fd00cf76ec074cbdac4b83075212760"
TRAINER_SHA256 = "cd05fe4983bb9edac77d76ebc32f1b59a1aa0697392abb15ff096608a7d6d6b1"
OFFICIAL_SHA256 = "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646"
TRAINING_SOURCES_START_SHA256 = "ee16b1290c514b62db74527d080601d86473c4c4626f0cce560c397a5c371b33"
PINNED_ULTRALYTICS = "8.4.36"
INFERENCE = {"imgsz": 1280, "batch": 1, "conf": .05, "iou": .7, "max_det": 10,
             "end2end": True, "half": False, "augment": False, "device": "0"}
POINT_NAMES = ["far_left_doubles", "far_right_doubles", "near_left_doubles", "near_right_doubles",
               "far_left_singles", "near_left_singles", "far_right_singles", "near_right_singles",
               "far_service_left", "far_service_right", "near_service_left", "near_service_right",
               "far_service_center", "near_service_center"]
LINEAGE = {
    "frozen_manifest_sha256": ("data/yolo26/court_development_frozen_v1/grouped_development_manifest.json", "35ae2e741c6e8abdd6eb9b0ed516316be2ec5b1f73ef257b61867ea9e5cd851e"),
    "decisions_sha256": ("data/yolo26/court_review_decisions_v1.json", "b085770df3330b89a8371806d2376ea9106db9f2f82d4e5436c78f5af7a634cb"),
    "root_approval_sha256": ("artifacts/validation/yolo26_execution_2026-10-14/court_root_export_approval.json", "26b77be89c57f07ee60a5a3b5303409e23c39a651d0d26627ff6620d77374869"),
    "inventory_sha256": ("data/yolo26/court_review_expansion_v1/cache_inventory.json", "02cc85fa195c4ea5e0c1990f71a0d7f2954c561940f01dde0112c17077beeeae"),
    "native_detail_receipt_sha256": ("data/yolo26/court_native_detail_v1/native_detail_receipt.json", "e811e52595f3d02f2545ba891d768425c74c8f3c138d80f05886108300648e42"),
}
INSTALLED_KEYPOINT_SOURCES = {
    "ultralytics/engine/results.py": "66f2f9e3f72335a1015f15cc06581361682bbb97e24902b628f7c44baed4b3bc",
    "ultralytics/models/yolo/pose/predict.py": "6bf0fafb4a824614cbfdc7057ba513341d6381a077329d7079d2c342978b3de3",
}


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Diagnostic paths must remain inside the private workspace")
    return path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_value(value):
    """Retain every returned slot; replace nonfinite numbers by explicit null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported diagnostic JSON type: {type(value).__name__}")


def box_iou(first, second):
    if len(first) != 4 or len(second) != 4 or not all(finite(value) for value in [*first, *second]):
        return None
    if first[2] <= first[0] or first[3] <= first[1] or second[2] <= second[0] or second[3] <= second[1]:
        return None
    intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(0., min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if math.isfinite(union) and math.isfinite(intersection) and union > 0 else None


def raw_candidates(boxes, confidences, classes, keypoints):
    """Preserve result order and all fourteen raw x/y/confidence triples."""
    if not len(boxes) == len(confidences) == len(classes):
        raise ValueError("Returned box/confidence/class lengths differ")
    if keypoints is not None and len(keypoints) != len(boxes):
        raise ValueError("Returned box/keypoint instance lengths differ")
    rows = []
    for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
        points = keypoints[index] if keypoints is not None else [[None, None, None] for _ in range(14)]
        if len(box) != 4 or len(points) != 14 or any(len(point) != 3 for point in points):
            raise ValueError("Court result must retain four box values and all fourteen x/y/confidence slots")
        nonfinite_fields = ([f"box[{i}]" for i, value in enumerate(box) if not finite(value)]
                            + ([] if finite(confidence) else ["confidence"])
                            + ([] if finite(class_id) else ["class_id"])
                            + [f"keypoint[{i}][{j}]" for i, point in enumerate(points)
                               for j, value in enumerate(point) if not finite(value)])
        eligible = finite(class_id) and class_id == 0 and finite(confidence) and confidence >= INFERENCE["conf"]
        rows.append({"returned_index": index, "box_xyxy_px": safe_value(box),
                     "confidence": safe_value(confidence), "class_id": safe_value(class_id),
                     "keypoints_xy_confidence_px": safe_value(points), "nonfinite_fields": nonfinite_fields,
                     "keypoint_data_returned": keypoints is not None, "eligible_class0_instance": eligible})
    return rows


def score_image(ground_truth, candidates):
    """Select by confidence first, then apply the predeclared box/point gates."""
    if len(ground_truth["keypoints_px"]) != 14:
        raise ValueError("Ground truth requires fourteen fixed semantic slots")
    eligible = [item for item in candidates if item["eligible_class0_instance"]]
    selected = min(eligible, key=lambda item: (-item["confidence"], item["returned_index"])) if eligible else None
    overlap = box_iou(selected["box_xyxy_px"], ground_truth["bbox_xyxy_px"]) if selected else None
    matched = overlap is not None and overlap >= .5
    image_reason = ("NO_RETURNED_INSTANCE" if not candidates else "NO_ELIGIBLE_CLASS0_INSTANCE") if selected is None else (
        "SELECTED_BOX_INVALID_OR_NONFINITE" if overlap is None else "SELECTED_BOX_IOU_BELOW_0_5" if not matched else None)
    points = []
    for index, target in enumerate(ground_truth["keypoints_px"]):
        prediction = selected["keypoints_xy_confidence_px"][index] if selected else None
        error, reason = None, None
        if target is None:
            reason = "UNKNOWN_GT_EXCLUDED"
        elif len(target) != 3 or target[2] != 2 or not all(finite(value) for value in target):
            raise ValueError("Ground truth may contain only visible finite coordinates or unknown/null")
        elif not matched:
            reason = image_reason
        elif not selected["keypoint_data_returned"]:
            reason = "KEYPOINT_DATA_NOT_RETURNED"
        elif not all(finite(value) for value in prediction):
            reason = "NONFINITE_KEYPOINT_COORDINATE_OR_CONFIDENCE"
        elif prediction[2] < .25:
            reason = "KEYPOINT_CONFIDENCE_BELOW_0_25"
        else:
            error = math.hypot(prediction[0] - target[0], prediction[1] - target[1])
            if not math.isfinite(error):
                error, reason = None, "NONFINITE_LOCALIZATION_ERROR"
        points.append({"point_index": index, "point_name": POINT_NAMES[index], "visible_gt": target is not None,
                       "gt_xy_visibility_px": target, "selected_prediction_xy_confidence_px": prediction,
                       "error_native_px": error, "missing_or_exclusion_reason": reason,
                       "within_10_px": error is not None and error <= 10,
                       "within_20_px": error is not None and error <= 20})
    return {"gt_bbox_xyxy_px": list(ground_truth["bbox_xyxy_px"]),
            "selected_returned_index": selected["returned_index"] if selected else None,
            "selected_bbox_iou": overlap, "selected_instance_matched": matched,
            "instance_missing_reason": image_reason, "returned_instances": len(candidates),
            "unmatched_extra_instances": len(candidates) - int(matched),
            "complete_instance_miss": not matched,
            "complete_visible_keypoint_miss": not any(point["error_native_px"] is not None for point in points),
            "raw_predictions": candidates, "points": points}


def percentile(values, probability):
    """Linear interpolation at (n - 1) * probability, including every error."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_points(points):
    visible = [point for point in points if point["visible_gt"]]
    errors = [point["error_native_px"] for point in visible if point["error_native_px"] is not None]
    count = len(visible)
    return {"visible_gt_denominator": count, "unknown_gt_excluded": len(points) - count,
            "finite_confident_matched_predictions": len(errors), "missing_visible_keypoints": count - len(errors),
            "coverage": len(errors) / count if count else None,
            "within_10_px": sum(point["within_10_px"] for point in visible),
            "within_20_px": sum(point["within_20_px"] for point in visible),
            "recall_at_10_px": sum(point["within_10_px"] for point in visible) / count if count else None,
            "recall_at_20_px": sum(point["within_20_px"] for point in visible) / count if count else None,
            "median_error_native_px": percentile(errors, .5),
            "p95_error_native_px": percentile(errors, .95), "errors_native_px": errors,
            "missing_reasons": dict(Counter(point["missing_or_exclusion_reason"] for point in visible if point["error_native_px"] is None)),
            "conditional_error_population": "ALL_FINITE_CONFIDENT_MATCHED_VISIBLE_GT_POINTS_INCLUDING_ERRORS_OVER20PX"}


def aggregate(rows):
    points = [point for row in rows for point in row["assessment"]["points"]]
    contexts = list(dict.fromkeys(row["context_group"] for row in rows))
    return {"images": len(rows), "all_visible_keypoints": summarize_points(points),
            "matched_instances": sum(row["assessment"]["selected_instance_matched"] for row in rows),
            "complete_instance_misses": sum(row["assessment"]["complete_instance_miss"] for row in rows),
            "complete_visible_keypoint_misses": sum(row["assessment"]["complete_visible_keypoint_miss"] for row in rows),
            "returned_instances": sum(row["assessment"]["returned_instances"] for row in rows),
            "unmatched_extra_instances": sum(row["assessment"]["unmatched_extra_instances"] for row in rows),
            "instance_missing_reasons": dict(Counter(row["assessment"]["instance_missing_reason"] for row in rows if row["assessment"]["instance_missing_reason"])),
            "per_landmark": [{"point_index": index, "point_name": name,
                              **summarize_points([point for point in points if point["point_index"] == index])}
                             for index, name in enumerate(POINT_NAMES)],
            "per_context": [{"context_group": context, "images": sum(row["context_group"] == context for row in rows),
                             **summarize_points([point for row in rows if row["context_group"] == context
                                                 for point in row["assessment"]["points"]])} for context in contexts]}


def check_resources(resources):
    for key, minimum, message in (("free_disk_bytes", 1024 ** 3, "1 GiB free disk"),
                                  ("free_ram_bytes", 1536 * 1024 ** 2, "1.5 GiB available RAM after CUDA initialization"),
                                  ("free_vram_bytes", 2 * 1024 ** 3, "2 GiB free VRAM")):
        if not finite(resources.get(key)) or resources[key] < minimum:
            raise ValueError(f"Court diagnostic guard requires {message}; no unrelated processes are stopped")


def bind_file(path, expected=None):
    path = local_path(path)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Diagnostic input checksum changed: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def bind_installed_keypoint_sources():
    """Hash source text only; never import Ultralytics to inspect masking."""
    from importlib.metadata import distribution

    installed = distribution("ultralytics")
    bindings = {}
    for relative, expected in INSTALLED_KEYPOINT_SOURCES.items():
        path = Path(installed.locate_file(relative)).resolve()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("Installed keypoint postprocessing changed; review confidence masking before inference")
        bindings[relative] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    return bindings


def png_dimensions(path):
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Expected original native PNG image")
    return list(struct.unpack(">II", header[16:24]))


def validate_protocol(protocol):
    if (protocol.get("schema") != "tennis.yolo26.court-development-pixel-diagnostic.v1"
            or protocol.get("source_manifest_sha256") != MANIFEST_SHA256
            or protocol.get("inference") != INFERENCE or protocol.get("split") != "val"
            or protocol.get("expected_images") != 20 or protocol.get("matched_instance_bbox_iou_minimum") != .5
            or protocol.get("keypoint_confidence_minimum") != .25
            or protocol.get("visible_keypoint_error_thresholds_px") != [10, 20]
            or protocol.get("instance_selection") != "HIGHEST_CLASS0_CONFIDENCE_STABLE_ORIGINAL_INDEX_TIE"
            or any(protocol.get(key) is not False for key in ("qualification_evidence", "final_test_used", "independent_human_reviewed"))):
        raise ValueError("Require the unchanged predeclared twenty-DEV fixed-threshold protocol")


def validate_completed_execution(receipt, metadata_sha256, before, after):
    """The root monitor, rather than training metadata alone, proves the freeze."""
    if (receipt.get("schema") != "tennis.yolo26.controlled-court-execution.v1"
            or receipt.get("status") != "COMPLETED_PRIVATE_TRAINED_UNQUALIFIED"
            or receipt.get("exit_code") != 0 or isinstance(receipt.get("exit_code"), bool)
            or receipt.get("source_freeze_verified") is not True
            or receipt.get("changed_runtime_inputs") != [] or receipt.get("added_runtime_sources") != []
            or receipt.get("trainer_status") != "TRAINED_UNQUALIFIED"
            or receipt.get("metadata_sha256") != metadata_sha256
            or receipt.get("runtime_input_count") != 861 or len(before) != 861 or before != after
            or receipt.get("qualification_evidence") is not False or receipt.get("final_test_used") is not False):
        raise ValueError("Require completed exit0 root monitoring with all861 frozen inputs unchanged and exact completed metadata")


def validate_initial20_metadata(metadata):
    audit, arguments = metadata.get("data_audit", {}), metadata.get("training_arguments", {})
    if (metadata.get("status") != "TRAINED_UNQUALIFIED" or metadata.get("role") != "court"
            or metadata.get("mode") != "REVIEWED_STATIC_COURT_DEVELOPMENT"
            or metadata.get("training_script_sha256") != TRAINER_SHA256 or metadata.get("extension_review") is not None
            or metadata.get("initial_epoch_protocol") != 20 or arguments.get("epochs") != 20
            or audit.get("manifest_sha256") != MANIFEST_SHA256 or audit.get("split_image_counts") != {"train": 100, "val": 20, "test": 0}
            or audit.get("static_image_timing_is_unknown") is not True
            or any(metadata.get(key) is not False for key in ("qualification_evidence", "final_test_used", "independent_human_reviewed"))
            or audit.get("final_test_used") is not False):
        raise ValueError("Diagnostic requires the completed unqualified initial20 court run, never an active run or extension")
    updates = metadata.get("optimizer_update_receipts", [])
    if not updates or updates[-1].get("epoch") != 20:
        raise ValueError("Require retained evidence that initial20 actually completed")


def validate_best_checkpoint_binding(metadata, checkpoint_binding):
    if not any(local_path(item.get("path", "")) == CHECKPOINT.resolve() and item.get("sha256") == checkpoint_binding["sha256"]
               for item in metadata.get("checkpoints", [])):
        raise ValueError("best.pt must be bound to the completed initial20 metadata")


def validate_initial20_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        epochs = [int(row["epoch"].strip()) for row in csv.DictReader(handle)]
    if epochs != list(range(1, 21)):
        raise ValueError("Completed initial20 requires exactly the retained chronological epochs1 through20")


def prepare_inputs():
    """Read and bind completed-run and native DEV inputs without runtime imports."""
    bindings = {"protocol": bind_file(PROTOCOL, PROTOCOL_SHA256), "source_manifest": bind_file(MANIFEST, MANIFEST_SHA256),
                "trainer_script": bind_file(ROOT / "scripts/train/train_yolo26_tennis.py", TRAINER_SHA256),
                "diagnostic_script": bind_file(__file__)}
    protocol, manifest, metadata = read_json(PROTOCOL), read_json(MANIFEST), read_json(RUN / "run_metadata.json")
    validate_protocol(protocol)
    validate_initial20_metadata(metadata)
    audit = metadata["data_audit"]
    bindings["run_metadata"] = bind_file(RUN / "run_metadata.json")
    execution = read_json(PROTOCOL.parent / "execution_receipt.json")
    bindings["monitored_training_execution"] = bind_file(PROTOCOL.parent / "execution_receipt.json")
    bindings["training_sources_start_snapshot"] = bind_file(PROTOCOL.parent / "sources_start.json", TRAINING_SOURCES_START_SHA256)
    bindings["training_sources_end_snapshot"] = bind_file(PROTOCOL.parent / "sources_end.json", execution.get("sources_end_sha256"))
    if (execution.get("sources_start_sha256") != TRAINING_SOURCES_START_SHA256
            or execution.get("sources_end_sha256") != bindings["training_sources_end_snapshot"]["sha256"]):
        raise ValueError("Root monitored source-start/end snapshot bindings changed")
    before, after = read_json(PROTOCOL.parent / "sources_start.json"), read_json(PROTOCOL.parent / "sources_end.json")
    validate_completed_execution(execution, bindings["run_metadata"]["sha256"], before, after)
    runtime_sources = {}
    for filename, expected in before.items():
        frozen_path = Path(filename)
        if sha256_file(frozen_path) != expected["sha256"] or frozen_path.stat().st_size != expected["bytes"]:
            raise ValueError(f"A root-frozen training runtime input changed after training: {filename}")
        runtime_sources[filename] = {"path": filename, **expected}
    bindings["frozen_training_runtime_sources"] = runtime_sources
    bindings["checkpoint"] = bind_file(CHECKPOINT)
    validate_best_checkpoint_binding(metadata, bindings["checkpoint"])
    initial, transfer = metadata.get("initial_weights", {}), metadata.get("transfer", {})
    official = ROOT / "models/yolo26/yolo26s-pose.pt"
    if (initial.get("sha256") != OFFICIAL_SHA256 or local_path(initial.get("path", "")) != official.resolve()
            or transfer.get("weights_sha256") != OFFICIAL_SHA256 or transfer.get("kpt_shape") != [14, 3]
            or not transfer.get("transferred_tensor_keys") or not transfer.get("initialized_tensor_keys")
            or metadata.get("initialization") != "OFFICIAL_YOLO26S_POSE_REBUILT_NC1_COURT14_HEAD"):
        raise ValueError("Require official YOLO26s-pose transfer into the separate court14 head")
    bindings["official_initial_weights"] = bind_file(official, OFFICIAL_SHA256)
    bindings["official_weight_provenance"] = bind_file(ROOT / "models/yolo26/provenance.json")
    if not any(item.get("sha256") == OFFICIAL_SHA256 and local_path(item["file"]) == official.resolve()
               for item in read_json(ROOT / "models/yolo26/provenance.json").get("artifacts", [])):
        raise ValueError("Official checkpoint origin receipt does not bind its bytes")
    architecture = local_path(transfer.get("architecture", ""))
    if architecture != RUN / "yolo26s-court-pose.yaml":
        raise ValueError("Require the initial run's rebuilt court architecture")
    bindings["court_architecture"] = bind_file(architecture)
    bindings["training_dataset_yaml"] = bind_file(RUN / "dataset.yaml", metadata.get("dataset_yaml_sha256"))
    bindings["training_arguments_yaml"] = bind_file(RUN / "training/args.yaml")
    bindings["training_metrics_csv"] = bind_file(RUN / "training/results.csv")
    validate_initial20_csv(RUN / "training/results.csv")
    bindings["root_initial20_approval"] = bind_file(
        ROOT / "artifacts/validation/yolo26_execution_2026-10-15/court_development_run1/root_initial20_approval.json",
        "8169f448a78e981e109afe85a914f42ab963764cad2b18275e3a29da52452feb")
    bindings["installed_keypoint_sources"] = bind_installed_keypoint_sources()
    for key, (relative, expected) in LINEAGE.items():
        if manifest.get("lineage", {}).get(key) != expected or audit.get("lineage_sha256", {}).get(key) != expected:
            raise ValueError("Frozen review lineage differs from the completed training run")
        bindings[key] = bind_file(ROOT / relative, expected)
    if (manifest.get("counts") != {"train": 100, "val": 20, "test": 0}
            or manifest.get("qualification_evidence") is not False or manifest.get("source_independence_verified") is not False):
        raise ValueError("Only the unchanged reviewed private development export is allowed")
    sources = {source["source_id"]: source for source in manifest["sources"]}
    seen_groups = {}
    for sample in manifest["samples"]:
        source = sources[sample["source_id"]]
        if sample["split"] not in ("train", "val"):
            raise ValueError("No final-test sample may enter the diagnostic")
        for kind in ("original_source_video_id", "context_group"):
            group = (kind, source[kind])
            if seen_groups.setdefault(group, sample["split"]) != sample["split"]:
                raise ValueError("Original video or transitive event context crosses train/DEV")
    samples = [sample for sample in manifest["samples"] if sample["split"] == "val"]
    if len(samples) != 20:
        raise ValueError("Require all twenty DEV rows in manifest order")
    prepared, annotation_bindings = [], {}
    for sample in samples:
        source = sources[sample["source_id"]]
        if (sample["role"] != "court" or source["source_type"] != "PUBLISHER_STILL_IMAGE"
                or source.get("training_allowed") is not True or source.get("resolution") != [1280, 720]
                or any(source.get(key) is not None for key in ("fps", "source_pts", "source_time_base", "match_id"))
                or any(sample.get(key) is not None for key in ("timestamp_seconds", "source_pts", "source_time_base"))
                or sample.get("landmark_schema") != "tennis_court_14_v1"
                or source.get("context_id_is_verified_match_identity") is not False):
            raise ValueError("Court still-image timing and match identity must remain unknown")
        image = bind_file(sample["image"], sample["sha256"])
        native = bind_file(source["source_path"], source["sha256"])
        label = bind_file(sample["label"], sample["label_sha256"])
        if image["sha256"] != native["sha256"] or png_dimensions(image["path"]) != [1280, 720]:
            raise ValueError("DEV image must preserve original native1280x720 source bytes")
        values = [float(value) for value in Path(label["path"]).read_text(encoding="utf-8").split()]
        points = sample["keypoints_px"]
        if len(values) != 47 or len(points) != 14 or values[0] != 0 or not all(finite(value) for value in values):
            raise ValueError("Require exactly one court instance with fourteen finite encoded semantic slots")
        bbox = sample["bbox_xyxy_px"]
        expected_box = [(bbox[0] + bbox[2]) / 2560, (bbox[1] + bbox[3]) / 1440,
                        (bbox[2] - bbox[0]) / 1280, (bbox[3] - bbox[1]) / 720]
        if any(abs(actual - expected) > 1e-9 for actual, expected in zip(values[1:5], expected_box)):
            raise ValueError("Court label box differs from the reviewed native coordinates")
        for index, point in enumerate(points):
            encoded = values[5 + index * 3:8 + index * 3]
            expected = [0., 0., 0.] if point is None else [point[0] / 1280, point[1] / 720, 2.]
            if point is not None and (point[2] != 2 or not all(finite(value) for value in point)):
                raise ValueError("Ground truth must retain observed points or unknown/null")
            if any(abs(actual - target) > 1e-9 for actual, target in zip(encoded, expected)):
                raise ValueError("Court label visibility/native coordinate roundtrip changed")
        annotation_name = sample["publisher_annotation_file"]
        annotation_path = ROOT / "artifacts/validation/vision_upgrade/court_heatmap_training_source" / annotation_name
        if annotation_name not in annotation_bindings:
            annotation_bindings[annotation_name] = bind_file(annotation_path, sample["publisher_annotation_sha256"])
        elif annotation_bindings[annotation_name]["sha256"] != sample["publisher_annotation_sha256"]:
            raise ValueError("Publisher annotation file has conflicting hashes")
        prepared.append({"sample": sample, "source": source, "image": image, "native_source_image": native, "label": label})
    bindings["publisher_annotation_files"] = annotation_bindings
    return protocol, prepared, bindings


def validate_model(model):
    architecture = model.model.yaml
    head = model.model.model[-1]
    if (model.task != "pose" or architecture.get("scale") != "s"
            or "yolo26" not in str(architecture.get("yaml_file", ""))
            or architecture.get("nc") != 1 or list(architecture.get("kpt_shape", [])) != [14, 3]
            or list(getattr(head, "kpt_shape", [])) != [14, 3]
            or type(head).__name__ != "Pose26" or getattr(head, "end2end", None) is not True
            or model.names != {0: "court"}):
        raise ValueError("Only the initial run's single-class YOLO26s court Pose14 model is permitted")
    return {"task": model.task, "names": model.names, "scale": architecture["scale"], "nc": architecture["nc"],
            "kpt_shape": list(head.kpt_shape), "head_class": type(head).__name__, "end2end": head.end2end,
            "architecture": safe_value(architecture)}


def render_overlay(image_path, assessment, output, caption):
    """Keep native dimensions; G labels are green, returned P labels magenta."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != (1280, 720):
        raise ValueError("Overlay must retain native1280x720 dimensions")
    draw = ImageDraw.Draw(image)
    for candidate in assessment["raw_predictions"]:
        box = candidate["box_xyxy_px"]
        selected = candidate["returned_index"] == assessment["selected_returned_index"]
        color = "#ff4df0" if selected else "#eaa640"
        if all(finite(value) for value in box) and box[2] >= box[0] and box[3] >= box[1]:
            draw.rectangle(box, outline=color, width=3 if selected else 1)
        for index, point in enumerate(candidate["keypoints_xy_confidence_px"]):
            if finite(point[0]) and finite(point[1]):
                x, y = point[:2]
                if 0 <= x < image.width and 0 <= y < image.height:
                    draw.line((x - 4, y, x + 4, y), fill=color, width=2)
                    draw.line((x, y - 4, x, y + 4), fill=color, width=2)
                    draw.text((x + 5, y + 3), f"P{candidate['returned_index']}:{index}", fill=color, stroke_width=1, stroke_fill="black")
    for point in assessment["points"]:
        if point["visible_gt"]:
            x, y = point["gt_xy_visibility_px"][:2]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#4dff6e", width=2)
            draw.text((x + 6, y - 13), f"G{point['point_index']}", fill="#4dff6e", stroke_width=1, stroke_fill="black")
    gt_box = assessment["gt_bbox_xyxy_px"]
    draw.rectangle(gt_box, outline="#4dff6e", width=2)
    draw.text((gt_box[0] + 5, gt_box[1] + 5), "GT court box", fill="#4dff6e", stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, 1280, 40), fill="black")
    draw.text((5, 4), caption, fill="white")
    draw.text((5, 21), "Green G: visible GT | Magenta P: selected prediction | Orange P: extra | Unknown GT excluded; low-confidence predictions still shown", fill="white")
    image.save(output)
    return {"path": str(output), "sha256": sha256_file(output), "resolution": [1280, 720]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Root-released post-terminal inference; fixed twenty DEV images only")
    args = parser.parse_args(argv)
    output = local_path(args.output)
    if output.exists():
        raise FileExistsError("Never overwrite a diagnostic receipt; choose a fresh output directory")
    protocol, inputs, bindings = prepare_inputs()
    if version("ultralytics") != PINNED_ULTRALYTICS:
        raise ValueError("Use pinned Ultralytics8.4.36")
    output.mkdir(parents=True)
    receipt = {"status": "PREFLIGHT_PASSED_NOT_INFERRED", "created_utc": datetime.now(timezone.utc).isoformat(),
               "qualification_evidence": False, "final_test_used": False, "independent_human_reviewed": False,
               "protocol": protocol, "bindings": bindings, "runtime_probe_performed": False,
               "resource_guard_pending": True, "selected_dev_ids": [item["sample"]["id"] for item in inputs],
               "inference_arguments": dict(INFERENCE, nms=False, verbose=False), "rows": [],
               "keypoint_coordinate_policy": "PINNED_RESULTS_DATA_NO_CONFIDENCE_MASK; POSE_SCALE_COORDS_TO_NATIVE_INCLUDES_BOUND_CLIPPING; NOT_UNCLIPPED_NETWORK_HEAD_OUTPUT",
               "not_established": protocol["not_established"]}
    path = output / "diagnostic.json"
    write_json(path, receipt)
    if not args.execute:
        print(json.dumps({"status": receipt["status"], "receipt": str(path), "images": len(inputs)}))
        return
    started = time.perf_counter()
    try:
        import torch
        import psutil
        from ultralytics import YOLO

        available = torch.cuda.is_available()
        free_vram = torch.cuda.mem_get_info()[0] if available else None
        resources = {"free_vram_bytes": free_vram, "free_ram_bytes": psutil.virtual_memory().available,
                     "free_disk_bytes": shutil.disk_usage(output).free,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if available else None,
                     "torch": torch.__version__, "ultralytics": version("ultralytics")}
        receipt.update(runtime_probe_performed=True, resources_after_cuda_initialization=resources)
        write_json(path, receipt)
        check_resources(resources)
        receipt["resource_guard_pending"] = False
        model = YOLO(str(CHECKPOINT), task="pose")
        receipt["model_config"] = validate_model(model)
        receipt["model_config_sha256"] = hashlib.sha256(json.dumps(receipt["model_config"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        receipt["status"] = "DIAGNOSTIC_STARTED_UNQUALIFIED"
        write_json(path, receipt)
        tensor_shapes = []
        hook = model.model.register_forward_pre_hook(lambda _model, arguments: tensor_shapes.append(list(arguments[0].shape)))
        torch.cuda.reset_peak_memory_stats()
        try:
            for index, item in enumerate(inputs):
                sample, source = item["sample"], item["source"]
                result = model.predict(item["image"]["path"], **receipt["inference_arguments"])[0]
                if list(result.orig_shape) != [720, 1280]:
                    raise ValueError("Predictions must use original native1280x720 image coordinates")
                boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
                confidences = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
                classes = result.boxes.cls.detach().cpu().tolist() if result.boxes is not None else []
                keypoints = result.keypoints.data.detach().cpu().tolist() if result.keypoints is not None else None
                assessment = score_image(sample, raw_candidates(boxes, confidences, classes, keypoints))
                overlay = render_overlay(item["image"]["path"], assessment, output / f"dev_{index + 1:02d}_{sample['id']}.png",
                                         f"DEV {index + 1}/20 {sample['id']} | {source['context_group']} | selectedIoU={assessment['selected_bbox_iou']}")
                receipt["rows"].append({"id": sample["id"], "split": "val", "context_group": source["context_group"],
                                        "original_source_video_id": source["original_source_video_id"],
                                        "publisher_frame_ordinal": source["publisher_frame_ordinal"],
                                        "source_fps": None, "source_pts": None, "source_time_base": None,
                                        "image": item["image"], "native_source_image": item["native_source_image"], "label": item["label"],
                                        "publisher_annotation_file": sample["publisher_annotation_file"],
                                        "publisher_annotation_sha256": sample["publisher_annotation_sha256"],
                                        "native_resolution": [1280, 720], "gt_bbox_xyxy_px": sample["bbox_xyxy_px"],
                                        "assessment": assessment, "overlay": overlay})
                write_json(path, receipt)
        finally:
            hook.remove()
        if len(receipt["rows"]) != 20:
            raise ValueError("A completed diagnostic must retain all twenty DEV rows and overlays")
        receipt["aggregates"] = aggregate(receipt["rows"])
        receipt["actual_network_tensor_shapes"] = [list(shape) for shape in sorted(set(tuple(shape) for shape in tensor_shapes))]
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        changed = []
        for binding in bindings.values():
            candidates = [binding] if "path" in binding else list(binding.values())
            changed.extend(item["path"] for item in candidates if sha256_file(item["path"]) != item["sha256"])
        for item in inputs:
            changed.extend(item[key]["path"] for key in ("image", "native_source_image", "label")
                           if sha256_file(item[key]["path"]) != item[key]["sha256"])
        if changed:
            receipt["changed_inputs"] = changed
            raise ValueError("Frozen diagnostic source/input bytes changed during inference")
        receipt.update(status="COMPLETED_FIXED_PROTOCOL_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION",
                       completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                       input_hashes_unchanged=True)
        write_json(path, receipt)
    except Exception as exc:
        receipt.update(status="DIAGNOSTIC_BLOCKED_OR_FAILED_NOT_QUALIFICATION", failure_type=type(exc).__name__,
                       failure_reason=str(exc), failed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        write_json(path, receipt)
        raise
    summary = {key: value for key, value in receipt["aggregates"]["all_visible_keypoints"].items() if key != "errors_native_px"}
    print(json.dumps({"status": receipt["status"], "receipt": str(path), "all_visible_keypoint_summary": summary}, allow_nan=False))


if __name__ == "__main__":
    main()

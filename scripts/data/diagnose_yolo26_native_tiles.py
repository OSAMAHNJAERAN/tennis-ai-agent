"""Frozen native-tile discovery check on ten already-seen development frames.

Default/--prepare verifies local provenance and freezes a protocol without
importing Torch or running inference. --execute consumes that exact protocol.
No training, oracle tile selection, threshold search, final-test use or model
promotion is performed. Returned proposals are retained before deduplication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/yolo26/pilot_broadcast_native_crop_v1/source_manifest.json"
RUN_RECEIPT = ROOT / "outputs/yolo26_ball_native_crop_pilot_v1/run_metadata.json"
OLD_DIAGNOSTIC = ROOT / "outputs/yolo26_ball_native_crop_diagnostic_v1/diagnostic.json"
CHECKPOINT = ROOT / "outputs/yolo26_ball_native_crop_pilot_v1/training/weights/best.pt"
OUTPUT = ROOT / "outputs/yolo26_ball_native_tile_diagnostic_v1"
CHECKPOINT_SHA256 = "3fbbadc1431ada8b1f7be9387d6b3ccfbc49120f5fa6979d14b84830dd5f8a78"
MANIFEST_SHA256 = "10184dd0ccea015484cb663d59438751430e1cd3a6d38cd741baf95d052c06b6"
PINNED_ULTRALYTICS = "8.4.36"
DEV_FRAMES = (56, 103, 163, 234, 281, 329, 376, 435, 483, 530)
TILE_SIZE = 640
TILE_OVERLAP = .2
CONFIDENCE = .01
MATCH_IOU = .5
DEDUP_IOU = .5
DEDUP_CENTER_PX = 2.
GRID = tuple((x, y, x + 640, y + 640)
             for y in (0, 440) for x in (0, 512, 1024, 1280))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def local_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Diagnostic inputs must remain inside the private workspace")
    return path


def tile_grid(width, height):
    """The frozen edge-anchored grid depends only on native image dimensions."""
    if (width, height) != (1920, 1080):
        raise ValueError("Frozen diagnostic requires native1920x1080 frames")
    return list(GRID)


def box_iou(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def box_center(box):
    return ((np.asarray(box[:2]) + np.asarray(box[2:])) / 2).tolist()


def safe_numbers(values):
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def map_candidates(boxes, scores, classes, region, tile_id, width=1920, height=1080):
    """Retain every returned row, its raw mapping, and its scoring eligibility."""
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError("Returned boxes, confidence and class lengths differ")
    left, top, _, _ = region
    candidates = []
    for index, (local_box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        box = np.asarray(local_box, dtype=float)
        if box.shape != (4,):
            raise ValueError("Returned proposal must have four box coordinates")
        raw_global = box + np.array([left, top, left, top])
        clipped = np.clip(raw_global, [0, 0, 0, 0], [width, height, width, height])
        finite = np.isfinite(box).all() and np.isfinite(score) and np.isfinite(class_id)
        eligible = bool(finite and clipped[2] > clipped[0] and clipped[3] > clipped[1]
                        and score >= CONFIDENCE and class_id == 0)
        candidates.append({
            "candidate_id": f"{tile_id}:p{index:03d}", "tile_id": tile_id,
            "tile_origin_xy_px": [left, top], "returned_index": index,
            "local_box_xyxy_px": safe_numbers(box),
            "raw_global_box_xyxy_px": safe_numbers(raw_global),
            "global_box_xyxy_px": safe_numbers(clipped),
            "confidence": float(score) if np.isfinite(score) else None,
            "class_id": float(class_id) if np.isfinite(class_id) else None,
            "eligible_for_scoring": eligible,
            "exclusion_reason": None if eligible else "NONFINITE_EMPTY_WRONG_CLASS_OR_BELOW_FIXED_CONFIDENCE",
        })
    return candidates


def deduplicate(candidates):
    """Match the adapter's stable confidence-order IoU OR center suppression."""
    accepted, associations = [], []
    for proposal in sorted((item for item in candidates if item["eligible_for_scoring"]),
                           key=lambda item: item["confidence"], reverse=True):
        suppressed = None
        for other in accepted:
            overlap = box_iou(proposal["global_box_xyxy_px"], other["global_box_xyxy_px"])
            distance = float(np.linalg.norm(np.asarray(box_center(proposal["global_box_xyxy_px"]))
                                            - np.asarray(box_center(other["global_box_xyxy_px"]))))
            if overlap >= DEDUP_IOU or distance <= DEDUP_CENTER_PX:
                suppressed = {"candidate_id": proposal["candidate_id"],
                              "representative_candidate_id": other["candidate_id"],
                              "pair_iou": overlap, "pair_center_distance_px": distance,
                              "iou_rule_triggered": overlap >= DEDUP_IOU,
                              "center_rule_triggered": distance <= DEDUP_CENTER_PX}
                break
        if suppressed is None:
            accepted.append(proposal)
        else:
            associations.append(suppressed)
    return accepted, associations


def score_candidates(candidates, target_box, publisher_center):
    """One reviewed target can match at most one prediction at the fixed IoU."""
    overlaps = [box_iou(item["global_box_xyxy_px"], target_box) for item in candidates]
    errors = [float(np.linalg.norm(np.asarray(box_center(item["global_box_xyxy_px"]))
                                  - np.asarray(publisher_center))) for item in candidates]
    best = max(range(len(candidates)), key=lambda i: (overlaps[i], candidates[i]["confidence"], -i)) if candidates else None
    closest = min(range(len(candidates)), key=lambda i: (errors[i], -candidates[i]["confidence"], i)) if candidates else None
    matched = best is not None and overlaps[best] >= MATCH_IOU
    return {
        "candidate_count_at_conf001": len(candidates), "matched_target_at_iou05": matched,
        "matched_candidate_id": candidates[best]["candidate_id"] if matched else None,
        "unmatched_or_duplicate_candidates": len(candidates) - int(matched),
        "maximum_iou": overlaps[best] if best is not None else 0.,
        "maximum_iou_candidate_id": candidates[best]["candidate_id"] if best is not None else None,
        "confidence_at_maximum_iou": candidates[best]["confidence"] if best is not None else None,
        "nearest_candidate_center_error_native_px": errors[closest] if closest is not None else None,
        "nearest_candidate_id": candidates[closest]["candidate_id"] if closest is not None else None,
        "nearest_candidate_confidence": candidates[closest]["confidence"] if closest is not None else None,
        "matched_candidate_center_error_native_px": errors[best] if matched else None,
        "candidate_target_assessments": [
            {"candidate_id": item["candidate_id"], "target_iou": overlap,
             "publisher_center_error_native_px": error, "one_to_one_match": matched and i == best}
            for i, (item, overlap, error) in enumerate(zip(candidates, overlaps, errors))
        ],
    }


def aggregate(rows, key="native_tiles"):
    matches = sum(row[key]["matched_target_at_iou05"] for row in rows)
    errors = [row[key]["nearest_candidate_center_error_native_px"] for row in rows
              if row[key]["nearest_candidate_center_error_native_px"] is not None]
    matched_errors = [row[key].get("matched_candidate_center_error_native_px") for row in rows
                      if row[key].get("matched_candidate_center_error_native_px") is not None]
    return {
        "reviewed_targets": len(rows), "matched_targets_conf001_iou05": matches,
        "missed_targets_conf001_iou05": len(rows) - matches,
        "total_returned_candidates_conf001": sum(row[key]["candidate_count_at_conf001"] for row in rows),
        "unmatched_or_duplicate_candidates": sum(row[key]["unmatched_or_duplicate_candidates"] for row in rows),
        "median_nearest_candidate_center_error_native_px": float(np.median(errors)) if errors else None,
        "median_matched_candidate_center_error_native_px": float(np.median(matched_errors)) if matched_errors else None,
    }


def compare_with_baseline(old, current, rows):
    """Use saved baseline metrics without rerunning or selecting thresholds."""
    keys = ("matched_targets_conf001_iou05", "missed_targets_conf001_iou05",
            "total_returned_candidates_conf001", "unmatched_or_duplicate_candidates")
    previous = old["aggregates"]["full_native_frame"]
    old_by_frame = {row["frame_index"]: row for row in old["rows"]}
    return {"baseline_full_native_frame_imgsz1280": previous,
            "context_only_oracle_crop_imgsz640": old["aggregates"]["oracle_crop"],
            "native_tiles_imgsz640_after_dedup": current,
            "native_tiles_minus_baseline_counts": {key: current[key] - previous[key] for key in keys},
            "frame_match_transitions": [
                {"frame_index": row["frame_index"],
                 "before_full_native_frame_matched": old_by_frame[row["frame_index"]]["full_native_frame"]["matched_target_at_iou05"],
                 "after_native_tiles_matched": row["native_tiles"]["matched_target_at_iou05"]}
                for row in rows],
            "interpretation": "POSITIVE_ONLY_DEVELOPMENT_COMPARISON_NO_NEGATIVE_FALSE_ALARM_RATE_OR_QUALIFICATION"}


def check_resources(resources):
    for key, floor, label in (("free_disk_bytes", 1024 ** 3, "1 GiB free disk"),
                              ("free_ram_bytes", 1024 ** 3, "1 GiB available RAM after CUDA initialization"),
                              ("free_vram_bytes", 2 * 1024 ** 3, "2 GiB free VRAM")):
        if resources.get(key) is None or resources[key] < floor:
            raise ValueError(f"Native tile inference guard requires {label}; no unrelated processes are stopped")


def prepare_protocol():
    if version("ultralytics") != PINNED_ULTRALYTICS:
        raise ValueError("Use the pinned local Ultralytics runtime")
    if digest(MANIFEST) != MANIFEST_SHA256 or digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Frozen crop manifest or unpromoted checkpoint changed")
    manifest, run, old = read_json(MANIFEST), read_json(RUN_RECEIPT), read_json(OLD_DIAGNOSTIC)
    if run["status"] != "TRAINED_UNQUALIFIED" or run["data_audit"]["final_test_used"]:
        raise ValueError("Expected the completed unqualified crop development run")
    if run["data_audit"]["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Training receipt manifest does not match")
    registered_best = next(item for item in run["checkpoints"] if Path(item["path"]).name == "best.pt")
    if registered_best["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Training receipt checkpoint does not match")
    if (old["qualification_evidence"] or old["final_test_used"] or old["confidence"] != CONFIDENCE
            or old["iou_match"] != MATCH_IOU or old["checkpoint_sha256"] != CHECKPOINT_SHA256
            or old["manifest_sha256"] != MANIFEST_SHA256):
        raise ValueError("Saved baseline must use the same frozen checkpoint, samples and criteria")
    split_counts = {split: sum(item["split"] == split for item in manifest["samples"])
                    for split in ("train", "val", "test")}
    samples = sorted((item for item in manifest["samples"] if item["split"] == "val"),
                     key=lambda item: item["frame_index"])
    if (split_counts != {"train": 71, "val": 10, "test": 0}
            or tuple(item["frame_index"] for item in samples) != DEV_FRAMES
            or any(item["source_id"] != "match182" or item["role"] != "ball" for item in samples)):
        raise ValueError("Only the frozen ten match182 development samples may be consumed")
    if tuple(sorted(row["frame_index"] for row in old["rows"])) != DEV_FRAMES:
        raise ValueError("Saved baseline sample coverage differs")
    source = next(item for item in manifest["sources"] if item["source_id"] == "match182")
    if source["development_split"] != "val" or source["resolution"] != [1920, 1080]:
        raise ValueError("Require the registered native development source")
    verified_source = {}
    for field, hash_field in (("source_path", "sha256"), ("annotation_path", "annotation_sha256")):
        path = local_path(source[field])
        if digest(path) != source[hash_field]:
            raise ValueError(f"Registered source changed: {path}")
        verified_source[field] = str(path)
        verified_source[hash_field] = source[hash_field]
    old_rows = {row["frame_index"]: row for row in old["rows"]}
    frozen_samples = []
    for sample in samples:
        native_image, native_label = local_path(sample["native_frame_image"]), local_path(sample["native_label"])
        if digest(native_image) != sample["native_frame_sha256"] or digest(native_label) != sample["native_label_sha256"]:
            raise ValueError("Native development image or label changed")
        values = np.asarray(list(map(float, native_label.read_text().split())))
        if len(values) != 5 or values[0] != 0 or not np.isfinite(values).all():
            raise ValueError("Each positive development frame must have exactly one valid ball label")
        cx, cy, width, height = values[1:] * np.array([1920, 1080, 1920, 1080])
        native_box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
        if not np.allclose(native_box, sample["native_box_xyxy_px"], atol=1e-5, rtol=0):
            raise ValueError("Native target label mapping differs")
        baseline = old_rows[sample["frame_index"]]
        for key in ("source_id", "source_pts", "source_time_base", "publisher_center_xy_px", "native_box_xyxy_px"):
            if baseline[key] != sample[key]:
                raise ValueError("Native evaluation target or source timing differs from saved baseline")
        if baseline["native_image_sha256"] != sample["native_frame_sha256"]:
            raise ValueError("Native image differs from saved baseline")
        if abs(float(sample["source_pts"] * Fraction(sample["source_time_base"])) - sample["timestamp_seconds"]) > 1e-9:
            raise ValueError("Original PTS/time-base mapping changed")
        frozen_samples.append({key: sample[key] for key in (
            "source_id", "frame_index", "source_pts", "source_time_base", "timestamp_seconds",
            "native_frame_image", "native_frame_sha256", "native_label", "native_label_sha256",
            "publisher_center_xy_px", "native_box_xyxy_px", "independent_human_reviewed")})
    return {"schema_version": "1.0", "status": "FROZEN_NATIVE_TILE_DEVELOPMENT_PROTOCOL",
            "qualification_evidence": False, "final_test_used": False, "training_performed": False,
            "checkpoint_promoted": False, "private_local_only": True,
            "confidence": CONFIDENCE, "iou_match": MATCH_IOU,
            "inference": {"imgsz": 640, "classes": [0], "device": 0, "end2end": True,
                          "nms": False, "iou": .7, "max_det": 300, "half": False},
            "tiling": {"tile_size_px": 640, "nominal_overlap": .2, "native_pixel_scale": 1.0,
                       "edge_anchored_regions_xyxy_px": [list(region) for region in GRID],
                       "selection": "ALL_EIGHT_FIXED_LABEL_INDEPENDENT_TILES_NO_FULL_FRAME_PASS"},
            "deduplication": {"iou": DEDUP_IOU, "center_distance_px": DEDUP_CENTER_PX,
                              "rule": "IOU_GE_0_5_OR_CENTER_DISTANCE_LE_2_NATIVE_PX",
                              "ordering": "STABLE_DESCENDING_CONFIDENCE_FIRST_ACCEPTED_REPRESENTATIVE"},
            "matching": "ONE_TARGET_ONE_PREDICTION_MAX_IOU_THEN_CONFIDENCE_AT_FIXED_IOU_0_5",
            "checkpoint_sha256": CHECKPOINT_SHA256, "manifest_sha256": MANIFEST_SHA256,
            "script_sha256": digest(__file__), "training_receipt_sha256": digest(RUN_RECEIPT),
            "baseline_receipt_sha256": digest(OLD_DIAGNOSTIC),
            "ultralytics_version": PINNED_ULTRALYTICS, "source": verified_source, "samples": frozen_samples,
            "resource_guard_floors_bytes": {"free_disk_bytes": 1024 ** 3, "free_ram_bytes": 1024 ** 3,
                                           "free_vram_bytes": 2 * 1024 ** 3},
            "limitations": ["Ten positive development frames from one already-seen indoor broadcast context",
                            "Agent-selected boxes have no independent human box review",
                            "Nearest-center errors can describe false objects and exclude frames with no candidates",
                            "Unmatched proposals are positive-frame false/duplicate candidates, not a negative-video false-alarm rate",
                            "No final-test, reserved NKUST reference, speed, tracking or qualification evidence"]}


def execute(protocol):
    # Imported only after an explicit execution request and frozen protocol check.
    import cv2
    import psutil
    import torch
    from ultralytics import YOLO

    receipt_path = OUTPUT / "diagnostic.json"
    if receipt_path.exists():
        raise ValueError("Preserve existing completed diagnostic; no overwrite is permitted")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.init()
    resources = {"free_disk_bytes": shutil.disk_usage(ROOT).free,
                 "free_ram_bytes": psutil.virtual_memory().available,
                 "free_vram_bytes": int(torch.cuda.mem_get_info(0)[0]) if cuda_available else None}
    try:
        check_resources(resources)
    except ValueError as error:
        receipt = {"schema_version": "1.0", "status": "RESOURCE_GUARD_BLOCKED_NOT_INFERRED",
                   "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "qualification_evidence": False, "final_test_used": False, "training_performed": False,
                   "checkpoint_promoted": False, "private_local_only": True,
                   "confidence": CONFIDENCE, "iou_match": MATCH_IOU,
                   "checkpoint_sha256": CHECKPOINT_SHA256, "manifest_sha256": MANIFEST_SHA256,
                   "protocol_path": str(OUTPUT / "protocol.json"), "protocol_sha256": digest(OUTPUT / "protocol.json"),
                   "script_sha256": digest(__file__), "resource_guard": resources,
                   "resource_guard_floors_bytes": protocol["resource_guard_floors_bytes"],
                   "guard_error": str(error), "model_constructed": False, "inference_performed": False,
                   "limitations": protocol["limitations"]}
        write_json(receipt_path, receipt)
        print(json.dumps({"receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
                          "status": receipt["status"], "resource_guard": resources, "guard_error": str(error)}, indent=2))
        return
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats(0)
    model = YOLO(str(CHECKPOINT), task="detect")
    config = model.model.yaml
    if (model.task != "detect" or config.get("scale") != "m" or not config.get("end2end")
            or "yolo26" not in Path(str(config.get("yaml_file", ""))).name.lower()
            or model.names != {0: "tennis-ball"}):
        raise ValueError("Require the registered custom end-to-end YOLO26m tennis-ball checkpoint")
    tensor_shapes = []
    hook = model.model.register_forward_pre_hook(lambda _, args: tensor_shapes.append(list(args[0].shape)))
    rows, started = [], time.perf_counter()
    try:
        for sample in protocol["samples"]:
            image = cv2.imread(sample["native_frame_image"])
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError("Native development image must decode at original1920x1080")
            raw, tile_rows = [], []
            frame_started = time.perf_counter()
            for tile_index, region in enumerate(tile_grid(image.shape[1], image.shape[0])):
                left, top, right, bottom = region
                tile = image[top:bottom, left:right]
                if tile.shape != (640, 640, 3):
                    raise ValueError("Every fixed tile must preserve640x640 native pixels")
                tile_id = f"f{sample['frame_index']:09d}:t{tile_index}"
                shape_start = len(tensor_shapes)
                result = model.predict(tile, verbose=False, **protocol["inference"])[0]
                shapes = tensor_shapes[shape_start:]
                if not shapes or any(shape != [1, 3, 640, 640] for shape in shapes):
                    raise ValueError("Actual per-tile network tensor must remain1x3x640x640")
                boxes = result.boxes
                candidates = map_candidates(boxes.xyxy.detach().cpu().numpy(), boxes.conf.detach().cpu().numpy(),
                                            boxes.cls.detach().cpu().numpy(), region, tile_id) if boxes is not None else []
                raw.extend(candidates)
                tile_rows.append({"tile_id": tile_id, "region_xyxy_px": list(region),
                                  "tile_origin_xy_px": [left, top], "imgsz": 640,
                                  "native_tile_pixel_sha256": hashlib.sha256(tile.tobytes()).hexdigest(),
                                  "actual_network_tensor_shapes": shapes,
                                  "raw_candidate_ids": [item["candidate_id"] for item in candidates]})
            accepted, associations = deduplicate(raw)
            row = dict(sample)
            row.update({"tiles": tile_rows, "raw_candidates": raw,
                        "raw_returned_candidate_count": len(raw),
                        "eligible_raw_candidate_count": sum(item["eligible_for_scoring"] for item in raw),
                        "deduplicated_candidate_ids": [item["candidate_id"] for item in accepted],
                        "dedup_associations": associations,
                        "native_tiles": score_candidates(accepted, sample["native_box_xyxy_px"], sample["publisher_center_xy_px"]),
                        "raw_native_tiles": score_candidates([item for item in raw if item["eligible_for_scoring"]],
                                                             sample["native_box_xyxy_px"], sample["publisher_center_xy_px"]),
                        "elapsed_seconds": time.perf_counter() - frame_started})
            rows.append(row)
    finally:
        hook.remove()
    current = aggregate(rows)
    old = read_json(OLD_DIAGNOSTIC)
    receipt = {"schema_version": "1.0", "status": "FIXED_NATIVE_TILE_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION",
               "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
               "qualification_evidence": False, "final_test_used": False, "training_performed": False,
               "checkpoint_promoted": False, "private_local_only": True,
               "confidence": CONFIDENCE, "iou_match": MATCH_IOU,
               "checkpoint_sha256": CHECKPOINT_SHA256, "manifest_sha256": MANIFEST_SHA256,
               "protocol_path": str(OUTPUT / "protocol.json"), "protocol_sha256": digest(OUTPUT / "protocol.json"),
               "baseline_receipt_path": str(OLD_DIAGNOSTIC), "baseline_receipt_sha256": digest(OLD_DIAGNOSTIC),
               "script_sha256": digest(__file__), "resource_guard": resources,
               "resource_usage": {"elapsed_seconds": time.perf_counter() - started,
                                  "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                                  "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))},
               "aggregates": {"native_tiles": current, "raw_native_tiles_before_dedup": aggregate(rows, "raw_native_tiles"),
                              "total_raw_returned_candidates": sum(row["raw_returned_candidate_count"] for row in rows),
                              "total_dedup_suppressed_candidates": sum(len(row["dedup_associations"]) for row in rows)},
               "before_after": compare_with_baseline(old, current, rows), "rows": rows,
               "actual_network_tensor_shapes": [list(shape) for shape in sorted({tuple(shape) for shape in tensor_shapes})],
               "limitations": protocol["limitations"]}
    write_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "receipt_sha256": digest(receipt_path),
                      "aggregates": receipt["aggregates"], "before_after": receipt["before_after"]}, indent=2))


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Freeze protocol; no Torch import or inference")
    mode.add_argument("--execute", action="store_true", help="Execute the already-frozen protocol")
    parser.add_argument("--output", default=str(OUTPUT), help="Private workspace output directory; existing receipts are immutable")
    args = parser.parse_args(argv)
    OUTPUT = local_path(args.output)
    if (OUTPUT / "diagnostic.json").exists():
        raise ValueError("Preserve existing terminal diagnostic; select a fresh output directory")
    protocol = prepare_protocol()
    protocol_path = OUTPUT / "protocol.json"
    if args.execute:
        if not protocol_path.is_file() or read_json(protocol_path) != protocol:
            raise ValueError("Execution requires the exact prepared protocol; inputs or script changed")
        execute(protocol)
    else:
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Preserve the frozen protocol; do not silently replace it")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if not protocol_path.exists():
            write_json(protocol_path, protocol)
        print(json.dumps({"status": "PREPARED_NO_GPU_INFERENCE", "protocol": str(protocol_path),
                          "protocol_sha256": digest(protocol_path), "development_samples": len(protocol["samples"]),
                          "fixed_tiles_per_frame": len(GRID)}, indent=2))


if __name__ == "__main__":
    main()

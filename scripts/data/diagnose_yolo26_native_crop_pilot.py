"""Fixed-threshold error review on the ten already-seen development frames.

One custom YOLO26m checkpoint is applied to its oracle crops and original
native frames. No final-test frame, training, threshold search or promotion.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO, __version__

from scripts.train.train_yolo26_tennis import check_resource_guard
from src.detection.yolo26_bundle import PINNED_ULTRALYTICS, REPO_ROOT, sha256_file

cv2.setNumThreads(1)
RUN = REPO_ROOT / "outputs/yolo26_ball_native_crop_pilot_v1"
MANIFEST = REPO_ROOT / "data/yolo26/pilot_broadcast_native_crop_v1/source_manifest.json"
OUTPUT = REPO_ROOT / "outputs/yolo26_ball_native_crop_diagnostic_v1"


def iou(boxes, target):
    if not len(boxes):
        return np.empty(0)
    intersection = np.maximum(np.minimum(boxes[:, 2:4], target[2:]) - np.maximum(boxes[:, :2], target[:2]), 0).prod(axis=1)
    return intersection / np.maximum((boxes[:, 2:4] - boxes[:, :2]).prod(axis=1)
                                     + (target[2] - target[0]) * (target[3] - target[1]) - intersection, 1e-12)


def main():
    if __version__ != PINNED_ULTRALYTICS or OUTPUT.exists():
        raise ValueError("Require pinned runtime and fresh diagnostic output")
    manifest = json.loads(MANIFEST.read_text())
    metadata = json.loads((RUN / "run_metadata.json").read_text())
    if metadata["status"] != "TRAINED_UNQUALIFIED" or metadata["data_audit"]["manifest_sha256"] != sha256_file(MANIFEST):
        raise ValueError("Completed run/data integrity mismatch")
    samples = [sample for sample in manifest["samples"] if sample["split"] == "val"]
    if len(samples) != 10 or any(sample["source_id"] != "match182" for sample in samples):
        raise ValueError("Only the registered ten development frames may be inspected")
    resources = {"free_disk_bytes": __import__("shutil").disk_usage(RUN).free,
                 "free_ram_bytes": psutil.virtual_memory().available,
                 "free_vram_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None}
    check_resource_guard(resources, gpu_requested=True)
    if resources["free_ram_bytes"] < 1024 ** 3:
        raise ValueError("Inference diagnostic requires1GiB availableRAM afterCUDAinitialization")
    checkpoint = RUN / "training/weights/best.pt"
    model = YOLO(str(checkpoint))
    if model.task != "detect" or model.model.yaml.get("scale") != "m" or model.names != {0: "tennis-ball"}:
        raise ValueError("Use only the registered custom YOLO26m ball model")
    OUTPUT.mkdir(parents=True)
    rows, panels, tensor_shapes = [], [], []
    hook = model.model.register_forward_pre_hook(lambda _, args: tensor_shapes.append(list(args[0].shape)))
    for sample in samples:
        row = {"source_id": sample["source_id"], "frame_index": sample["frame_index"],
               "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
               "native_image": sample["native_frame_image"], "native_image_sha256": sample["native_frame_sha256"],
               "publisher_center_xy_px": sample["publisher_center_xy_px"],
               "native_box_xyxy_px": sample["native_box_xyxy_px"],
               "native_ball_width_height_px": (np.array(sample["native_box_xyxy_px"])[2:] - np.array(sample["native_box_xyxy_px"])[:2]).tolist()}
        views = []
        for kind, image_path, image_size in [("oracle_crop", sample["image"], 640),
                                             ("full_native_frame", sample["native_frame_image"], 1280)]:
            image = cv2.imread(image_path)
            target = np.array(sample["native_box_xyxy_px"], dtype=float)
            center = np.array(sample["publisher_center_xy_px"], dtype=float)
            if kind == "oracle_crop":
                origin = np.array(sample["crop_origin_xy_px"])
                target -= np.tile(origin, 2)
                center -= origin
            result = model.predict(image, imgsz=image_size, device=0, conf=.01, iou=.7, nms=False,
                                   max_det=300, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            overlaps = iou(boxes, target)
            errors = np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - center, axis=1)
            best = int(np.argmax(overlaps)) if len(boxes) else None
            closest = int(np.argmin(errors)) if len(boxes) else None
            matches = np.flatnonzero(overlaps >= .5)
            # There is exactly one reviewed target; at most one prediction
            # can be matched. Additional predictions are false/duplicates.
            row[kind] = {"imgsz": image_size, "candidate_count_at_conf001": len(boxes),
                         "matched_target_at_iou05": bool(len(matches)),
                         "unmatched_or_duplicate_candidates": len(boxes) - bool(len(matches)),
                         "maximum_iou": float(overlaps[best]) if best is not None else 0.,
                         "confidence_at_maximum_iou": float(confidences[best]) if best is not None else None,
                         "nearest_candidate_center_error_native_px": float(errors[closest]) if closest is not None else None,
                         "nearest_candidate_confidence": float(confidences[closest]) if closest is not None else None}
            cv2.rectangle(image, tuple(np.round(target[:2]).astype(int)), tuple(np.round(target[2:]).astype(int)), (0, 255, 0), 2)
            if best is not None:
                cv2.rectangle(image, tuple(np.round(boxes[best, :2]).astype(int)), tuple(np.round(boxes[best, 2:]).astype(int)), (0, 0, 255), 2)
            thumb = cv2.resize(image, (320, 320) if kind == "oracle_crop" else (568, 320))
            views.append(thumb)
        caption = np.full((28, 888, 3), 245, np.uint8)
        cv2.putText(caption, f"match182 f{sample['frame_index']} crop640 | native1280 greenGT redmaxIoU conf.01", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (15, 15, 15), 1)
        panels.append(np.vstack([np.hstack(views), caption]))
        rows.append(row)
    hook.remove()
    for start in range(0, len(panels), 5):
        cv2.imwrite(str(OUTPUT / f"review_{start // 5:02d}.jpg"), np.vstack(panels[start:start + 5]))
    aggregates = {}
    for kind in ("oracle_crop", "full_native_frame"):
        matches = sum(row[kind]["matched_target_at_iou05"] for row in rows)
        aggregates[kind] = {"reviewed_targets": len(rows), "matched_targets_conf001_iou05": matches,
                            "missed_targets_conf001_iou05": len(rows) - matches,
                            "total_returned_candidates_conf001": sum(row[kind]["candidate_count_at_conf001"] for row in rows),
                            "unmatched_or_duplicate_candidates": sum(row[kind]["unmatched_or_duplicate_candidates"] for row in rows),
                            "median_nearest_candidate_center_error_native_px": float(np.median([row[kind]["nearest_candidate_center_error_native_px"] for row in rows if row[kind]["nearest_candidate_center_error_native_px"] is not None]))}
    receipt = {"status": "FIXED_THRESHOLD_DEVELOPMENT_ERROR_DIAGNOSTIC_NOT_QUALIFICATION",
               "qualification_evidence": False, "final_test_used": False, "confidence": .01, "iou_match": .5,
               "manifest_sha256": sha256_file(MANIFEST), "checkpoint_sha256": sha256_file(checkpoint),
               "resource_guard": resources, "aggregates": aggregates, "rows": rows,
               "actual_network_tensor_shapes": sorted({tuple(shape) for shape in tensor_shapes}),
               "limitations": ["Agent-expanded boxes, no independent human review", "Only one development indoor broadcast context",
                               "Oracle crops selected using ground truth; crop scores do not show full-frame coverage",
                               "Nearest-center errors are conditional on a returned candidate and may describe false detections",
                               "Unmatched predictions on positive-only frames are diagnostic false/duplicate candidates, not negative-video false-alarm rate"]}
    (OUTPUT / "diagnostic.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(OUTPUT / "diagnostic.json"), "aggregates": aggregates}))


if __name__ == "__main__":
    main()

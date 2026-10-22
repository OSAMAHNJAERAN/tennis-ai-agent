"""Create one reversible native640 crop pilot after full-frame RAM rejection.

The same YOLO26m is used. Ball pixel sizes stay native; deterministic crop
jitter varies target placement. These targeted crops are a learning diagnostic,
not a complete-frame test, production validation or new independent truth.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.data.prepare_yolo26_data import encode_instances
from scripts.train.train_yolo26_tennis import validate_labels
from src.detection.yolo26_bundle import REPO_ROOT, sha256_file

cv2.setNumThreads(1)


def main():
    parent = REPO_ROOT / "data/yolo26/pilot_broadcast_v1/source_manifest.json"
    rejection = REPO_ROOT / "outputs/yolo26_ball_broadcast_pilot_v1/run_metadata.json"
    output = REPO_ROOT / "data/yolo26/pilot_broadcast_native_crop_v1"
    if output.exists():
        raise FileExistsError("Crop dataset exists; preserve its manifest")
    failed = json.loads(rejection.read_text())
    if failed["status"] != "RESOURCE_GUARD_BLOCKED_NOT_TRAINED":
        raise ValueError("Require the measured full-frame resource rejection before this limited transform")
    manifest = deepcopy(json.loads(parent.read_text()))
    records = []
    crop_positions = []
    for sample in manifest["samples"]:
        image_path, label_path = Path(sample["image"]), Path(sample["label"])
        if sha256_file(image_path) != sample["sha256"] or sha256_file(label_path) != sample["label_sha256"]:
            raise ValueError("Reviewed parent image/box changed")
        image = cv2.imread(str(image_path))
        values = np.loadtxt(label_path, ndmin=2)
        if values.shape != (1, 5):
            raise ValueError("Native-crop check requires one reviewed visible ball per image")
        cx, cy, width, height = values[0, 1:5] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        rng = np.random.default_rng(int(hashlib.sha256(f"709|{sample['source_id']}|{sample['frame_index']}".encode()).hexdigest()[:16], 16))
        target_x, target_y = rng.integers(96, 545, size=2)
        left = int(np.clip(round(cx) - target_x, 0, image.shape[1] - 640))
        top = int(np.clip(round(cy) - target_y, 0, image.shape[0] - 640))
        crop = image[top:top + 640, left:left + 640]
        native_box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
        crop_box = [native_box[0] - left, native_box[1] - top, native_box[2] - left, native_box[3] - top]
        stem = image_path.stem
        destination = output / "ball/images" / sample["split"] / (stem + ".png")
        destination_label = output / "ball/labels" / sample["split"] / (stem + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), crop)
        destination_label.write_text(encode_instances([{"bbox_xyxy_px": crop_box, "visibility": "visible"}], "ball", 640, 640), encoding="utf-8")
        validate_labels(destination_label, "ball")
        record = deepcopy(sample)
        record.update(image=str(destination), label=str(destination_label), sha256=sha256_file(destination),
                      label_sha256=sha256_file(destination_label), native_frame_image=str(image_path),
                      native_frame_sha256=sample["sha256"], native_label=str(label_path), native_label_sha256=sample["label_sha256"],
                      crop_origin_xy_px=[left, top], crop_size_px=[640, 640], native_box_xyxy_px=native_box,
                      native_pixel_scale=1.0, target_center_xy_in_crop=[cx - left, cy - top])
        crop_positions.append(record["target_center_xy_in_crop"])
        records.append(record)
    manifest["samples"] = records
    manifest["parent_manifest"] = str(parent.relative_to(REPO_ROOT))
    manifest["parent_manifest_sha256"] = sha256_file(parent)
    manifest["training_transform"] = {
        "kind": "NATIVE_640_JITTERED_DEVELOPMENT_CROPS", "native_pixel_scale": 1.0,
        "reason": "Measured full-frame YOLO26m1280 CUDA-initialized availableRAM848060416B rejected by1.5GiBguard; tiny full-frame balls shrink by2/3. One native crop experiment preserves pixels and reduces buffers/VRAM.",
        "resource_rejection_receipt": str(rejection.relative_to(REPO_ROOT)), "resource_rejection_sha256": sha256_file(rejection),
        "seed": 709, "crop_size_px": [640, 640], "random_target_interval_px": [96, 544],
        "target_center_min_xy_px": np.min(crop_positions, axis=0).tolist(),
        "target_center_max_xy_px": np.max(crop_positions, axis=0).tolist(),
        "validation_limitation": "Ground-truth-targeted jittered crops are easier than full-frame discovery. Do not report their scores as full-frame accuracy, independent qualification, or real crop-tracker performance.",
    }
    manifest["source_independence_verified"] = False
    path = output / "source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "sha256": sha256_file(path), "samples": len(records)}))


if __name__ == "__main__":
    main()

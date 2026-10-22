"""Verify actual crop pixels, native-coordinate inversion, timing and splits."""
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.train.train_yolo26_tennis import audit_manifest
from src.detection.yolo26_bundle import REPO_ROOT, sha256_file

cv2.setNumThreads(1)
path = REPO_ROOT / "data/yolo26/pilot_broadcast_native_crop_v1/source_manifest.json"
manifest, records, data_audit = audit_manifest(path, "ball", pilot=True)
width_heights = []
centers = []
for sample in manifest["samples"]:
    native_path, crop_path = Path(sample["native_frame_image"]), Path(sample["image"])
    if sha256_file(native_path) != sample["native_frame_sha256"]:
        raise ValueError("Native source image changed")
    if sha256_file(Path(sample["native_label"])) != sample["native_label_sha256"]:
        raise ValueError("Native reviewed box changed")
    native, crop = cv2.imread(str(native_path)), cv2.imread(str(crop_path))
    left, top = sample["crop_origin_xy_px"]
    if crop.shape != (640, 640, 3) or not np.array_equal(crop, native[top:top + 640, left:left + 640]):
        raise ValueError("Crop no longer preserves exact native pixels")
    row = np.loadtxt(Path(sample["label"]), ndmin=2)[0]
    cx, cy, width, height = row[1:5] * 640
    restored = [left + cx - width / 2, top + cy - height / 2,
                left + cx + width / 2, top + cy + height / 2]
    if not np.allclose(restored, sample["native_box_xyxy_px"], atol=1e-5):
        raise ValueError("Crop label cannot be inverted to the original reviewed rectangle")
    if sample["source_pts"] is None or sample["source_time_base"] is None:
        raise ValueError("Native PTS/time base was lost")
    width_heights.append([width, height])
    centers.append([cx, cy])
receipt = {"status": "EXACT_NATIVE_CROP_PIXELS_LABEL_TRANSFORM_PTS_AND_GROUPS_VERIFIED",
           "qualification_evidence": False, "final_test_used": False,
           "manifest_sha256": sha256_file(path), "data_audit": data_audit,
           "sample_count": len(records), "ball_native_width_height_min_px": np.min(width_heights, axis=0).tolist(),
           "ball_native_width_height_median_px": np.median(width_heights, axis=0).tolist(),
           "ball_native_width_height_max_px": np.max(width_heights, axis=0).tolist(),
           "target_center_min_xy_px": np.min(centers, axis=0).tolist(),
           "target_center_max_xy_px": np.max(centers, axis=0).tolist(),
           "reviewed_negative_count": 0,
           "validation_limitation": "Oracle-targeted jittered crops; no full-frame discovery, trajectory or qualification claim"}
output = REPO_ROOT / "outputs/yolo26_native_crop_transform_audit_v1.json"
output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))

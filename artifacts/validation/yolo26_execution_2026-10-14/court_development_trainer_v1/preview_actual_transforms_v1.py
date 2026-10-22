"""Render three actual CPU Ultralytics TRAIN transformations, never models/DEV."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "transform_preview_v1"
PREFLIGHT = BASE / "preflight_data_only_v2/run_metadata.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve the prior actual-transform preview")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/yolo26/court_reviewed_development_v1/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(manifest_path) != preflight["data_audit"]["manifest_sha256"]:
        raise ValueError("Preflight manifest changed")
    if sha(ROOT / "scripts/train/train_yolo26_tennis.py") != preflight["training_script_sha256"]:
        raise ValueError("Trainer changed since frozen settings")
    selected = [sample for sample in manifest["samples"] if sample["split"] == "train"][:3]
    if len(selected) != 3 or any(sample["role"] != "court" for sample in selected):
        raise ValueError("Preview only the first three exact ordered frozen TRAIN rows")
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import psutil
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    import yaml

    if ultralytics.__version__ != "8.4.36" or torch.cuda.is_initialized():
        raise ValueError("Require pinned CPU-only transform import without CUDA initialization")
    resources = {"free_ram_bytes": psutil.virtual_memory().available,
                 "free_disk_bytes": shutil.disk_usage(BASE).free}
    if resources["free_ram_bytes"] < 1536 * 1024 ** 2 or resources["free_disk_bytes"] < 1024 ** 3:
        raise ValueError(f"Unchanged 1.5GiB RAM/1GiB disk preview launch floors: {resources}")
    OUTPUT.mkdir()
    (OUTPUT / "images/train").mkdir(parents=True)
    (OUTPUT / "labels/train").mkdir(parents=True)
    initial_bindings = {str(PREFLIGHT): sha(PREFLIGHT), str(manifest_path): sha(manifest_path)}
    by_id = {}
    for sample in selected:
        by_id[sample["id"]] = sample
        for field, digest in (("image", sample["sha256"]), ("label", sample["label_sha256"])):
            source = ROOT / sample[field]
            if sha(source) != digest:
                raise ValueError("Original selected bytes changed")
            initial_bindings[str(source)] = digest
            destination = OUTPUT / ("images/train" if field == "image" else "labels/train") / source.name
            shutil.copyfile(source, destination)
            if sha(destination) != digest:
                raise ValueError("Staged native bytes differ")
    seed = 709
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    settings = preflight["training_arguments"]
    cfg = get_cfg(overrides=dict(settings, task="pose", mode="train"))
    data = yaml.safe_load((ROOT / "data/yolo26/court_reviewed_development_v1/court/data.yaml").read_text(encoding="utf-8"))
    # This is the exact installed DetectionTrainer.build_dataset factory,
    # with the official YOLO26 maximum stride32 and train padding0/rectFalse.
    dataset = build_yolo_dataset(cfg, str(OUTPUT / "images/train"), 1, data, mode="train", rect=False, stride=32)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    receipt = {"status": "ACTUAL_CPU_ULTRALYTICS_TRAIN_TRANSFORM_PREVIEW_NOT_TRAINED_NOT_QUALIFICATION",
               "created_utc": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
               "ultralytics": ultralytics.__version__, "torch": torch.__version__, "seed": seed,
               "selection": "FIRST_THREE_MANIFEST_TRAIN_ROWS_ONLY_NO_DEV_OR_TEST",
               "seeding": "PYTHON_NUMPY_TORCH_CPU_SEED709_NO_CUDA_CALLS",
               "factory": "ultralytics.data.build.build_yolo_dataset(mode=train,batch=1,rect=False,stride=32)",
               "not_training_batch_order": True, "model_constructed": False, "training_executed": False,
               "cuda_initialized": torch.cuda.is_initialized(), "resource_launch": resources,
               "settings": settings, "training_arguments_sha256": preflight["training_arguments_sha256"],
               "preflight_sha256": sha(PREFLIGHT), "trainer_sha256": preflight["training_script_sha256"],
               "preview_script_sha256": sha(__file__), "actual_transform_repr": repr(dataset.transforms),
               "package_source_sha256": {str(Path(module.__file__)): sha(module.__file__) for module in
                                           (sys.modules["ultralytics.data.build"], sys.modules["ultralytics.data.dataset"],
                                            sys.modules["ultralytics.data.augment"])}, "samples": []}
    for index in range(len(dataset)):
        item = dataset[index]
        ident = Path(item["im_file"]).stem
        source = by_id[ident]
        pixels = item["img"].permute(1, 2, 0).numpy()
        keypoints = item["keypoints"].numpy()
        boxes = item["bboxes"].numpy()
        classes = item["cls"].numpy()
        if pixels.shape != (1280, 1280, 3) or keypoints.shape != (1, 14, 3) or classes.tolist() != [[0.0]]:
            raise ValueError("Actual transform unexpectedly changed image/court14/class0 shape")
        for point_index, point in enumerate(source["keypoints_px"]):
            if point is None and keypoints[0, point_index, 2] != 0:
                raise ValueError("Actual transform turned an unknown slot visible")
        tensor_path = OUTPUT / f"{ident}_actual_train_tensor.png"
        Image.fromarray(pixels).save(tensor_path)
        overlay = Image.fromarray(pixels.copy())
        draw = ImageDraw.Draw(overlay)
        cx, cy, width, height = boxes[0] * 1280
        draw.rectangle((cx-width/2, cy-height/2, cx+width/2, cy+height/2), outline="#ffcf4b", width=2)
        null_indices = []
        for point_index, (x, y, visibility) in enumerate(keypoints[0]):
            if visibility == 0:
                null_indices.append(point_index)
                continue
            x, y = float(x * 1280), float(y * 1280)
            draw.ellipse((x-5, y-5, x+5, y+5), fill="#00ffff", outline="black", width=1)
            draw.text((x+7, y-18), str(point_index), fill="#00ffff", stroke_fill="black", stroke_width=2, font=font)
        draw.rectangle((0, 0, 1280, 61), fill="#121818")
        draw.text((12, 5), f"TRAIN ONLY | {ident} | actual Ultra8.4.36 transform | seed709", font=font, fill="white")
        draw.text((12, 31), f"court14 class0 | transformed null indices {null_indices} | no model/GPU/training", font=font, fill="#ffcf4b")
        overlay_path = OUTPUT / f"{ident}_actual_train_overlay.png"
        overlay.save(overlay_path)
        receipt["samples"].append({"id": ident, "split": "train", "source_image": source["image"],
                                    "source_image_sha256": source["sha256"], "source_label_sha256": source["label_sha256"],
                                    "actual_tensor_image": str(tensor_path), "actual_tensor_image_sha256": sha(tensor_path),
                                    "overlay": str(overlay_path), "overlay_sha256": sha(overlay_path),
                                    "keypoints_normalized": keypoints.tolist(), "boxes_xywh_normalized": boxes.tolist(),
                                    "null_indices_after_transform": null_indices})
    if torch.cuda.is_initialized():
        raise ValueError("CPU transform preview unexpectedly initialized CUDA")
    receipt["source_binding_changes"] = [path for path, digest in initial_bindings.items() if sha(path) != digest]
    if receipt["source_binding_changes"]:
        raise ValueError("Original/preflight bytes changed during preview")
    receipt["process_rss_bytes_at_completion"] = psutil.Process().memory_info().rss
    receipt["cuda_initialized"] = False
    with (OUTPUT / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": receipt["status"], "output": str(OUTPUT), "samples": len(receipt["samples"]),
                      "cuda_initialized": False, "source_binding_changes": receipt["source_binding_changes"]}))


if __name__ == "__main__":
    main()

"""Actual CPU-only native640 TRAIN transforms; no model or DEV/test access."""
from datetime import datetime, timezone
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "transform_preview_v1"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=["v1", "v2", "v3"], default="v1")
    args = parser.parse_args()
    OUTPUT = BASE / ("transform_preview_" + args.attempt)
    if OUTPUT.exists():
        raise FileExistsError("Preserve previous actual preview")
    preflight_path = BASE / "trainer_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    manifest_path = Path(preflight["data_audit"]["manifest_path"])
    if sha(manifest_path) != preflight["data_audit"]["manifest_sha256"]:
        raise ValueError("Reviewed export changed after preflight")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = [row for row in manifest["sources"] if row["development_split"] == "train"]
    positives = [row for row in train if not row["negative_visible_absence_only"]]
    area = lambda row: (row["box_xyxy_crop_px"][2]-row["box_xyxy_crop_px"][0]) * (row["box_xyxy_crop_px"][3]-row["box_xyxy_crop_px"][1])
    selected = [min(positives, key=lambda row: (area(row), row["sample_id"])),
                min(positives, key=lambda row: (-area(row), row["sample_id"])),
                next(row for row in train if row["negative_visible_absence_only"])]
    if len({row["sample_id"] for row in selected}) != 3:
        raise ValueError("Require three distinct fixed TRAIN examples")
    class Memory(ctypes.Structure):
        _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong)] + [(name,ctypes.c_ulonglong) for name in
                    ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
    available = Memory()
    available.length = ctypes.sizeof(available)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(available)):
        raise ctypes.WinError()
    if available.avail_phys < 1536*1024**2:
        refusal = dict(status="RESOURCE_ADMISSION_REFUSED_BEFORE_IMAGE_OR_TORCH_IMPORT", available_ram_bytes=available.avail_phys,
                       required_available_ram_bytes=1536*1024**2, measured_at_utc=datetime.now(timezone.utc).isoformat(),
                       preview_script_sha256=sha(__file__), model_constructed=False, image_decoded=False,
                       cuda_initialized=False, training_performed=False, qualification_evidence=False)
        with (BASE / ("preview_" + args.attempt + "_admission_refusal.json")).open("x",encoding="utf-8") as handle:
            json.dump(refusal,handle,indent=2)
        print(json.dumps(refusal))
        return 2
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    if ultralytics.__version__ != "8.4.36" or torch.cuda.is_initialized():
        raise ValueError("Require pinned CPU-only transformation without CUDA")
    resources = dict(free_ram_bytes=psutil.virtual_memory().available, free_disk_bytes=shutil.disk_usage(BASE).free)
    if resources["free_ram_bytes"] < 1536*1024**2 or resources["free_disk_bytes"] < 1024**3:
        raise ValueError(f"Unchanged CPU preview1.5GiB RAM/1GiB disk floors: {resources}")
    OUTPUT.mkdir()
    (OUTPUT / "images/train").mkdir(parents=True)
    (OUTPUT / "labels/train").mkdir(parents=True)
    bindings = {str(preflight_path): sha(preflight_path), str(manifest_path): sha(manifest_path),
                str(ROOT / "scripts/train/train_yolo26_reviewed_ball.py"): sha(ROOT / "scripts/train/train_yolo26_reviewed_ball.py")}
    by_id = {row["sample_id"]: row for row in selected}
    for row in selected:
        for source, directory, expected in ((row["image_path"], "images/train", row["image_source_sha256"]),
                                             (row["label_path"], "labels/train", row["label_sha256"])):
            if sha(source) != expected:
                raise ValueError("Selected native source bytes changed")
            bindings[source] = expected
            target = OUTPUT / directory / Path(source).name
            with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024*1024)
            if sha(target) != expected:
                raise ValueError("Working preview bytes differ")
    random.seed(709)
    np.random.seed(709)
    torch.random.default_generator.manual_seed(709)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    settings = preflight["training_arguments"]
    cfg = get_cfg(overrides=dict(settings, task="detect", mode="train"))
    dataset = build_yolo_dataset(cfg, str(OUTPUT / "images/train"), 1, {"names": {0: "tennis-ball"}}, mode="train", rect=False, stride=32)
    receipt = dict(status="ACTUAL_CPU_TRAIN_TRANSFORM_PREVIEW_NOT_TRAINED_NOT_QUALIFICATION",
                   created_at_utc=datetime.now(timezone.utc).isoformat(), preview_script_sha256=sha(__file__),
                   selection="SMALLEST_TRAIN_POSITIVE_AREA_THEN_LARGEST_THEN_FIRST_TRAIN_NEGATIVE; ID_TIES; NO_MODEL_OUTPUT_SELECTION",
                   seed=709, settings=settings, actual_transform_repr=repr(dataset.transforms), resources=resources,
                   factory="ultralytics.data.build.build_yolo_dataset(mode=train,batch1,rectFalse,stride32)",
                   model_constructed=False, cuda_initialized=False, training_performed=False, qualification_evidence=False, samples=[])
    for index in range(len(dataset)):
        item = dataset[index]
        ident = Path(item["im_file"]).stem
        source = by_id[ident]
        image = item["img"].permute(1,2,0).numpy()
        boxes = item["bboxes"].numpy()
        expected_boxes = 0 if source["negative_visible_absence_only"] else 1
        if image.shape != (640,640,3) or boxes.shape != (expected_boxes,4) or item["cls"].shape != (expected_boxes,1):
            raise ValueError("Actual transformation changed native image or visible/absence box count")
        roundtrip_error = 0.
        if expected_boxes:
            cx, cy, width, height = boxes[0] * 640
            restored = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
            roundtrip_error = max(abs(a-b) for a,b in zip(restored,source["box_xyxy_crop_px"]))
            if roundtrip_error > .0001:
                raise ValueError("HSV-only preview moved or clipped the reviewed tiny-ball box")
        tensor_path = OUTPUT / (ident+"_actual_train_tensor.png")
        if not cv2.imwrite(str(tensor_path), image[...,::-1]):
            raise ValueError("Unable to save actual RGB train tensor")
        overlay = image[...,::-1].copy()
        if expected_boxes:
            x1,y1,x2,y2 = map(int,source["box_xyxy_crop_px"])
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,0),1)
        overlay_path = OUTPUT / (ident+"_actual_train_overlay.png")
        if not cv2.imwrite(str(overlay_path),overlay):
            raise ValueError("Unable to save actual train overlay")
        receipt["samples"].append(dict(id=ident, split="train", negative=source["negative_visible_absence_only"],
                  tensor_image=str(tensor_path), tensor_image_sha256=sha(tensor_path), overlay=str(overlay_path),
                  overlay_sha256=sha(overlay_path), actual_boxes_xywh_normalized=boxes.tolist(), coordinate_roundtrip_error_px=float(roundtrip_error)))
    receipt["input_changes"] = [path for path,value in bindings.items() if sha(path)!=value]
    if receipt["input_changes"] or torch.cuda.is_initialized():
        raise ValueError("Preview changed original inputs or initialized CUDA")
    receipt["owned_rss_bytes_at_completion"] = psutil.Process().memory_info().rss
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"samples":len(receipt["samples"]),"cuda_initialized":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

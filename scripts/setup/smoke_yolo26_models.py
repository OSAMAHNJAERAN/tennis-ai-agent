"""Offline model/API smoke only; never a tennis accuracy qualification."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from scripts.train.train_yolo26_tennis import build_training_model
from src.detection.yolo26_adapters import YOLO26PlayerPoseDetector
from src.detection.yolo26_bundle import PINNED_ULTRALYTICS, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo26_model_smoke"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if ultralytics.__version__ != PINNED_ULTRALYTICS:
        raise ValueError("Run the smoke in the pinned runtime")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    image = cv2.imread(str(args.image)) if args.image else np.zeros((1080, 1920, 3), np.uint8)
    if image is None:
        raise ValueError("Image could not be read")
    evidence = {"date_utc": datetime.now(timezone.utc).isoformat(), "scope": "ARCHITECTURE_API_SMOKE_ONLY_NOT_TENNIS_ACCURACY",
                "input": str(args.image) if args.image else "SYNTHETIC_BLACK_FRAME_NO_DATA_RIGHTS_REQUIRED",
                "input_sha256": sha256_file(args.image) if args.image else None, "image_shape": list(image.shape),
                "device": args.device, "python_runtime": "LOCAL_WORKING_ENVIRONMENT_NOT_REPLACED",
                "ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "free_vram_before_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
                "models": {}, "independent_accuracy_validated": False}
    for role, name, imgsz in [("players", "yolo26s-pose.pt", 960), ("ball_initial", "yolo26m.pt", 1280)]:
        path = Path("models/yolo26") / name
        start = time.perf_counter()
        model = YOLO(str(path))
        result = model.predict(image, imgsz=imgsz, device=args.device, end2end=True, nms=False, verbose=False)[0]
        evidence["models"][role] = {"path": str(path), "sha256": sha256_file(path), "task": model.task,
                                    "yaml_file": model.model.yaml.get("yaml_file"), "scale": model.model.yaml.get("scale"),
                                    "kpt_shape": model.model.yaml.get("kpt_shape"), "imgsz": imgsz,
                                    "end2end": True, "nms": False, "boxes": len(result.boxes),
                                    "output_original_shape": list(result.orig_shape), "speed_ms": result.speed,
                                    "load_and_infer_seconds": time.perf_counter() - start}
        if role == "players":
            detector = YOLO26PlayerPoseDetector(path, tracker="configs/production/botsort_no_reid.yaml", device=args.device, _model=model)
            boxes = detector.detect_and_track(image)
            evidence["models"][role]["tracker_smoke_boxes"] = len(boxes)
            detector.reset()
            evidence["models"][role]["tracker_reset"] = "PASSED"
        del model, result
        gc.collect()
    start = time.perf_counter()
    model, transfer = build_training_model("court", "models/yolo26/yolo26s-pose.pt", output)
    model.model.eval()
    result = model.predict(image, imgsz=1280, device=args.device, end2end=True, nms=False, verbose=False)[0]
    transfer.update({"task": model.task, "scale": model.model.yaml.get("scale"), "model_kpt_shape": model.model.yaml.get("kpt_shape"),
                     "load_and_infer_seconds": time.perf_counter() - start, "speed_ms": result.speed,
                     "qualification": "RANDOM_COURT_HEAD_UNTRAINED_NOT_A_PRODUCTION_ARTIFACT",
                     "compatible_tensor_count": len(transfer["transferred_tensor_keys"]),
                     "initialized_tensor_count": len(transfer["initialized_tensor_keys"])})
    evidence["models"]["court_head_only"] = transfer
    path = output / "evidence.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"evidence": str(path), "models": list(evidence["models"]), "scope": evidence["scope"]}))


if __name__ == "__main__":
    main()

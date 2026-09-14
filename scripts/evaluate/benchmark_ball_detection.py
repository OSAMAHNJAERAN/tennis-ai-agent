"""Reproducible fixed-threshold ball detector benchmark on a YOLO image split.

No interpolation, pseudo-label creation, training or threshold optimization.
This is diagnostic unless source-match independence has separately been proven.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from src.evaluation.detection_metrics import aggregate_counts, match_detections


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels(path: Path, width: int, height: int) -> list[list[float]]:
    # Missing labels are a data failure, never silently treated as negatives.
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or not np.isfinite(values).all() or values[0] != 0:
            raise ValueError(f"Expected single-class ball YOLO label: {path}:{number}")
        _, x, y, w, h = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid normalized box: {path}:{number}")
        result.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/tennis_ball_dataset")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1024, 1280])
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--class-id", type=int, default=0, help="Model class (0 for dedicated ball; 32 for COCO sports ball)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve prior evidence")
    if not 0 <= args.confidence <= 1 or any(size <= 0 for size in args.imgsz):
        parser.error("Invalid confidence or image size")
    if not args.model.is_file():
        parser.error("Checkpoint must exist locally; no automatic model downloads")
    images = sorted(path for path in (args.dataset / "images" / args.split).iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        parser.error("No evaluation images")
    manifest = [{"image": str(path.relative_to(args.dataset)), "sha256": sha256(path),
                 "label_sha256": sha256(args.dataset / "labels" / args.split / (path.stem + ".txt"))}
                for path in images]
    training_hashes = {sha256(path) for path in (args.dataset / "images/train").iterdir() if path.is_file()}
    duplicates = [row["image"] for row in manifest if row["sha256"] in training_hashes]
    started = time.perf_counter()
    model = YOLO(str(args.model))
    load_seconds = time.perf_counter() - started
    if args.class_id not in model.names:
        parser.error("Model does not contain the requested class")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize()

    experiments = []
    for size in args.imgsz:
        predict_args = dict(imgsz=size, conf=args.confidence, classes=[args.class_id],
                            device=args.device, verbose=False, augment=False, iou=.7)
        sample = cv2.imread(str(images[0]))
        for _ in range(3):
            model.predict(sample, **predict_args)
        synchronize()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        rows, counts, latencies = [], [], []
        loop_start = time.perf_counter()
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode {path}")
            height, width = image.shape[:2]
            target = labels(args.dataset / "labels" / args.split / (path.stem + ".txt"), width, height)
            synchronize()
            start = time.perf_counter()
            prediction = model.predict(image, **predict_args)[0]
            synchronize()
            latencies.append(time.perf_counter() - start)
            boxes = prediction.boxes.xyxy.cpu().numpy().tolist()
            conf = prediction.boxes.conf.cpu().numpy().tolist()
            count = match_detections(boxes, target, iou_threshold=args.iou)
            counts.append(count)
            rows.append({"image": str(path.relative_to(args.dataset)), "width": width, "height": height,
                         "targets_xyxy": target, "predictions_xyxy": boxes,
                         "confidences": conf, "counts": count.to_dict()})
        elapsed = time.perf_counter() - loop_start
        report = {"imgsz": size, "confidence": args.confidence, "match_iou": args.iou, "nms_iou": .7,
                  "images": len(images), "negative_images": sum(not row["targets_xyxy"] for row in rows),
                  "metrics": aggregate_counts(counts).to_dict(),
                  "prediction_seconds": sum(latencies), "evaluation_loop_seconds": elapsed,
                  "prediction_fps": len(images) / sum(latencies),
                  "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
                  "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
                  "per_image": rows}
        experiments.append(report)
        print(json.dumps({key: value for key, value in report.items() if key != "per_image"}), flush=True)
    output = {"schema_version": "1.0", "scientific_split": "DIAGNOSTIC_IMAGE_SPLIT",
              "qualification_evidence": False, "source_match_independence": "UNVERIFIED",
              "split": args.split, "exact_train_overlap_images": duplicates, "dataset_manifest": manifest,
              "model": str(args.model), "model_sha256": sha256(args.model),
              "class_id": args.class_id, "model_class_name": model.names[args.class_id],
              "model_load_seconds": load_seconds,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "evaluator_sha256": sha256(Path(__file__)),
              "matcher_sha256": sha256(ROOT / "src/evaluation/detection_metrics.py"),
              "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                              "opencv": cv2.__version__, "ultralytics": ultralytics.__version__,
                              "device": args.device, "gpu": torch.cuda.get_device_name() if use_cuda else None},
              "matching": "MAX_CARDINALITY_THEN_TOTAL_IOU_ONE_TO_ONE; NOT_COCO_AP",
              "timing_scope": "warm prediction includes preprocess/forward/postprocess; excludes model loading, image decoding and evaluation",
              "experiments": experiments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

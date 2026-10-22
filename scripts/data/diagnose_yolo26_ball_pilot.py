"""Read-only development diagnostics for a completed ball pilot; never trains.

Quantifies native labels, actual pinned preprocessing/augmentation tensors,
optimizer updates and low-confidence localization. Results are not independent
accuracy, threshold-selection or production-qualification evidence.
"""
import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import yaml

from src.detection.yolo26_bundle import REPO_ROOT, PINNED_ULTRALYTICS, sha256_file


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def box_iou(boxes, target):
    if not len(boxes):
        return np.zeros(0)
    a = np.maximum(boxes[:, :2], target[:2])
    b = np.minimum(boxes[:, 2:4], target[2:4])
    intersection = np.maximum(b - a, 0).prod(axis=1)
    area = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + (target[2] - target[0]) * (target[3] - target[1]) - intersection, 1e-12)


def diagnose(manifest_path, run, output, *, checkpoint=None):
    from ultralytics import __version__
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import LetterBox
    from src.detection.yolo26_adapters import _load

    if __version__ != PINNED_ULTRALYTICS:
        raise ValueError("Diagnostic preprocessing must use the pinned runtime")
    manifest_path, run, output = resolve(manifest_path), resolve(run), resolve(output)
    manifest = json.loads(manifest_path.read_text())
    samples = [s for s in manifest["samples"] if s["role"] == "ball" and s["split"] == "train"]
    if not samples or any(s.get("split") == "test" for s in samples):
        raise ValueError("Only existing development training samples may be inspected")
    if len(samples) > 32:
        raise ValueError("This is a bounded diagnostic, limited to 32 training images")
    output.mkdir(parents=True, exist_ok=True)
    args = yaml.safe_load((run / "training/args.yaml").read_text())
    metadata = json.loads((run / "run_metadata.json").read_text())
    cfg = get_cfg(overrides=args)
    random.seed(709)
    np.random.seed(709)
    torch.manual_seed(709)
    dataset = YOLODataset(img_path=str(run / "train.txt"), imgsz=args["imgsz"], batch_size=1,
                          augment=True, hyp=cfg, rect=False, stride=32, cache=False,
                          data={"names": {0: "tennis-ball"}, "nc": 1}, task="detect")
    by_path = {str(resolve(s["image"])): s for s in samples}
    rows = []
    for index, file in enumerate(dataset.im_files):
        sample = by_path[str(resolve(file))]
        image = cv2.imread(str(resolve(file)))
        values = np.loadtxt(resolve(sample["label"]), ndmin=2)
        if len(values) != 1:
            raise ValueError("Pilot diagnostic expects one manually reviewed ball per image")
        cx, cy, width, height = values[0, 1:5] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        box = np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
        letterbox = LetterBox((args["imgsz"], args["imgsz"]), auto=False)(image=image)
        ratio = min(args["imgsz"] / image.shape[1], args["imgsz"] / image.shape[0])
        augmentation = dataset[index]
        transformed = augmentation["bboxes"].numpy()
        tensor_height, tensor_width = augmentation["img"].shape[1:]
        transformed_sizes = (transformed[:, 2:4] * [tensor_width, tensor_height]).tolist()
        rows.append({"source_id": sample["source_id"], "frame_index": sample["frame_index"],
                     "image": sample["image"], "image_sha256": sample["sha256"],
                     "native_image_width_height": [image.shape[1], image.shape[0]],
                     "native_box_xyxy": box.tolist(), "native_box_width_height": [width, height],
                     "square_letterbox_width_height": [letterbox.shape[1], letterbox.shape[0]],
                     "letterbox_ratio": ratio, "letterbox_box_width_height": [width * ratio, height * ratio],
                     "augmented_tensor_width_height": [tensor_width, tensor_height],
                     "augmented_label_count": len(transformed), "augmented_box_width_height": transformed_sizes,
                     "augmentation_probe": "Seed709 dry dataset traversal, not an exact replay of the original shuffled trainer RNG"})
        # Unmodified pixels at the native and resized ball location, enlarged
        # solely for visual review. No detector-derived label is introduced.
        crops = []
        for frame, px, py in [(image, cx, cy), (letterbox, cx * ratio,
                                  cy * ratio + (args["imgsz"] - round(image.shape[0] * ratio)) // 2)]:
            x, y = round(px), round(py)
            crop = frame[max(0, y - 20):y + 21, max(0, x - 20):x + 21]
            crops.append(cv2.resize(crop, (246, 246), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(output / f"{sample['source_id']}_{sample['frame_index']}_native_resized_crop.png"), cv2.hconcat(crops))
    epoch_path = run / "training/weights/epoch0.pt"
    saved = torch.load(epoch_path, map_location="cpu", weights_only=False)
    optimizer = saved.get("optimizer")
    steps = sorted({float(state["step"]) for state in optimizer["state"].values() if "step" in state}) if optimizer else []
    evidence = {"status": "DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION", "qualification_evidence": False,
                "final_test_used": False, "manifest_sha256": sha256_file(manifest_path),
                "runtime": PINNED_ULTRALYTICS, "image_size": args["imgsz"], "frames": rows,
                "initial_terminal_classifier_tensors": len(metadata["transfer"]["initialized_tensor_keys"]),
                "epoch_batches": len(samples), "optimizer_step_values": steps,
                "warmup_iterations": max(round(args["warmup_epochs"] * len(samples)), 100),
                "nominal_accumulation_batch": args["nbs"],
                "observed_training_previews": [str(run / f"training/train_batch{i}.jpg") for i in range(3)],
                "limitations": ["Six positive neighboring frames from two near-static courts; no reviewed negative images",
                                "Only two optimizer updates; classifier head was freshly initialized and warmup is unfinished",
                                "Target boxes become approximately3–6px in the full-frame network tensor",
                                "No independent human qualification review, source diversity or held-out accuracy evidence"]}
    if checkpoint:
        model = _load(resolve(checkpoint), "detect", "m", None, "cpu")
        input_shapes = []
        hook = model.model.register_forward_pre_hook(lambda module, inputs: input_shapes.append(list(inputs[0].shape))
                                                     if inputs and isinstance(inputs[0], torch.Tensor) else None)
        for row in rows:
            frame = cv2.imread(str(resolve(row["image"])))
            result = model.predict(frame, imgsz=args["imgsz"], device="cpu", conf=.00001,
                                   max_det=300, nms=False, end2end=True, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            target = np.asarray(row["native_box_xyxy"])
            overlaps = box_iou(boxes, target)
            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
            distances = np.linalg.norm(centers - (target[:2] + target[2:4]) / 2, axis=1)
            nearest = int(np.argmin(distances)) if len(distances) else None
            row["low_confidence_probe"] = {"threshold": .00001, "returned_candidates": len(boxes),
                                          "maximum_box_iou_with_training_label": float(overlaps.max()) if len(overlaps) else 0,
                                          "nearest_candidate_center_distance_px": float(distances[nearest]) if nearest is not None else None,
                                          "nearest_candidate_confidence": float(scores[nearest]) if nearest is not None else None,
                                          "nearest_candidate_xyxy": boxes[nearest].tolist() if nearest is not None else None}
        hook.remove()
        evidence["predictor_tensor_shapes"] = sorted({tuple(shape) for shape in input_shapes})
        evidence["checkpoint_sha256"] = sha256_file(resolve(checkpoint))
    path = output / "diagnostic.json"
    path.write_text(json.dumps(evidence, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Optional existing pilot checkpoint, CPU localization only")
    args = parser.parse_args()
    print(diagnose(args.manifest, args.run, args.output, checkpoint=args.checkpoint))


if __name__ == "__main__":
    main()

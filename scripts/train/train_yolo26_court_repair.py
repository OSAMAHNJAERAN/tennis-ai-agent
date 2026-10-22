"""Exact court175 warmstart: pure preflight, CPU preview, reviewed finite GPU run.

Default commands never import Torch or initialize CUDA. The first GPU stage
has total_epochs20 but stops after epoch2 for a concrete learning review.
Only its registered unstripped state can continue to that same total in a new
directory. Historical trainers, weights, data and production remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from scripts.data.export_yolo26_court_repair_v2 import (
    ROOT, FLIP_IDX, POINT_NAMES, canonical_hash, digest, read_json, verify_export, write_json,
)

EXPORT = ROOT / "data/yolo26/court_reviewed_development_v2"
MANIFEST_SHA = "4029e043f0fd67ac1b7458543cef4e12d73b6dcb01a88c7570fc66a79ba3be28"
INITIAL = ROOT / "outputs/yolo26_court_reviewed_development_run1/training/weights/best.pt"
INITIAL_SHA = "e47db3784adb39b82be6df7fca73ed0206f67f4f5d1bb029f0f862a34a4feefa"
PRIOR_METADATA = ROOT / "outputs/yolo26_court_reviewed_development_run1/run_metadata.json"
PRIOR_METADATA_SHA = "d5b35180e9399a30412cb486296d2590efadbdfbcc40516237fd8d6c0266c7f4"
OFFICIAL_ANCESTOR_SHA = "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646"
MODE = "REVIEWED_NATIVE_COURT175_FINITE_REPAIR"
VERSIONS = {"torch": "2.13.0+cu126", "ultralytics": "8.4.36"}
PREVIEW_IDS = ["TcXqTFdDaTA_2250", "dEsWEIct7Fk_2550", "t0WOL-1c9sQ_1550"]
TOTAL_EPOCHS = 20
REVIEW_EPOCH = 2
GPU_APPROVAL = "ROOT_APPROVED_COURT175_INITIAL2_OF20_GPU_LEARNING_CHECK"
RESUME_APPROVAL = "ROOT_APPROVED_COURT175_SAME20_UNFINISHED_STATE_CONTINUATION"


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def validate_binding(binding):
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("Require an explicit retained evidence path and SHA256")
    path = Path(binding["path"]).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != binding.get("sha256"):
        raise ValueError("Retained evidence path/hash changed")
    return path


def fresh_output(path):
    path = Path(path).resolve()
    parent = (ROOT / "outputs").resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError("Choose a fresh child of workspace outputs")
    if path.exists():
        raise FileExistsError("Never overwrite or reuse a retained output, including resume")
    return path


def training_settings(output):
    output = Path(output).resolve()
    return dict(data=str(output / "training_inputs/dataset.yaml"), epochs=20, imgsz=1280, batch=1,
        device="0", seed=709, deterministic=True, patience=20, save=True, save_period=5,
        amp=False, workers=0, cache=False, optimizer="AdamW", lr0=.0005, nbs=1, warmup_epochs=0.,
        project=str(output), name="training", exist_ok=False, val=True, plots=False,
        mosaic=0., close_mosaic=0, mixup=0., cutmix=0., copy_paste=0., scale=.05, translate=0.,
        degrees=0., shear=0., perspective=0., fliplr=0., flipud=0., bgr=0., multi_scale=0., rect=False,
        auto_augment=None, augmentations=[], erasing=0., hsv_h=.01, hsv_s=.15, hsv_v=.15)


def source_bindings():
    from scripts.train.train_yolo26_reviewed_ball import _source_bindings
    bindings = _source_bindings()
    for path in (Path(__file__), ROOT / "scripts/data/export_yolo26_court_repair_v2.py",
                 ROOT / "scripts/data/consolidate_yolo26_court_source_wave.py",
                 ROOT / "scripts/data/acquire_yolo26_court_source_wave.py",
                 ROOT / "scripts/evaluate/diagnose_yolo26_court_repair.py",
                 ROOT / "scripts/evaluate/diagnose_yolo26_reviewed_court.py",
                 ROOT / "scripts/evaluate/diagnose_yolo26_court_input_transform.py"):
        bindings[str(path.resolve())] = digest(path)
    return dict(sorted(bindings.items()))


def runtime_bindings():
    from scripts.train.train_yolo26_reviewed_ball import _runtime_source_bindings
    package = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    return _runtime_source_bindings(package)


def binding_changes(before, after):
    from scripts.train.train_yolo26_reviewed_ball import _binding_changes
    return _binding_changes(before, after)


def _changed(before, after):
    return any(binding_changes(before, after).values())


def audit_initialization():
    if digest(INITIAL) != INITIAL_SHA or digest(PRIOR_METADATA) != PRIOR_METADATA_SHA:
        raise ValueError("Pinned prior court14 best or metadata changed")
    prior = read_json(PRIOR_METADATA)
    if (prior.get("status") != "TRAINED_UNQUALIFIED" or prior.get("role") != "court"
            or prior.get("mode") != "REVIEWED_STATIC_COURT_DEVELOPMENT"
            or prior.get("initial_weights", {}).get("sha256") != OFFICIAL_ANCESTOR_SHA
            or prior.get("final_test_used") is not False or prior.get("qualification_evidence") is not False
            or not any(Path(row.get("path", "")).resolve() == INITIAL and row.get("sha256") == INITIAL_SHA
                       for row in prior.get("checkpoints", []))):
        raise ValueError("Warmstart must be the registered official-derived private court14 checkpoint")
    return {"checkpoint": _binding(INITIAL), "prior_metadata": _binding(PRIOR_METADATA),
            "official_ancestor_sha256": OFFICIAL_ANCESTOR_SHA,
            "kind": "NEW_OPTIMIZER_FINE_TUNE_FROM_TRAINED_COURT14_NOT_OLD_DATA_RESUME"}


def diagnostic_protocol(manifest):
    dev = [row for row in manifest["samples"] if row["split"] == "val"]
    if len(dev) != 26:
        raise ValueError("Diagnostic must cover all26 frozen development images")
    return {"schema": "tennis.yolo26.court175-native-diagnostic.v1", "qualification_evidence": False,
        "final_test_used": False, "independent_human_review": False, "manifest_sha256": MANIFEST_SHA,
        "baseline_checkpoint": _binding(INITIAL), "selection": "ALL26_DEV_IN_MANIFEST_ORDER_NO_SWEEP",
        "inference": {"imgsz": 1280, "batch": 1, "device": "0", "conf": .05, "iou": .7,
                      "max_det": 10, "rect": True, "half": False, "augment": False, "end2end": True, "nms": False},
        "court_instance": "HIGHEST_CLASS0_CONFIDENCE_STABLE_RETURNED_INDEX_TIE",
        "bbox_iou_minimum": .5, "keypoint_confidence_minimum": .25,
        "native_pixel_error_thresholds": [10, 20],
        "denominator": "ALL_VISIBLE_REFERENCE_SLOTS_IN_ALL26_INPUTS_INCLUDING_MISSES",
        "missing_policy": "MISSING_INSTANCE_LOW_CONFIDENCE_OR_NONFINITE_POINT_FAILS_VISIBLE_THRESHOLD_RECALL",
        "null_policy": "REPORT_UNKNOWN_SLOTS_SEPARATELY_EXCLUDE_FROM_LOCALIZATION_REFERENCE_NO_HIDDEN_POINT_CLAIM",
        "required_outputs": ["EVERY_IMAGE_LABEL_HASH_NATIVE_PREPROCESSING_AND_INVERSE",
            "EVERY_RETURNED_BOX_AND_ALL14_RAW_KEYPOINT_COORDINATES_AND_CONFIDENCES",
            "EVERY_VISIBLE_REFERENCE_ERROR_OR_MISSING_REASON", "MEDIAN_P95_WITH_CONDITIONAL_DENOMINATOR_AND_COVERAGE",
            "ALL_VISIBLE_DENOMINATOR_RECALL_AT10_AND20PX", "PER_LANDMARK_AND_CONTEXT_COUNTS_AND_ERRORS",
            "COMPLETE_IMAGE_MISSES_EXTRA_INSTANCES", "ALL26_NATIVE_TARGET_AND_PREDICTION_OVERLAYS",
            "NO_RECT_OR_THRESHOLD_SEARCH_NO_TEST_NO_METRIC_GROUND_OR_LINE_CALL_CLAIM"],
        "posttraining_selection": "REGISTERED_ENGINEERING_DEV_FITNESS_BEST_ONLY_NO_PROMOTION",
        "baseline_before_any_training_required": True,
        "preprocessing_observation": "EVERY_NATIVE_RGB_PIXEL_AND114_PADDING; ACTUAL_IMAGE_FORWARDS_SEPARATE_FROM_LIBRARY_WARMUP; INDEPENDENT_SCALE_COORDS_ROUNDTRIP",
        "samples": [{"id": row["id"], "context_group": row["context_group"],
            "visual_event_family": row["visual_event_family"], "image": _binding(EXPORT / row["image"]),
            "label": _binding(EXPORT / row["label"]), "bbox_xyxy_px": row["bbox_xyxy_px"],
            "keypoints_px": row["keypoints_px"]} for row in dev]}


def preflight(output):
    output = fresh_output(output)
    verification = verify_export(EXPORT, expected_manifest_sha256=MANIFEST_SHA)
    manifest = read_json(EXPORT / "source_manifest.json")
    observed = {name: importlib.metadata.version(name) for name in VERSIONS}
    if observed != VERSIONS or sys.version_info[:3] != (3, 13, 5):
        raise ValueError("Retain pinned Python3.13.5/Torch/Ultralytics runtime")
    return {"schema_version": "1.0", "status": "PURE_COURT175_REPAIR_PREFLIGHT_PENDING_CPU_AND_GPU_REVIEW",
        "mode": MODE, "role": "court", "data_audit": verification, "initialization": audit_initialization(),
        "training_arguments": training_settings(output), "output": str(output),
        "source_bindings": source_bindings(), "runtime_source_bindings": runtime_bindings(),
        "runtime_versions": observed, "python": sys.version, "python_executable": sys.executable,
        "diagnostic_protocol": diagnostic_protocol(manifest), "cpu_preview_ids": PREVIEW_IDS,
        "maximum_total_new_epochs": 20, "initial_learning_review_stop_epoch": 2,
        "gpu_resource_guard": {"measurement": "AFTER_TORCH_CUDA_INITIALIZATION_BEFORE_CHECKPOINT_MODEL",
            "minimum_available_ram_bytes": 1536 * 1024**2, "minimum_free_vram_bytes": 2 * 1024**3,
            "minimum_free_disk_bytes": 1024**3},
        "before_gpu_runtime_guard": {"minimum_available_ram_bytes": 4608 * 1024**2,
            "minimum_free_disk_bytes": 1024**3,
            "reason": "Actual30frame startup with about3GiB pre-runtime RAM left61,718,528B after Torch/CUDA; reserve about3GiB startup plus unchanged1.5GiB post-CUDA floor"},
        "cpu_guard": {"available_ram_before_and_after_torch_bytes": 1536 * 1024**2,
                      "free_disk_bytes": 1024**3, "cuda_initialization_allowed": False},
        "qualification_evidence": False, "final_test_used": False, "independent_human_review": False,
        "training_performed": False, "output_files_written": 0}


def _resources(path, *, torch=None):
    from scripts.data.acquire_yolo26_court_source_wave import current_resources
    resources = current_resources(Path(path))
    return {"free_ram_bytes": resources["available_physical_ram_bytes"],
            "free_disk_bytes": resources["free_disk_bytes"], "owned_rss_bytes": resources["owned_rss_bytes"],
            "peak_owned_rss_bytes": resources["peak_owned_rss_bytes"],
            "free_vram_bytes": None if torch is None else torch.cuda.mem_get_info(0)[0]}


def cpu_guard(resources):
    if resources["free_ram_bytes"] < 1536 * 1024**2 or resources["free_disk_bytes"] < 1024**3:
        raise ValueError("Unchanged CPU court1280 admission requires1.5GiB available RAM/1GiB disk")


def before_gpu_runtime_guard(resources):
    if resources["free_ram_bytes"] < 4608 * 1024**2 or resources["free_disk_bytes"] < 1024**3:
        raise ValueError("Before Torch/CUDA require4.5GiB available RAM/1GiB disk; unchanged post-CUDA1.5GiB floor follows")


def validate_model(model, *, require_names=True):
    # YOLO.__getattr__ proxies yaml from its inner PoseModel. Checking only
    # hasattr(wrapper, 'yaml') therefore mistakes the wrapper for the core.
    nested = getattr(model, "model", None)
    core = nested if hasattr(nested, "yaml") else model
    config = getattr(core, "yaml", {})
    head = core.model[-1] if hasattr(core, "model") else None
    if (not isinstance(config, dict) or config.get("scale") != "s" or config.get("nc") != 1
            or config.get("kpt_shape") != [14, 3]
            or Path(config.get("yaml_file", "")).name != "yolo26s-court-pose.yaml"
            or getattr(core, "end2end", None) is not True or list(getattr(head, "kpt_shape", [])) != [14, 3]
            or getattr(head, "nc", None) != 1
            or require_names and getattr(core, "names", None) != {0: "court"}):
        raise ValueError("Require the exact YOLO26s end-to-end court14 one-class pose specialist")
    return core


def head_fingerprints(core):
    """Hash trainable head parameters; running BN counters cannot prove learning."""
    prefix = f"model.{len(core.model) - 1}."
    values = {key: hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
              for key, value in core.named_parameters() if key.startswith(prefix) and value.requires_grad}
    if not values:
        raise ValueError("No court-head tensors found for the learning-change receipt")
    return values


def transfer_receipt(source, target, torch):
    """Prove complete exact FP32 reuse; partial compatible loading fails closed."""
    left, right = source.float().state_dict(), target.float().state_dict()
    missing = sorted(set(left) - set(right))
    unexpected = sorted(set(right) - set(left))
    shape = sorted(key for key in left.keys() & right.keys() if tuple(left[key].shape) != tuple(right[key].shape))
    unequal = sorted(key for key in left.keys() & right.keys() if key not in shape
                     and not torch.equal(left[key].detach().cpu(), right[key].detach().cpu()))
    if missing or unexpected or shape or unequal:
        raise ValueError(f"Incomplete trained-court transfer: missing={missing}, unexpected={unexpected}, shape={shape}, unequal={unequal}")
    return {"source_tensor_count": len(left), "target_tensor_count": len(right),
            "all_transferred_tensor_keys": sorted(left), "missing_keys": [], "unexpected_keys": [],
            "shape_mismatches": [], "unequal_initial_values": [], "randomly_initialized_keys": [],
            "exact_fp32_values_before_optimizer": True}


def repair_trainer_class(settings, *, resume=False, receipt_sink=None):
    from ultralytics.models.yolo.pose import PoseTrainer
    import torch
    class CourtRepairTrainer(PoseTrainer):
        def preprocess_batch(self, batch):
            # Pinned trainer retries first-epoch OOMs, even with batch1. This
            # controlled experiment must preserve the first failure instead.
            self._oom_retries = 3
            return super().preprocess_batch(batch)

        def _handle_nan_recovery(self, epoch):
            if (self.fitness is not None and not math.isfinite(float(self.fitness))
                    or any(not math.isfinite(float(value)) for value in self.metrics.values())):
                raise ValueError("Nonfinite actual court validation; automatic recovery forbidden")
            return False

        def check_resume(self, overrides):
            super().check_resume(overrides)
            if resume:
                for key, value in settings.items():
                    setattr(self.args, key, value)
                self.args.save_dir = str(Path(settings["project"]) / settings["name"])

        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is None:
                raise ValueError("Court repair never starts a random initialized pose head")
            source = weights["model"] if isinstance(weights, dict) else weights
            validate_model(source)
            target = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            validate_model(target, require_names=False)
            receipt = transfer_receipt(source, target, torch)
            if receipt_sink is not None:
                receipt_sink.append(receipt)
            return target
    return CourtRepairTrainer


def _copy_inputs(samples, output):
    records, bindings = [], []
    for row in samples:
        image = output / "court/images" / row["split"] / (row["id"] + ".png")
        label = output / "court/labels" / row["split"] / (row["id"] + ".txt")
        for relative, target in ((row["image"], image), (row["label"], label)):
            source = EXPORT / relative
            expected = digest(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if digest(target) != expected:
                raise ValueError("Training/preview working copy differs from immutable export")
            bindings.append(_binding(target))
        records.append({"image": str(image), "label": str(label), "split": row["split"], "instances": 1})
    return records, bindings


def _find_affine(transform):
    if type(transform).__name__ == "RandomPerspective":
        return transform
    for child in getattr(transform, "transforms", []):
        found = _find_affine(child)
        if found is not None:
            return found
    return None


def cpu_preview(plan, output):
    """Actual three TRAIN transforms and complete CPU court-head weight reuse."""
    output = fresh_output(output)
    # Plan values must still match live source/runtime/export facts.
    current = preflight(plan["output"])
    # Other authorized analytics SRC work may proceed during CPU-only review;
    # a fresh settled full SRC snapshot is still required before GPU approval.
    if {k: v for k, v in current.items() if k != "source_bindings"} != {k: v for k, v in plan.items() if k != "source_bindings"}:
        raise ValueError("Training inputs/runtime/config changed before CPU inspection")
    plan = current
    before = _resources(output)
    cpu_guard(before)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"mode": MODE, "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "CPU_PREVIEW_STARTED_NOT_TRAINING", "plan_sha256": canonical_hash(plan),
               "manifest_sha256": MANIFEST_SHA, "initialization": plan["initialization"],
               "training_arguments": plan["training_arguments"], "diagnostic_protocol_sha256": canonical_hash(plan["diagnostic_protocol"]),
               "resources_before_torch": before, "qualification_evidence": False, "training_performed": False,
               "final_test_used": False, "cuda_initialized": False, "samples": []}
    try:
        import torch
        import numpy as np
        import cv2
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        if torch.cuda.is_initialized():
            raise ValueError("CPU inspection cannot use initialized CUDA")
        receipt["resources_after_torch"] = _resources(output)
        cpu_guard(receipt["resources_after_torch"])
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        random.seed(709)
        np.random.seed(709)
        torch.random.default_generator.manual_seed(709)
        manifest = read_json(EXPORT / "source_manifest.json")
        by_id = {r["id"]: r for r in manifest["samples"]}
        selected = [by_id[ident] for ident in PREVIEW_IDS]
        if any(row["split"] != "train" for row in selected):
            raise ValueError("CPU preview must use only the frozen three TRAIN examples")
        _copy_inputs(selected, output / "preview_inputs")
        cfg = get_cfg(overrides=dict(plan["training_arguments"], task="pose", mode="train"))
        data = {"names": {0: "court"}, "nc": 1, "channels": 3, "kpt_shape": [14, 3], "flip_idx": FLIP_IDX}
        dataset = build_yolo_dataset(cfg, str(output / "preview_inputs/court/images/train"), 1,
                                     data, mode="train", rect=False, stride=32)
        affine = _find_affine(dataset.transforms)
        if affine is None:
            raise ValueError("Actual pinned training affine transform was not observed")
        original_apply = affine.apply_keypoints
        original_boxes = affine.apply_bboxes
        observations = []
        box_observations = []
        def observed_apply(keypoints, matrix):
            record = {"input": keypoints.copy(), "matrix": matrix.copy()}
            result = original_apply(keypoints, matrix)
            record["output"] = result.copy()
            observations.append(record)
            return result
        affine.apply_keypoints = observed_apply
        def observed_boxes(boxes, matrix):
            record = {"input": boxes.copy(), "matrix": matrix.copy()}
            result = original_boxes(boxes, matrix)
            record["output"] = result.copy()
            box_observations.append(record)
            return result
        affine.apply_bboxes = observed_boxes
        receipt["actual_transform_repr"] = repr(dataset.transforms)
        for index in range(len(dataset)):
            observations.clear()
            box_observations.clear()
            item = dataset[index]
            ident = Path(item["im_file"]).stem
            row = by_id[ident]
            image = item["img"].permute(1, 2, 0).numpy()
            keypoints = item["keypoints"].numpy()
            if (image.shape != (1280, 1280, 3) or keypoints.shape != (1, 14, 3)
                    or len(observations) != 1 or len(box_observations) != 1):
                raise ValueError("Actual TRAIN tensor/head count changed")
            observed = observations[0]
            box_record = box_observations[0]
            x1, y1, x2, y2 = row["bbox_xyxy_px"]
            corners = np.asarray([[x1, y1 + 280, 1.], [x2, y1 + 280, 1.],
                                  [x2, y2 + 280, 1.], [x1, y2 + 280, 1.]])
            mapped_box = corners @ observed["matrix"].T
            mapped_box = mapped_box[:, :2] / mapped_box[:, 2:]
            expected_box = np.clip([mapped_box[:, 0].min(), mapped_box[:, 1].min(),
                                    mapped_box[:, 0].max(), mapped_box[:, 1].max()], 0, 1280)
            actual_box = item["bboxes"].numpy()[0] * 1280
            actual_xyxy = np.asarray([actual_box[0] - actual_box[2] / 2, actual_box[1] - actual_box[3] / 2,
                                      actual_box[0] + actual_box[2] / 2, actual_box[1] + actual_box[3] / 2])
            bbox_error = float(max(abs(expected_box - actual_xyxy)))
            if bbox_error > .002 or max(abs(box_record["output"][0] - expected_box)) > .002:
                raise ValueError("Actual court box and14slot affine correspondence changed")
            maximum_error = 0.
            for slot, source in enumerate(row["keypoints_px"]):
                actual = keypoints[0, slot]
                if source is None:
                    if actual[2] != 0:
                        raise ValueError("Unknown reference became a visible transformed keypoint")
                    continue
                # Native image width already1280: actual letterbox adds280px on top.
                expected_input = np.asarray([source[0], source[1] + 280], dtype=float)
                if max(abs(expected_input - observed["input"][0, slot, :2])) > .001:
                    raise ValueError("Actual letterbox transform moved semantic source coordinates")
                mapped = observed["matrix"] @ np.asarray([*expected_input, 1.])
                expected = mapped[:2] / mapped[2]
                error = float(max(abs(actual[:2] * 1280 - expected)))
                maximum_error = max(maximum_error, error)
                if actual[2] != 2 or error > .002:
                    raise ValueError("Visible14slot affine correspondence/visibility changed")
            raw = output / (ident + "_actual_train_tensor.png")
            overlay_path = output / (ident + "_actual_train_overlay.png")
            bgr = image[..., ::-1].copy()
            if not cv2.imwrite(str(raw), bgr):
                raise ValueError("Cannot save actual unmarked train tensor")
            overlay = bgr.copy()
            for slot, point in enumerate(keypoints[0]):
                if point[2] != 0:
                    x, y = map(lambda v: int(round(float(v) * 1280)), point[:2])
                    cv2.circle(overlay, (x, y), 4, (0, 255, 255), 1)
                    cv2.putText(overlay, str(slot), (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            if not cv2.imwrite(str(overlay_path), overlay):
                raise ValueError("Cannot save train keypoint overlay")
            receipt["samples"].append({"id": ident, "split": "train", "unmarked_tensor": _binding(raw),
                "overlay": _binding(overlay_path), "actual_keypoints_normalized": keypoints.tolist(),
                "actual_bboxes_xywh_normalized": item["bboxes"].numpy().tolist(),
                "actual_affine_matrix": observed["matrix"].tolist(), "maximum_visible_coordinate_error_px": maximum_error,
                "maximum_bbox_coordinate_error_px": bbox_error,
                "unknown_slots": [slot for slot, p in enumerate(row["keypoints_px"]) if p is None]})
        cpu_guard(_resources(output))
        model = YOLO(str(INITIAL))
        source = validate_model(model)
        # Exercise the same actual get_model implementation that GPU training uses,
        # without constructing a trainer, optimizer, data loader or CUDA context.
        trainer_type = repair_trainer_class(plan["training_arguments"])
        trainer = object.__new__(trainer_type)
        trainer.data = data
        target = trainer.get_model(cfg=source.yaml, weights=source, verbose=False)
        receipt["complete_cpu_transfer"] = transfer_receipt(source, target, torch)
        receipt["actual_checkpoint_model"] = {"task": model.task, "scale": source.yaml["scale"],
            "nc": source.yaml["nc"], "kpt_shape": source.yaml["kpt_shape"], "names": source.names,
            "end2end": source.end2end, "checkpoint_sha256": digest(INITIAL)}
        after = preflight(plan["output"])
        if (torch.cuda.is_initialized()
                or {k: v for k, v in after.items() if k != "source_bindings"} != {k: v for k, v in plan.items() if k != "source_bindings"}):
            raise ValueError("CPU inspection initialized CUDA or training inputs/runtime changed")
        receipt["concurrent_source_changes"] = binding_changes(plan["source_bindings"], after["source_bindings"])
        for changed in (receipt["concurrent_source_changes"]["changed"]
                        + receipt["concurrent_source_changes"]["added"] + receipt["concurrent_source_changes"]["removed"]):
            if not Path(changed).is_relative_to(ROOT / "src"):
                raise ValueError("Training script changed during CPU inspection")
        receipt["full_source_snapshot_requires_refresh_before_gpu"] = True
        receipt.update(status="PASS_ACTUAL_CPU_TRAIN_TRANSFORMS_AND_COMPLETE_COURT14_TRANSFER_NOT_TRAINED",
                       resources_at_end=_resources(output), actual_visual_review_pending=True)
    except BaseException as error:
        receipt.update(status="CPU_INSPECTION_FAILED_PRESERVED_NO_RETRY", failure_type=type(error).__name__,
                       failure_reason=str(error), resources_at_end=_resources(output))
        write_json(output / "receipt.json", receipt)
        raise
    write_json(output / "receipt.json", receipt)
    return receipt


def same_settings(left, right):
    ignored = {"data", "project", "name"}
    return {k: v for k, v in left.items() if k not in ignored} == {k: v for k, v in right.items() if k not in ignored}


def validate_unfinished_state(state):
    epoch = state.get("epoch") if isinstance(state, dict) else None
    if (type(epoch) is not int or not 0 <= epoch < 19 or state.get("optimizer") is None
            or state.get("ema") is None or type(state.get("updates")) is not int or state["updates"] < 1
            or state.get("train_args", {}).get("epochs") != 20):
        raise ValueError("Resume requires registered unfinished unstripped optimizer/EMA state with total20")
    return epoch + 1


def audit_resume(binding, plan):
    checkpoint = validate_binding(binding)
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != "training":
        raise ValueError("Resume only a checkpoint from a retained court repair training output")
    metadata_path = checkpoint.parents[2] / "run_metadata.json"
    metadata = read_json(metadata_path)
    if (metadata.get("mode") != MODE or metadata.get("role") != "court"
            or metadata.get("manifest_sha256") != MANIFEST_SHA
            or metadata.get("runtime_versions") != VERSIONS
            or metadata.get("source_bindings") != plan["source_bindings"]
            or metadata.get("runtime_source_bindings") != plan["runtime_source_bindings"]
            or not same_settings(metadata.get("training_arguments", {}), plan["training_arguments"])
            or metadata.get("qualification_evidence") is not False
            or metadata.get("status") not in {"STOPPED_AFTER2_FOR_ROOT_LEARNING_REVIEW_UNQUALIFIED", "TRAINING_FAILED_UNQUALIFIED"}
            or not any(row.get("path") == str(checkpoint) and row.get("sha256") == binding["sha256"]
                       for row in metadata.get("unstripped_checkpoints", []))):
        raise ValueError("Resume data/settings/implementation/registered court state changed")
    return {"checkpoint": binding, "metadata": _binding(metadata_path), "prior": metadata}


def validate_gpu_approval(approval, plan, *, resume=False):
    if (approval.get("status") != (RESUME_APPROVAL if resume else GPU_APPROVAL)
            or approval.get("reviewer") != "ROOT_AGENT" or not approval.get("root_review_note")
            or approval.get("plan_sha256") != canonical_hash(plan)
            or approval.get("manifest_sha256") != MANIFEST_SHA or approval.get("total_epochs") != 20
            or approval.get("qualification_evidence") is not False or approval.get("final_test_used") is not False):
        raise ValueError("Require concrete root approval of this exact finite court175 GPU plan")
    for key in ("cpu_inspection", "diagnostic_protocol", "baseline26_review"):
        validate_binding(approval.get(key))
    protocol = read_json(approval["diagnostic_protocol"]["path"])
    if protocol.get("protocol") != plan["diagnostic_protocol"]:
        raise ValueError("Frozen native26 diagnostic protocol changed")
    cpu = read_json(approval["cpu_inspection"]["path"])
    if (cpu.get("status") != "PASS_ACTUAL_CPU_TRAIN_TRANSFORMS_AND_COMPLETE_COURT14_TRANSFER_NOT_TRAINED"
            or cpu.get("cuda_initialized") is not False or cpu.get("training_performed") is not False
            or cpu.get("manifest_sha256") != MANIFEST_SHA or cpu.get("initialization") != plan["initialization"]
            or not same_settings(cpu.get("training_arguments", {}), plan["training_arguments"])
            or cpu.get("diagnostic_protocol_sha256") != canonical_hash(plan["diagnostic_protocol"])
            or sorted(row.get("id", "") for row in cpu.get("samples", [])) != sorted(PREVIEW_IDS)
            or cpu.get("complete_cpu_transfer", {}).get("exact_fp32_values_before_optimizer") is not True):
        raise ValueError("Actual guarded CPU transform/complete transfer receipt required")
    baseline = read_json(approval["baseline26_review"]["path"])
    if (baseline.get("status") != "ROOT_REVIEWED_ACTUAL_ALL26_COURT175_BASELINE_NOT_QUALIFICATION"
            or baseline.get("manifest_sha256") != MANIFEST_SHA or baseline.get("checkpoint_sha256") != INITIAL_SHA
            or baseline.get("protocol_sha256") != approval["diagnostic_protocol"]["sha256"]
            or baseline.get("all26_dev_inputs_evaluated") is not True or baseline.get("qualification_evidence") is not False):
        raise ValueError("Actual complete26 prior-checkpoint native baseline root review required")
    diagnostic = read_json(validate_binding(baseline.get("diagnostic")))
    if (diagnostic.get("status") != "COMPLETED_ALL26_NATIVE_COURT175_FIXED_DIAGNOSTIC_NOT_QUALIFICATION"
            or diagnostic.get("manifest_sha256") != MANIFEST_SHA
            or diagnostic.get("qualification_evidence") is not False or diagnostic.get("final_test_used") is not False
            or diagnostic.get("plan", {}).get("phase") != "baseline"
            or diagnostic.get("plan", {}).get("protocol") != plan["diagnostic_protocol"]
            or diagnostic.get("plan", {}).get("checkpoint", {}).get("sha256") != INITIAL_SHA
            or [row.get("id") for row in diagnostic.get("rows", [])] != [row["id"] for row in plan["diagnostic_protocol"]["samples"]]):
        raise ValueError("Baseline root review must reference actual complete26 fixed-protocol diagnostic results")
    if resume:
        audit_resume(approval.get("resume_checkpoint"), plan)


def execute(plan, approval_path, approval_sha256):
    """GPU path requires separately reviewed CPU evidence and actual26 baseline."""
    output = fresh_output(plan["output"])
    if preflight(output) != plan or digest(approval_path) != approval_sha256:
        raise ValueError("Approved plan or release receipt changed")
    approval = read_json(approval_path)
    resume_mode = approval.get("status") == RESUME_APPROVAL
    validate_gpu_approval(approval, plan, resume=resume_mode)
    resumed = audit_resume(approval["resume_checkpoint"], plan) if resume_mode else None
    from scripts.train.train_yolo26_reviewed_ball import _write_metadata
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run_metadata.json"
    metadata = {"mode": MODE, "role": "court", "status": "NOT_STARTED_ADMISSION_PENDING",
        "started_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": MANIFEST_SHA,
        "runtime_versions": VERSIONS, "training_arguments": plan["training_arguments"],
        "source_bindings": plan["source_bindings"], "runtime_source_bindings": plan["runtime_source_bindings"],
        "plan_sha256": canonical_hash(plan), "root_approval": _binding(approval_path),
        "initialization": plan["initialization"], "qualification_evidence": False, "final_test_used": False,
        "production_promoted": False, "unstripped_checkpoints": [], "complete_transfer_receipts": [],
        "resume": None if resumed is None else {k: v for k, v in resumed.items() if k != "prior"},
        "maximum_total_new_epochs": 20,
        "resume_limits": "EMA-derived FP16 checkpoint model/optimizer; RNG/data cursor/stopper history not fully checkpointed, not bitwise reproducibility"}
    _write_metadata(path, metadata)
    started = None
    try:
        metadata["resources_before_gpu_runtime"] = _resources(output)
        before_gpu_runtime_guard(metadata["resources_before_gpu_runtime"])
        _write_metadata(path, metadata)
        import torch
        import psutil
        from ultralytics import YOLO
        from scripts.train.train_yolo26_tennis import prepare_dataset, check_resource_guard
        if not torch.cuda.is_available():
            raise ValueError("CUDA0 required; no CPU training fallback")
        torch.cuda.set_device(0)
        torch.cuda.init()
        resources = _resources(output, torch=torch)
        check_resource_guard(resources, gpu_requested=True, role="court", image_size=1280, development=True)
        metadata.update(resources_after_cuda=resources, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                        python=sys.version, python_executable=sys.executable)
        if torch.__version__ != VERSIONS["torch"] or torch.version.cuda != "12.6":
            raise ValueError("Pinned Torch/CUDA runtime changed")
        manifest = read_json(EXPORT / "source_manifest.json")
        records, working = _copy_inputs(manifest["samples"], output / "training_inputs")
        yaml = prepare_dataset(records, "court", output / "training_inputs", development=True)
        if str(yaml) != plan["training_arguments"]["data"]:
            raise ValueError("Actual working dataset path differs from approved settings")
        for p in (yaml, yaml.parent / "train.txt", yaml.parent / "val.txt", INITIAL):
            working.append(_binding(p))
        metadata["working_input_bindings"] = working
        model = YOLO(str(INITIAL if resumed is None else resumed["checkpoint"]["path"]))
        validate_model(model)
        if resumed is not None:
            metadata["resume_start_epoch"] = validate_unfinished_state(model.ckpt)
        trainer_type = repair_trainer_class(plan["training_arguments"], resume=resume_mode,
                                           receipt_sink=metadata["complete_transfer_receipts"])
        usage = {"peak_owned_rss_bytes": psutil.Process().memory_info().rss,
                 "minimum_available_ram_bytes": psutil.virtual_memory().available}
        def sample(trainer):
            usage["peak_owned_rss_bytes"] = max(usage["peak_owned_rss_bytes"], psutil.Process().memory_info().rss)
            usage["minimum_available_ram_bytes"] = min(usage["minimum_available_ram_bytes"], psutil.virtual_memory().available)
        def training_started(trainer):
            metadata["initial_court_head_fingerprints"] = head_fingerprints(validate_model(trainer.model))
            _write_metadata(path, metadata)
        def epoch_end(trainer):
            sample(trainer)
            if not torch.isfinite(trainer.tloss).all():
                raise ValueError("Nonfinite actual court learning loss")
            epoch = int(trainer.epoch) + 1
            steps = sorted({float(s["step"]) for s in trainer.optimizer.state.values() if "step" in s})
            metadata.setdefault("optimizer_update_receipts", []).append({"epoch": epoch, "step_values": steps,
                "batches_per_epoch": len(trainer.train_loader), "accumulation": int(trainer.accumulate),
                "finite_supervised_losses": trainer.tloss.detach().cpu().tolist(),
                "loss_names": list(trainer.loss_names)})
            if not resume_mode and epoch >= REVIEW_EPOCH:
                trainer.stop = True
            _write_metadata(path, metadata)
        def saved(trainer):
            epoch = int(trainer.epoch) + 1
            after_head = head_fingerprints(validate_model(trainer.model))
            changed_head = sorted(key for key, value in after_head.items()
                                  if value != metadata["initial_court_head_fingerprints"].get(key))
            if not changed_head:
                raise ValueError("No actual court-head tensor changed during supervised learning")
            metadata.setdefault("learning_epoch_receipts", []).append({"epoch": epoch,
                "validation_metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "fitness": float(trainer.fitness), "court_head_fingerprints": after_head,
                "changed_court_head_tensor_keys": changed_head})
            if trainer.stop or epoch == 20:
                target = Path(trainer.wdir) / f"review_unstripped_epoch{epoch}.pt"
                with Path(trainer.last).open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                metadata["unstripped_checkpoints"].append(_binding(target) | {"epoch": epoch})
            _write_metadata(path, metadata)
        model.add_callback("on_train_batch_end", sample)
        model.add_callback("on_train_start", training_started)
        model.add_callback("on_train_epoch_end", epoch_end)
        model.add_callback("on_model_save", saved)
        metadata["status"] = "TRAINING_STARTED_UNQUALIFIED"
        _write_metadata(path, metadata)
        arguments = dict(plan["training_arguments"], trainer=trainer_type)
        if resumed is not None:
            arguments["resume"] = resumed["checkpoint"]["path"]
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        model.train(**arguments)
        metadata.update(training_elapsed_seconds=time.perf_counter() - started, resource_samples=usage,
            peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_torch_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
        metadata["status"] = ("STOPPED_AFTER2_FOR_ROOT_LEARNING_REVIEW_UNQUALIFIED" if not resume_mode
                              else "FINISHED_SAME20_FINITE_REPAIR_UNQUALIFIED")
        metadata["checkpoints"] = [_binding(p) for p in sorted((output / "training/weights").glob("*.pt"))]
        if (_changed(plan["source_bindings"], source_bindings())
                or _changed(plan["runtime_source_bindings"], runtime_bindings())
                or any(digest(x["path"]) != x["sha256"] for x in working)
                or digest(approval_path) != approval_sha256):
            raise ValueError("Training source/runtime/working input/approval bytes changed")
        verify_export(EXPORT, expected_manifest_sha256=MANIFEST_SHA)
        metadata["input_source_runtime_changes"] = []
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_metadata(path, metadata)
    except BaseException as error:
        metadata.update(status="TRAINING_FAILED_UNQUALIFIED" if started is not None else "NOT_STARTED_ADMISSION_OR_RUNTIME_FAILURE",
            failure_type=type(error).__name__, failure_reason=str(error), failed_utc=datetime.now(timezone.utc).isoformat())
        _write_metadata(path, metadata)
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "cpu-preview", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-preflight", type=Path)
    parser.add_argument("--freeze-diagnostic-protocol", type=Path)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    plan = preflight(args.output)
    if args.save_preflight:
        write_json(args.save_preflight, plan)
    if args.freeze_diagnostic_protocol:
        write_json(args.freeze_diagnostic_protocol, {"status": "FROZEN_BEFORE_ANY_COURT175_GPU_DIAGNOSTIC_OR_TRAINING",
            "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": plan["diagnostic_protocol"]})
    if args.command == "cpu-preview":
        if args.preview_output is None:
            parser.error("CPU preview requires a separate fresh --preview-output in outputs")
        result = cpu_preview(plan, args.preview_output)
        print(json.dumps({"status": result["status"], "samples": len(result["samples"]), "cuda_initialized": False}))
    elif args.command == "execute":
        if args.approval is None or args.approval_sha256 is None:
            parser.error("GPU execution requires separately reviewed --approval and exact --approval-sha256")
        result = execute(plan, args.approval, args.approval_sha256)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
    else:
        print(json.dumps({"status": plan["status"], "plan_sha256": canonical_hash(plan), "torch_imported": "torch" in sys.modules,
                          "input_bindings": plan["data_audit"]["input_hashes_verified"], "training_performed": False}))


if __name__ == "__main__":
    main()

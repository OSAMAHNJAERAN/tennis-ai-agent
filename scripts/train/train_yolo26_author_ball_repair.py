"""Private ball101 warmstart preparation and separately approved finite execution.

Pure preflight never imports Torch, numerical/image libraries or CUDA. CPU
inspection, baseline inference and GPU training each need their own exact ROOT
release. The initial learning stage completes two of twenty epochs, then stops.
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

from scripts.data import export_yolo26_author_ball_repair_v2 as exported

ROOT = exported.ROOT
EXPORT = exported.OUTPUT
MODE = "PRIVATE_NATIVE_BALL101_FINITE_REPAIR"
MANIFEST_SHA = "fdd63635665ebd03426ab452496cf0b93ad52d462f9c32886e6ff7f3ab564c22"
RECEIPT_SHA = "554fb861957b130f65b3936f63ac159dc65e2fc9fe0ed9a1eb5bc667f50568d6"
YAML_SHA = "0941286e0cd90efab514b033378be4738f47549ab15e94ae7d123f8f5ebca0c8"
INITIAL = ROOT / "outputs/yolo26_ball_reviewed_development_run1/training/weights/best.pt"
INITIAL_SHA = "90a141489dc7198789043466d7ab3c0245b86cb0135122f7b78d62115a8a9fcc"
PRIOR_METADATA = INITIAL.parents[2] / "run_metadata.json"
PRIOR_METADATA_SHA = "8bd1a064a5379f0730d743da3c2bd0209679c65240ea967d2b92febf3f7fc5a3"
OFFICIAL_ANCESTOR_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
VERSIONS = {"torch": "2.13.0+cu126", "ultralytics": "8.4.36"}
TOTAL_EPOCHS, REVIEW_EPOCH = 20, 2
WALL_SECONDS = 900
CPU_RELEASE = "ROOT_APPROVED_BALL101_CPU_TRANSFER_AND_THREE_TRAIN_FACTORY_PREVIEWS"
GPU_RELEASE = "ROOT_APPROVED_BALL101_INITIAL_TWO_OF_TWENTY_EPOCHS"
RESUME_RELEASE = "ROOT_APPROVED_BALL101_SAME_TWENTY_EPOCH_CONTINUATION"


def digest(path):
    return exported.parent.digest(path)


def read(path):
    return exported.read_json(path)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def validate_binding(item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Explicit retained evidence path/hash required")
    path = Path(item["path"]).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != item.get("sha256"):
        raise ValueError("Retained evidence is outside workspace or changed")
    return path


def fresh_output(path):
    path = Path(path).resolve()
    if path == ROOT / "outputs" or not path.is_relative_to(ROOT / "outputs"):
        raise ValueError("Choose a fresh child of workspace outputs")
    if path.exists():
        raise FileExistsError("Preserve all previous attempts; output must be fresh")
    return path


def training_settings(output):
    from scripts.train.train_yolo26_reviewed_ball import training_settings as original
    settings = original(output, epochs=TOTAL_EPOCHS, seed=709)
    settings["patience"] = 20
    return settings


def audit_export():
    result = exported.verify_export(EXPORT, expected_manifest_sha256=MANIFEST_SHA)
    if (result["receipt_sha256"] != RECEIPT_SHA or result["data_yaml_sha256"] != YAML_SHA
            or result["counts"] != exported.EXPECTED_COUNTS
            or result["manifest"].get("qualification_evidence") is not False):
        raise ValueError("Require the exact ROOT-approved private101 export without TEST")
    return result


def audit_initialization():
    if digest(INITIAL) != INITIAL_SHA or digest(PRIOR_METADATA) != PRIOR_METADATA_SHA:
        raise ValueError("Prior trained ball best or its registered provenance changed")
    prior = read(PRIOR_METADATA)
    if (prior.get("status") != "TRAINED_UNQUALIFIED" or prior.get("role") != "ball"
            or prior.get("mode") != "REVIEWED_NATIVE_BALL_DEVELOPMENT"
            or prior.get("initial_weights", {}).get("sha256") != OFFICIAL_ANCESTOR_SHA
            or prior.get("qualification_evidence") is not False
            or not any(Path(row["path"]).resolve() == INITIAL and row["sha256"] == INITIAL_SHA for row in prior["checkpoints"])):
        raise ValueError("Warmstart must be the registered official-derived one-class ball specialist")
    return {"checkpoint": binding(INITIAL), "prior_metadata": binding(PRIOR_METADATA),
            "official_ancestor_sha256": OFFICIAL_ANCESTOR_SHA,
            "required_transfer": "ALL_TENSOR_KEYS_SHAPES_AND_EXACT_FP32_VALUES_NO_HEAD_REINITIALIZATION",
            "actual_tensor_count": None, "runtime_transfer_not_yet_executed": True}


def preview_ids(samples):
    train = [row for row in samples if row["development_split"] == "train"]
    positives = [row for row in train if not row["negative_visible_absence_only"] and row.get("inherited_from_old79") is False]
    negatives = [row for row in train if row["negative_visible_absence_only"] and row.get("inherited_from_old79") is True]
    def area(row):
        x1, y1, x2, y2 = row["box_xyxy_crop_px"]
        return (x2 - x1) * (y2 - y1)
    if not positives or not negatives:
        raise ValueError("Require NEW22 TRAIN positives and an inherited reviewed TRAIN absence crop")
    return [min(positives, key=lambda row: (area(row), row["sample_id"]))["sample_id"],
            min(positives, key=lambda row: (-area(row), row["sample_id"]))["sample_id"],
            min(negatives, key=lambda row: row["sample_id"])["sample_id"]]


def parameter_bindings():
    files = (Path(__file__), ROOT / "scripts/evaluate/diagnose_yolo26_author_ball_repair.py",
             ROOT / "scripts/data/export_yolo26_author_ball_repair_v2.py",
             ROOT / "scripts/train/train_yolo26_reviewed_ball.py", ROOT / "scripts/train/train_yolo26_tennis.py",
             ROOT / "scripts/data/export_yolo26_reviewed_ball_wave.py",
             ROOT / "scripts/data/prepare_yolo26_reviewed_ball_wave.py",
             ROOT / "scripts/data/prepare_yolo26_ball_review_wave.py",
             ROOT / "scripts/data/acquire_yolo26_court_source_wave.py",
             ROOT / "scripts/data/diagnose_yolo26_native_tiles.py",
             ROOT / "scripts/evaluate/diagnose_yolo26_court_input_transform.py",
             ROOT / "scripts/evaluate/diagnose_yolo26_reviewed_ball.py")
    return {str(path.resolve()): digest(path) for path in files}


def preflight(output):
    output = fresh_output(output)
    result = audit_export()
    initialization = audit_initialization()
    versions = {key: importlib.metadata.version(key) for key in VERSIONS}
    if versions != VERSIONS:
        raise ValueError("Pinned installed training runtime changed")
    from scripts.evaluate.diagnose_yolo26_author_ball_repair import prepare_protocol
    return {"schema": "tennis.private-ball101-controlled-training.v1", "mode": MODE,
        "status": "PURE_PREFLIGHT_PENDING_SEPARATE_CPU_BASELINE_AND_GPU_ROOT_RELEASES",
        "output": str(output), "dataset": {"manifest": binding(EXPORT / "source_manifest.json"),
            "export_receipt": binding(EXPORT / "export_receipt.json"), "yaml": binding(EXPORT / "ball/data.yaml"),
            "counts": result["counts"]}, "initialization": initialization, "versions": versions,
        "training_arguments": training_settings(output), "initial_completed_epoch_budget": REVIEW_EPOCH,
        "total_epoch_ceiling": TOTAL_EPOCHS, "initial_maximum_batch1_optimizer_steps": 160,
        "continuation": "ONLY_REGISTERED_UNSTRIPPED_EPOCH2_EMA_OPTIMIZER_STATE_SAME20_AFTER_ROOT_REVIEW",
        "owned_process_wall_seconds": WALL_SECONDS, "cpu_preview_train_ids": preview_ids(result["samples"]),
        "cpu_preview_selection": "MIN_AND_MAX_VISIBLE_BOX_AREA_AMONG_NEW22_TRAIN_PLUS_FIRST_INHERITED_TRAIN_ABSENCE",
        "diagnostic_protocol": prepare_protocol(output), "parameter_logic_bindings": parameter_bindings(),
        "global_src_bindings": None, "global_src_snapshot_policy": "DEFER_UNTIL_ROOT_RELEASE_AFTER_CONCURRENT_SEMANTIC_EDITS_SETTLE",
        "cpu_admission": {"ram_before_and_after_torch_bytes": 1536 * 1024**2, "disk_bytes": 1024**3, "cuda_allowed": False},
        "gpu_admission": {"ram_before_torch_bytes": 4608 * 1024**2,
            "ram_after_cuda_bytes": 1024**3, "free_vram_bytes": 2 * 1024**3, "disk_bytes": 1024**3},
        "source_scope": "OLD79_TRAINED_PLUS22_TRAIN_DATA_REPAIR_UNCHANGED21DEV_NOT_INDEPENDENT_TEST",
        "qualification_evidence": False, "final_test_used": False, "training_performed": False,
        "raw_or_weight_public_release_cleared": False, "commercial_clearance": False}


def resource_snapshot(path, torch=None):
    from scripts.data.acquire_yolo26_court_source_wave import current_resources
    values = current_resources(Path(path))
    return {"ram": values["available_physical_ram_bytes"], "disk": values["free_disk_bytes"],
            "rss": values["owned_rss_bytes"], "peak_rss": values["peak_owned_rss_bytes"],
            "vram": None if torch is None else int(torch.cuda.mem_get_info(0)[0])}


def resource_guard(values, stage):
    required_ram = {"cpu": 1536 * 1024**2, "before_gpu": 4608 * 1024**2, "after_cuda": 1024**3}[stage]
    if values["ram"] < required_ram or values["disk"] < 1024**3:
        raise ValueError(f"Unchanged {stage} RAM/disk resource guard refused: {values}")
    if stage == "after_cuda" and (values.get("vram") is None or values["vram"] < 2 * 1024**3):
        raise ValueError("Unchanged post-CUDA2GiB free VRAM guard refused")


def validate_model(model, require_names=True):
    nested = getattr(model, "model", None)
    core = nested if hasattr(nested, "yaml") else model
    config = getattr(core, "yaml", {})
    head = core.model[-1] if hasattr(core, "model") else None
    if (not isinstance(config, dict) or config.get("scale") != "m" or config.get("nc") != 1
            or Path(config.get("yaml_file", "")).name not in {"yolo26m.yaml", "yolo26m-tennis-ball.yaml"}
            or getattr(core, "end2end", None) is not True or getattr(head, "nc", None) != 1
            or require_names and getattr(core, "names", None) != {0: "tennis-ball"}):
        raise ValueError("Require the exact end-to-end YOLO26m one-class trained tennis-ball detector")
    return core


def transfer_receipt(source, target, torch):
    left, right = source.float().state_dict(), target.float().state_dict()
    missing, unexpected = sorted(set(left) - set(right)), sorted(set(right) - set(left))
    shapes = sorted(key for key in left.keys() & right.keys() if tuple(left[key].shape) != tuple(right[key].shape))
    unequal = sorted(key for key in left.keys() & right.keys() if key not in shapes
        and not torch.equal(left[key].detach().cpu(), right[key].detach().cpu()))
    non_fp32 = sorted(key for key, value in list(left.items()) + list(right.items())
        if value.is_floating_point() and str(value.dtype) != "torch.float32")
    if missing or unexpected or shapes or unequal or non_fp32 or not left:
        raise ValueError(f"Incomplete exact ball transfer: {missing=}, {unexpected=}, {shapes=}, {unequal=}, {non_fp32=}")
    return {"source_tensor_count": len(left), "target_tensor_count": len(right), "all_tensor_keys": sorted(left),
        "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [], "unequal_values": [],
        "randomly_initialized_keys": [], "all_floating_tensors_fp32": True, "exact_values_before_optimizer": True}


def head_fingerprints(core):
    prefix = f"model.{len(core.model) - 1}."
    values = {key: hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        for key, value in core.named_parameters() if key.startswith(prefix) and value.requires_grad}
    if not values:
        raise ValueError("No trainable ball-head parameters found")
    return values


def trainer_class(settings, transfer_sink=None, resume=False):
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch
    class BallRepairTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            self._oom_retries = 3  # Pinned trainer allows retries only while this is <3.
            return super().preprocess_batch(batch)

        def _handle_nan_recovery(self, epoch):
            if (self.fitness is not None and not math.isfinite(float(self.fitness))
                    or any(not math.isfinite(float(value)) for value in self.metrics.values())):
                raise ValueError("Nonfinite actual validation; automatic recovery forbidden")
            return False

        def check_resume(self, overrides):
            super().check_resume(overrides)
            if resume:
                for key, value in settings.items():
                    setattr(self.args, key, value)
                self.args.save_dir = str(Path(settings["project"]) / settings["name"])

        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is None:
                raise ValueError("Ball101 repair cannot initialize a random head")
            source = weights["model"] if isinstance(weights, dict) else weights
            validate_model(source)
            target = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            validate_model(target, require_names=False)
            receipt = transfer_receipt(source, target, torch)
            if transfer_sink is not None:
                transfer_sink.append(receipt)
            return target
    return BallRepairTrainer


def validate_release(approval, plan, status):
    if (approval.get("status") != status or approval.get("reviewer_type") != "ROOT_AGENT"
            or approval.get("plan_sha256") != exported.canonical_hash(plan)
            or not isinstance(approval.get("review_note"), str) or not approval["review_note"].strip()):
        raise ValueError("Separate exact ROOT review of this concrete stage is required")
    if any(approval.get(key) is not False for key in ("qualification_evidence", "final_test_used", "public_release_cleared")):
        raise ValueError("Private preparation approval cannot grant TEST, qualification or public rights")


def frozen_runtime_sources():
    from scripts.train.train_yolo26_reviewed_ball import _source_bindings
    sources = _source_bindings()
    sources.update(parameter_bindings())
    return sources, runtime_bindings()


def runtime_bindings():
    """Read installed model implementation hashes without importing it."""
    from scripts.train.train_yolo26_reviewed_ball import _runtime_source_bindings
    package = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    return _runtime_source_bindings(package)


def copy_inputs(samples, output):
    from scripts.train.train_yolo26_reviewed_ball import _copy_training_inputs
    return _copy_training_inputs(samples, output)


def save(path, value):
    from scripts.train.train_yolo26_reviewed_ball import _write_metadata
    _write_metadata(path, value)


def experiment_semantics(plan):
    """A continuation may change only fresh working/output locations."""
    ignored = {"data", "project", "name"}
    return exported.canonical_hash({"mode": plan["mode"], "dataset": plan["dataset"],
        "initialization": plan["initialization"], "versions": plan["versions"],
        "training_arguments": {key: value for key, value in plan["training_arguments"].items() if key not in ignored},
        "diagnostic_protocol": plan["diagnostic_protocol"], "cpu_preview_train_ids": plan["cpu_preview_train_ids"],
        "parameter_logic_bindings": plan["parameter_logic_bindings"], "total_epoch_ceiling": plan["total_epoch_ceiling"]})


def cpu_preview(plan, output, approval):
    validate_release(approval, plan, CPU_RELEASE)
    if preflight(plan["output"]) != plan:
        raise ValueError("CPU-approved parameter/data plan changed")
    output = fresh_output(output)
    before = resource_snapshot(output)
    resource_guard(before, "cpu")
    sources_before, runtime_before = frozen_runtime_sources()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "CPU_PREVIEW_STARTED", "plan_sha256": exported.canonical_hash(plan),
        "manifest_sha256": MANIFEST_SHA, "experiment_semantics_sha256": experiment_semantics(plan),
        "resources_before_torch": before, "samples": [],
        "observed_source_bindings_before": sources_before, "runtime_source_bindings": runtime_before,
        "parameter_logic_bindings": plan["parameter_logic_bindings"],
        "training_performed": False, "cuda_initialized": None, "qualification_evidence": False, "final_test_used": False}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        import numpy as np
        import cv2
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        if torch.cuda.is_initialized():
            raise ValueError("CPU inspection cannot initialize CUDA")
        receipt["resources_after_torch"] = resource_snapshot(output)
        resource_guard(receipt["resources_after_torch"], "cpu")
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        random.seed(709)
        np.random.seed(709)
        torch.random.default_generator.manual_seed(709)
        by_id = {row["sample_id"]: row for row in audit_export()["samples"]}
        selected = [by_id[key] for key in plan["cpu_preview_train_ids"]]
        if any(row["development_split"] != "train" for row in selected):
            raise ValueError("Preview cannot inspect DEV or TEST")
        copy_inputs(selected, output / "preview_inputs")
        cfg = get_cfg(overrides=dict(plan["training_arguments"], task="detect", mode="train"))
        data = {"names": {0: "tennis-ball"}, "nc": 1, "channels": 3}
        dataset = build_yolo_dataset(cfg, str(output / "preview_inputs/training_inputs/ball/images/train"), 1,
            data, mode="train", rect=False, stride=32)
        # HSV-only changes are permitted; no geometry or crop can alter labels.
        receipt["actual_factory_transform_repr"] = repr(dataset.transforms)
        for index in range(len(dataset)):
            item = dataset[index]
            key = Path(item["im_file"]).stem
            row = by_id[key]
            if list(item["img"].shape) != [3, 640, 640]:
                raise ValueError("Actual factory tensor changed native640 geometry")
            actual = item["bboxes"].numpy()
            expected = [] if row["negative_visible_absence_only"] else [list(map(float, row["label_text"].split()[1:]))]
            if actual.shape != (len(expected), 4) or expected and not np.allclose(actual, expected, atol=1e-6, rtol=0):
                raise ValueError("Factory transform changed/dropped the reviewed ball or absence label")
            if not np.isfinite(item["img"].numpy()).all():
                raise ValueError("Actual preview contains nonfinite pixels")
            path = output / (key + "_actual_train_tensor.png")
            image = item["img"].permute(1, 2, 0).numpy()[..., ::-1].copy()
            if not cv2.imwrite(str(path), image):
                raise ValueError("Cannot retain actual unmarked train tensor")
            overlay = image.copy()
            for cx, cy, width, height in actual:
                x1, y1, x2, y2 = [int(round(float(value) * 640)) for value in (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)
            overlay_path = output / (key + "_actual_train_overlay.png")
            if not cv2.imwrite(str(overlay_path), overlay):
                raise ValueError("Cannot retain actual factory label overlay")
            receipt["samples"].append({"id": key, "split": "train", "tensor": binding(path),
                "overlay": binding(overlay_path), "input_image": binding(row["image_path"]), "input_label": binding(row["label_path"]),
                "actual_shape_chw": list(item["img"].shape), "actual_labels_xywh": actual.tolist(),
                "native_geometry_and_label_extent_preserved": True})
        resource_guard(resource_snapshot(output), "cpu")
        wrapper = YOLO(str(INITIAL))
        source = validate_model(wrapper)
        kind = trainer_class(plan["training_arguments"])
        instance = object.__new__(kind)
        instance.data = data
        target = instance.get_model(cfg=source.yaml, weights=source, verbose=False)
        receipt["complete_transfer"] = transfer_receipt(source, target, torch)
        if torch.cuda.is_initialized() or preflight(plan["output"]) != plan:
            raise ValueError("CPU inspection initialized CUDA or changed approved inputs")
        sources_after, runtime_after = frozen_runtime_sources()
        from scripts.train.train_yolo26_reviewed_ball import _binding_changes
        changes = _binding_changes(sources_before, sources_after)
        receipt["observed_source_changes"] = changes
        receipt["observed_source_bindings_after"] = sources_after
        receipt["runtime_source_bindings_after"] = runtime_after
        if runtime_after != runtime_before or any(not Path(key).is_relative_to(ROOT / "src")
                for paths in changes.values() for key in paths):
            raise ValueError("CPU transfer/transform implementation changed during inspection")
        receipt.update(status="PASS_ACTUAL_CPU_BALL101_FACTORY_TRANSFORMS_AND_COMPLETE_FP32_TRANSFER", actual_visual_review_pending=True)
    except BaseException as error:
        receipt.update(status="FAILED_CPU_PREVIEW_NO_AUTOMATIC_RETRY", failure_type=type(error).__name__, failure_reason=str(error))
        raise
    finally:
        actual_torch = sys.modules.get("torch")
        final_error = None
        try:
            receipt["cuda_initialized"] = bool(actual_torch.cuda.is_initialized()) if actual_torch is not None else None
        except Exception as error:
            receipt["cuda_initialized"] = None
            receipt["cuda_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            receipt["resources_at_end"] = resource_snapshot(output)
            receipt["owned_peak_working_set_bytes"] = receipt["resources_at_end"]["peak_rss"]
        except Exception as error:
            receipt["end_resource_observation_error"] = {"type": type(error).__name__, "message": str(error)}
            if receipt["status"].startswith("PASS"):
                final_error = ValueError("Final CPU resource state could not be observed")
        try:
            final_sources, final_runtime = frozen_runtime_sources()
            receipt["runtime_source_bindings_observed_finally"] = final_runtime
            from scripts.train.train_yolo26_reviewed_ball import _binding_changes
            receipt["observed_source_changes_finally"] = _binding_changes(sources_before, final_sources)
            if receipt["status"].startswith("PASS") and receipt["runtime_source_bindings_observed_finally"] != runtime_before:
                final_error = ValueError("Installed runtime changed at the final CPU inspection boundary")
            if receipt["status"].startswith("PASS") and any(not Path(key).is_relative_to(ROOT / "src")
                    for paths in receipt["observed_source_changes_finally"].values() for key in paths):
                final_error = ValueError("CPU helper or parameter code changed at the final inspection boundary")
        except Exception as error:
            receipt["end_runtime_observation_error"] = {"type": type(error).__name__, "message": str(error)}
            if receipt["status"].startswith("PASS"):
                final_error = ValueError("Final installed runtime integrity could not be observed")
        if receipt["status"].startswith("PASS") and receipt["cuda_initialized"] is not False:
            final_error = ValueError("Final CPU CUDA state is initialized or unavailable")
        if final_error is not None:
            receipt.update(status="FAILED_CPU_PREVIEW_NO_AUTOMATIC_RETRY", failure_type=type(final_error).__name__, failure_reason=str(final_error))
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "receipt.json", receipt)
        if final_error is not None:
            raise final_error
    return receipt


def audit_gpu_release(approval, plan):
    continuing = approval.get("status") == RESUME_RELEASE
    validate_release(approval, plan, RESUME_RELEASE if continuing else GPU_RELEASE)
    cpu = read(validate_binding(approval["cpu_preview"]))
    if (cpu.get("status") != "PASS_ACTUAL_CPU_BALL101_FACTORY_TRANSFORMS_AND_COMPLETE_FP32_TRANSFER"
            or cpu.get("experiment_semantics_sha256") != experiment_semantics(plan)
            or sorted(row["id"] for row in cpu.get("samples", [])) != sorted(plan["cpu_preview_train_ids"])
            or any(cpu.get(key) is not False for key in ("training_performed", "cuda_initialized", "qualification_evidence", "final_test_used"))
            or approval.get("actual_preview_images_reviewed") is not True):
        raise ValueError("GPU release requires actual transfer/factory previews and separate image review")
    transfer = cpu.get("complete_transfer", {})
    count = transfer.get("source_tensor_count")
    keys = transfer.get("all_tensor_keys", [])
    if (type(count) is not int or count <= 0 or transfer.get("target_tensor_count") != count
            or not isinstance(keys, list) or len(keys) != count or len(set(keys)) != count
            or any(not isinstance(key, str) or not key for key in keys)
            or any(transfer.get(key) != [] for key in ("missing_keys", "unexpected_keys", "shape_mismatches", "unequal_values", "randomly_initialized_keys"))
            or transfer.get("all_floating_tensors_fp32") is not True or transfer.get("exact_values_before_optimizer") is not True):
        raise ValueError("CPU receipt lacks a complete measured exact FP32 tensor transfer")
    for row in cpu["samples"]:
        if row.get("split") != "train" or row.get("actual_shape_chw") != [3, 640, 640] or row.get("native_geometry_and_label_extent_preserved") is not True:
            raise ValueError("CPU receipt lacks actual native640 TRAIN factory evidence")
        for key in ("tensor", "overlay", "input_image", "input_label"):
            validate_binding(row[key])
        expected_image = EXPORT / "ball/images/train" / (row["id"] + ".png")
        expected_label = EXPORT / "ball/labels/train" / (row["id"] + ".txt")
        if row["input_image"] != binding(expected_image) or row["input_label"] != binding(expected_label):
            raise ValueError("CPU factory image/label evidence belongs to another sample")
    baseline = read(validate_binding(approval["fixed_native_baseline"]))
    if (baseline.get("status") != "COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED"
            or baseline.get("phase") != "before" or baseline.get("checkpoint", {}).get("sha256") != INITIAL_SHA
            or baseline.get("protocol_sha256") != exported.canonical_hash(plan["diagnostic_protocol"])
            or baseline.get("all_target_denominator") != 20 or baseline.get("actual_image_forward_count") != 161):
        raise ValueError("GPU release needs the actual fixed20DEV old-checkpoint baseline")
    from scripts.evaluate.diagnose_yolo26_author_ball_repair import validate_completed_receipt
    validate_completed_receipt(baseline, plan["diagnostic_protocol"], "before")
    sources, runtime = frozen_runtime_sources()
    if (cpu.get("runtime_source_bindings") != runtime or cpu.get("runtime_source_bindings_after") != runtime
            or cpu.get("runtime_source_bindings_observed_finally") != runtime
            or cpu.get("parameter_logic_bindings") != plan["parameter_logic_bindings"]
            or baseline.get("runtime_source_bindings") != runtime or baseline.get("runtime_source_bindings_after") != runtime):
        raise ValueError("CPU preview or fixed baseline used different installed runtime/parameter code")
    if not approval.get("source_bindings") or approval["source_bindings"] != sources or approval.get("runtime_source_bindings") != runtime:
        raise ValueError("GPU release requires a fresh settled global source/runtime freeze")
    return continuing


def observe_training_terminal(receipt, output, fixed_inputs, working):
    """Observe every terminal boundary without masking a primary run failure."""
    errors = []
    try:
        sources, runtime = frozen_runtime_sources()
        from scripts.train.train_yolo26_reviewed_ball import _binding_changes
        receipt["source_bindings_after"] = sources
        receipt["runtime_source_bindings_after"] = runtime
        receipt["full_source_integrity_changes"] = _binding_changes(receipt["source_bindings"], sources)
        receipt["runtime_integrity_changes"] = _binding_changes(receipt["runtime_source_bindings"], runtime)
        if any(receipt["full_source_integrity_changes"].values()) or any(receipt["runtime_integrity_changes"].values()):
            errors.append("Training helper/source/runtime bytes changed")
    except Exception as error:
        receipt["end_integrity_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        errors.append("Final training source/runtime integrity could not be observed")
    try:
        inputs = dict(fixed_inputs)
        for row in working:
            exported.bind(inputs, row["path"], row["sha256"])
        receipt["fixed_and_working_input_bindings_observed_finally"] = inputs
        receipt["input_hashes_changed"] = [path for path, sha256 in inputs.items()
            if not Path(path).is_file() or digest(path) != sha256]
        if receipt["input_hashes_changed"]:
            errors.append("Training model/data/working input bytes changed")
    except Exception as error:
        receipt["end_input_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        errors.append("Final training inputs could not be observed")
    actual_torch = sys.modules.get("torch")
    try:
        receipt["cuda_initialized"] = bool(actual_torch.cuda.is_initialized()) if actual_torch is not None else None
    except Exception as error:
        receipt["cuda_initialized"] = None
        receipt["cuda_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        errors.append("Final training CUDA state could not be observed")
    try:
        receipt["resources_at_end"] = resource_snapshot(output)
        receipt["owned_peak_working_set_bytes"] = receipt["resources_at_end"]["peak_rss"]
    except Exception as error:
        receipt["end_resource_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        errors.append("Final training resources could not be observed")
    try:
        receipt["terminal_checkpoint_bindings"] = [binding(path) for path in sorted((output / "training/weights").glob("*.pt"))]
    except Exception as error:
        receipt["end_checkpoint_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        errors.append("Terminal training checkpoint bytes could not be observed")
    receipt["terminal_observation_failures"] = errors
    return errors


def continuation_inputs(approval, plan):
    """Bind the registered unfinished run before importing the model runtime."""
    checkpoint = validate_binding(approval["resume_checkpoint"])
    prior_path = validate_binding(approval["initial_two_epoch_metadata"])
    prior = read(prior_path)
    if (prior.get("mode") != MODE or prior.get("status") != "STOPPED_AFTER_TWO_COMPLETED_EPOCHS_FOR_ROOT_REVIEW"
            or prior.get("experiment_semantics_sha256") != experiment_semantics(plan)
            or prior.get("source_bindings") != approval["source_bindings"]
            or prior.get("runtime_source_bindings") != approval["runtime_source_bindings"]
            or prior.get("dataset") != plan["dataset"]
            or any(prior.get(key) is not False for key in ("qualification_evidence", "final_test_used", "production_promoted"))
            or not any(Path(row["path"]).resolve() == checkpoint and row["sha256"] == digest(checkpoint)
                and row.get("epoch") == REVIEW_EPOCH for row in prior.get("unstripped_checkpoints", []))):
        raise ValueError("Continuation requires the registered actual epoch2 checkpoint and same exact data")
    ignored = {"data", "project", "name"}
    if ({k: v for k, v in prior["training_arguments"].items() if k not in ignored}
            != {k: v for k, v in plan["training_arguments"].items() if k not in ignored}):
        raise ValueError("Continuation cannot extend epochs or alter optimization settings")
    return checkpoint, {"checkpoint": binding(checkpoint), "metadata": binding(prior_path)}


def execute_owned_child(plan, approval):
    continuing = audit_gpu_release(approval, plan)
    if preflight(plan["output"]) != plan:
        raise ValueError("Approved ball101 plan changed")
    output = fresh_output(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    path = output / "run_metadata.json"
    receipt = {"mode": MODE, "role": "ball", "status": "ADMISSION_PENDING", "plan_sha256": exported.canonical_hash(plan),
        "dataset": plan["dataset"], "initialization": plan["initialization"], "training_arguments": plan["training_arguments"],
        "versions": VERSIONS, "experiment_semantics_sha256": experiment_semantics(plan),
        "started_utc": datetime.now(timezone.utc).isoformat(), "complete_transfer_receipts": [],
        "optimizer_update_receipts": [], "unstripped_checkpoints": [], "qualification_evidence": False,
        "final_test_used": False, "production_promoted": False, "source_bindings": approval["source_bindings"],
        "runtime_source_bindings": approval["runtime_source_bindings"]}
    save(path, receipt)
    started = time.perf_counter()
    fixed_inputs, working = {}, []
    try:
        receipt["resources_before_torch"] = resource_snapshot(output)
        resource_guard(receipt["resources_before_torch"], "before_gpu")
        audited = audit_export()
        fixed_inputs.update(audited["manifest"]["contract"]["input_bindings"])
        for sample in audited["samples"]:
            exported.bind(fixed_inputs, sample["image_path"], sample["image_source_sha256"])
            exported.bind(fixed_inputs, sample["label_path"], sample["label_sha256"])
        for item in [plan["dataset"][key] for key in ("manifest", "export_receipt", "yaml")] + [plan["initialization"][key] for key in ("checkpoint", "prior_metadata")]:
            exported.bind(fixed_inputs, item["path"], item["sha256"])
        checkpoint = INITIAL
        if continuing:
            checkpoint, receipt["resumed_from"] = continuation_inputs(approval, plan)
            for item in receipt["resumed_from"].values():
                exported.bind(fixed_inputs, item["path"], item["sha256"])
        exported.check_bindings(fixed_inputs)
        receipt["fixed_input_bindings_before"] = dict(fixed_inputs)
        import torch
        from ultralytics import YOLO
        from scripts.train.train_yolo26_tennis import prepare_dataset
        if torch.__version__ != VERSIONS["torch"] or not torch.cuda.is_available():
            raise ValueError("Pinned CUDA0 training required; CPU fallback forbidden")
        torch.cuda.set_device(0)
        torch.cuda.init()
        receipt["resources_after_cuda"] = resource_snapshot(output, torch)
        resource_guard(receipt["resources_after_cuda"], "after_cuda")
        if torch.version.cuda != "12.6":
            raise ValueError("Pinned CUDA version changed")
        samples = audited["samples"]
        records, working = copy_inputs(samples, output)
        data = prepare_dataset(records, "ball", output / "training_inputs", pilot=True)
        if str(data) != plan["training_arguments"]["data"]:
            raise ValueError("Prepared working train/DEV lists changed")
        working.extend(binding(p) for p in (data, data.parent / "train.txt", data.parent / "val.txt", INITIAL))
        model = YOLO(str(checkpoint))
        validate_model(model)
        if continuing:
            state = model.ckpt
            if (state.get("epoch") != 1 or state.get("optimizer") is None or state.get("ema") is None
                    or type(state.get("updates")) is not int or state["train_args"].get("epochs") != TOTAL_EPOCHS):
                raise ValueError("Require unfinished epoch2 optimizer/EMA state with the original20 ceiling")
        kind = trainer_class(plan["training_arguments"], receipt["complete_transfer_receipts"], resume=continuing)
        batch_counts = {}
        def on_start(trainer):
            receipt["initial_head_parameters"] = head_fingerprints(validate_model(trainer.model))
        def batch_end(trainer):
            if time.perf_counter() - started > WALL_SECONDS:
                raise TimeoutError("Finite owned ball101 learning budget exhausted")
            if not torch.isfinite(trainer.loss).all() or not torch.isfinite(trainer.loss_items).all():
                raise ValueError("Nonfinite supervised loss; recovery/retry forbidden")
            steps = sorted({float(state["step"]) for state in trainer.optimizer.state.values() if "step" in state})
            epoch = int(trainer.epoch) + 1
            batch_index = batch_counts.get(epoch, 0)
            batch_counts[epoch] = batch_index + 1
            receipt["optimizer_update_receipts"].append({"epoch": epoch,
                "batch_index": batch_index, "optimizer_step_values": steps,
                "accumulation": int(trainer.accumulate)})
            save(path, receipt)
        def epoch_end(trainer):
            if not torch.isfinite(trainer.tloss).all():
                raise ValueError("Nonfinite epoch loss")
            if not continuing and int(trainer.epoch) + 1 >= REVIEW_EPOCH:
                trainer.stop = True
        def saved(trainer):
            epoch = int(trainer.epoch) + 1
            current = head_fingerprints(validate_model(trainer.model))
            changed = sorted(key for key, value in current.items() if value != receipt["initial_head_parameters"].get(key))
            if not changed:
                raise ValueError("No trainable ball-head parameter changed")
            receipt.setdefault("completed_epoch_receipts", []).append({"epoch": epoch,
                "changed_trainable_head_keys": changed, "validation_metrics": {k: float(v) for k, v in trainer.metrics.items()},
                "fitness": float(trainer.fitness), "finite_supervised_losses": trainer.tloss.detach().cpu().tolist()})
            if trainer.stop or epoch == TOTAL_EPOCHS:
                kept = Path(trainer.wdir) / f"review_unstripped_epoch{epoch}.pt"
                with Path(trainer.last).open("rb") as incoming, kept.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                receipt["unstripped_checkpoints"].append(binding(kept) | {"epoch": epoch})
            save(path, receipt)
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_batch_end", batch_end)
        model.add_callback("on_train_epoch_end", epoch_end)
        model.add_callback("on_model_save", saved)
        receipt["status"] = "TRAINING_STARTED_UNQUALIFIED"
        save(path, receipt)
        arguments = dict(plan["training_arguments"], trainer=kind)
        if continuing:
            arguments["resume"] = str(checkpoint)
        torch.cuda.reset_peak_memory_stats(0)
        model.train(**arguments)
        expected_end = TOTAL_EPOCHS if continuing else REVIEW_EPOCH
        if not receipt.get("completed_epoch_receipts") or receipt["completed_epoch_receipts"][-1]["epoch"] != expected_end:
            raise ValueError("Actual controlled training did not complete the approved epoch boundary")
        if frozen_runtime_sources() != (approval["source_bindings"], approval["runtime_source_bindings"]) or any(digest(row["path"]) != row["sha256"] for row in working):
            raise ValueError("Model execution source/runtime/working inputs changed")
        audit_export()
        receipt.update(status="FINISHED_SAME_TWENTY_EPOCH_REPAIR_UNQUALIFIED" if continuing else "STOPPED_AFTER_TWO_COMPLETED_EPOCHS_FOR_ROOT_REVIEW",
            checkpoints=[binding(p) for p in sorted((output / "training/weights").glob("*.pt"))],
            input_changes=[], peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
    except BaseException as error:
        receipt.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", failure_type=type(error).__name__, failure_reason=str(error))
        raise
    finally:
        was_completed = receipt["status"] in {"FINISHED_SAME_TWENTY_EPOCH_REPAIR_UNQUALIFIED", "STOPPED_AFTER_TWO_COMPLETED_EPOCHS_FOR_ROOT_REVIEW"}
        final_errors = observe_training_terminal(receipt, output, fixed_inputs, working)
        if was_completed and receipt["cuda_initialized"] is not True:
            final_errors.append("Completed training lacks an observed initialized CUDA runtime")
        if was_completed and final_errors:
            receipt.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", failure_type="ValueError", failure_reason="; ".join(final_errors))
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        save(path, receipt)
        if was_completed and final_errors:
            raise ValueError(receipt["failure_reason"])
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "cpu-preview", "execute", "execute-owned-child"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight" and args.plan:
        raise ValueError("Preflight must reconstruct live metadata; a saved plan is for reviewed runtime stages only")
    plan = preflight(args.output) if args.plan is None else read(args.plan)
    if args.command == "preflight":
        if args.approval or args.approval_sha256:
            raise ValueError("Pure preflight cannot consume a runtime release")
        if args.write_plan:
            exported.write_new_json(args.write_plan, plan)
        print(json.dumps({"status": plan["status"], "plan_sha256": exported.canonical_hash(plan), "counts": plan["dataset"]["counts"]}, indent=2))
        return 0
    if not args.plan or not args.approval or digest(args.approval) != args.approval_sha256:
        raise ValueError("Runtime needs exact saved plan and separate ROOT stage approval")
    approval = read(args.approval)
    if args.command.startswith("execute") and Path(args.output).resolve() != Path(plan["output"]).resolve():
        raise ValueError("Owned training output must match the exact approved plan")
    if args.command == "cpu-preview":
        cpu_preview(plan, args.output, approval)
    elif args.command == "execute-owned-child":
        execute_owned_child(plan, approval)
    else:
        audit_gpu_release(approval, plan)
        resource_guard(resource_snapshot(args.output), "before_gpu")
        supervisor = fresh_output(str(args.output) + "_supervisor")
        supervisor.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "scripts.train.train_yolo26_author_ball_repair", "execute-owned-child",
            "--output", str(args.output), "--plan", str(args.plan), "--approval", str(args.approval), "--approval-sha256", args.approval_sha256]
        result = {"status": "OWNED_CHILD_STARTING", "wall_limit_seconds": WALL_SECONDS, "automatic_retry": False}
        try:
            with (supervisor / "child.log").open("xb") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=WALL_SECONDS, check=False)
            result.update(status="OWNED_CHILD_TERMINAL", exit_code=completed.returncode)
            if completed.returncode:
                raise RuntimeError("Owned training child failed; preserve original failure without retry")
        except subprocess.TimeoutExpired:
            result["status"] = "OWNED_CHILD_TERMINATED_AT_FINITE_WALL_LIMIT"
            raise
        finally:
            exported.write_new_json(supervisor / "receipt.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

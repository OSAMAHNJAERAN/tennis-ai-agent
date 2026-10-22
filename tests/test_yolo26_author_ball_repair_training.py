"""Pure admission/protocol tests; fake tensor objects never claim runtime transfer."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts.train import train_yolo26_author_ball_repair as train
from scripts.evaluate import diagnose_yolo26_author_ball_repair as diagnostic


@pytest.fixture(scope="module")
def plan():
    return train.preflight(train.ROOT / "outputs/yolo26_ball_author_repair_run1")


def test_actual_export_preflight_retains_old_dev_and_new_train_only(plan):
    assert plan["dataset"]["counts"] == {"images": 101, "positive_boxes": 97, "negative_tiles": 4,
        "train": 80, "val": 21, "test": 0, "inherited_images": 79, "new_positive_images": 22}
    assert plan["dataset"]["manifest"]["sha256"] == train.MANIFEST_SHA
    assert plan["initialization"]["checkpoint"]["sha256"] == train.INITIAL_SHA
    assert plan["initialization"]["actual_tensor_count"] is None
    assert plan["initial_completed_epoch_budget"] == 2 and plan["total_epoch_ceiling"] == 20
    assert plan["initial_maximum_batch1_optimizer_steps"] == 160
    assert plan["global_src_bindings"] is None
    assert plan["qualification_evidence"] is False and plan["final_test_used"] is False
    assert not Path(plan["output"]).exists()


def test_settings_preserve_native_small_ball_and_use_actual_batch1_steps(plan):
    args = plan["training_arguments"]
    assert (args["imgsz"], args["batch"], args["workers"], args["amp"], args["nbs"], args["warmup_epochs"]) == (640, 1, 0, False, 1, 0)
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "scale", "translate", "degrees", "shear", "perspective", "fliplr", "flipud", "bgr", "erasing", "multi_scale"):
        assert args[key] == 0, key
    assert args["augmentations"] == [] and args["auto_augment"] is None
    assert args["hsv_h"] == .01 and args["hsv_s"] == .15 and args["hsv_v"] == .15
    assert args["epochs"] == 20 and args["optimizer"] == "AdamW" and args["lr0"] == .0005


def test_factory_preview_selection_exercises_new_positive_sources_and_inherited_absence():
    def row(key, size, inherited, split="train", absent=False):
        return {"sample_id": key, "box_xyxy_crop_px": [0, 0, size, size], "inherited_from_old79": inherited,
                "development_split": split, "negative_visible_absence_only": absent}
    samples = [row("old_smaller", 1, True), row("old_larger", 500, True), row("new_small", 3, False),
        row("new_large", 20, False), row("new_DEV_ignored", 2, False, "val"), row("old_absence", 0, True, absent=True)]
    assert train.preview_ids(samples) == ["new_small", "new_large", "old_absence"]
    with pytest.raises(ValueError, match="NEW22"):
        train.preview_ids([row("old_only", 10, True), row("old_absence", 0, True, absent=True)])


@pytest.mark.parametrize("stage,ram", [("cpu", 1536 * 1024**2), ("before_gpu", 4608 * 1024**2), ("after_cuda", 1024**3)])
def test_runtime_guards_admit_exact_boundary_and_reject_one_byte_short(stage, ram):
    values = {"ram": ram, "disk": 1024**3, "vram": 2 * 1024**3}
    train.resource_guard(values, stage)
    with pytest.raises(ValueError):
        train.resource_guard(dict(values, ram=ram - 1), stage)
    with pytest.raises(ValueError):
        train.resource_guard(dict(values, disk=1024**3 - 1), stage)
    if stage == "after_cuda":
        with pytest.raises(ValueError):
            train.resource_guard(dict(values, vram=2 * 1024**3 - 1), stage)


class Tensor:
    def __init__(self, values, shape=(2,), floating=True):
        self.values, self.shape, self.floating = values, shape, floating
        self.dtype = "torch.float32" if floating else "torch.int64"
    def detach(self): return self
    def cpu(self): return self
    def is_floating_point(self): return self.floating


class Model:
    def __init__(self, state): self.state = state
    def float(self): return self
    def state_dict(self): return self.state


FAKE_TORCH = SimpleNamespace(equal=lambda a, b: a.values == b.values and a.dtype == b.dtype)


@pytest.mark.parametrize("count", [1, 9, 23])
def test_complete_transfer_counts_are_measured_from_all_keys_not_assumed(count):
    source = Model({f"tensor.{i}": Tensor([i, i + 1]) for i in range(count)})
    target = Model(deepcopy(source.state))
    result = train.transfer_receipt(source, target, FAKE_TORCH)
    assert result["source_tensor_count"] == count == result["target_tensor_count"]
    assert len(result["all_tensor_keys"]) == count
    assert result["randomly_initialized_keys"] == []


@pytest.mark.parametrize("damage", ["missing_head", "extra_head", "wrong_shape", "different_head_values", "non_fp32", "empty"])
def test_partial_or_randomized_head_transfer_fails_closed(damage):
    left = {"backbone": Tensor([1, 2]), "head": Tensor([3, 4])}
    right = deepcopy(left)
    if damage == "missing_head": right.pop("head")
    elif damage == "extra_head": right["another_head"] = Tensor([3, 4])
    elif damage == "wrong_shape": right["head"].shape = (1, 2)
    elif damage == "different_head_values": right["head"].values = [0, 0]
    elif damage == "non_fp32": left["head"].dtype = right["head"].dtype = "torch.float16"
    elif damage == "empty": left = right = {}
    with pytest.raises(ValueError, match="Incomplete"):
        train.transfer_receipt(Model(left), Model(right), FAKE_TORCH)


def _core():
    return SimpleNamespace(yaml={"scale": "m", "nc": 1, "yaml_file": "yolo26m-tennis-ball.yaml"},
        model=[SimpleNamespace(nc=1)], end2end=True, names={0: "tennis-ball"})


@pytest.mark.parametrize("damage", ["scale", "classes", "architecture", "end2end", "head_classes", "names"])
def test_wrong_checkpoint_model_is_not_accepted_as_the_trained_ball_specialist(damage):
    core = _core()
    assert train.validate_model(SimpleNamespace(model=core)) is core
    if damage == "scale": core.yaml["scale"] = "s"
    elif damage == "classes": core.yaml["nc"] = 80
    elif damage == "architecture": core.yaml["yaml_file"] = "yolo11m.yaml"
    elif damage == "end2end": core.end2end = False
    elif damage == "head_classes": core.model[0].nc = 80
    elif damage == "names": core.names = {0: "person"}
    with pytest.raises(ValueError):
        train.validate_model(core)


def _release(plan, status):
    return {"status": status, "reviewer_type": "ROOT_AGENT", "plan_sha256": train.exported.canonical_hash(plan),
        "review_note": "Concrete stage reviewed.", "qualification_evidence": False, "final_test_used": False, "public_release_cleared": False}


@pytest.mark.parametrize("field,value", [("status", train.GPU_RELEASE), ("reviewer_type", "HUMAN"), ("plan_sha256", "other"),
    ("review_note", ""), ("qualification_evidence", True), ("final_test_used", True), ("public_release_cleared", True)])
def test_cpu_runtime_cannot_start_on_wrong_stage_or_broader_authority(plan, field, value):
    approval = _release(plan, train.CPU_RELEASE)
    train.validate_release(approval, plan, train.CPU_RELEASE)
    approval[field] = value
    with pytest.raises(ValueError):
        train.validate_release(approval, plan, train.CPU_RELEASE)


def test_same20_continuation_changes_only_fresh_paths(plan):
    newer = deepcopy(plan)
    newer["output"] += "_continued"
    newer["training_arguments"].update(project=newer["output"], data=newer["output"] + "/training_inputs/dataset.yaml")
    assert train.experiment_semantics(newer) == train.experiment_semantics(plan)
    for key, value in (("epochs", 21), ("lr0", .001), ("amp", True), ("imgsz", 1280), ("nbs", 64)):
        changed = deepcopy(newer)
        changed["training_arguments"][key] = value
        assert train.experiment_semantics(changed) != train.experiment_semantics(plan)


def test_protocol_keeps_all_old_targets_negatives_and_native_policy(plan):
    protocol = plan["diagnostic_protocol"]
    reference = train.read(diagnostic.REFERENCE)
    for key in ("positive_samples", "negative_samples", "inference", "fixed_native_tile_grid", "target_iou_minimum", "dedup_iou_or_center"):
        assert protocol[key] == reference[key]
    assert protocol["inference"]["conf"] == .01
    assert protocol["all_target_denominator"] == 20 and protocol["actual_image_forward_count"] == 161
    assert protocol["full_frame_precision"] is None and protocol["negative_video_false_alarms_per_minute"] is None


def _completed(protocol):
    rows = [{"sample": deepcopy(sample), "tiles": [{"region": deepcopy(region),
        "observed": {"preprocessing": {"native_pixels_preserved": True}}} for region in protocol["fixed_native_tile_grid"]]}
        for sample in protocol["positive_samples"]]
    ids = [f"{sample['id']}:t{i}" for sample in protocol["positive_samples"] for i in range(8)] + [protocol["negative_samples"][0]["id"]]
    return {"status": "COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED", "phase": "before",
        "protocol_sha256": train.exported.canonical_hash(protocol), "all_target_denominator": 20,
        "actual_image_forward_count": 161, "rows": rows, "negative_crop_rows": [{"sample": deepcopy(protocol["negative_samples"][0])}],
        "forward_events": [{"phase": "ACTUAL_IMAGE_INFERENCE", "id": key, "shape_nchw": [1, 3, 640, 640], "dtype": "torch.float32"} for key in ids],
        "aggregates": {"reviewed_targets": 20, "matched_targets_conf001_iou05": 7, "missed_targets_conf001_iou05": 13},
        "qualification_evidence": False, "final_test_used": False, "training_performed": False,
        "runtime_source_bindings": {"synthetic_runtime": "bound"}, "runtime_source_bindings_after": {"synthetic_runtime": "bound"},
        "full_source_bindings_before": {"synthetic_source": "bound"}, "full_source_bindings_after": {"synthetic_source": "bound"},
        "fixed_input_bindings_before": {"synthetic_input": "bound"},
        "input_hashes_changed": [], "full_source_integrity_changes": {"changed": [], "added": [], "removed": []},
        "runtime_integrity_changes": {"changed": [], "added": [], "removed": []},
        "full_frame_precision": None, "negative_video_false_alarms_per_minute": None}


@pytest.mark.parametrize("damage", ["dropped_miss", "changed_target", "missing_tile", "wrong_region", "resized_pixels", "duplicate_forward", "missing_forward", "wrong_tensor", "half_tensor", "changed_absence", "wrong_denominator", "pretend_precision", "pretend_video_fpm", "qualification_evidence", "final_test_used", "training_performed", "missing_input_pins", "source_changed", "input_changed", "runtime_changed"])
def test_plausible_baseline_status_cannot_hide_missing_inputs_or_unsupported_accuracy(plan, damage):
    protocol = plan["diagnostic_protocol"]
    receipt = _completed(protocol)
    diagnostic.validate_completed_receipt(receipt, protocol, "before")
    if damage == "dropped_miss": receipt["rows"].pop()
    elif damage == "changed_target": receipt["rows"][0]["sample"]["target_box"][0] += 1
    elif damage == "missing_tile": receipt["rows"][0]["tiles"].pop()
    elif damage == "wrong_region": receipt["rows"][0]["tiles"][0]["region"][0] += 1
    elif damage == "resized_pixels": receipt["rows"][0]["tiles"][0]["observed"]["preprocessing"]["native_pixels_preserved"] = False
    elif damage == "duplicate_forward": receipt["forward_events"][1]["id"] = receipt["forward_events"][0]["id"]
    elif damage == "missing_forward": receipt["forward_events"].pop()
    elif damage == "wrong_tensor": receipt["forward_events"][0]["shape_nchw"] = [1, 3, 384, 640]
    elif damage == "half_tensor": receipt["forward_events"][0]["dtype"] = "torch.float16"
    elif damage == "changed_absence": receipt["negative_crop_rows"][0]["sample"]["id"] = "other"
    elif damage == "wrong_denominator": receipt["aggregates"]["missed_targets_conf001_iou05"] = 12
    elif damage == "pretend_precision": receipt["full_frame_precision"] = .95
    elif damage == "pretend_video_fpm": receipt["negative_video_false_alarms_per_minute"] = 0
    elif damage == "missing_input_pins": receipt.pop("fixed_input_bindings_before")
    elif damage == "source_changed": receipt["full_source_bindings_after"] = {"synthetic_source": "changed"}
    elif damage == "input_changed": receipt["input_hashes_changed"] = ["synthetic_input"]
    elif damage == "runtime_changed": receipt["runtime_source_bindings_after"] = {"synthetic_runtime": "changed"}
    elif damage in ("qualification_evidence", "final_test_used", "training_performed"): receipt[damage] = True
    with pytest.raises(ValueError):
        diagnostic.validate_completed_receipt(receipt, protocol, "before")


@pytest.mark.parametrize("damage", [None, "cpu_status_only", "missing_sample", "no_measured_count", "random_head", "fp16_transfer", "wrong_tensor_geometry", "cpu_cuda", "cpu_test", "visual_review_missing", "incomplete_baseline", "baseline_precision", "unsettled_sources", "runtime_changed", "prior_cpu_runtime", "cpu_runtime_changed_after", "cpu_final_runtime_changed", "helper_code_changed", "baseline_runtime_changed"])
def test_gpu_stage_release_needs_complete_cpu_and_fixed_baseline_evidence(plan, tmp_path, monkeypatch, damage):
    # These explicit software fixtures test admission logic only. Fake preview
    # bytes and forwards are never passed to model, image or training execution.
    samples = []
    for index, key in enumerate(plan["cpu_preview_train_ids"]):
        tensor, overlay = tmp_path / f"tensor{index}.png", tmp_path / f"overlay{index}.png"
        tensor.write_bytes(b"synthetic software fixture, not an actual preview")
        overlay.write_bytes(b"synthetic software fixture, not an actual overlay")
        samples.append({"id": key, "split": "train", "actual_shape_chw": [3, 640, 640],
            "native_geometry_and_label_extent_preserved": True, "tensor": train.binding(tensor), "overlay": train.binding(overlay),
            "input_image": train.binding(train.EXPORT / "ball/images/train" / (key + ".png")),
            "input_label": train.binding(train.EXPORT / "ball/labels/train" / (key + ".txt"))})
    cpu = {"status": "PASS_ACTUAL_CPU_BALL101_FACTORY_TRANSFORMS_AND_COMPLETE_FP32_TRANSFER",
        "experiment_semantics_sha256": train.experiment_semantics(plan), "samples": samples,
        "training_performed": False, "cuda_initialized": False, "qualification_evidence": False, "final_test_used": False,
        "runtime_source_bindings": {"synthetic_runtime": "bound"}, "runtime_source_bindings_after": {"synthetic_runtime": "bound"},
        "runtime_source_bindings_observed_finally": {"synthetic_runtime": "bound"},
        "parameter_logic_bindings": plan["parameter_logic_bindings"],
        "complete_transfer": {"source_tensor_count": 3, "target_tensor_count": 3, "all_tensor_keys": ["a", "b", "head"],
            "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [], "unequal_values": [],
            "randomly_initialized_keys": [], "all_floating_tensors_fp32": True, "exact_values_before_optimizer": True}}
    baseline = _completed(plan["diagnostic_protocol"])
    baseline["checkpoint"] = {"path": str(train.INITIAL), "sha256": train.INITIAL_SHA}
    if damage == "cpu_status_only": cpu.pop("complete_transfer")
    elif damage == "missing_sample": cpu["samples"].pop()
    elif damage == "no_measured_count": cpu["complete_transfer"]["source_tensor_count"] = 0
    elif damage == "random_head": cpu["complete_transfer"]["randomly_initialized_keys"] = ["head"]
    elif damage == "fp16_transfer": cpu["complete_transfer"]["all_floating_tensors_fp32"] = False
    elif damage == "wrong_tensor_geometry": cpu["samples"][0]["actual_shape_chw"] = [3, 384, 640]
    elif damage == "cpu_cuda": cpu["cuda_initialized"] = True
    elif damage == "cpu_test": cpu["final_test_used"] = True
    elif damage == "incomplete_baseline": baseline["rows"].pop()
    elif damage == "baseline_precision": baseline["full_frame_precision"] = .95
    elif damage == "prior_cpu_runtime": cpu["runtime_source_bindings"] = {"synthetic_runtime": "prior"}
    elif damage == "cpu_runtime_changed_after": cpu["runtime_source_bindings_after"] = {"synthetic_runtime": "changed"}
    elif damage == "cpu_final_runtime_changed": cpu["runtime_source_bindings_observed_finally"] = {"synthetic_runtime": "changed"}
    elif damage == "helper_code_changed": cpu["parameter_logic_bindings"] = {}
    elif damage == "baseline_runtime_changed": baseline["runtime_source_bindings"] = {"synthetic_runtime": "changed"}
    cpu_path, baseline_path = tmp_path / "synthetic_cpu.json", tmp_path / "synthetic_baseline.json"
    cpu_path.write_text(json.dumps(cpu), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    sources, runtime = {"synthetic_source": "bound"}, {"synthetic_runtime": "bound"}
    monkeypatch.setattr(train, "frozen_runtime_sources", lambda: (sources, runtime))
    approval = _release(plan, train.GPU_RELEASE)
    approval.update(cpu_preview=train.binding(cpu_path), fixed_native_baseline=train.binding(baseline_path),
        actual_preview_images_reviewed=damage != "visual_review_missing", source_bindings=sources, runtime_source_bindings=runtime)
    if damage == "unsettled_sources": approval["source_bindings"] = {}
    elif damage == "runtime_changed": approval["runtime_source_bindings"] = {"synthetic_runtime": "changed"}
    if damage is None:
        assert train.audit_gpu_release(approval, plan) is False
    else:
        with pytest.raises(ValueError):
            train.audit_gpu_release(approval, plan)


def test_unapproved_cpu_and_training_stages_do_not_create_output_or_import_torch(plan, tmp_path):
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="ROOT"):
        train.cpu_preview(plan, output, {})
    with pytest.raises(ValueError, match="ROOT"):
        train.execute_owned_child(plan, {})
    assert not output.exists()
    assert "torch" not in sys.modules


def test_failed_synthetic_cpu_import_records_actual_initialized_cuda_and_end_resources(plan, tmp_path, monkeypatch):
    # Exercise only the failure boundary using a fake module; no real Torch,
    # image decoding, transfer, preview or resource snapshot is performed.
    output = tmp_path / "synthetic_cpu_failure"
    resources = {"ram": 2 * 1024**3, "disk": 2 * 1024**3, "rss": 123, "peak_rss": 456, "vram": None}
    monkeypatch.setattr(train, "preflight", lambda _: plan)
    monkeypatch.setattr(train, "fresh_output", lambda path: Path(path).resolve())
    monkeypatch.setattr(train, "resource_snapshot", lambda *_: resources)
    monkeypatch.setattr(train, "frozen_runtime_sources", lambda: ({"synthetic_source": "bound"}, {"synthetic_runtime": "bound"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic_original")
    original = builtins.__import__
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    def guarded(name, *args, **kwargs):
        if name == "torch":
            monkeypatch.setitem(sys.modules, "torch", fake_torch)
            return fake_torch
        if name == "numpy":
            raise RuntimeError("synthetic numerical import failure")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="synthetic numerical import failure"):
        train.cpu_preview(plan, output, _release(plan, train.CPU_RELEASE))
    receipt = train.read(output / "receipt.json")
    assert receipt["status"] == "FAILED_CPU_PREVIEW_NO_AUTOMATIC_RETRY"
    assert receipt["cuda_initialized"] is True
    assert receipt["failure_reason"] == "synthetic numerical import failure"
    assert receipt["resources_at_end"] == resources and receipt["owned_peak_working_set_bytes"] == 456
    assert receipt["runtime_source_bindings_observed_finally"] == {"synthetic_runtime": "bound"}
    assert receipt["samples"] == [] and receipt["training_performed"] is False


@pytest.mark.parametrize("changed", ["input", "source", "runtime", "end_resource"])
def test_synthetic_diagnostic_import_failure_rechecks_actual_bound_bytes_and_preserves_primary_error(plan, tmp_path, monkeypatch, changed):
    # A deliberately failing synthetic import cannot reach inference. The tiny
    # temporary input verifies real hashing at both diagnostic boundaries.
    checkpoint = tmp_path / "synthetic_checkpoint.bin"
    checkpoint.write_bytes(b"before")
    export_dir = tmp_path / "synthetic_export"
    (export_dir / "ball").mkdir(parents=True)
    (export_dir / "export_receipt.json").write_text("{}", encoding="utf-8")
    (export_dir / "ball/data.yaml").write_text("synthetic: true", encoding="utf-8")
    protocol = deepcopy(plan["diagnostic_protocol"])
    output = tmp_path / "synthetic_diagnostic_failure"
    snapshot = {"source": "before", "runtime": "before", "resources": "before"}
    resources = {"ram": 5 * 1024**3, "disk": 2 * 1024**3, "rss": 123, "peak_rss": 456, "vram": None}
    monkeypatch.setattr(train, "fresh_output", lambda path: Path(path).resolve())
    monkeypatch.setattr(diagnostic, "validate_member", lambda *_: (checkpoint, {"checkpoint": train.binding(checkpoint)}))
    monkeypatch.setattr(train.exported, "OUTPUT", export_dir)
    monkeypatch.setattr(train.exported, "verify_export", lambda **_: {"manifest": {"contract": {"input_bindings": {}}}, "samples": []})
    monkeypatch.setattr(train, "frozen_runtime_sources", lambda: ({"synthetic_source": snapshot["source"]}, {"synthetic_runtime": snapshot["runtime"]}))
    def resource_snapshot(*_):
        if snapshot["resources"] != "before":
            raise RuntimeError("synthetic resource observation failure")
        return resources
    monkeypatch.setattr(train, "resource_snapshot", resource_snapshot)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "cv2":
            if changed == "input": checkpoint.write_bytes(b"after")
            elif changed in ("source", "runtime"): snapshot[changed] = "after"
            else: snapshot["resources"] = "after"
            raise RuntimeError("synthetic diagnostic import failure")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    approval = {"status": diagnostic.RELEASE, "reviewer_type": "ROOT_AGENT", "phase": "before",
        "protocol_sha256": train.exported.canonical_hash(protocol), "review_note": "Synthetic failure-boundary test only.",
        "qualification_evidence": False, "final_test_used": False, "public_release_cleared": False}
    with pytest.raises(RuntimeError, match="synthetic diagnostic import failure"):
        diagnostic.execute_member(protocol, "before", output, approval, "2026-10-16T00:00:00+00:00")
    receipt = train.read(output / "diagnostic.json")
    assert receipt["status"] == "FAILED_FIXED_MEMBER_PRESERVED_NO_AUTOMATIC_RETRY"
    assert receipt["failure_reason"] == "synthetic diagnostic import failure"
    assert receipt["end_integrity_failure"]
    assert receipt["cuda_initialized"] is None and receipt["rows"] == []
    assert receipt["forward_events"] == [] and receipt["actual_image_forward_count"] == 0
    if changed == "input": assert receipt["input_hashes_changed"] == [str(checkpoint)]
    elif changed == "source": assert receipt["full_source_integrity_changes"]["changed"] == ["synthetic_source"]
    elif changed == "runtime": assert receipt["runtime_integrity_changes"]["changed"] == ["synthetic_runtime"]
    else: assert receipt["end_resource_observation_error"]["type"] == "RuntimeError"


@pytest.mark.parametrize("changed", [None, "input", "working", "source", "runtime", "resources", "cuda_observation"])
def test_synthetic_training_terminal_observations_check_each_boundary_without_raising(tmp_path, monkeypatch, changed):
    # This directly tests a stdlib terminal observer with fake Torch state and
    # tiny files, without executing a model, optimizer, CUDA or training stage.
    source_file, working_file = tmp_path / "synthetic_input.bin", tmp_path / "synthetic_working.bin"
    source_file.write_bytes(b"before")
    working_file.write_bytes(b"before")
    fixed_inputs = {str(source_file): train.digest(source_file)}
    working = [train.binding(working_file)]
    output = tmp_path / "synthetic_training_terminal"
    checkpoints = output / "training/weights"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes, not a model")
    receipt = {"status": "FAILED_PRESERVED_NO_AUTOMATIC_RETRY", "failure_reason": "synthetic original failure",
        "source_bindings": {"synthetic_source": "before"}, "runtime_source_bindings": {"synthetic_runtime": "before"}}
    resources = {"ram": 123, "disk": 456, "rss": 789, "peak_rss": 987, "vram": None}
    monkeypatch.setattr(train, "frozen_runtime_sources", lambda: ({"synthetic_source": "after" if changed == "source" else "before"},
        {"synthetic_runtime": "after" if changed == "runtime" else "before"}))
    def resource_snapshot(*_):
        if changed == "resources": raise RuntimeError("synthetic resource error")
        return resources
    def cuda_state():
        if changed == "cuda_observation": raise RuntimeError("synthetic CUDA observation error")
        return True
    monkeypatch.setattr(train, "resource_snapshot", resource_snapshot)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=cuda_state)))
    if changed == "input": source_file.write_bytes(b"after")
    elif changed == "working": working_file.write_bytes(b"after")
    errors = train.observe_training_terminal(receipt, output, fixed_inputs, working)
    assert bool(errors) is (changed is not None)
    assert receipt["status"] == "FAILED_PRESERVED_NO_AUTOMATIC_RETRY" and receipt["failure_reason"] == "synthetic original failure"
    assert receipt["terminal_checkpoint_bindings"] == [train.binding(checkpoint)]
    if changed == "input": assert receipt["input_hashes_changed"] == [str(source_file)]
    elif changed == "working": assert receipt["input_hashes_changed"] == [str(working_file)]
    elif changed == "source": assert receipt["full_source_integrity_changes"]["changed"] == ["synthetic_source"]
    elif changed == "runtime": assert receipt["runtime_integrity_changes"]["changed"] == ["synthetic_runtime"]
    elif changed == "resources": assert receipt["end_resource_observation_error"]["type"] == "RuntimeError"
    elif changed == "cuda_observation": assert receipt["cuda_initialized"] is None
    else:
        assert receipt["cuda_initialized"] is True and receipt["owned_peak_working_set_bytes"] == 987
        assert receipt["resources_at_end"] == resources and receipt["input_hashes_changed"] == []


def _synthetic_continuation(plan, tmp_path):
    checkpoint = tmp_path / "synthetic_epoch2.pt"
    checkpoint.write_bytes(b"synthetic unfinished checkpoint, not a model")
    metadata_path = tmp_path / "synthetic_epoch2_metadata.json"
    sources, runtime = {"synthetic_source": "bound"}, {"synthetic_runtime": "bound"}
    prior = {"mode": train.MODE, "status": "STOPPED_AFTER_TWO_COMPLETED_EPOCHS_FOR_ROOT_REVIEW",
        "experiment_semantics_sha256": train.experiment_semantics(plan), "dataset": plan["dataset"],
        "source_bindings": sources, "runtime_source_bindings": runtime, "training_arguments": plan["training_arguments"],
        "qualification_evidence": False, "final_test_used": False, "production_promoted": False,
        "unstripped_checkpoints": [train.binding(checkpoint) | {"epoch": 2}]}
    metadata_path.write_text(json.dumps(prior), encoding="utf-8")
    approval = {"resume_checkpoint": train.binding(checkpoint), "initial_two_epoch_metadata": train.binding(metadata_path),
        "source_bindings": sources, "runtime_source_bindings": runtime}
    return checkpoint, metadata_path, prior, approval


@pytest.mark.parametrize("damage", [None, "checkpoint_bytes", "metadata_bytes", "missing_registration", "wrong_epoch",
    "completed_run", "other_dataset", "other_source", "other_runtime", "schedule", "qualification", "test", "promotion"])
def test_continuation_requires_exact_registered_unfinished_inputs_before_runtime(plan, tmp_path, damage):
    checkpoint, metadata_path, prior, approval = _synthetic_continuation(plan, tmp_path)
    if damage == "checkpoint_bytes": checkpoint.write_bytes(b"changed")
    elif damage == "metadata_bytes": metadata_path.write_text("{}", encoding="utf-8")
    elif damage is not None:
        if damage == "missing_registration": prior["unstripped_checkpoints"] = []
        elif damage == "wrong_epoch": prior["unstripped_checkpoints"][0]["epoch"] = 1
        elif damage == "completed_run": prior["status"] = "FINISHED_SAME_TWENTY_EPOCH_REPAIR_UNQUALIFIED"
        elif damage == "other_dataset": prior["dataset"] = {}
        elif damage == "other_source": prior["source_bindings"] = {}
        elif damage == "other_runtime": prior["runtime_source_bindings"] = {}
        elif damage == "schedule": prior["training_arguments"] = dict(prior["training_arguments"], epochs=21)
        elif damage == "qualification": prior["qualification_evidence"] = True
        elif damage == "test": prior["final_test_used"] = True
        elif damage == "promotion": prior["production_promoted"] = True
        metadata_path.write_text(json.dumps(prior), encoding="utf-8")
        approval["initial_two_epoch_metadata"] = train.binding(metadata_path)
    if damage is None:
        actual, bindings = train.continuation_inputs(approval, plan)
        assert actual == checkpoint
        assert bindings == {"checkpoint": train.binding(checkpoint), "metadata": train.binding(metadata_path)}
    else:
        with pytest.raises(ValueError):
            train.continuation_inputs(approval, plan)


@pytest.mark.parametrize("changed", [None, "checkpoint", "metadata"])
def test_synthetic_continuation_import_failure_retains_both_actual_input_pins(plan, tmp_path, monkeypatch, changed):
    # Only the stdlib admission/finally paths run. The intercepted Torch import
    # fails before any model, CUDA, numerical runtime or training is executed.
    synthetic_plan = deepcopy(plan)
    for group, keys in (("dataset", ("manifest", "export_receipt", "yaml")), ("initialization", ("checkpoint", "prior_metadata"))):
        for key in keys:
            tiny = tmp_path / (group + "_" + key + ".bin")
            tiny.write_bytes(b"synthetic source bytes")
            synthetic_plan[group][key] = train.binding(tiny)
    output = tmp_path / "synthetic_owned_continuation"
    synthetic_plan["output"] = str(output)
    checkpoint, metadata_path, _, approval = _synthetic_continuation(synthetic_plan, tmp_path)
    original_bindings = {str(checkpoint): train.digest(checkpoint), str(metadata_path): train.digest(metadata_path)}
    monkeypatch.setattr(train, "audit_gpu_release", lambda *_: True)
    monkeypatch.setattr(train, "preflight", lambda *_: synthetic_plan)
    monkeypatch.setattr(train, "fresh_output", lambda path: Path(path))
    monkeypatch.setattr(train, "audit_export", lambda: {"manifest": {"contract": {"input_bindings": {}}}, "samples": []})
    monkeypatch.setattr(train, "frozen_runtime_sources", lambda: (approval["source_bindings"], approval["runtime_source_bindings"]))
    monkeypatch.setattr(train, "resource_snapshot", lambda *_: {"ram": 4608 * 1024**2, "disk": 1024**3,
        "rss": 123, "peak_rss": 456, "vram": None})
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch":
            if changed == "checkpoint": checkpoint.write_bytes(b"changed during intercepted import")
            elif changed == "metadata": metadata_path.write_bytes(b"changed during intercepted import")
            raise RuntimeError("synthetic continuation import failure")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="synthetic continuation import failure"):
        train.execute_owned_child(synthetic_plan, approval)
    receipt = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED_PRESERVED_NO_AUTOMATIC_RETRY"
    assert receipt["failure_reason"] == "synthetic continuation import failure"
    assert receipt["resumed_from"] == {"checkpoint": approval["resume_checkpoint"], "metadata": approval["initial_two_epoch_metadata"]}
    for path, sha256 in original_bindings.items():
        assert receipt["fixed_input_bindings_before"][path] == sha256
        assert receipt["fixed_and_working_input_bindings_observed_finally"][path] == sha256
    assert receipt["input_hashes_changed"] == ([] if changed is None else [str(checkpoint if changed == "checkpoint" else metadata_path)])
    assert receipt["complete_transfer_receipts"] == [] and receipt["optimizer_update_receipts"] == []
    assert receipt["cuda_initialized"] is None


def test_import_tripwire_and_live_preflight_load_no_model_image_or_numerical_runtime():
    source = '''
import builtins,json,sys
original=builtins.__import__
forbidden={'torch','ultralytics','numpy','cv2','PIL'}
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in forbidden:raise RuntimeError('Forbidden runtime import:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from scripts.train import train_yolo26_author_ball_repair as train
p=train.preflight(train.ROOT/'outputs/yolo26_ball_author_repair_run1')
print(json.dumps({'mode':p['mode'],'forbidden':[n for n in sys.modules if n.split('.')[0] in forbidden]}))
'''
    result = subprocess.run([sys.executable, "-c", source], cwd=train.ROOT, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"mode": train.MODE, "forbidden": []}


def test_no_torch_imported_by_the_pure_suite():
    assert not any(key in sys.modules for key in ("torch", "ultralytics", "numpy", "cv2", "PIL"))

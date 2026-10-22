"""Pure admission/provenance tests; synthetic objects are never model evidence."""
from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train import train_yolo26_court_repair as training


HEAVY = {"torch", "ultralytics", "numpy", "cv2", "PIL", "psutil"}


def test_import_has_no_ml_or_image_runtime(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in HEAVY:
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("synthetic_court175_import", training.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TOTAL_EPOCHS == 20 and module.REVIEW_EPOCH == 2


def test_fixed_training_settings_do_not_extend_or_add_transforms(tmp_path):
    args = training.training_settings(tmp_path)
    assert {key: args[key] for key in ("epochs", "imgsz", "batch", "amp", "workers", "optimizer", "lr0", "nbs", "warmup_epochs")} == {
        "epochs": 20, "imgsz": 1280, "batch": 1, "amp": False, "workers": 0, "optimizer": "AdamW", "lr0": .0005, "nbs": 1, "warmup_epochs": 0.}
    assert args["augmentations"] == [] and args["auto_augment"] is None and args["scale"] == .05
    assert all(args[k] == 0 for k in ("mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "shear", "perspective", "fliplr", "flipud"))


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_gpu_preruntime_floor_preserves4point5gib(offset):
    resources = {"free_ram_bytes": 4608 * 1024**2 + offset, "free_disk_bytes": 1024**3}
    if offset < 0:
        with pytest.raises(ValueError, match="4.5GiB"):
            training.before_gpu_runtime_guard(resources)
    else:
        training.before_gpu_runtime_guard(resources)


@pytest.mark.parametrize("key", ["free_ram_bytes", "free_disk_bytes"])
def test_cpu_floor_not_relaxed(key):
    values = {"free_ram_bytes": 1536 * 1024**2, "free_disk_bytes": 1024**3}
    training.cpu_guard(values)
    values[key] -= 1
    with pytest.raises(ValueError):
        training.cpu_guard(values)


def test_fresh_output_rejects_existing_root_and_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    assert training.fresh_output(root / "new") == root / "new"
    for path in (root, tmp_path / "outside", root / "../outside"):
        with pytest.raises(ValueError):
            training.fresh_output(path)
    existing = root / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        training.fresh_output(existing)


def core_model():
    head = SimpleNamespace(kpt_shape=[14, 3], nc=1)
    return SimpleNamespace(yaml={"scale": "s", "nc": 1, "kpt_shape": [14, 3], "yaml_file": "yolo26s-court-pose.yaml"},
        model=[object(), head], end2end=True, names={0: "court"})


def test_model_accepts_core_and_actual_yolo_proxy_semantics():
    core = core_model()
    class Proxy:
        def __init__(self):
            self.model = core
        def __getattr__(self, key):
            return getattr(self.model, key)
    assert training.validate_model(core) is core
    assert training.validate_model(Proxy()) is core


@pytest.mark.parametrize(("key", "bad"), [("scale", "m"), ("nc", 2), ("kpt_shape", [17, 3]), ("yaml_file", "yolo26s-pose.yaml")])
def test_non_court_or_wrong_head_fails(key, bad):
    core = core_model()
    core.yaml[key] = bad
    with pytest.raises(ValueError, match="court14"):
        training.validate_model(core)


class Tensor:
    def __init__(self, values, shape=(2,)):
        self.values, self.shape = values, shape
    def detach(self):
        return self
    def cpu(self):
        return self


class State:
    def __init__(self, mapping):
        self.mapping = mapping
    def float(self):
        return self
    def state_dict(self):
        return self.mapping


def test_full_transfer_accepts_every_key_and_exact_fp32_value():
    original = State({"backbone": Tensor([1., 2.]), "court_head": Tensor([3., 4.])})
    target = State({"court_head": Tensor([3., 4.]), "backbone": Tensor([1., 2.])})
    torch = SimpleNamespace(equal=lambda a, b: a.values == b.values)
    receipt = training.transfer_receipt(original, target, torch)
    assert receipt["source_tensor_count"] == receipt["target_tensor_count"] == 2
    assert receipt["all_transferred_tensor_keys"] == ["backbone", "court_head"]
    assert receipt["randomly_initialized_keys"] == []


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "value"])
def test_partial_or_reinitialized_head_transfer_fails(change):
    original = State({"backbone": Tensor([1., 2.]), "court_head": Tensor([3., 4.])})
    target = deepcopy(original)
    if change == "missing":
        del target.mapping["court_head"]
    elif change == "extra":
        target.mapping["human_head"] = Tensor([1., 2.])
    elif change == "shape":
        target.mapping["court_head"].shape = (17,)
    else:
        target.mapping["court_head"].values = [0., 0.]
    with pytest.raises(ValueError, match="Incomplete trained-court transfer"):
        training.transfer_receipt(original, target, SimpleNamespace(equal=lambda a, b: a.values == b.values))


def valid_state():
    return {"epoch": 1, "optimizer": {"state": "synthetic"}, "ema": "synthetic", "updates": 298, "train_args": {"epochs": 20}}


def test_unstripped_state_has18_epochs_left_at_review2():
    assert training.validate_unfinished_state(valid_state()) == 2


@pytest.mark.parametrize(("key", "value"), [("epoch", -1), ("epoch", 19), ("epoch", True), ("optimizer", None),
    ("ema", None), ("updates", 0), ("updates", True), ("train_args", {"epochs": 21})])
def test_resume_rejects_finished_stripped_unregistered_total(key, value):
    state = valid_state()
    state[key] = value
    with pytest.raises(ValueError, match="unfinished"):
        training.validate_unfinished_state(state)


def test_resume_same_settings_ignores_only_new_working_output_paths(tmp_path):
    first = training.training_settings(tmp_path / "one")
    second = training.training_settings(tmp_path / "two")
    assert training.same_settings(first, second)
    for key, changed in (("epochs", 21), ("scale", .1), ("imgsz", 960), ("amp", True)):
        other = dict(second, **{key: changed})
        assert not training.same_settings(first, other)


def test_all26_protocol_keeps_unknowns_and_no_test(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "EXPORT", tmp_path)
    monkeypatch.setattr(training, "INITIAL", tmp_path / "weights.pt")
    (tmp_path / "weights.pt").write_bytes(b"synthetic")
    samples = []
    for n in range(26):
        (tmp_path / f"image{n}.png").write_bytes(b"not decoded")
        (tmp_path / f"label{n}.txt").write_bytes(b"not parsed")
        samples.append({"id": str(n), "split": "val", "context_group": "synthetic_dev", "visual_event_family": "unknown",
            "image": f"image{n}.png", "label": f"label{n}.txt", "bbox_xyxy_px": [0, 0, 100, 100], "keypoints_px": [None] * 14})
    protocol = training.diagnostic_protocol({"samples": samples})
    assert len(protocol["samples"]) == 26 and protocol["samples"][0]["keypoints_px"] == [None] * 14
    assert protocol["inference"]["iou"] == .7 and protocol["bbox_iou_minimum"] == .5 and protocol["inference"]["nms"] is False
    assert protocol["final_test_used"] is False and protocol["baseline_before_any_training_required"] is True
    with pytest.raises(ValueError, match="all26"):
        training.diagnostic_protocol({"samples": samples[:-1]})


def test_registered_evidence_cannot_escape_or_change_hash(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(training, "ROOT", root)
    evidence = root / "evidence.json"
    evidence.write_bytes(b"first")
    binding = {"path": str(evidence), "sha256": training.digest(evidence)}
    assert training.validate_binding(binding) == evidence
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError):
        training.validate_binding(binding)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError):
        training.validate_binding({"path": str(outside), "sha256": training.digest(outside)})


def test_learning_receipt_uses_trainable_head_parameters_not_bn_counters():
    class Parameter:
        def __init__(self, content, trainable=True):
            self.content, self.requires_grad = content, trainable
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def contiguous(self): return self
        def numpy(self): return SimpleNamespace(tobytes=lambda: self.content)
    core = SimpleNamespace(model=[object(), object()], named_parameters=lambda: iter([
        ("model.0.weight", Parameter(b"backbone")), ("model.1.weight", Parameter(b"head")),
        ("model.1.frozen", Parameter(b"frozen", False))]))
    assert set(training.head_fingerprints(core)) == {"model.1.weight"}


@pytest.fixture
def approval_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    protocol = {"samples": [{"id": str(n)} for n in range(26)], "synthetic": True}
    plan = {"initialization": {"synthetic": "exact_initialization"},
            "training_arguments": training.training_settings(tmp_path / "outputs/new"), "diagnostic_protocol": protocol}
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": training.digest(path)}
    diagnostic = {"status": "COMPLETED_ALL26_NATIVE_COURT175_FIXED_DIAGNOSTIC_NOT_QUALIFICATION",
        "manifest_sha256": training.MANIFEST_SHA, "qualification_evidence": False, "final_test_used": False,
        "plan": {"phase": "baseline", "protocol": protocol, "checkpoint": {"sha256": training.INITIAL_SHA}},
        "rows": [{"id": str(n)} for n in range(26)]}
    diagnostic_binding = save("diagnostic.json", diagnostic)
    frozen = save("protocol.json", {"protocol": protocol})
    cpu = {"status": "PASS_ACTUAL_CPU_TRAIN_TRANSFORMS_AND_COMPLETE_COURT14_TRANSFER_NOT_TRAINED",
        "cuda_initialized": False, "training_performed": False, "manifest_sha256": training.MANIFEST_SHA,
        "initialization": plan["initialization"], "training_arguments": plan["training_arguments"],
        "diagnostic_protocol_sha256": training.canonical_hash(protocol), "samples": [{"id": ident} for ident in training.PREVIEW_IDS],
        "complete_cpu_transfer": {"exact_fp32_values_before_optimizer": True}}
    baseline = {"status": "ROOT_REVIEWED_ACTUAL_ALL26_COURT175_BASELINE_NOT_QUALIFICATION",
        "manifest_sha256": training.MANIFEST_SHA, "checkpoint_sha256": training.INITIAL_SHA,
        "protocol_sha256": frozen["sha256"], "all26_dev_inputs_evaluated": True,
        "qualification_evidence": False, "diagnostic": diagnostic_binding}
    approval = {"status": training.GPU_APPROVAL, "reviewer": "ROOT_AGENT", "root_review_note": "synthetic boundary test",
        "plan_sha256": training.canonical_hash(plan), "manifest_sha256": training.MANIFEST_SHA, "total_epochs": 20,
        "qualification_evidence": False, "final_test_used": False,
        "cpu_inspection": save("cpu.json", cpu), "diagnostic_protocol": frozen, "baseline26_review": save("baseline.json", baseline)}
    return plan, approval, cpu, diagnostic, save


def test_gpu_release_requires_actual_cpu_and_complete26_baseline(approval_boundary):
    plan, approval, *_ = approval_boundary
    training.validate_gpu_approval(approval, plan)


@pytest.mark.parametrize(("key", "value"), [("reviewer", "SUBAGENT"), ("plan_sha256", "0" * 64),
    ("total_epochs", 21), ("qualification_evidence", True), ("root_review_note", "")])
def test_gpu_release_rejects_wrong_reviewer_plan_scope_or_total(approval_boundary, key, value):
    plan, approval, *_ = approval_boundary
    approval[key] = value
    with pytest.raises(ValueError, match="root approval"):
        training.validate_gpu_approval(approval, plan)


@pytest.mark.parametrize("change", ["cuda", "wrong_manifest", "missing_preview", "transfer"])
def test_gpu_release_rejects_incomplete_cpu_receipt(approval_boundary, change):
    plan, approval, cpu, _, save = approval_boundary
    if change == "cuda": cpu["cuda_initialized"] = True
    elif change == "wrong_manifest": cpu["manifest_sha256"] = "f" * 64
    elif change == "missing_preview": cpu["samples"].pop()
    else: cpu["complete_cpu_transfer"]["exact_fp32_values_before_optimizer"] = False
    approval["cpu_inspection"] = save("cpu_modified.json", cpu)
    with pytest.raises(ValueError, match="Actual guarded CPU"):
        training.validate_gpu_approval(approval, plan)


@pytest.mark.parametrize("change", ["drop", "order", "not_completed", "another_checkpoint"])
def test_gpu_release_rejects_plausible_but_incomplete_baseline(approval_boundary, change):
    plan, approval, _, diagnostic, save = approval_boundary
    if change == "drop": diagnostic["rows"].pop()
    elif change == "order": diagnostic["rows"].reverse()
    elif change == "not_completed": diagnostic["status"] = "PREFLIGHT_ONLY"
    else: diagnostic["plan"]["checkpoint"]["sha256"] = "a" * 64
    baseline = training.read_json(approval["baseline26_review"]["path"])
    baseline["diagnostic"] = save("diagnostic_modified.json", diagnostic)
    approval["baseline26_review"] = save("baseline_modified.json", baseline)
    with pytest.raises(ValueError, match="actual complete26"):
        training.validate_gpu_approval(approval, plan)


@pytest.fixture
def resume_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    checkpoint = tmp_path / "outputs/prior/training/weights/review_unstripped_epoch2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic registered state, never loaded")
    binding = {"path": str(checkpoint), "sha256": training.digest(checkpoint)}
    plan = {"source_bindings": {"synthetic_source": "a" * 64}, "runtime_source_bindings": {"synthetic_runtime": "b" * 64},
            "training_arguments": training.training_settings(tmp_path / "outputs/new")}
    metadata = {"mode": training.MODE, "role": "court", "manifest_sha256": training.MANIFEST_SHA,
        "runtime_versions": training.VERSIONS, "source_bindings": deepcopy(plan["source_bindings"]),
        "runtime_source_bindings": deepcopy(plan["runtime_source_bindings"]),
        "training_arguments": training.training_settings(tmp_path / "outputs/prior"),
        "qualification_evidence": False, "status": "STOPPED_AFTER2_FOR_ROOT_LEARNING_REVIEW_UNQUALIFIED",
        "unstripped_checkpoints": [binding]}
    metadata_path = checkpoint.parents[2] / "run_metadata.json"
    def save():
        metadata_path.write_text(json.dumps(metadata))
    save()
    return binding, plan, metadata, save


def test_registered_unfinished_resume_retains_same_data_settings_and_new_output(resume_boundary):
    binding, plan, metadata, _ = resume_boundary
    result = training.audit_resume(binding, plan)
    assert result["checkpoint"] == binding and result["prior"] == metadata


@pytest.mark.parametrize("change", ["manifest", "source_addition", "runtime_removal", "epochs", "unregistered", "finished", "qualification"])
def test_resume_rejects_data_implementation_extension_or_state_substitution(resume_boundary, change):
    binding, plan, metadata, save = resume_boundary
    if change == "manifest": metadata["manifest_sha256"] = "f" * 64
    elif change == "source_addition": metadata["source_bindings"]["added_source"] = "c" * 64
    elif change == "runtime_removal": metadata["runtime_source_bindings"] = {}
    elif change == "epochs": metadata["training_arguments"]["epochs"] = 21
    elif change == "unregistered": metadata["unstripped_checkpoints"] = []
    elif change == "finished": metadata["status"] = "FINISHED_SAME20_FINITE_REPAIR_UNQUALIFIED"
    else: metadata["qualification_evidence"] = True
    save()
    with pytest.raises(ValueError, match="Resume data/settings"):
        training.audit_resume(binding, plan)

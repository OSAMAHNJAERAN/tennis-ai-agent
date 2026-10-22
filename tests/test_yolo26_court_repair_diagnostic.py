"""Pure fixed26 diagnostic boundary and denominator checks, never model evidence."""
from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from scripts.evaluate import diagnose_yolo26_court_repair as diagnostic
from scripts.train import train_yolo26_court_repair as training


def test_import_does_not_load_image_torch_or_cuda(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "ultralytics", "numpy", "PIL", "cv2", "psutil"}:
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    spec = importlib.util.spec_from_file_location("synthetic_court175_diagnostic_import", diagnostic.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.COMPLETED.endswith("NOT_QUALIFICATION")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    export = tmp_path / "export"
    export.mkdir()
    monkeypatch.setattr(training, "EXPORT", export)
    checkpoint = tmp_path / "old_best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint, never loaded")
    monkeypatch.setattr(training, "INITIAL", checkpoint)
    monkeypatch.setattr(training, "INITIAL_SHA", training.digest(checkpoint))
    samples = []
    for n in range(26):
        (export / f"i{n}.png").write_bytes(b"synthetic image, never decoded")
        (export / f"l{n}.txt").write_bytes(b"synthetic label")
        samples.append({"id": str(n), "split": "val", "context_group": "synthetic", "visual_event_family": "unknown",
            "image": f"i{n}.png", "label": f"l{n}.txt", "bbox_xyxy_px": [0, 0, 100, 100], "keypoints_px": [[10, 20, 2]] * 13 + [None]})
    (export / "source_manifest.json").write_text(json.dumps({"samples": samples}))
    protocol = {"status": "FROZEN_BEFORE_ANY_COURT175_GPU_DIAGNOSTIC_OR_TRAINING", "protocol": training.diagnostic_protocol({"samples": samples})}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    calls = []
    monkeypatch.setattr(training, "verify_export", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(training, "audit_initialization", lambda: {"synthetic": True})
    monkeypatch.setattr(training, "source_bindings", lambda: {"synthetic_local_source": "a" * 64})
    monkeypatch.setattr(training, "runtime_bindings", lambda: {"synthetic_runtime": "b" * 64})
    return path, protocol, checkpoint, calls


def run_prepare(frozen):
    path, _, checkpoint, _ = frozen
    return diagnostic.prepare(path, training.digest(path), checkpoint, training.digest(checkpoint), phase="baseline")


def test_baseline_covers_exact26_order_and_requires_deep_export_check(frozen):
    plan = run_prepare(frozen)
    assert [row["id"] for row in plan["protocol"]["samples"]] == [str(n) for n in range(26)]
    assert len(frozen[3]) == 1 and frozen[3][0][1] == {"expected_manifest_sha256": training.MANIFEST_SHA}
    assert plan["phase"] == "baseline" and plan["training_metadata"] is None


@pytest.mark.parametrize("change", ["conf", "rect", "bbox_iou", "unknown", "drop_row", "order", "test_claim"])
def test_frozen26_protocol_rejects_threshold_geometry_reference_or_order_changes(frozen, change):
    path, protocol, _, _ = frozen
    if change == "conf":
        protocol["protocol"]["inference"]["conf"] = .1
    elif change == "rect":
        protocol["protocol"]["inference"]["rect"] = False
    elif change == "bbox_iou":
        protocol["protocol"]["bbox_iou_minimum"] = .4
    elif change == "unknown":
        protocol["protocol"]["samples"][0]["keypoints_px"][-1] = [0, 0, 2]
    elif change == "drop_row":
        protocol["protocol"]["samples"].pop()
    elif change == "order":
        protocol["protocol"]["samples"].reverse()
    else:
        protocol["protocol"]["final_test_used"] = True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="exact all26"):
        run_prepare(frozen)


def test_baseline_rejects_an_alternate_checkpoint_even_if_hash_known(frozen):
    path, _, checkpoint, _ = frozen
    alternate = checkpoint.with_name("another.pt")
    alternate.write_bytes(b"another model")
    with pytest.raises(ValueError, match="pinned prior court14"):
        diagnostic.prepare(path, training.digest(path), alternate, training.digest(alternate), phase="baseline")


def test_missing_instance_fails_all_visible_recall_but_excludes_unknowns():
    sample = {"bbox_xyxy_px": [0, 0, 100, 100], "keypoints_px": [[10, 20, 2]] * 13 + [None]}
    assessment = diagnostic.scoring.score_image(sample, [])
    summary = diagnostic.scoring.aggregate([{"context_group": "synthetic", "assessment": assessment}])
    points = summary["all_visible_keypoints"]
    assert points["visible_gt_denominator"] == 13 and points["unknown_gt_excluded"] == 1
    assert points["missing_visible_keypoints"] == 13 and points["recall_at_20_px"] == 0
    assert points["median_error_native_px"] is None


def test_highest_confidence_box_is_selected_before_iou_no_gt_oracle():
    sample = {"bbox_xyxy_px": [0, 0, 100, 100], "keypoints_px": [[10, 20, 2]] * 14}
    candidates = diagnostic.scoring.raw_candidates([[200, 200, 300, 300], [0, 0, 100, 100]], [.9, .8], [0, 0],
        [[[10, 20, .9]] * 14, [[10, 20, .9]] * 14])
    assessment = diagnostic.scoring.score_image(sample, candidates)
    assert assessment["selected_returned_index"] == 0 and assessment["complete_instance_miss"] is True
    assert all(point["error_native_px"] is None for point in assessment["points"])


def test_failed_preruntime_admission_never_imports_torch_and_retains_receipt(frozen, tmp_path, monkeypatch):
    plan = run_prepare(frozen)
    monkeypatch.setattr(training, "_resources", lambda *args, **kwargs: {"free_ram_bytes": 1, "free_disk_bytes": 2 * 1024**3})
    original = builtins.__import__
    attempted = []
    def guard(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "ultralytics"}:
            attempted.append(name)
            raise AssertionError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    output = tmp_path / "outputs/failed_admission"
    with pytest.raises(ValueError, match="4.5GiB"):
        diagnostic.execute(plan, output)
    assert attempted == []
    receipt = json.loads((output / "diagnostic.json").read_text())
    assert receipt["runtime_probe_performed"] is False
    assert receipt["status"] == "DIAGNOSTIC_NOT_STARTED_OR_FAILED_PRESERVED_NO_RETRY"
    assert receipt["rows"] == []

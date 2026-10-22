"""Reconstruct the current private101 plan using metadata and hashes only."""
import builtins
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FORBIDDEN = {"torch", "ultralytics", "numpy", "cv2", "PIL"}
original_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split(".")[0] in FORBIDDEN:
        raise RuntimeError("Pure preflight forbids runtime import: " + name)
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded
from scripts.train import train_yolo26_author_ball_repair as trainer

before = trainer.parameter_bindings()
started = datetime.now(timezone.utc).isoformat()
plan = trainer.preflight(ROOT / "outputs/yolo26_ball_author_repair_run1")
trainer.exported.write_new_json(OUTPUT / "plan.json", plan)
protocol_path = OUTPUT / "native20_protocol.json"
trainer.exported.write_new_json(protocol_path, {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": plan["diagnostic_protocol"]})
audit = trainer.audit_export()
selected = {row["sample_id"]: row for row in audit["samples"]}
preview_rows = []
for key in plan["cpu_preview_train_ids"]:
    row = selected[key]
    preview_rows.append({"sample_id": key, "split": row["development_split"],
        "inherited_from_old79": row["inherited_from_old79"],
        "negative_visible_absence_only": row["negative_visible_absence_only"],
        "box_xyxy_crop_px": row.get("box_xyxy_crop_px"),
        "image": trainer.binding(row["image_path"]), "label": trainer.binding(row["label_path"])})
after = trainer.parameter_bindings()
loaded = [name for name in sys.modules if name.split(".")[0] in FORBIDDEN]
receipt = {"status": "PASS_CURRENT_PRIVATE101_PURE_PREFLIGHT_PENDING_ROOT_CPU_REVIEW"
    if before == after and not loaded else "FAILED_PRESERVED",
    "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
    "plan": trainer.binding(OUTPUT / "plan.json"),
    "plan_canonical_sha256": trainer.exported.canonical_hash(plan),
    "fixed_native_protocol": trainer.binding(protocol_path),
    "protocol_canonical_sha256": trainer.exported.canonical_hash(plan["diagnostic_protocol"]),
    "dataset_counts": plan["dataset"]["counts"], "preview_selection_metadata": preview_rows,
    "parameter_bindings_before": before, "parameter_bindings_after": after,
    "forbidden_runtime_modules_loaded": loaded, "actual_cpu_transfer_executed": False,
    "actual_factory_images_generated": False, "actual_diagnostic_inference_executed": False,
    "actual_cuda_or_training_executed": False, "qualification_evidence": False, "final_test_used": False}
trainer.exported.write_new_json(OUTPUT / "receipt.json", receipt)
print(json.dumps({"status": receipt["status"], "plan": receipt["plan"],
    "protocol": receipt["fixed_native_protocol"], "preview_ids": plan["cpu_preview_train_ids"]}, indent=2))
raise SystemExit(0 if receipt["status"].startswith("PASS") else 1)

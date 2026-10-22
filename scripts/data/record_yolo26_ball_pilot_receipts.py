"""Bind the completed broadcast development runs to the pilot-only ledgers.

This reads receipts and writes metadata ledgers only; it imports no ML runtime,
opens no reserved frames, and never promotes a checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ledger_path = ROOT / "data/yolo26/pilot_training_receipts.json"
    ledger = read_json(ledger_path)
    diagnostic_path = ROOT / "outputs/yolo26_ball_native_crop_diagnostic_v1/diagnostic.json"
    diagnostic = read_json(diagnostic_path)
    run_relative = "outputs/yolo26_ball_native_crop_pilot_v1/run_metadata.json"
    run_path = ROOT / run_relative
    run = read_json(run_path)
    if run["status"] != "TRAINED_UNQUALIFIED":
        raise ValueError("Expected a completed unqualified development run")
    if run["data_audit"]["final_test_used"] or diagnostic["final_test_used"]:
        raise ValueError("Final-test use cannot be recorded as this development pilot")
    if diagnostic["qualification_evidence"]:
        raise ValueError("This diagnostic provides no independent qualification")
    if diagnostic["manifest_sha256"] != run["data_audit"]["manifest_sha256"]:
        raise ValueError("Diagnostic manifest does not match the trained manifest")
    best = next(item for item in run["checkpoints"] if Path(item["path"]).name == "best.pt")
    if best["sha256"] != diagnostic["checkpoint_sha256"]:
        raise ValueError("Diagnostic checkpoint does not match the trained best checkpoint")
    for item in run["checkpoints"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Checkpoint changed: {item['path']}")

    diagnostic_summary = {
        key: diagnostic[key]
        for key in ("status", "qualification_evidence", "final_test_used", "confidence", "iou_match", "aggregates", "limitations")
    }
    diagnostic_summary.update({
        "receipt_path": diagnostic_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": digest(diagnostic_path),
        "script_sha256": digest(ROOT / "scripts/data/diagnose_yolo26_native_crop_pilot.py"),
        "review_sheets": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "reviewer": "AGENT_VISUAL_REVIEW", "independent_human_review": False}
            for path in sorted(diagnostic_path.parent.glob("review_*.jpg"))
        ],
    })
    if len(diagnostic_summary["review_sheets"]) != 2 or len(diagnostic["rows"]) != 10:
        raise ValueError("Expected both reviewed sheets and all ten development samples")

    run.pop("transfer")  # Full tensor-level transfer evidence stays in the immutable run receipt.
    results_relative = "outputs/yolo26_ball_native_crop_pilot_v1/training/results.csv"
    results_path = ROOT / results_relative
    with results_path.open(encoding="utf-8", newline="") as stream:
        epoch_results = list(csv.DictReader(stream))
    if len(epoch_results) != 4:
        raise ValueError("Expected the bounded four-epoch receipt")
    run.update({
        "metadata_path": run_relative,
        "metadata_sha256": digest(run_path),
        "results_path": results_relative,
        "results_sha256": digest(results_path),
        "epoch_results": epoch_results,
        "qualification_evidence": False,
        "fixed_threshold_development_diagnostic": diagnostic_summary,
    })
    native_tile_summaries = []
    experiment_protocol = None
    for native_tile_version in (1, 2):
        native_tile_path = ROOT / f"outputs/yolo26_ball_native_tile_diagnostic_v{native_tile_version}/diagnostic.json"
        if not native_tile_path.is_file():
            continue
        native_tile = read_json(native_tile_path)
        protocol_path = native_tile_path.parent / "protocol.json"
        protocol = read_json(protocol_path)
        script_path = native_tile_path.parent / "diagnostic_script.py"
        if not script_path.is_file():
            script_path = ROOT / "scripts/data/diagnose_yolo26_native_tiles.py"
        if (native_tile["qualification_evidence"] or native_tile["final_test_used"]
                or native_tile["training_performed"] or native_tile["checkpoint_promoted"]
                or native_tile["checkpoint_sha256"] != best["sha256"]
                or native_tile["manifest_sha256"] != diagnostic["manifest_sha256"]
                or native_tile["confidence"] != .01 or native_tile["iou_match"] != .5
                or native_tile["protocol_sha256"] != digest(protocol_path)
                or native_tile["script_sha256"] != protocol["script_sha256"]
                or native_tile["script_sha256"] != digest(script_path)
                or len(protocol["samples"]) != 10
                or any(sample["source_id"] != "match182" for sample in protocol["samples"])):
            raise ValueError("Native tile diagnostic must preserve the fixed private development protocol")
        settings = {key: value for key, value in protocol.items() if key != "script_sha256"}
        if experiment_protocol is not None and settings != experiment_protocol:
            raise ValueError("Fresh native tile attempts cannot change frozen experimental settings or inputs")
        experiment_protocol = settings
        if native_tile["status"] not in ("RESOURCE_GUARD_BLOCKED_NOT_INFERRED",
                                          "FIXED_NATIVE_TILE_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION"):
            raise ValueError("Unexpected native tile diagnostic terminal status")
        native_tile_summary = {
            key: native_tile[key]
            for key in ("status", "qualification_evidence", "final_test_used", "training_performed",
                        "checkpoint_promoted", "private_local_only", "confidence", "iou_match",
                        "checkpoint_sha256", "manifest_sha256", "protocol_sha256", "script_sha256",
                        "resource_guard", "limitations")
        }
        native_tile_summary.update({
            "receipt_path": native_tile_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": digest(native_tile_path),
            "protocol_path": protocol_path.relative_to(ROOT).as_posix(),
            "script_path": script_path.relative_to(ROOT).as_posix(),
            "tiling": protocol["tiling"], "deduplication": protocol["deduplication"],
        })
        for key in ("resource_guard_floors_bytes", "guard_error", "model_constructed", "inference_performed",
                    "aggregates", "before_after", "actual_network_tensor_shapes", "resource_usage"):
            if key in native_tile:
                native_tile_summary[key] = native_tile[key]
        if native_tile["status"] == "RESOURCE_GUARD_BLOCKED_NOT_INFERRED" and (
                native_tile["model_constructed"] or native_tile["inference_performed"]):
            raise ValueError("A resource-blocked attempt must not construct a model or infer")
        native_tile_summaries.append(native_tile_summary)
    if native_tile_summaries:
        run["fixed_native_tile_development_diagnostic"] = native_tile_summaries[-1]
        run["fixed_native_tile_development_diagnostic_attempts"] = native_tile_summaries
    blocked_relative = "outputs/yolo26_ball_broadcast_pilot_v1/run_metadata.json"
    blocked_path = ROOT / blocked_relative
    blocked = read_json(blocked_path)
    if blocked["status"] != "RESOURCE_GUARD_BLOCKED_NOT_TRAINED" or "checkpoints" in blocked:
        raise ValueError("Expected the preserved pre-model resource rejection")
    blocked.update({"metadata_path": blocked_relative, "metadata_sha256": digest(blocked_path), "qualification_evidence": False})

    new_entries = {item["metadata_path"]: item for item in (blocked, run)}
    ledger["runs"] = [
        item for item in ledger["runs"]
        if item.get("metadata_path", "").replace("\\", "/") not in new_entries
    ] + list(new_entries.values())
    ledger["checked_on"] = "2026-10-09"
    ledger["qualification_evidence"] = False
    write_json(ledger_path, ledger)

    manifest_path = ROOT / "data/yolo26/source_manifest.json"
    manifest = read_json(manifest_path)
    entry = next(item for item in manifest["pilot_datasets"] if item["manifest"] == "data/yolo26/pilot_broadcast_native_crop_v1/source_manifest.json")
    entry.update({
        "training_receipt": run_relative,
        "training_receipt_sha256": run["metadata_sha256"],
        "training_status": run["status"],
        "completed_epochs": 4,
        "maximum_optimizer_state_step": 284,
        "fixed_threshold_development_diagnostic": diagnostic_summary,
    })
    if native_tile_summaries:
        entry["fixed_native_tile_development_diagnostic"] = native_tile_summaries[-1]
        entry["fixed_native_tile_development_diagnostic_attempts"] = native_tile_summaries
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "PILOT_RECEIPTS_RECORDED_NO_PROMOTION", "runs": len(ledger["runs"]), "diagnostic_sha256": digest(diagnostic_path), "run_metadata_sha256": digest(run_path)}, indent=2))


if __name__ == "__main__":
    main()

"""One private reviewed court training run, with immutable inputs and owned-process sampling.

This launcher imports no vision runtime. The training entrypoint performs its
own post-CUDA launch guard. Resource samples describe capacity, not accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/yolo26_court_reviewed_development_run1"
EXPORT = ROOT / "data/yolo26/court_reviewed_development_v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def input_paths():
    manifest = json.loads((EXPORT / "source_manifest.json").read_text(encoding="utf-8"))
    paths = set((ROOT / "src").rglob("*.py"))
    for relative in (
        "scripts/train/train_yolo26_tennis.py",
        "scripts/data/export_yolo26_reviewed_court.py",
        "scripts/data/prepare_yolo26_court_review.py",
        "data/yolo26/court_development_frozen_v1/grouped_development_manifest.json",
        "data/yolo26/court_review_decisions_v1.json",
        "artifacts/validation/yolo26_execution_2026-10-14/court_root_export_approval.json",
        "models/yolo26/yolo26s-pose.pt",
        "configs/production/yolo26.yaml",
        "configs/production/yolo26.artifacts.json",
    ):
        paths.add(ROOT / relative)
    paths.update(EXPORT.glob("*.json"))
    paths.add(EXPORT / "court/data.yaml")
    for sample in manifest["samples"]:
        paths.update(ROOT / sample[key] for key in ("image", "label"))
    for source in manifest["sources"]:
        paths.add(ROOT / source["source_path"])
    review = ROOT / "data/yolo26/court_review_expansion_v1"
    paths.update(review.glob("*.json"))
    detail = ROOT / "data/yolo26/court_native_detail_v1/native_detail_receipt.json"
    paths.add(detail)
    for board in json.loads(detail.read_text(encoding="utf-8"))["boards"]:
        paths.add(ROOT / board["image"])
    package = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    paths.update(package.rglob("*.py"))
    paths.update((package / "cfg").rglob("*.yaml"))
    paths.add(Path(__file__).resolve())
    return sorted(paths, key=str)


def snapshot(paths):
    return {str(path): {"sha256": digest(path), "bytes": path.stat().st_size} for path in paths}


def sample(pid):
    owned = []
    try:
        parent = psutil.Process(pid)
        for process in [parent, *parent.children(recursive=True)]:
            try:
                memory = process.memory_info()
                owned.append({"pid": process.pid, "rss_bytes": memory.rss,
                              "working_set_bytes": getattr(memory, "wset", memory.rss)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return {"utc": utc(), "available_ram_bytes": psutil.virtual_memory().available,
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "owned_processes": owned, "owned_tree_rss_bytes": sum(row["rss_bytes"] for row in owned)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    if ROOT != Path.cwd().resolve():
        raise ValueError("Run in the intended workspace")
    if OUTPUT.exists() or any((EVIDENCE / name).exists() for name in
                             ("execution_receipt.json", "sources_start.json", "console.log", "resources.jsonl")):
        raise ValueError("Retain prior attempt; this launcher is for one fresh run only")
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    trainer = ROOT / "scripts/train/train_yolo26_tennis.py"
    if (approval.get("status") != "ROOT_REVIEWED_COURT_DEVELOPMENT_INITIAL20_APPROVED"
            or approval.get("trainer_sha256") != digest(trainer)
            or approval.get("source_manifest_sha256") != digest(EXPORT / "source_manifest.json")
            or approval.get("output") != str(OUTPUT)
            or approval.get("qualification_evidence") is not False):
        raise ValueError("Require exact root-reviewed development release bindings")
    paths = input_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing pinned runtime inputs: {missing}")
    before = snapshot(paths)
    write_json(EVIDENCE / "sources_start.json", before)
    command = [sys.executable, "-u", "-m", "scripts.train.train_yolo26_tennis",
               "--role", "court", "--development", "--weights", "models/yolo26/yolo26s-pose.pt",
               "--epochs", "20", "--patience", "20", "--output", str(OUTPUT), "--execute"]
    receipt = {"schema": "tennis.yolo26.controlled-court-execution.v1",
               "status": "LAUNCHING_PRIVATE_UNQUALIFIED", "started_utc": utc(),
               "approval": {"path": str(args.approval.resolve()), "sha256": digest(args.approval)},
               "command": command, "output": str(OUTPUT), "runtime_input_count": len(before),
               "sources_start_sha256": digest(EVIDENCE / "sources_start.json"),
               "qualification_evidence": False, "final_test_used": False,
               "resource_guard": "TRAINER_POST_CUDA_LAUNCH_GUARD_UNCHANGED",
               "sampling": "2_SECOND_SYSTEM_AND_OWNED_PROCESS_TREE_OBSERVATIONS_NO_OTHER_APPS_STOPPED"}
    write_json(EVIDENCE / "execution_receipt.json", receipt)
    minimum_ram, maximum_rss, sample_count = None, 0, 0
    started = time.perf_counter()
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    with (EVIDENCE / "console.log").open("wb") as console, (EVIDENCE / "resources.jsonl").open("w", encoding="utf-8") as observations:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        receipt.update(status="RUNNING_PRIVATE_UNQUALIFIED", owned_child_pid=child.pid)
        write_json(EVIDENCE / "execution_receipt.json", receipt)
        while True:
            observation = sample(child.pid)
            observations.write(json.dumps(observation, allow_nan=False) + "\n")
            observations.flush()
            sample_count += 1
            minimum_ram = observation["available_ram_bytes"] if minimum_ram is None else min(minimum_ram, observation["available_ram_bytes"])
            maximum_rss = max(maximum_rss, observation["owned_tree_rss_bytes"])
            exit_code = child.poll()
            if exit_code is not None:
                break
            time.sleep(2)
    after = snapshot(paths)
    write_json(EVIDENCE / "sources_end.json", after)
    changed = [path for path in before if before[path] != after[path]]
    new_runtime_sources = set(input_paths()) - set(paths)
    metadata_path = OUTPUT / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    receipt.update(completed_utc=utc(), exit_code=exit_code, elapsed_seconds=time.perf_counter() - started,
                   sources_end_sha256=digest(EVIDENCE / "sources_end.json"), changed_runtime_inputs=changed,
                   added_runtime_sources=sorted(map(str, new_runtime_sources)),
                   source_freeze_verified=not changed and not new_runtime_sources,
                   sampled_minimum_available_ram_bytes=minimum_ram, sampled_peak_owned_tree_rss_bytes=maximum_rss,
                   resource_sample_count=sample_count, resource_samples_sha256=digest(EVIDENCE / "resources.jsonl"),
                   console_sha256=digest(EVIDENCE / "console.log"), trainer_status=metadata.get("status"),
                   metadata_sha256=digest(metadata_path) if metadata_path.is_file() else None)
    success = exit_code == 0 and metadata.get("status") == "TRAINED_UNQUALIFIED" and receipt["source_freeze_verified"]
    receipt["status"] = "COMPLETED_PRIVATE_TRAINED_UNQUALIFIED" if success else "TERMINAL_NOT_QUALIFIED_REVIEW_REQUIRED"
    write_json(EVIDENCE / "execution_receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "exit_code", "elapsed_seconds", "trainer_status",
                                                  "source_freeze_verified", "sampled_minimum_available_ram_bytes",
                                                  "sampled_peak_owned_tree_rss_bytes")}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

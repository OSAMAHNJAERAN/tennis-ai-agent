"""Small owned clean-env redirector regression; no models or heavy imports."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
OWN = Path(__file__).resolve().parent
MONITOR_SOURCE = ROOT / "scripts/audit/measure_candidate_storage.py"
CLEAN = ROOT / "artifacts/validation/product_installation/clean_env/Scripts/python.exe"
ALLOCATION_BYTES = 64 * 1024 * 1024
THRESHOLD_BYTES = 32 * 1024 * 1024
CHILD_CODE = r'''
import json, os, pathlib, sys, time
import psutil
directory = pathlib.Path(sys.argv[1])
process = psutil.Process()
buffer = bytearray(64 * 1024 * 1024)
buffer[::4096] = b"x" * (len(buffer) // 4096)
memory = process.memory_info()._asdict()
identity = {"pid": process.pid, "parentPid": process.ppid(), "createTimeUnixSeconds": process.create_time(),
            "allocatedBytes": len(buffer), "memoryBytesAfterAllocation": memory,
            "windowsPeakWorkingSetBytes": memory.get("peak_wset"),
            "windowsPeakPrivateCommitBytes": memory.get("peak_pagefile"),
            "heavyModulesImported": [name for name in ("torch", "numpy", "pandas", "cv2") if name in sys.modules]}
with (directory / "actual_child_identity.json").open("x", encoding="utf-8") as stream:
    json.dump(identity, stream, indent=2)
print(json.dumps(identity), flush=True)
time.sleep(30)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    directory = OWN / "redirector_monitor_regression"
    directory.mkdir(exist_ok=False)
    source_hash = digest(MONITOR_SOURCE)
    report = {"startedAt": stamp(), "passed": False, "allocationBytes": ALLOCATION_BYTES,
              "controlledAggregateRssThresholdBytes": THRESHOLD_BYTES,
              "sourceFile": str(MONITOR_SOURCE), "sourceSha256": source_hash,
              "helperSha256": digest(Path(__file__)), "cleanInterpreter": str(CLEAN),
              "scope": "One controlled 64MiB bytearray child; no models/inference/imports of numpy/torch/pandas/cv2. Proof that the actual clean-env descendant is sampled and stopped by exact identity, with wrapper retirement."}
    process = owned = monitor = None
    began = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("candidate_monitor_regression_target", MONITOR_SOURCE)
        monitor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = monitor
        spec.loader.exec_module(monitor)
        report["heavyModulesImportedByHelper"] = [name for name in ("torch", "numpy", "pandas", "cv2") if name in sys.modules]
        if report["heavyModulesImportedByHelper"]:
            raise ValueError("Regression unexpectedly imported a heavy library")
        with (directory / "child_stdout.txt").open("x", encoding="utf-8") as log, (directory / "resource_samples.jsonl").open("x", encoding="utf-8") as samples:
            argv = [str(CLEAN), "-c", CHILD_CODE, str(directory)]
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            report["redirectorIdentity"] = {"pid": owned.root_identity[0], "createTimeUnixSeconds": owned.root_identity[1]}
            while process.poll() is None and time.perf_counter() - began < 8:
                rows = owned.sample()
                sample = {"at": stamp(), "elapsedSeconds": time.perf_counter() - began,
                          "processes": rows, "ownedTreeRssBytes": sum(row["memoryBytes"]["rss"] for row in rows)}
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                identity_path = directory / "actual_child_identity.json"
                if identity_path.is_file():
                    # The child may still have its small JSON open; do not
                    # treat an incomplete concurrent write as a test retry.
                    try:
                        identity = json.loads(identity_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        identity = None
                    if identity is not None:
                        report["actualWorkloadIdentity"] = identity
                        actual = [row for row in rows if row["pid"] == identity["pid"] and row["createTimeUnixSeconds"] == identity["createTimeUnixSeconds"]]
                        if actual and actual[0]["memoryBytes"]["rss"] > THRESHOLD_BYTES:
                            report["triggeringSample"] = sample
                            report["actualWorkloadSampledByExactIdentity"] = owned.observed_workload(identity)
                            report["actualWorkloadRssBytesAtTrigger"] = actual[0]["memoryBytes"]["rss"]
                            report["distinctRedirectorAndActualInterpreter"] = identity["pid"] != process.pid
                            report["ownedTerminationActions"] = monitor.stop_owned(process, owned)
                            break
                time.sleep(0.05)
            else:
                raise RuntimeError("Controlled actual child was not sampled above the RSS threshold")
        report["ownedProcessIdentities"] = owned.identities()
        report["redirectorReturncode"] = process.poll()
        report["remainingOwnedIdentities"] = [row for row in owned.identities()
                                             if owned._matches(owned._owned[(row["pid"], row["createTimeUnixSeconds"])])]
        report["terminationDepths"] = [row["depth"] for row in report["ownedTerminationActions"]]
        report["deepestFirstTermination"] = report["terminationDepths"] == sorted(report["terminationDepths"], reverse=True)
        report["sourceUnchanged"] = digest(MONITOR_SOURCE) == source_hash
        child_identity = report["actualWorkloadIdentity"]
        report["passed"] = (report["actualWorkloadSampledByExactIdentity"]
                            and report["distinctRedirectorAndActualInterpreter"]
                            and report["actualWorkloadRssBytesAtTrigger"] > THRESHOLD_BYTES
                            and report["redirectorReturncode"] is not None
                            and not report["remainingOwnedIdentities"]
                            and report["deepestFirstTermination"]
                            and not child_identity["heavyModulesImported"]
                            and child_identity["allocatedBytes"] == ALLOCATION_BYTES
                            and report["sourceUnchanged"])
        if not report["passed"]:
            raise ValueError("Controlled redirector RSS/retirement conditions failed")
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if process is not None and monitor is not None:
            try:
                final_actions = monitor.stop_owned(process, owned)
                report["finalCleanupActions"] = final_actions
            except BaseException as cleanup_error:
                report["finalCleanupError"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
        report["completedAt"] = stamp()
        report["elapsedSeconds"] = time.perf_counter() - began
        with (directory / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

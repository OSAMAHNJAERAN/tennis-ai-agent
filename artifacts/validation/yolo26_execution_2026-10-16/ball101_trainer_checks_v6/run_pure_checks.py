"""Owned lightweight tests; never authorizes a model or image runtime."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
TEMP = OUTPUT / "temp"
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
tempfile.tempdir = str(TEMP)
sys.path.insert(0, str(ROOT))
FILES = ("scripts/train/train_yolo26_author_ball_repair.py",
         "scripts/evaluate/diagnose_yolo26_author_ball_repair.py",
         "tests/test_yolo26_author_ball_repair_training.py")
def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
before = hashes()
forbidden = {"torch", "ultralytics", "numpy", "cv2", "PIL"}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise RuntimeError("Pure suite forbids actual runtime import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pytest
class Results:
    def __init__(self): self.passed, self.failures = 0, []
    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed: self.passed += 1
        if report.failed: self.failures.append({"node": report.nodeid, "phase": report.when, "failure": str(report.longrepr)})
results = Results()
start = time.perf_counter()
code = pytest.main(["-q", "--noconftest", "--basetemp=" + str(TEMP / "pytest"),
    str(ROOT / FILES[-1])], plugins=[results])
after = hashes()
loaded = [name for name in sys.modules if name.split(".")[0] in forbidden]
receipt = {"status": "PASS" if code == 0 and before == after and not loaded else "FAILED_PRESERVED",
    "pytest_exit_code": int(code), "seconds": time.perf_counter() - start,
    "test_counts": {"passed": results.passed}, "failures": results.failures,
    "source_bindings_before": before, "source_bindings_after": after,
    "forbidden_runtime_modules_loaded": loaded,
    "actual_model_transfer_executed": False, "actual_image_preview_executed": False,
    "actual_diagnostic_inference_executed": False, "gpu_or_training_executed": False,
    "qualification_evidence": False,
    "synthetic_failure_boundary_tests": True}
(OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": receipt["status"], "passed": results.passed, "receipt": str(OUTPUT / "receipt.json")}))
raise SystemExit(0 if receipt["status"] == "PASS" else 1)

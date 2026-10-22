"""Retained pure actual-definition test execution; no video or model loading."""
from __future__ import annotations
import builtins
import contextlib
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
run_dir = Path(__file__).resolve().parent / sys.argv[1]
run_dir.mkdir(exist_ok=False)
temp_dir = run_dir / "temp"
temp_dir.mkdir()
tempfile.tempdir = str(temp_dir)
os.environ.update(TEMP=str(temp_dir), TMP=str(temp_dir), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
sources = [ROOT / name for name in (
    "src/line_calling/line_call_engine.py", "src/line_calling/line_geometry.py", "src/scoring/match_state.py", "src/pipeline/phase4_pipeline.py",
    "src/pipeline/phase5_pipeline.py", "tests/test_service_target_abstention.py", "src/scoring/point_timeline.py", "src/utils/point_analysis_sequence.py",
    "src/pipeline/phase6_pipeline.py", "src/utils/analytics_json_artifact.py", "tests/test_point_timeline.py", "tests/test_point_timeline_storage_integration.py", "tests/test_analytics_capacity_seam.py")]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path.relative_to(ROOT)): sha(path) for path in sources}
forbidden = {"torch", "ultralytics"}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise RuntimeError("Pure focused execution refuses runtime: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pytest
class Results:
    def __init__(self):
        self.passed = self.failed = self.skipped = self.errors = 0
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed: self.passed += 1
            elif report.failed: self.failed += 1
            elif report.skipped: self.skipped += 1
        elif report.failed:
            self.errors += 1
counts = Results()
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
with (run_dir / "pytest.txt").open("x", encoding="utf-8") as output:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = int(pytest.main(["--noconftest", "-q", "--tb=short", "-p", "no:cacheprovider",
            "--basetemp=" + str(temp_dir / "pytest"),
            *sys.argv[2:]], plugins=[counts]))
builtins.__import__ = original
after = {str(path.relative_to(ROOT)): sha(path) for path in sources}
class Memory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
        "PagefileUsage", "PeakPagefileUsage")]
memory = Memory()
memory.cb = ctypes.sizeof(memory)
get_process = ctypes.windll.kernel32.GetCurrentProcess
get_process.restype = ctypes.c_void_p
get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
get_memory.restype = ctypes.c_int
handle = get_process()
assert get_memory(handle, ctypes.byref(memory), memory.cb), "Process-memory observation failed"
assert memory.PeakWorkingSetSize > 0
receipt = dict(started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=time.perf_counter()-clock, exit_code=code, counts=vars(counts),
    peak_owned_working_set_bytes=memory.PeakWorkingSetSize, source_before=before,
    source_after=after, sources_unchanged=before == after,
    forbidden_imports_present=sorted(forbidden.intersection(sys.modules)),
    actual_class_definitions_with_synthetic_routing_fixtures=True, numpy_cv2_pandas_imported=[name for name in ("numpy","cv2","pandas") if name in sys.modules],
    models_used=False, media_used=False, gpu_used=False, qualification_evidence=False,
    runner_sha256=sha(Path(__file__)))
with (run_dir / "receipt.json").open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(receipt))
raise SystemExit(code)

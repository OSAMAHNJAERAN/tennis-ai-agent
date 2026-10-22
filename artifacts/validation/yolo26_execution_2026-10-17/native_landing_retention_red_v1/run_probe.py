"""Retain actual native landing storage/geometry RED; no model imports."""
import builtins
import contextlib
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent / sys.argv[1]
RUN.mkdir(exist_ok=False)
temporary = RUN / 'temp'
temporary.mkdir()
tempfile.tempdir = str(temporary)
os.environ.update(TEMP=str(temporary), TMP=str(temporary), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0, str(ROOT))
sources = ('src/pipeline/phase6_pipeline.py', 'src/visualization/landing_zones.py',
    'src/court/court_geometry.py', 'src/utils/json_array_artifact.py',
    'tests/test_native_landing_retention.py', 'tests/test_event_result_pipeline_seam.py')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {name: sha(ROOT / name) for name in sources}
for name in sources:
    snapshot = RUN / 'source_snapshot' / name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((ROOT / name).read_bytes())
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        raise RuntimeError('Landing storage check refuses model runtime ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import pytest
class Counts:
    passed = failed = errors = skipped = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            if report.passed: self.passed += 1
            elif report.failed: self.failed += 1
            elif report.skipped: self.skipped += 1
        elif report.failed:
            self.errors += 1
counts = Counts()
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
with (RUN / 'pytest.txt').open('x', encoding='utf-8') as output:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = int(pytest.main(['--noconftest', '-q', '--tb=short', '-p', 'no:cacheprovider',
            '--basetemp=' + str(temporary / 'pytest'), str(ROOT / 'tests/test_native_landing_retention.py'),
            *sys.argv[2:]], plugins=[counts]))
module = sys.modules.get('tests.test_native_landing_retention') or sys.modules.get('test_native_landing_retention')
rows = [] if module is None else module.OBSERVATIONS
after = {name: sha(ROOT / name) for name in sources}
class Memory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
memory = Memory()
memory.cb = ctypes.sizeof(memory)
handle = ctypes.windll.kernel32.GetCurrentProcess
handle.restype = ctypes.c_void_p
observe = ctypes.windll.psapi.GetProcessMemoryInfo
observe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
assert observe(handle(), ctypes.byref(memory), memory.cb)
receipt = dict(status='PASS' if code == 0 else 'RED_OR_FAILED', pytest_exit_code=code,
    started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
    seconds=time.perf_counter() - clock, counts={name: getattr(counts, name) for name in ('passed', 'failed', 'errors', 'skipped')},
    rows=rows, source_before=before, source_after=after, sources_unchanged=before == after,
    model_runtime_imports_observed=sorted({'torch', 'ultralytics'}.intersection(sys.modules)),
    numerical_runtime_imports_observed=sorted({'numpy', 'cv2', 'pandas', 'PIL'}.intersection(sys.modules)),
    numpy_version=importlib.metadata.version('numpy'), peak_owned_working_set_bytes=memory.PeakWorkingSetSize,
    video_used=False, gpu_used=False, synthetic_geometry_storage_only=True,
    qualification_evidence=False, runner_sha256=sha(Path(__file__)))
with (RUN / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({key: receipt[key] for key in ('status', 'pytest_exit_code', 'seconds', 'counts', 'sources_unchanged', 'peak_owned_working_set_bytes')}))
raise SystemExit(code)

"""Retained completed-inference checkpoint worker integration; no real inference."""
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
RUN = Path(__file__).resolve().parent / sys.argv[1]
RUN.mkdir(exist_ok=False)
SCOPE = sys.argv[2]
TESTS = ('tests/test_pipeline_checkpoint.py', 'tests/test_analysis_resume_identity.py', 'tests/test_worker_result_publication.py', 'tests/test_checkpoint_process_restart.py', 'tests/test_product_api.py', 'tests/test_result_document.py')

temporary = RUN / 'temp'
temporary.mkdir()
tempfile.tempdir = str(temporary)
os.environ.update(TEMP=str(temporary), TMP=str(temporary), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
os.environ['TENNIS_STORAGE_ROOT'] = str(temporary / 'app_bootstrap')
sys.path.insert(0, str(ROOT))
sources = (str(Path(__file__).relative_to(ROOT)), 'src/api/pipeline_checkpoint.py',
    'src/api/analysis_resume.py', 'src/api/worker.py', 'src/api/jobs.py',
    'src/api/result_storage.py', 'src/api/result_document.py', 'src/api/results.py',
    'src/api/media_timing.py', 'src/api/app.py', 'src/api/product.py', 'tests/helpers/checkpoint_worker_child.py', *TESTS)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {name: sha(ROOT / name) for name in sources}
for name in sources:
    snapshot = RUN / 'source_snapshot' / name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((ROOT / name).read_bytes())
forbidden = {'torch', 'ultralytics'}
if SCOPE in ('pure', 'native_exports'): forbidden |= {'numpy', 'cv2', 'pandas', 'PIL'}
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in forbidden:
        raise RuntimeError('Activity ' + SCOPE + ' check refuses runtime ' + name)
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
        elif report.failed: self.errors += 1
counts = Counts()
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
with (RUN / 'pytest.txt').open('x', encoding='utf-8') as output:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = int(pytest.main(['--noconftest', '-q', '--tb=short', '-p', 'no:cacheprovider',
            '--basetemp=' + str(temporary / 'pytest'), *[str(ROOT / name) for name in TESTS], *sys.argv[3:]], plugins=[counts]))
module_name = 'test_worker_result_publication'
module = sys.modules.get('tests.' + module_name) or sys.modules.get(module_name)
rows = [] if module is None else getattr(module, 'OBSERVATIONS', [])
capacity_rows = [] if module is None else getattr(module, 'CAPACITY_OBSERVATIONS', [])
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
receipt = dict(status='PASS' if code == 0 else 'FAIL', scope=SCOPE, pytest_exit_code=code,
    started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - clock,
    counts={name: getattr(counts, name) for name in ('passed', 'failed', 'errors', 'skipped')},
    rows=rows, capacity_rows=capacity_rows, source_before=before, source_after=after, sources_unchanged=before == after,
    forbidden_runtime_imports_observed=sorted(forbidden.intersection(sys.modules)),
    numerical_runtime_imports_observed=sorted({'numpy', 'cv2', 'pandas', 'PIL'}.intersection(sys.modules)),
    peak_owned_working_set_bytes=memory.PeakWorkingSetSize, video_used=False, gpu_used=False,
    synthetic_checkpoint_worker_only=True, qualification_evidence=False,
    default_app_bootstrap_confined=str(temporary / 'app_bootstrap'),
    runner_sha256=sha(Path(__file__)))
with (RUN / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({key: receipt[key] for key in ('status', 'pytest_exit_code', 'seconds', 'counts', 'sources_unchanged', 'peak_owned_working_set_bytes')}))
raise SystemExit(code)

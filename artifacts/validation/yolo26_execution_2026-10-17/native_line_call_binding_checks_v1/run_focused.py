"""Retain pure native line-call tests and exact source pins; no inference."""
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
TEMP = RUN / 'temp'
TEMP.mkdir()
tempfile.tempdir = str(TEMP)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'pandas', 'PIL'}
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        raise RuntimeError('Pure native line-call binding checks refuse ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded

SOURCE_NAMES = (
    'src/scoring/scoring_engine.py', 'src/scoring/point_timeline.py',
    'src/utils/scoring_journal.py', 'src/utils/point_analysis_sequence.py',
    'src/utils/mapped_sequence_view.py', 'src/scoring/event_history.py',
    'src/scoring/match_state.py', 'src/scoring/scoring_rules.py',
    'src/scoring/point_outcome_resolver.py', 'src/line_calling/line_call_engine.py',
    'src/line_calling/line_geometry.py', 'tests/test_scoring_service_phase.py',
    'tests/test_point_timeline.py', 'tests/test_native_line_call_binding.py',
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {name: sha(ROOT / name) for name in SOURCE_NAMES}
for name in SOURCE_NAMES:
    target = RUN / 'source_snapshot' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / name).read_bytes())

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
            '--basetemp=' + str(TEMP / 'pytest'), str(ROOT / 'tests/test_native_line_call_binding.py')], plugins=[counts]))
after = {name: sha(ROOT / name) for name in SOURCE_NAMES}
class Memory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
memory = Memory()
memory.cb = ctypes.sizeof(memory)
process = ctypes.windll.kernel32.GetCurrentProcess
process.restype = ctypes.c_void_p
memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
memory_info.restype = ctypes.c_int
assert memory_info(process(), ctypes.byref(memory), memory.cb)
receipt = dict(status='PASS' if code == 0 else 'FAILED', started_utc=started,
    ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - clock,
    pytest_exit_code=code,
    counts={key: getattr(counts, key) for key in ('passed', 'failed', 'errors', 'skipped')},
    source_before=before, source_after=after, sources_unchanged=before == after,
    peak_owned_working_set_bytes=memory.PeakWorkingSetSize,
    forbidden_imports_present=sorted(FORBIDDEN.intersection(sys.modules)),
    actual_rules_and_sqlite_owners=True, source_extraction_omits_transitive_package_imports=True,
    input_scope='EXPLICIT_SYNTHETIC_RULE_INPUTS_ONLY', models_used=False,
    media_used=False, gpu_used=False, qualification_evidence=False,
    runner_sha256=sha(Path(__file__)))
with (RUN / 'receipt.json').open('x', encoding='utf-8') as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
    output.write('\n')
print(json.dumps({key: value for key, value in receipt.items() if key not in ('source_before', 'source_after')}))
assert before == after and not receipt['forbidden_imports_present']
raise SystemExit(code)

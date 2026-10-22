"""Actual local ASGI/auth/saved-result integration; never decode media or load models."""
from __future__ import annotations
import contextlib
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ATTEMPT = BASE / (sys.argv[1] if len(sys.argv) > 1 else 'attempt1')
ATTEMPT.mkdir(parents=True, exist_ok=False)
TEMP = ATTEMPT/'temp'
TEMP.mkdir()
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TEMP=str(TEMP), TMP=str(TEMP),
                  TENNIS_STORAGE_ROOT=str(TEMP/'bootstrap'), TENNIS_REQUIRE_AUTH='1')
tempfile.tempdir = str(TEMP)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
FORBIDDEN = ('torch', 'ultralytics')
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('Model runtime forbidden in saved-result HTTP checks: '+fullname)
sys.meta_path.insert(0, NoModels())

FILES = [str(path.relative_to(ROOT)).replace('\\', '/') for path in sorted((ROOT/'src').rglob('*.py'))]
FILES += ['tests/test_point_timeline_http_contract.py', 'tests/test_point_timeline_web_contract.py']
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes():
    return {name: sha(ROOT/name) for name in FILES}
before = hashes()
start = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
code, error = 1, None
totals = {}
try:
    import pytest
    with (ATTEMPT/'pytest.txt').open('x', encoding='utf-8') as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = int(pytest.main(['-q', '--tb=short', '--noconftest', '-p', 'no:cacheprovider',
                '--basetemp='+str(TEMP/'pytest'), '--junitxml='+str(ATTEMPT/'junit.xml'),
                'tests/test_point_timeline_http_contract.py']))
    report = ET.parse(ATTEMPT/'junit.xml').getroot()
    totals = {key: sum(int(node.get(key, '0')) for node in report.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
except BaseException:
    error = traceback.format_exc()
    (ATTEMPT/'failure.txt').write_text(error, encoding='utf-8')

class Memory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
memory = Memory()
memory.cb = ctypes.sizeof(memory)
get_process = ctypes.windll.kernel32.GetCurrentProcess
get_process.restype = ctypes.c_void_p
get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
get_memory.restype = ctypes.c_int
memory_ok = bool(get_memory(get_process(), ctypes.byref(memory), memory.cb))
after = hashes()
outputs = {str(path.relative_to(ATTEMPT)).replace('\\', '/'): sha(path)
           for path in sorted(TEMP.rglob('http_observations/*')) if path.is_file()}
receipt = {'scope': 'ACTUAL_FASTAPI_TESTCLIENT_TWO_ACCOUNTS_REAL_UPLOAD_JOB_RESULTSTORAGE_PUBLICATION_HTTP_JSON_HTML_CSV_AND_RESTART',
    'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.perf_counter()-clock,
    'exit_code': code, **totals, 'error': error, 'source_sha256_before': before,
    'source_sha256_after': after, 'sources_unchanged': before == after,
    'source_changes': [name for name in before if before[name] != after[name]],
    'peak_owned_working_set_bytes': int(memory.PeakWorkingSetSize) if memory_ok else None,
    'memory_observation_succeeded': memory_ok,
    'forbidden_runtime_imports_observed': [name for name in FORBIDDEN if name in sys.modules],
    'other_numeric_runtime_imports_observed': [name for name in ('numpy', 'cv2', 'pandas', 'PIL') if name in sys.modules],
    'fixture_kind': 'SUPPLIED_SYNTHETIC_PHASE6_RULE_DECISIONS_AND_BYTE_IDENTITY; NOT_PLAYABLE_VIDEO',
    'media_decode_executed': False, 'worker_executed': False, 'model_inference_executed': False,
    'gpu_used': False, 'independent_accuracy_qualified': False,
    'retained_http_outputs_sha256': outputs, 'runner_sha256': sha(Path(__file__))}
(ATTEMPT/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items() if key not in ('source_sha256_before', 'source_sha256_after')}))
raise SystemExit(code)

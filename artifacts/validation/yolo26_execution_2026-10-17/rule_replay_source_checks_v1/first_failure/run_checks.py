"""Retained model-free rule replay integration verification, not qualification."""
import builtins
from contextlib import redirect_stdout, redirect_stderr
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
RUN = Path(__file__).resolve().parent
TEMP = RUN / 'temp'
TEMP.mkdir(exist_ok=True)
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                  TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(TEMP/'app_bootstrap'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
TESTS = ['test_rule_replay_source', 'test_rule_reproduction']
SOURCES = ['src/scoring/match_state_codec.py', 'src/scoring/match_state.py',
           'src/scoring/scoring_engine.py', 'src/scoring/scoring_rules.py',
           'src/scoring/point_outcome_resolver.py', 'src/scoring/event_history.py',
           'src/scoring/point_timeline.py', 'src/scoring/rule_replay.py',
           'src/scoring/rule_reproduction.py', 'src/scoring/rule_replay_source.py',
           'src/utils/point_analysis_sequence.py',
           'src/utils/scoring_journal.py', 'src/utils/json_array_artifact.py',
           'src/utils/analytics_json_artifact.py', 'src/utils/point_timeline_artifact.py',
           'src/shot_analysis/shot_types.py', 'src/line_calling/line_geometry.py',
           'src/line_calling/line_call_engine.py', 'src/api/result_storage.py',
           'src/api/analysis_resume.py', 'src/api/pipeline_checkpoint.py',
           'src/pipeline/phase6_pipeline.py'] + ['tests/'+name+'.py' for name in TESTS] + ['tests/test_point_timeline.py', 'tests/test_scoring_service_phase.py']
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'PIL', 'pandas'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    return {name: sha(ROOT/name) for name in SOURCES}

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (key, ctypes.c_ulonglong) for key in ('totalPhysical', 'availablePhysical',
        'totalPage', 'availablePage', 'totalVirtual', 'availableVirtual', 'extended')]

class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ('peakWorkingSet', 'workingSet', 'peakPaged',
        'paged', 'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]

def resources():
    status = Memory()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return dict(available_physical_bytes=status.availablePhysical,
                total_physical_bytes=status.totalPhysical)

before = snapshot()
for name in SOURCES:
    target = RUN/'source_snapshot'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/name, target)
assert not {name.split('.')[0] for name in sys.modules} & FORBIDDEN
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('Rule integration cannot load model/image runtime: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
started = datetime.now(timezone.utc).isoformat()
admission = resources()
begin = time.perf_counter()
import pytest
with (RUN/'pytest.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
    code = pytest.main(['-q', '-k', 'source_decoder_retains_exact_raw_states', '--noconftest', '-p', 'no:cacheprovider',
                        '--basetemp='+str(TEMP/'pytest'),
                        '--junitxml='+str(RUN/'junit.xml')] + ['tests/'+name+'.py' for name in TESTS] + ['tests/test_point_timeline.py', 'tests/test_scoring_service_phase.py'])
seconds = time.perf_counter()-begin
after = snapshot()
counts = dict(passed=0, failed=0, errors=0, skipped=0)
tree = ET.parse(RUN/'junit.xml')
for case in tree.iter('testcase'):
    outcome = ('failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None
               else 'skipped' if case.find('skipped') is not None else 'passed')
    counts[outcome] += 1
memory = ProcessMemory()
memory.cb = ctypes.sizeof(memory)
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
if not process_memory(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
    raise ctypes.WinError()
present = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
receipt = dict(status='PASS' if code == 0 and before == after and not attempts and not present else 'FAIL',
    scope='Shared strict original source decoder and unchanged rule reproduction; synthetic actual rule evidence only; no video/inference/accuracy/live qualification',
    pytest_exit_code=int(code), counts=counts, started_utc=started,
    ended_utc=datetime.now(timezone.utc).isoformat(), seconds=seconds, source_before=before,
    source_after=after, sources_unchanged=before == after, forbidden_import_attempts=attempts,
    forbidden_modules_present=present, peak_parent_working_set_bytes=memory.peakWorkingSet,
    host_before=admission, host_after=resources(), video_used=False, gpu_used=False,
    correction_publication_implemented=False, independent_accuracy_validated=False,
    deselected=None,
    runner_sha256=sha(Path(__file__)))
(RUN/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status', 'counts', 'seconds', 'sources_unchanged',
    'forbidden_import_attempts', 'forbidden_modules_present', 'peak_parent_working_set_bytes', 'host_after')}))
raise SystemExit(0 if receipt['status']=='PASS' else 1)

"""Retained isolated actual corrected shot and analytics integration checks."""
from contextlib import redirect_stderr, redirect_stdout
import builtins
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
os.chdir(ROOT)
TEMP = ART / 'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP),
    TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'),
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
TESTS = [
    'tests/test_corrected_shot_rebuild.py', 'tests/test_corrected_shot_sources.py',
    'tests/test_shot_classifier_lazy_import.py', 'tests/test_phase6_shot_analytics.py',
    'tests/test_shot_evidence_sequence.py', 'tests/test_point_timeline.py',
    'tests/test_point_timeline_revision_guards.py',
    'tests/test_point_timeline_storage_integration.py',
    'tests/test_native_line_call_binding.py', 'tests/test_corrected_rule_rebuild.py',
    'tests/test_correction_sources.py',
]
DESELECT = [
    'tests/test_corrected_shot_sources.py::test_adapter_duration_capacity_retains_no_python_frame_or_event_cache',
    'tests/test_shot_evidence_sequence.py::test_native_linker_keeps_live_event_and_shot_objects_constant_across_commit_batches',
    'tests/test_point_timeline.py::test_indexed_semantic_ledger_scales_completed_points_without_live_membership_arrays',
    'tests/test_corrected_rule_rebuild.py::test_combined_corrected_builder_keeps_buffers_bounded_as_source_duration_grows',
    'tests/test_correction_sources.py::test_spilled_sources_and_lazy_slices_keep_duration_out_of_retained_memory',
]
PATHS = sorted(set([*ROOT.joinpath('src').rglob('*.py'),
    *ROOT.joinpath('tests').rglob('*.py'), Path(__file__).resolve()]))
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PATHS}
before = pins()
for path in PATHS:
    destination = ART / 'sources' / path.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Model-free shot integration imported inference: ' + name)
    return original_import(name, *args, **kwargs)
started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART / 'pytest.txt').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
                '--basetemp', str(TEMP / 'pytest'), '--junitxml', str(ART / 'junit.xml'),
                *TESTS, *[item for test in DESELECT for item in ('--deselect', test)]])
finally:
    builtins.__import__ = original_import
from tests.test_corrected_shot_rebuild import OBSERVATIONS
after = pins()
suites = ET.parse(ART / 'junit.xml').getroot().findall('testsuite')
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ('peakWorkingSet', 'workingSet', 'peakPaged',
        'paged', 'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]
memory = ProcessMemory()
memory.cb = ctypes.sizeof(memory)
if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                               ctypes.byref(memory), memory.cb):
    raise ctypes.WinError()
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'selected_test_modules': TESTS, 'deselected_capacity_functions': DESELECT,
    'source_before': before, 'source_after': after,
    'source_stable': before == after, 'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'numerical_image_loaded': sorted({'numpy', 'cv2', 'pandas'} & set(sys.modules)),
    'parent_peak_working_set_bytes': memory.peakWorkingSet,
    'observations': OBSERVATIONS,
    'scope': 'ACTUAL NATIVE RULES/SHOTS/CLASSIFIER/POINT ANALYTICS AND TWO TOY VIDEO CROP DECODE CASES; SYNTHETIC MEASUREMENTS AND TRUSTED CALLBACK; NO VISION ACCURACY',
    'published': False}
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded', 'observations')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART / 'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

"""Retained isolated native corrected causal forecast integration checks."""
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
    'tests/test_corrected_forecasts.py', 'tests/test_forecast_sequence.py',
    'tests/test_prediction_point_index.py', 'tests/test_scoring_prediction.py',
    'tests/test_prediction_evaluation.py', 'tests/test_corrected_shot_rebuild.py',
    'tests/test_point_timeline.py', 'tests/test_product_matches.py',
    'tests/test_product_api.py',
    'tests/test_point_timeline_revision_guards.py',
    'tests/test_point_timeline_storage_integration.py',
    'tests/test_corrected_rule_rebuild.py',
]
DESELECT = [
    'tests/test_corrected_forecasts.py::test_growing_completed_points_bounds_forecast_history_and_deduplication_in_python',
    'tests/test_forecast_sequence.py::test_growing_history_and_point_cardinality_keeps_owned_python_memory_bounded',
    'tests/test_corrected_shot_rebuild.py::test_corrected_shot_pipeline_keeps_owned_buffers_bounded_as_frames_and_shots_grow',
    'tests/test_point_timeline.py::test_indexed_semantic_ledger_scales_completed_points_without_live_membership_arrays',
    'tests/test_corrected_rule_rebuild.py::test_combined_corrected_builder_keeps_buffers_bounded_as_source_duration_grows',
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
from tests.test_corrected_forecasts import OBSERVATIONS
from tests.test_corrected_shot_rebuild import OBSERVATIONS as SHOT_OBSERVATIONS
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
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
memory_error = None
try:
    if not process_memory(ctypes.windll.kernel32.GetCurrentProcess(),
                          ctypes.byref(memory), memory.cb):
        raise ctypes.WinError()
except OSError as error:
    memory_error = str(error)
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'selected_test_modules': TESTS, 'deselected_capacity_functions': DESELECT,
    'source_before': before, 'source_after': after,
    'source_stable': before == after, 'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'numerical_image_loaded': sorted({'numpy', 'cv2', 'pandas'} & set(sys.modules)),
    'parent_peak_working_set_bytes': memory.peakWorkingSet if memory_error is None else None,
    'optional_memory_observation_error': memory_error,
    'observations': OBSERVATIONS,
    'shot_crop_observations': SHOT_OBSERVATIONS,
    'separate_current_capacity_receipt': '../corrected_forecast_final_v1/receipt.json',
    'previous_attempt': 'final_v1: 344 passed including current completed-point capacity; one old profile fixture omitted required summary.id and was correctly excluded. Only that fixture and missing/wrong-ID regressions changed before this run.',
    'scope': 'ACTUAL NATIVE CORRECTED RULE/SHOT/FORECAST CHAIN, BOUNDED SQL POINT INDEX/HISTORY, PREFIX CAUSALITY, RAW SCORE/TERMINAL CONTEXT AND PRIVATE ATOMIC EXPORT; SYNTHETIC OBSERVATIONS AND TWO TOY CROP VIDEOS; NO INDEPENDENT PREDICTION/VISION ACCURACY',
    'published': False}
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded', 'observations')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART / 'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

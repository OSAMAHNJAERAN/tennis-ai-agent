"""Retained complete private corrected result/report and surrounding regressions."""
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
TEMP = ART/'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
TESTS = [
    'tests/test_corrected_preparation.py', 'tests/test_corrected_forecast_presentation.py',
    'tests/test_corrected_rendering.py', 'tests/test_corrected_render_capacity.py',
    'tests/test_corrected_exports.py', 'tests/test_corrected_landing_exports.py',
    'tests/test_landing_zone_sequence.py', 'tests/test_landing_zone_contract.py',
    'tests/test_bounded_landing_reader.py', 'tests/test_rule_reproduction.py',
    'tests/test_corrected_forecasts.py', 'tests/test_forecast_sequence.py',
    'tests/test_prediction_point_index.py', 'tests/test_scoring_prediction.py',
    'tests/test_prediction_evaluation.py', 'tests/test_corrected_shot_rebuild.py',
    'tests/test_point_timeline.py', 'tests/test_product_matches.py', 'tests/test_product_api.py',
    'tests/test_point_timeline_revision_guards.py', 'tests/test_point_timeline_storage_integration.py',
    'tests/test_corrected_rule_rebuild.py', 'tests/test_bounded_reporting.py',
    'tests/test_streaming_telemetry_csv.py', 'tests/test_bounded_result_contract.py',
    'tests/test_result_builder_reads.py', 'tests/test_result_storage.py',
    'tests/test_result_document.py', 'tests/test_result_contract.py',
    'tests/test_result_landing_zones.py', 'tests/test_analysis_generation.py',
    'tests/test_generation_http_contract.py', 'tests/test_generation_commit_contract.py',
    'tests/test_worker_result_publication.py',
]
DESELECT = [
    'tests/test_rule_reproduction.py::test_source_frame_reproduction_keeps_traced_python_memory_bounded',
    'tests/test_bounded_landing_reader.py::test_bounded_reader_retention_does_not_scale_with_all_calibrations_and_event_ids',
    'tests/test_corrected_forecasts.py::test_growing_completed_points_bounds_forecast_history_and_deduplication_in_python',
    'tests/test_forecast_sequence.py::test_growing_history_and_point_cardinality_keeps_owned_python_memory_bounded',
    'tests/test_corrected_shot_rebuild.py::test_corrected_shot_pipeline_keeps_owned_buffers_bounded_as_frames_and_shots_grow',
    'tests/test_point_timeline.py::test_indexed_semantic_ledger_scales_completed_points_without_live_membership_arrays',
    'tests/test_corrected_rule_rebuild.py::test_combined_corrected_builder_keeps_buffers_bounded_as_source_duration_grows',
    'tests/test_result_storage.py::test_configured_432000_frame_series_streams_without_full_materialization',
]
PATHS = sorted(set([*ROOT.joinpath('src').rglob('*.py'),
    *ROOT.joinpath('tests').rglob('*.py'), Path(__file__).resolve()]))
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS}
before = pins()
for path in PATHS:
    target = ART/'sources'/path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Private corrected result imported inference: '+name)
    return native_import(name, *args, **kwargs)
started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
            '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'), *TESTS,
            *[argument for name in DESELECT for argument in ('--deselect', name)]])
finally:
    builtins.__import__ = native_import
from tests.test_corrected_preparation import OBSERVATIONS as PREPARATION
from tests.test_corrected_forecast_presentation import OBSERVATIONS as PRESENTATION
from tests.test_corrected_exports import OBSERVATIONS as EXPORTS
from tests.test_corrected_render_capacity import OBSERVATIONS as RENDER_CAPACITY
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
          for key in ('tests', 'errors', 'failures', 'skipped')}
class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)]+[
        (key, ctypes.c_size_t) for key in ('peakWorkingSet', 'workingSet', 'peakPaged', 'paged',
        'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]
memory = ProcessMemory()
memory.cb = ctypes.sizeof(memory)
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
read_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
memory_error = None
try:
    if not read_memory(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
        raise ctypes.WinError()
except OSError as error:
    memory_error = str(error)
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'selected_test_modules': TESTS, 'deselected_previously_measured_capacity': DESELECT,
    'source_before': before, 'source_after': after, 'source_stable': before == after,
    'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'numerical_image_loaded': sorted({'numpy', 'cv2', 'pandas'} & set(sys.modules)),
    'parent_peak_working_set_bytes': memory.peakWorkingSet if memory_error is None else None,
    'optional_memory_observation_error': memory_error,
    'preparation_observations': PREPARATION, 'forecast_presentation_observations': PRESENTATION,
    'export_observations': EXPORTS, 'render_capacity_observations': RENDER_CAPACITY,
    'scope': 'ACTUAL COMPLETE PRIVATE CORRECTION RESULT/CAUSAL FORECAST/STREAMED REPORT/VIDEO CHAIN WITH SYNTHETIC SOURCE OBSERVATIONS; SOURCE-PINNED API/GENERATION/DEFAULT RESULT/REPORT REGRESSIONS; NO AUTHENTICATED CORRECTION PUBLICATION OR MODEL/INDEPENDENT ACCURACY/LONG-MATCH/LIVE QUALIFICATION',
    'published': False}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded',
    'preparation_observations', 'forecast_presentation_observations')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART/'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

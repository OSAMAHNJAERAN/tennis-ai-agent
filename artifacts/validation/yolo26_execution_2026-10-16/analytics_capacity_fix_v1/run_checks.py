"""Focused metadata event regression runner; no model or GPU is invoked."""
import contextlib
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import pytest

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/analytics_capacity_checks_v1'
OUT.mkdir(parents=True, exist_ok=False)
local_temp = OUT / 'temp'
local_temp.mkdir()
tempfile.tempdir = str(local_temp)
os.environ.update(TEMP=str(local_temp), TMP=str(local_temp), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
paths = [ROOT / value for value in ('src/utils/analytics_json_artifact.py', 'src/analytics/rally_analyzer.py', 'src/analytics/shot_statistics.py', 'src/analytics/serve_analyzer.py', 'src/analytics/match_analytics.py', 'tests/test_analytics_capacity_seam.py', 'src/utils/event_analysis_sequence.py', 'src/events/event_detector.py',
         'src/pipeline/phase6_pipeline.py', 'tests/test_event_analysis_sequence.py',
         'tests/test_event_result_pipeline_seam.py', 'src/utils/shot_evidence_sequence.py', 'src/shot_analysis/shot_linker.py', 'tests/test_shot_evidence_sequence.py', 'artifacts/validation/yolo26_execution_2026-10-16/shot_result_capacity_probe_v1/prechange_shot_linker.py', 'src/shot_analysis/shot_classifier.py', 'src/shot_analysis/shot_direction.py', 'src/shot_analysis/court_zones.py', 'src/utils/identity_frame_sequence.py',
         'src/visualization/landing_zones.py', 'src/utils/json_array_artifact.py')]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                               'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    status = Counters()
    status.cb = ctypes.sizeof(status)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(status), status.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'owned_rss_bytes': status.rss, 'peak_owned_rss_bytes': status.peak_rss}

class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def __getattr__(self, name):
        return getattr(self.stream, name)
    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.file.flush()
        return len(value)
    def flush(self):
        self.stream.flush()
        self.file.flush()

before = {str(path): digest(path) for path in paths}
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
tests = ['tests/test_analytics_capacity_seam.py', 'tests/test_json_array_artifact.py', 'tests/test_analytics.py', 'tests/test_temporal_ball_source_clock.py', 'tests/test_shot_evidence_sequence.py', 'tests/test_event_result_pipeline_seam.py', 'tests/test_event_analysis_sequence.py', 'tests/test_event_feature_storage.py',
         'tests/test_event_feature_ranking.py', 'tests/test_event_lazy_views.py',
         'tests/test_event_departure_lookup.py', 'tests/test_event_timestamp_lookup.py',
         'tests/test_phase6_4_event_pipeline_consistency.py',
         'tests/test_phase6_4_semantic_event_recovery.py']
arguments = ['--noconftest', '-q', *tests, '--basetemp', str(OUT / 'pytest_tmp'), '--junitxml', str(OUT / 'junit.xml')]
initial = memory()
with (OUT / 'console.txt').open('w', encoding='utf-8', newline='\n') as console:
    with contextlib.redirect_stdout(Tee(sys.stdout, console)), contextlib.redirect_stderr(Tee(sys.stderr, console)):
        result = pytest.main(arguments)
after = {str(path): digest(path) for path in paths}
suites = ET.parse(OUT / 'junit.xml').getroot().findall('testsuite')
receipt = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
    'elapsed_seconds': time.perf_counter() - clock, 'pytest_exit_code': int(result),
    'junit_counts': {key: sum(int(suite.get(key, '0')) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')},
    'python_executable': sys.executable, 'python_version': sys.version, 'pytest_version': pytest.__version__,
    'arguments': arguments, 'source_bindings_before': before, 'source_bindings_after': after,
    'source_files_changed': [path for path in before if before[path] != after[path]],
    'resources_before': initial, 'resources_after': memory(), 'torch_imported': 'torch' in sys.modules,
    'ultralytics_imported': 'ultralytics' in sys.modules, 'model_or_media_used': False,
    'training_performed': False, 'qualification_evidence': False, 'full_match_capacity_proven': False,
    'console_sha256': digest(OUT / 'console.txt'), 'junit_sha256': digest(OUT / 'junit.xml')}
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'receipt': str(OUT / 'receipt.json'), 'counts': receipt['junit_counts'],
                  'resources': receipt['resources_after']}))
raise SystemExit(result)








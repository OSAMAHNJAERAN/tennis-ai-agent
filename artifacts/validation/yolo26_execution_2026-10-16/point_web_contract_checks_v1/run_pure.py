"""Reproducible pure artifact/API-adapter/report checks; no media inference."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ATTEMPT = BASE / (sys.argv[1] if len(sys.argv) > 1 else 'attempt1')
ATTEMPT.mkdir(parents=True, exist_ok=False)
TEMP = ATTEMPT / 'temp'
TEMP.mkdir()
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', TEMP=str(TEMP), TMP=str(TEMP))
tempfile.tempdir = str(TEMP)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

FORBIDDEN = ('torch', 'ultralytics', 'numpy', 'cv2', 'pandas', 'PIL')
class NoHeavyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('Forbidden runtime import in pure contract checks: '+fullname)
sys.meta_path.insert(0, NoHeavyRuntime())
FILES = ('src/api/point_timeline_contract.py', 'src/utils/point_timeline_artifact.py',
         'src/api/results.py', 'src/api/reporting.py', 'tests/test_point_timeline_web_contract.py')
def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}

before = hashes()
start = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
import pytest
code = pytest.main(['-q', '--tb=short', '--noconftest', '--basetemp='+str(TEMP/'pytest'),
                    '--junitxml='+str(ATTEMPT/'junit.xml'), 'tests/test_point_timeline_web_contract.py'])
report = ET.parse(ATTEMPT/'junit.xml').getroot()
totals = {key: sum(int(node.get(key, '0')) for node in report.iter('testsuite'))
          for key in ('tests', 'failures', 'errors', 'skipped')}
receipt = {'scope': 'ACTUAL_SAVED_SYNTHETIC_PHASE6_BUNDLE_TO_NATIVE_WEB_CONTRACT_AND_HTML; NOT_VIDEO_OR_CV_ACCURACY',
    'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
    'wall_seconds': time.perf_counter()-clock, 'exit_code': code, **totals,
    'source_sha256_before': before, 'source_sha256_after': hashes(),
    'forbidden_runtime_imports_observed': [name for name in FORBIDDEN if name in sys.modules],
    'adapter_boundary': 'CURRENT_WHOLE_RESULTS_SOURCE_WITH_ONLY_POSE_CONSTANT_IMPORT_REPLACED_BY_ITS_EXACT_LITERAL_SOURCE; NO_NORMALIZATION_LOGIC_MOCKED',
    'http_media_inference_executed': False}
if code == 0:
    from tests.test_point_timeline_web_contract import SAVED, fixture_bundle, load
    fixture = fixture_bundle(ATTEMPT/'synthetic_native_bundle')
    value = load(fixture)
    (ATTEMPT/'pointAnalysis.json').write_text(json.dumps(value, allow_nan=False, indent=2), encoding='utf-8')
    (ATTEMPT/'fixture_description.json').write_text(json.dumps({
        'description': 'SYNTHETIC exact Phase6 rule fixtures, supplied decisions and source identity bytes; not decoded video and not independently qualified vision results.',
        'source_fixture': str((SAVED/'test_exact_native_phase6_scori0').relative_to(ROOT)).replace('\\', '/'),
        'point_analysis_sha256': hashlib.sha256((ATTEMPT/'pointAnalysis.json').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
(ATTEMPT/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
raise SystemExit(code)

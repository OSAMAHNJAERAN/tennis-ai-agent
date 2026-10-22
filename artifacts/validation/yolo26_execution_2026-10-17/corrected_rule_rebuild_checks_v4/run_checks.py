"""Retained isolated runner for the private correction integration contract."""
from contextlib import redirect_stderr, redirect_stdout
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
PATHS = sorted(set([
    *ROOT.joinpath('src/scoring').glob('*.py'),
    *ROOT.joinpath('src/line_calling').glob('*.py'),
    ROOT / 'src/api/analysis_generation.py', ROOT / 'src/api/result_storage.py',
    ROOT / 'src/utils/point_analysis_sequence.py', ROOT / 'src/utils/scoring_journal.py',
    ROOT / 'src/utils/mapped_sequence_view.py', ROOT / 'src/utils/json_array_artifact.py',
    ROOT / 'src/tracking/temporal_ball_tracker.py', ROOT / 'src/court/homography.py',
    ROOT / 'src/utils/bbox_utils.py', ROOT / 'tests/test_corrected_rule_rebuild.py',
    ROOT / 'tests/test_correction_sources.py', Path(__file__).resolve(),
]))


def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PATHS}


before = pins()
for path in PATHS:
    destination = ART / 'sources' / path.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
started = time.perf_counter()
import pytest
with (ART / 'pytest.txt').open('w', encoding='utf-8') as log:
    with redirect_stdout(log), redirect_stderr(log):
        code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
            '--basetemp', str(TEMP / 'pytest'), '--junitxml', str(ART / 'junit.xml'),
            'tests/test_corrected_rule_rebuild.py'])
from tests.test_corrected_rule_rebuild import OBSERVATIONS
after = pins()
suites = ET.parse(ART / 'junit.xml').getroot().findall('testsuite')
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
receipt = {
    'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'source_before': before, 'source_after': after,
    'source_stable': before == after,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'numerical_image_loaded': sorted({'numpy', 'cv2', 'pandas', 'PIL'} & set(sys.modules)),
    'observations': OBSERVATIONS,
    'scope': 'REAL RULES, CONTACT REFINER AND EVALUATOR WITH EXPLICIT SYNTHETIC SEALED OBSERVATIONS; NO VISION ACCURACY',
    'prior_attempts': {
        'v1': '30 passed; predates extra regressions and strict-role/provenance repair',
        'v2': '32 passed/1 test assertion failure: small detached source records remain readable after closing owned resources',
        'v3': '34 passed; includes combined capacity, predates strict-role/provenance repair and strengthened neighbour masks',
    },
}
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'forbidden_inference_loaded', 'observations')}, indent=2))
if code or not receipt['source_stable'] or receipt['forbidden_inference_loaded']:
    print((ART / 'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

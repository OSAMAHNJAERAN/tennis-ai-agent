"""Isolated actual native forecast presentation and bounded report verification."""
from contextlib import redirect_stdout, redirect_stderr
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
os.chdir(ROOT)
TEMP = ART/'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
names = ['src/scoring/corrected_forecast_presentation.py', 'src/api/reporting.py',
    'tests/test_corrected_forecast_presentation.py', 'src/scoring/corrected_exports.py',
    'src/scoring/corrected_forecasts.py', 'src/scoring/corrected_rule_rebuild.py',
    'src/scoring/corrected_shot_rebuild.py', 'src/scoring/corrected_shot_sources.py',
    'src/scoring/corrected_landing_exports.py', 'src/scoring/correction_sources.py',
    'src/scoring/point_timeline.py', 'src/scoring/rule_replay_source.py',
    'src/scoring/scoring_engine.py', 'src/scoring/match_state_codec.py',
    'src/api/result_storage.py', 'src/utils/forecast_sequence.py',
    'src/utils/scoring_journal.py', 'src/utils/point_analysis_sequence.py',
    'src/prediction/tennis.py', 'src/scoring/scoring_rules.py',
    'tests/test_corrected_forecasts.py', 'tests/test_corrected_exports.py',
    'tests/test_corrected_shot_rebuild.py', 'tests/test_corrected_rule_rebuild.py',
    'tests/test_correction_sources.py', 'tests/test_bounded_reporting.py',
    'tests/test_streaming_telemetry_csv.py']
paths = [ROOT/name for name in names]+[Path(__file__).resolve()]
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
before = pins()
for path in paths:
    target = ART/'sources'/path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Native forecast presentation imported model inference: '+name)
    return native_import(name, *args, **kwargs)
started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
            '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'),
            'tests/test_corrected_forecast_presentation.py', 'tests/test_bounded_reporting.py',
            'tests/test_streaming_telemetry_csv.py'])
finally:
    builtins.__import__ = native_import
from tests.test_corrected_forecast_presentation import OBSERVATIONS
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
          for key in ('tests', 'errors', 'failures', 'skipped')}
receipt = {'status': 'PASS' if code == 0 and before == after and not attempts else 'FAIL',
    'pytest_exit_code': int(code), 'counts': counts, 'seconds': time.perf_counter()-started,
    'source_before': before, 'source_after': after, 'source_unchanged': before == after,
    'forbidden_import_attempts': attempts, 'torch_imported': 'torch' in sys.modules,
    'ultralytics_imported': 'ultralytics' in sys.modules, 'observations': OBSERVATIONS,
    'scope': 'ACTUAL_CORRECTED_CAUSAL_FORECAST_REGENERATION_RESULT_OWNERSHIP_REPORTS_SYNTHETIC_RULE_CONTACTS'}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status','counts','seconds','source_unchanged')}))
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)

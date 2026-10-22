"""Retain actual owner HTTP, SQLite queue and generation guard checks."""
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
names = ['src/api/corrections.py', 'src/api/jobs.py', 'src/api/product.py', 'src/api/auth.py',
    'src/api/analysis_generation.py', 'src/api/result_document.py', 'src/api/result_storage.py',
    'src/scoring/corrected_rule_rebuild.py', 'src/scoring/rule_replay_source.py',
    'src/scoring/point_timeline.py', 'src/scoring/match_state_codec.py',
    'tests/test_analysis_corrections.py', 'tests/test_corrected_rule_rebuild.py',
    'tests/test_correction_sources.py']
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
        raise AssertionError('Correction HTTP imported inference: '+name)
    return native_import(name, *args, **kwargs)
started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
            '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'),
            'tests/test_analysis_corrections.py'])
finally:
    builtins.__import__ = native_import
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
          for key in ('tests', 'errors', 'failures', 'skipped')}
receipt = {'status': 'PASS' if code == 0 and before == after and not attempts else 'FAIL',
    'pytest_exit_code': int(code), 'counts': counts, 'seconds': time.perf_counter()-started,
    'source_before': before, 'source_after': after, 'source_unchanged': before == after,
    'forbidden_import_attempts': attempts, 'torch_imported': 'torch' in sys.modules,
    'ultralytics_imported': 'ultralytics' in sys.modules,
    'scope': 'ACTUAL_COOKIE_OWNER_HTTP_SQLITE_QUEUE_RESTART_CANCEL_AND_ATOMIC_POINTER_ACKNOWLEDGEMENT_SYNTHETIC_NATIVE_SOURCE; COMPLETE_CORRECTION_PRODUCTION_TESTED_SEPARATELY'}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status','counts','seconds','source_unchanged')}))
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)

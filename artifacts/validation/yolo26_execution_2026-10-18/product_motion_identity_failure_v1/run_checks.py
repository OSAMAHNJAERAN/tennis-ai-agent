"""Retain the exact pre-fix malformed legacy fixture failure, without edits."""
from contextlib import redirect_stdout, redirect_stderr
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
    TENNIS_STORAGE_ROOT=str(HERE / 'isolated_storage'), PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
(HERE / 'tmp').mkdir(exist_ok=False)
for name in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[name] = str(HERE / 'tmp')
FILES = ('src/api/product.py', 'src/api/result_document.py', 'src/api/result_storage.py',
         'tests/test_product_matches.py', 'docs/PRODUCT_DASHBOARD.md')
def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
before = pins()
for name in FILES:
    target = HERE/'source_snapshot'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, target)
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'ultralytics'):
        attempts.append(name)
        raise AssertionError('Identity regression attempted inference: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
buffer = io.StringIO()
started = time.perf_counter()
try:
    with redirect_stdout(buffer), redirect_stderr(buffer):
        import pytest
        code = pytest.main(['tests/test_product_matches.py::test_profiles_do_not_merge_names_and_unconfirmed_motion_is_excluded',
            '--noconftest', '-p', 'no:cacheprovider', '-q',
            '--basetemp', str(HERE/'pytest_tmp'), '--junitxml', str(HERE/'junit.xml')])
finally:
    builtins.__import__ = original
after = pins()
(HERE/'pytest.txt').write_text(buffer.getvalue(), encoding='utf-8')
junit = ET.parse(HERE/'junit.xml').getroot()
counts = {field: sum(int(suite.get(field, 0)) for suite in junit.iter('testsuite'))
          for field in ('tests', 'failures', 'errors', 'skipped')}
receipt = {'scope': 'EXACT_PRE_FIX_LEGACY_PROFILE_MOTION_FIXTURE_REPRODUCTION',
    'expected_failure_reproduced': int(code)==1 and counts=={'tests':1,'failures':1,'errors':0,'skipped':0},
    'pytest_exit_code': int(code), 'junit': counts, 'elapsed_seconds': time.perf_counter()-started,
    'source_before': before, 'source_after': after, 'source_pins_stable': before==after,
    'model_import_attempts': attempts,
    'forbidden_loaded': sorted({'torch','ultralytics'} & set(sys.modules)),
    'diagnosis': 'Fixture summary omits its analysis ID. open_result(output, analysis_id) rejects it before player metrics are read. Whole-clip role confirmation cannot bypass source identity.',
    'measurement_qualification_change': False}
(HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
print(buffer.getvalue()[-8000:])
raise SystemExit(0 if receipt['expected_failure_reproduced'] and before==after and not attempts else 2)

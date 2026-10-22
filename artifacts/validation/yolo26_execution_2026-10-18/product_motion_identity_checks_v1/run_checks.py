"""Current product regressions after the authorized fixture-only identity repair."""
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
        code = pytest.main(['tests/test_product_matches.py',
            '--noconftest', '-p', 'no:cacheprovider', '-q',
            '--basetemp', str(HERE/'pytest_tmp'), '--junitxml', str(HERE/'junit.xml')])
finally:
    builtins.__import__ = original
after = pins()
(HERE/'pytest.txt').write_text(buffer.getvalue(), encoding='utf-8')
junit = ET.parse(HERE/'junit.xml').getroot()
counts = {field: sum(int(suite.get(field, 0)) for suite in junit.iter('testsuite'))
          for field in ('tests', 'failures', 'errors', 'skipped')}
old = json.loads((HERE.parent/'product_motion_identity_failure_v1'/'receipt.json').read_text(encoding='utf-8'))
handler_files = ('src/api/product.py', 'src/api/result_document.py', 'src/api/result_storage.py')
receipt = {'scope': 'ACTUAL_PROFILE_ROLE_CONFIRMATION_AND_SOURCE_IDENTITY_PRODUCT_REGRESSIONS',
    'pytest_exit_code': int(code), 'junit': counts, 'elapsed_seconds': time.perf_counter()-started,
    'source_before': before, 'source_after': after, 'source_pins_stable': before==after,
    'model_import_attempts': attempts,
    'forbidden_loaded': sorted({'torch','ultralytics'} & set(sys.modules)),
    'handlers_unchanged_from_exact_failed_reproduction': all(
        before[name]==old['source_before'][name] for name in handler_files),
    'fixture_only_change': 'Add summary.id=match_test to the preexisting positive legacy fixture and its unchanged-result assertion.',
    'negative_controls': 'Missing and foreign summary.id cannot enter movement history even with explicit whole-clip role confirmation and linked profile; source bytes remain unchanged.',
    'measurement_qualification_change': False,
    'expected_old_failure_receipt_sha256': hashlib.sha256(
        (HERE.parent/'product_motion_identity_failure_v1'/'receipt.json').read_bytes()).hexdigest()}
(HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
print(buffer.getvalue()[-10000:])
raise SystemExit(int(code) if before==after and not attempts and receipt['handlers_unchanged_from_exact_failed_reproduction'] else 2)

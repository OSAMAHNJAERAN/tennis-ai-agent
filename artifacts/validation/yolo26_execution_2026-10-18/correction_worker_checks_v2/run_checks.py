"""Retained actual worker correction lifecycle verification on toy media."""
from contextlib import redirect_stderr, redirect_stdout
import builtins
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
PATHS = sorted({*ROOT.joinpath('src').rglob('*.py'), *ROOT.joinpath('tests').rglob('*.py'),
                Path(__file__).resolve()})

def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS}

before = pins()
for path in PATHS:
    target = ART/'sources'/path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
attempts = []
original_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Correction lifecycle verification imported inference: '+name)
    return original_import(name, *args, **kwargs)

started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
            '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'),
            'tests/test_correction_worker.py'])
finally:
    builtins.__import__ = original_import
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'source_before': before, 'source_after': after,
    'source_stable': before == after, 'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'REAL DURABLE QUEUE THROUGH NATIVE CORRECTION/RENDER/REPORT/COMPLETE GENERATION CAS; '
             'CANCELLATION, STALE BASE, FAILURE, LOST ACK, FAIR QUEUES AND FIXED PRE-TORCH REFUSAL; '
             'EXPLICIT SYNTHETIC VIDEO AND OBSERVATIONS; NO VISION ACCURACY, LONG MATCH OR LIVE CLAIM'}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART/'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

"""Retained synthetic correction source checks; no model imports."""
from pathlib import Path
import builtins
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['TENNIS_STORAGE_ROOT'] = str(HERE / 'isolated_storage')
for name in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[name] = str(HERE / 'tmp')
(HERE / 'tmp').mkdir(exist_ok=True)

PINS = ['src/scoring/correction_sources.py', 'tests/test_correction_sources.py',
        'src/api/analysis_generation.py', 'src/api/result_storage.py',
        'src/scoring/point_timeline.py', 'src/tracking/temporal_ball_tracker.py',
        'src/utils/bbox_utils.py']


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINS}


before = hashes()
actual = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        raise AssertionError('Correction sources must never import a model runtime: ' + name)
    return actual(name, *args, **kwargs)


builtins.__import__ = guarded
started = time.time()
buffer = io.StringIO()
with redirect_stdout(buffer), redirect_stderr(buffer):
    import pytest
    code = pytest.main(['tests/test_correction_sources.py', '--noconftest', '-p', 'no:cacheprovider',
                        '-q', '--basetemp', str(HERE / 'pytest_tmp'), '--junitxml', str(HERE / 'junit.xml')])
builtins.__import__ = actual
after = hashes()
(HERE / 'pytest.txt').write_text(buffer.getvalue(), encoding='utf-8')
receipt = {'scope': 'SYNTHETIC_SEALED_OBSERVATION_ADAPTERS_ONLY_NO_INFERENCE',
           'exit_code': int(code), 'elapsed_seconds': time.time() - started,
           'source_before': before, 'source_after': after, 'source_pins_stable': before == after,
           'imported': {name: name in sys.modules for name in ('torch', 'ultralytics', 'numpy', 'pandas', 'cv2')},
           'bootstrap_erratum': 'Earlier raw pytest invocation lacked an existing basetemp parent; all 39 cases failed fixture setup before test execution. This runner creates its isolated parent and retains actual execution.'}
(HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
print(buffer.getvalue()[-14000:])
raise SystemExit(int(code) if before == after else 2)

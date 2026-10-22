import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path.cwd()
output = Path(__file__).resolve().parent
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['TENNIS_STORAGE_ROOT'] = str(output / 'app-storage')
os.environ['TEMP'] = os.environ['TMP'] = str(output / 'temp')
(output / 'temp').mkdir(exist_ok=True)
sys.path.insert(0, str(root))
paths = sorted(str(path.relative_to(root)).replace('\\', '/')
               for directory in ('src', 'tests') for path in (root / directory).rglob('*.py'))
def pins():
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
before = pins()
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Forbidden model runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pytest
start = time.perf_counter()
code = pytest.main(['--noconftest', '-p', 'no:cacheprovider',
    'tests/test_corrected_forecasts.py', '-q', '--basetemp', str(output / 'pytest-temp'),
    '--junitxml', str(output / 'junit.xml')])
after = pins()
module = sys.modules.get('tests.test_corrected_forecasts')
value = {'status': 'PASS' if code == 0 and before == after and not attempts else 'FAIL',
    'exit_code': code, 'elapsed_seconds': time.perf_counter() - start,
    'source_pins_before': before, 'source_pins_after': after,
    'pins_unchanged': before == after, 'forbidden_import_attempts': attempts,
    'loaded_model_runtimes': [name for name in ('torch', 'ultralytics') if name in sys.modules],
    'loaded_numerical_runtimes': [name for name in ('numpy', 'cv2', 'pandas') if name in sys.modules],
    'observations': getattr(module, 'OBSERVATIONS', []),
    'scope': 'SYNTHETIC_PRIVATE_CAUSAL_FORECAST_SOFTWARE_ONLY; NO_ACCURACY_OR_MEDIA_QUALIFICATION',
    'published': False}
(output / 'receipt.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: val for key, val in value.items() if not key.startswith('source_pins')}, indent=2))
raise SystemExit(code if before == after else 3)

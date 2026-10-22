"""Retained synthetic original shot fixture admission, before driver integration."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
TEMP = ART / 'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'), TEMP=str(TEMP), TMP=str(TEMP),
                  PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        raise AssertionError('Synthetic native shot fixture imported inference: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
source = ROOT / 'tests/test_corrected_shot_rebuild.py'
before = hashlib.sha256(source.read_bytes()).hexdigest()
from tests.test_corrected_shot_rebuild import build_shot_base, unchanged
fixture = build_shot_base(TEMP)
unchanged(fixture)
receipt = {'status': 'PASS', 'scope': 'SYNTHETIC_ORIGINAL_NATIVE_SHOT_FIXTURE_ONLY',
    'source_before': before, 'source_after': hashlib.sha256(source.read_bytes()).hexdigest(),
    'original': fixture.original, 'replay_input_count': fixture.document['input_count'],
    'forbidden_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules))}
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))

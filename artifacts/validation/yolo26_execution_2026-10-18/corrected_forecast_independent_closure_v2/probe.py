"""Recheck the exact two public export mutations from the independent review."""
import builtins
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path.cwd()
output = root / 'artifacts/validation/yolo26_execution_2026-10-18/corrected_forecast_independent_closure_v2'
os.environ['TENNIS_STORAGE_ROOT'] = str(output / 'app-storage')
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['TEMP'] = os.environ['TMP'] = str(output / 'temp')
(output / 'temp').mkdir(exist_ok=True)
sys.path.insert(0, str(root))
paths = ['src/scoring/corrected_forecasts.py', 'src/utils/forecast_sequence.py',
         'src/prediction/tennis.py', 'tests/test_corrected_forecasts.py',
         'tests/test_corrected_rule_rebuild.py', 'tests/test_corrected_shot_rebuild.py']
def pins():
    return {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}
before = pins()
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Forbidden model import: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded

from tests.test_corrected_forecasts import fresh_shots
from tests.test_corrected_shot_rebuild import build_shot_base
from tests.test_corrected_rule_rebuild import unchanged
from src.scoring.corrected_forecasts import rebuild_corrected_forecasts, write_corrected_forecasts

observations = []
started = time.perf_counter()
for name in ('forged_audit', 'replaced_public_history'):
    case = output / name
    case.mkdir(exist_ok=False)
    fixture = build_shot_base(case)
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            trusted_audit = forecast.audit
            trusted_current = forecast.current
            trusted_history = list(forecast.history)
            if name == 'forged_audit':
                forecast.audit.update(analysis_id='OTHER_OWNER_ANALYSIS', analysis_revision=999,
                    source_sha256='b'*64, calibration_status='independently_calibrated',
                    independent_accuracy_validated=True, published=True)
                forecast.current.update(sample_count=12345)
                assignment_error = None
            else:
                forged = copy.deepcopy(forecast.history[-1])
                forged['forecast'].update(sample_count=12345, status='uncalibrated')
                forged['analysis_id'] = 'OTHER_OWNER_ANALYSIS'
                try:
                    forecast.history = [forged]
                    forecast.current = copy.deepcopy(forged['forecast'])
                except FrozenInstanceError as error:
                    assignment_error = str(error)
                else:
                    raise AssertionError('Public history replacement was accepted')
            destination = fixture.scratch/'corrected_forecasts.json'
            write_corrected_forecasts(destination, forecast)
            document = json.loads(destination.read_text(encoding='utf-8'))
            assert document['analysis_id'] == trusted_audit['analysis_id'] == 'fixture'
            assert document['analysis_revision'] == trusted_audit['analysis_revision'] == 1
            assert document['history'] == trusted_history and len(trusted_history) == 7
            assert document['current'] == trusted_current
            assert document['audit'] == trusted_audit
            assert document['audit']['independent_accuracy_validated'] is False
            assert document['audit']['published'] is False
            assert document['current']['sample_count'] == 1
            result = {'case': name, 'status': 'CLOSED', 'assignment_error': assignment_error,
                'export_analysis_id': document['analysis_id'],
                'export_revision': document['analysis_revision'],
                'export_history_count': len(document['history']),
                'export_sample_count': document['current']['sample_count'],
                'independent_accuracy_validated': False, 'published': False}
    unchanged(fixture)
    result['sealed_base_unchanged'] = True
    observations.append(result)

after = pins()
receipt = {'status': 'PASS' if before == after and not attempts else 'SOURCE_CHANGED',
    'source_pins_before': before, 'source_pins_after': after, 'observations': observations,
    'elapsed_seconds': time.perf_counter()-started, 'forbidden_import_attempts': attempts,
    'loaded_models': [name for name in ('torch', 'ultralytics') if name in sys.modules],
    'scope': 'SYNTHETIC_SOFTWARE_FIXTURES_ONLY; NO_PUBLICATION_OR_ACCURACY_CLAIM'}
(output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))

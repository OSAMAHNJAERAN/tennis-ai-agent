"""Independent nonpublishing probes of fresh forecast export provenance."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path.cwd()
output = root / 'artifacts/validation/yolo26_execution_2026-10-18/corrected_forecast_independent_review_v1'
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
snapshot = output / 'source_snapshot'
for path in paths:
    destination = snapshot / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((root/path).read_bytes())
attempts = []
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Model import is forbidden in independent software probe: '+name)
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
            expected_id = forecast.audit['analysis_id']
            expected_revision = forecast.audit['analysis_revision']
            expected_history_count = len(forecast.history)
            if name == 'forged_audit':
                forecast.audit.update(analysis_id='OTHER_OWNER_ANALYSIS', analysis_revision=999,
                    source_sha256='b'*64, calibration_status='independently_calibrated',
                    independent_accuracy_validated=True, published=True)
            else:
                forged = copy.deepcopy(forecast.history[-1])
                forged['forecast'].update(sample_count=12345, status='uncalibrated')
                forged['analysis_id'] = 'OTHER_OWNER_ANALYSIS'
                forecast.history = [forged]
                forecast.current = copy.deepcopy(forged['forecast'])
            destination = fixture.scratch / 'corrected_forecasts.json'
            try:
                write_corrected_forecasts(destination, forecast)
            except (ValueError, RuntimeError, AttributeError, TypeError) as error:
                result = {'case': name, 'accepted': False, 'error': repr(error)}
            else:
                document = json.loads(destination.read_text(encoding='utf-8'))
                (case/'exported.json').write_text(json.dumps(document, indent=2)+'\n', encoding='utf-8')
                result = {'case': name, 'accepted': True,
                    'trusted_analysis_id': expected_id, 'trusted_revision': expected_revision,
                    'trusted_history_count': expected_history_count,
                    'export_analysis_id': document['analysis_id'],
                    'export_revision': document['analysis_revision'],
                    'export_history_count': len(document['history']),
                    'export_first_row_analysis_id': document['history'][0]['analysis_id'],
                    'export_last_sample_count': document['history'][-1]['forecast']['sample_count'],
                    'export_audit_independent_accuracy_validated': document['audit']['independent_accuracy_validated'],
                    'export_audit_published': document['audit']['published']}
    unchanged(fixture)
    result['sealed_base_unchanged'] = True
    observations.append(result)

receipt = {'status': 'REPRODUCED_EXPORT_PROVENANCE_GAPS' if all(row['accepted'] for row in observations) else 'REVIEW',
    'source_pins_before': before, 'source_pins_after': pins(), 'observations': observations,
    'elapsed_seconds': time.perf_counter()-started, 'forbidden_import_attempts': attempts,
    'loaded_models': [name for name in ('torch', 'ultralytics') if name in sys.modules],
    'scope': 'SYNTHETIC_SOFTWARE_FIXTURES_ONLY; NO_PUBLICATION_OR_ACCURACY_CLAIM'}
(output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))

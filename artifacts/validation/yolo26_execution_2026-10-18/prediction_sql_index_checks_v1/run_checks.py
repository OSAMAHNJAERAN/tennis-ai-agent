"""Retain actual SQLite predictor-index integration and mathematical regressions."""
from contextlib import redirect_stderr, redirect_stdout
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
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['TENNIS_STORAGE_ROOT'] = str(HERE / 'isolated_storage')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
for name in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[name] = str(HERE / 'tmp')
(HERE / 'tmp').mkdir(exist_ok=False)
PINS = [
    'src/prediction/tennis.py', 'src/utils/forecast_sequence.py',
    'tests/test_prediction_point_index.py', 'tests/test_scoring_prediction.py',
    'tests/test_phase5_scoring.py', 'src/scoring/match_state.py',
    'src/scoring/scoring_rules.py', 'src/scoring/point_ledger.py',
    'src/scoring/scoring_engine.py', 'src/scoring/point_outcome_resolver.py',
    'src/line_calling/line_call_engine.py',
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINS}


before = hashes()
for name in PINS:
    destination = HERE / 'source_snapshot' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, destination)
attempts = []
original = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Predictor SQL integration attempted a model import: ' + name)
    return original(name, *args, **kwargs)


started = time.perf_counter()
buffer = io.StringIO()
builtins.__import__ = guarded
try:
    with redirect_stdout(buffer), redirect_stderr(buffer):
        import pytest
        code = pytest.main([
            'tests/test_prediction_point_index.py', 'tests/test_scoring_prediction.py',
            '--noconftest', '-p', 'no:cacheprovider', '-q',
            '--basetemp', str(HERE / 'pytest_tmp'), '--junitxml', str(HERE / 'junit.xml'),
        ])
finally:
    builtins.__import__ = original
after = hashes()
elapsed = time.perf_counter() - started
junit = ET.parse(HERE / 'junit.xml').getroot()
counts = {name: sum(int(suite.get(name, 0)) for suite in junit.iter('testsuite'))
          for name in ('tests', 'failures', 'errors', 'skipped')}
(HERE / 'pytest.txt').write_text(buffer.getvalue(), encoding='utf-8')
receipt = {
    'scope': 'ACTUAL_SQLITE_CALLER_OWNED_POINT_INDEX_AND_EXISTING_SCORING_PREDICTION_MATH; NO_MODEL_INFERENCE_OR_CALIBRATION',
    'exit_code': int(code), 'junit': counts, 'elapsed_seconds': elapsed,
    'source_before': before, 'source_after': after, 'source_pins_stable': before == after,
    'model_import_attempts': attempts,
    'model_runtimes_loaded': sorted(name for name in ('torch', 'ultralytics') if name in sys.modules),
    'actual_sql_index_cases': [
        'EXACT_SUPPLIED_OBJECT_AND_EVERY_CAUSAL_PREFIX_EQUALS_LEGACY_DICTIONARY',
        'IDEMPOTENT_EARLIER_RETRY_CONFLICT_AND_ORDER',
        'NONEMPTY_AND_CLOSED_OWNER_CONSTRUCTOR_REFUSAL',
        'CLOSED_OWNER_UPDATE_AND_SNAPSHOT_NO_FALLBACK',
        'NO_ITERATION_COPY_OR_ADDITIONAL_PREDICTOR_CACHE',
        'ACTUAL_SQLITE_AUTHORIZER_DENIES_COMMIT_ROLLBACK_RETAINS_ONLY_PRIOR_COMMIT_COUNTERS_UNCHANGED',
        'NONDEFAULT_PARTICIPANTS_EVERY_CAUSAL_PREFIX',
    ],
    'capacity_scope': 'No new capacity claim; dedicated growing native completed-point fixture belongs to separate integration scope.',
}
(HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
print(buffer.getvalue()[-14000:])
raise SystemExit(int(code) if before == after and not attempts else 2)

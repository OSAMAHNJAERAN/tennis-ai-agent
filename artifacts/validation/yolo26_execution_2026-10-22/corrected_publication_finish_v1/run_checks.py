"""Scoped actual complete authorized correction publisher verification."""
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
NAMES = ['src/scoring/corrected_publication.py', 'src/scoring/corrected_preparation.py',
    'src/scoring/corrected_rendering.py', 'src/scoring/corrected_exports.py',
    'src/scoring/corrected_forecast_presentation.py', 'src/scoring/corrected_forecasts.py',
    'src/scoring/corrected_rule_rebuild.py', 'src/scoring/corrected_shot_rebuild.py',
    'src/scoring/corrected_shot_sources.py', 'src/scoring/correction_sources.py',
    'src/scoring/corrected_landing_exports.py', 'src/scoring/rule_reproduction.py',
    'src/scoring/rule_replay.py', 'src/scoring/rule_replay_source.py',
    'src/scoring/point_timeline.py', 'src/scoring/scoring_engine.py',
    'src/scoring/match_state.py', 'src/scoring/match_state_codec.py',
    'src/api/analysis_generation.py', 'src/api/analysis_resume.py', 'src/api/result_document.py',
    'src/api/result_storage.py', 'src/api/results.py', 'src/api/reporting.py',
    'src/api/point_timeline_contract.py', 'src/api/landing_zone_contract.py',
    'src/api/media_source.py', 'src/api/media_revision.py', 'src/api/media_timing.py',
    'src/utils/event_analysis_sequence.py', 'src/utils/point_analysis_sequence.py',
    'src/utils/forecast_sequence.py', 'src/utils/scoring_journal.py',
    'src/utils/shot_evidence_sequence.py', 'src/utils/landing_zone_sequence.py',
    'src/utils/analytics_json_artifact.py', 'src/utils/json_array_artifact.py',
    'src/utils/point_timeline_artifact.py', 'tests/test_corrected_publication.py',
    'tests/test_corrected_preparation.py', 'tests/test_corrected_rendering.py',
    'tests/test_corrected_exports.py', 'tests/test_corrected_forecasts.py',
    'tests/test_corrected_shot_rebuild.py', 'tests/test_corrected_rule_rebuild.py',
    'tests/test_correction_sources.py']
PATHS = [ROOT/name for name in NAMES]+[Path(__file__).resolve()]
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS}
before = pins()
for path in PATHS:
    target = ART/'sources'/path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Complete correction publisher imported inference: '+name)
    return native_import(name, *args, **kwargs)
started = time.perf_counter()
builtins.__import__ = guarded
try:
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
                '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'),
                'tests/test_corrected_publication.py::test_pre_callback_native_seal_tampering_cannot_commit_an_unreadable_current_result',
                'tests/test_corrected_publication.py::test_valid_sqlite_root_tampering_before_initial_database_hash_is_rejected_against_report',
                'tests/test_corrected_publication.py::test_temporary_pointer_is_bound_to_exact_candidate_before_and_after_authorization',
                'tests/test_corrected_publication.py::test_critical_post_commit_interruption_propagates_while_actual_commit_stays_readable',
                'tests/test_corrected_publication.py::test_lost_acknowledgement_is_reconciled_from_actual_current_generation_and_retry_is_idempotent',
                'tests/test_corrected_publication.py::test_final_authorization_callback_cannot_change_the_complete_candidate[DATABASE]'])
finally:
    builtins.__import__ = native_import
from tests.test_corrected_publication import OBSERVATIONS
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'source_before': before, 'source_after': after,
    'source_stable': before == after, 'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'observations': OBSERVATIONS, 'published_scope': 'ISOLATED SYNTHETIC TEST GENERATIONS ONLY',
    'scope': 'NINE TARGETED ACTUAL COMPLETE CAS FAULT CASES: TRUSTED DB/ROOT/NATIVE MANIFEST/POINTER BEFORE AND AFTER AUTH, CRITICAL INTERRUPTS AND ORDINARY LOST ACK; SYNTHETIC IMAGERY AND OBSERVATIONS; NO MODEL OR INDEPENDENT ACCURACY QUALIFICATION',
    'pin_scope': '47 explicit publisher, semantic/report/storage dependency and fixture/runner files; concurrent unrelated HTTP/worker/store changes excluded'}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded', 'observations')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART/'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

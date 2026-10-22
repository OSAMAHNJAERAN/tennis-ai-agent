"""Scoped actual complete authorized correction publisher verification."""
from contextlib import redirect_stderr, redirect_stdout
import builtins
import hashlib
import importlib.util
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
PATHS = [ROOT/name for name in NAMES]+[Path(__file__).resolve(),
    ART/'corrected_publication_candidate.py', ART/'test_corrected_publication_proposed.py',
    ART/'test_candidate_publication.py']
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS}
before = pins()
for path in PATHS:
    target = (ART/'candidate_sources'/path.name if path.parent == ART
              else ART/'sources'/path.relative_to(ROOT))
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
    specification = importlib.util.spec_from_file_location('corrected_publication_candidate',
        ART/'corrected_publication_candidate.py')
    candidate = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = candidate
    specification.loader.exec_module(candidate)
    import pytest
    with (ART/'pytest.txt').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            code = pytest.main(['-q', '--noconftest', '-p', 'no:cacheprovider',
                '--basetemp', str(TEMP/'pytest'), '--junitxml', str(ART/'junit.xml'),
                str(ART/'test_candidate_publication.py')])
finally:
    builtins.__import__ = native_import
from test_candidate_publication import OBSERVATIONS
after = pins()
suites = ET.parse(ART/'junit.xml').getroot().findall('testsuite')
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
receipt = {'exit_code': int(code), 'elapsed_seconds': time.perf_counter()-started,
    'counts': counts, 'source_before': before, 'source_after': after,
    'source_stable': before == after, 'inference_import_attempts': attempts,
    'forbidden_inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'observations': OBSERVATIONS, 'published_scope': 'ISOLATED SYNTHETIC TEST GENERATIONS ONLY',
    'scope': 'CANDIDATE MODULE UNDER SEPARATE IMPORTLIB NAME; ACTUAL COMPLETE PRIVATE PREPARATION/OWNER AUTHORIZATION/DELIVERY REPORTS/CAS/LOST ACK/DB ROOT AND CANONICAL MANIFEST INTEGRITY/CRITICAL INTERRUPTION; SYNTHETIC IMAGERY AND OBSERVATIONS; NO MODEL OR INDEPENDENT ACCURACY QUALIFICATION',
    'pin_scope': '50 explicit production dependency/fixture/candidate/test/runner files; concurrent unrelated HTTP/worker/store changes excluded',
    'candidate_module': specification.name, 'candidate_source': str(ART/'corrected_publication_candidate.py'),
    'production_sources_modified_by_candidate_run': False}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'elapsed_seconds', 'counts',
    'source_stable', 'inference_import_attempts', 'forbidden_inference_loaded', 'observations')}, indent=2))
if code or before != after or attempts or receipt['forbidden_inference_loaded']:
    print((ART/'pytest.txt').read_text(encoding='utf-8'))
    raise SystemExit(code or 2)

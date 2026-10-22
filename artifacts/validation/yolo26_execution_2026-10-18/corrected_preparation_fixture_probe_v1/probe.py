"""Actual complete synthetic original/native corrected preparation fixture probe."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
(ART/'temp').mkdir(exist_ok=False)
os.environ.update(TEMP=str(ART/'temp'), TMP=str(ART/'temp'),
    TENNIS_STORAGE_ROOT=str(ART/'bootstrap'), PYTHONDONTWRITEBYTECODE='1')
paths = ['tests/test_corrected_preparation.py', 'tests/test_corrected_shot_rebuild.py',
         'tests/test_corrected_rule_rebuild.py', 'src/api/results.py',
         'src/scoring/corrected_rendering.py']
def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
before = pins()
attempts = []
native = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Complete fixture imported inference: '+name)
    return native(name, *args, **kwargs)
builtins.__import__ = guarded
result, code = {}, 0
started = time.perf_counter()
try:
    from tests.test_corrected_preparation import build_complete_base, complete_inputs
    from tests.test_corrected_rule_rebuild import unchanged
    from src.api.analysis_resume import REQUIRED_PIPELINE_FILES
    from src.api.result_document import ResultDocument
    directory = ART/'fixture'
    directory.mkdir(exist_ok=False)
    fixture = build_complete_base(directory)
    result['base_required_files'] = len(REQUIRED_PIPELINE_FILES)
    result['base_file_names'] = sorted(path.name for path in fixture.snapshot.files_directory.iterdir())
    with ResultDocument(fixture.directory) as base:
        result['base_points'] = base.value['pointAnalysis']['summary']
        result['base_provenance'] = base.value['provenance']
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        result.update(corrected_points=len(rebuilt.owner.rows('points', status='COMPLETED')),
            forecast_rows=len(forecast.history), sample_count=forecast.current['sample_count'],
            export_files=len(exports.files), render_files=len(rendered.files),
            render_manifest_sha256=rendered.manifest_sha256)
    unchanged(fixture)
except BaseException:
    code = 1
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = native
receipt = {'exit_code': code, 'result': result, 'elapsed_seconds': time.perf_counter()-started,
    'source_before': before, 'source_after': pins(), 'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Actual complete original native export + fresh corrected pipeline fixture; toy images and synthetic measurements, no accuracy claim'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
raise SystemExit(code)

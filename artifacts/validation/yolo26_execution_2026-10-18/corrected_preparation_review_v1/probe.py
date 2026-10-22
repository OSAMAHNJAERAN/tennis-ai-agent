"""Independent private preparation copy-binding probe; synthetic codec fixture."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
(ART / 'temp').mkdir(exist_ok=False)
os.environ.update(TEMP=str(ART / 'temp'), TMP=str(ART / 'temp'),
    TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'), PYTHONDONTWRITEBYTECODE='1')
NAMES = ['src/scoring/corrected_preparation.py', 'src/scoring/corrected_rendering.py',
    'src/scoring/corrected_exports.py', 'src/scoring/corrected_landing_exports.py',
    'src/scoring/corrected_forecast_presentation.py', 'src/api/results.py',
    'src/api/reporting.py', 'tests/test_corrected_preparation.py']
def pins():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in NAMES}
before = pins()
native_import = builtins.__import__
attempts = []
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Preparation review may not load inference: ' + name)
    return native_import(name, *args, **kwargs)
builtins.__import__ = guard
result = {}
try:
    from tests.test_corrected_preparation import build_complete_base, complete_inputs, prepare, read, digests
    from tests.test_corrected_rule_rebuild import unchanged
    from src.scoring import corrected_preparation as preparation
    parent = ART / 'fixture'
    parent.mkdir(exist_ok=False)
    fixture = build_complete_base(parent)
    changed = []
    original_report = preparation.readable_report_chunks
    def report_with_stage_change(value, directory):
        for index, chunk in enumerate(original_report(value, directory)):
            if index == 0:
                path = directory / 'match_state.json'
                document = read(path)
                original_points = document['player_2']['points_display']
                document['player_2']['points_display'] = 'FORGED 999'
                path.write_text(json.dumps(document), encoding='utf-8')
                changed.append({'path': str(path), 'before': original_points,
                    'after': document['player_2']['points_display']})
            yield chunk
    preparation.readable_report_chunks = report_with_stage_change
    try:
        with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
            export_hashes, render_hashes = digests(exports.directory), digests(rendered.directory)
            candidate = None
            try:
                with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as output:
                    candidate = output
                    copied_state = read(output.files['match_state.json'])
                    original_state = read(exports.files['match_state.json'])
                    result.update(preparation_entered=True,
                        copied_points=copied_state['player_2']['points_display'],
                        trusted_export_points=original_state['player_2']['points_display'],
                        current_native_points=rebuilt.engine.state.get_points_display_p2(),
                        stage_state_hash=hashlib.sha256(output.files['match_state.json'].read_bytes()).hexdigest(),
                        copied_state_matches_trusted_export=copied_state == original_state)
                result['preparation_successful_exit'] = True
            except (ValueError, RuntimeError, TypeError) as error:
                result.update(preparation_successful_exit=False, rejection=str(error))
            result.update(callback_stage_writes=changed,
                export_unchanged=digests(exports.directory) == export_hashes,
                render_unchanged=digests(rendered.directory) == render_hashes,
                final_seals=[str(path) for path in fixture.scratch.rglob('corrected_preparation_manifest.json')])
            if candidate is not None:
                result['storage_closed'] = candidate.storage._connection is None
                result['candidate_seal_exists'] = candidate.files['corrected_preparation_manifest.json'].is_file()
        unchanged(fixture)
        result['committed_base_unchanged'] = True
    finally:
        preparation.readable_report_chunks = original_report
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = native_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Actual complete synthetic correction/export/render/report chain with stage-copy corruption during streamed HTML; no publication/accuracy claim'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))

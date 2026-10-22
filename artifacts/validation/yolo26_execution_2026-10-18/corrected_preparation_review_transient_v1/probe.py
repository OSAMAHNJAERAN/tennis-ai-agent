"""Transient producer-file ABA across entry guard and copying, with real reports."""
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
    original_base, original_copy = preparation._base_inputs, preparation._copy
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        source = exports.files['metrics.json']
        original_bytes = source.read_bytes()
        before_export, before_render = digests(exports.directory), digests(rendered.directory)
        calls = []
        def base_after_guard(*args, **kwargs):
            value = original_base(*args, **kwargs)
            modified = read(source)
            modified['processing_fps'] = 999.
            source.write_text(json.dumps(modified), encoding='utf-8')
            calls.append('TRANSIENT_PRODUCER_METRICS_CHANGED_AFTER_ENTRY_GUARDS')
            return value
        def copy_restore(src, destination, expected):
            value = original_copy(src, destination, expected)
            if Path(src) == source:
                source.write_bytes(original_bytes)
                calls.append('PRODUCER_METRICS_RESTORED_AFTER_COPY')
            return value
        preparation._base_inputs, preparation._copy = base_after_guard, copy_restore
        candidate = None
        try:
            try:
                with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as output:
                    candidate = output
                    manifest = read(output.files['corrected_preparation_manifest.json'])
                    copied_metrics = read(output.files['metrics.json'])
                    original_metrics = read(source)
                    result.update(preparation_entered=True,
                        copied_processing_fps=copied_metrics['processing_fps'],
                        trusted_producer_processing_fps=original_metrics['processing_fps'],
                        normalized_pipeline_fps=output.result['pipelineMetrics']['pipelineFps'],
                        preparation_seals_transient_metrics=manifest['artifacts']['metrics.json']['sha256'] ==
                            hashlib.sha256(output.files['metrics.json'].read_bytes()).hexdigest())
                result['preparation_successful_exit'] = True
            except (ValueError, RuntimeError, TypeError) as error:
                result.update(preparation_successful_exit=False, rejection=str(error))
        finally:
            source.write_bytes(original_bytes)
            preparation._base_inputs, preparation._copy = original_base, original_copy
        result.update(callbacks=calls, export_unchanged=digests(exports.directory) == before_export,
            render_unchanged=digests(rendered.directory) == before_render,
            final_seals=[str(path) for path in fixture.scratch.rglob('corrected_preparation_manifest.json')])
        if candidate is not None:
            result['storage_closed'] = candidate.storage._connection is None
            result['candidate_seal_exists'] = candidate.files['corrected_preparation_manifest.json'].is_file()
    unchanged(fixture)
    result['committed_base_unchanged'] = True
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = native_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Complete synthetic preparation chain with transient producer metrics between entry guard and copy; no publication or accuracy claim'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))

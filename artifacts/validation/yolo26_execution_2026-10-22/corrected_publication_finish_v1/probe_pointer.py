"""Actual isolated pre-callback current-pointer damage probe; no inference."""
from contextlib import redirect_stderr, redirect_stdout
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent/'pointer_probe'
ART.mkdir(exist_ok=False)
TEMP = ART/'temp'
TEMP.mkdir()
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
paths = [ROOT/name for name in ('src/scoring/corrected_publication.py', 'src/api/analysis_generation.py',
    'src/api/result_document.py', 'src/api/result_storage.py', 'tests/test_corrected_publication.py')]
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
before = pins()
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('No inference permitted')
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded
observed = {}
started = time.perf_counter()
try:
    import pytest
    from tests.test_corrected_publication import ready, authorization
    from src.api import analysis_generation
    from src.api.result_document import ResultDocument
    from src.scoring.corrected_publication import publish_corrected_result
    with (ART/'probe.txt').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        workspace = ART/'fixture'
        workspace.mkdir()
        fixture, descriptor = ready(workspace)
        initial = (fixture.directory/'.result/current.store').read_bytes()
        native = analysis_generation._write_bytes
        changed = []
        def writing(path, payload, **kwargs):
            value = native(path, payload, **kwargs)
            if path.name.endswith('.store.tmp'):
                path.write_bytes(b'{"schemaVersion":2,"generation":"' + b'f'*64 + b'"}\n')
                changed.append(path.name)
            return value
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(analysis_generation, '_write_bytes', writing)
            try:
                observed['returned'] = publish_corrected_result(descriptor, fixture.directory,
                    authorization=authorization(fixture, descriptor))
            except BaseException as error:
                observed['exception'] = type(error).__name__+': '+str(error)
        observed['changed'] = changed
        observed['pointer_changed'] = (fixture.directory/'.result/current.store').read_bytes() != initial
        try:
            with ResultDocument(fixture.directory, 'fixture') as current:
                observed['current_readable'] = True
        except Exception as error:
            observed['current_readable'] = False
            observed['reader_error'] = type(error).__name__+': '+str(error)
finally:
    builtins.__import__ = native_import
after = pins()
receipt = {'status': 'REPRODUCED' if observed.get('pointer_changed') and not observed.get('current_readable') else 'NOT_REPRODUCED',
    'scope': 'ACTUAL_NATIVE_PRE_CALLBACK_POINTER_TMP_TAMPERING_ON_EXPLICIT_SYNTHETIC_GENERATION_ONLY',
    'elapsed_seconds': time.perf_counter()-started, 'observed': observed,
    'source_before': before, 'source_after': after, 'source_stable': before == after,
    'inference_import_attempts': attempts, 'inference_loaded': sorted({'torch','ultralytics'} & set(sys.modules))}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))

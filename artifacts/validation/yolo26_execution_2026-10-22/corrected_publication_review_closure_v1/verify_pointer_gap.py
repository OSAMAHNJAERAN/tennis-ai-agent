"""Independent actual pre-callback pointer-temporary corruption rejection."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
TEMP = ART/'pointer_temp'
TEMP.mkdir(exist_ok=False)
os.chdir(ROOT)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'pointer_bootstrap'),
    PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
NAMES = ['src/scoring/corrected_publication.py', 'src/api/correction_worker.py',
    'src/api/corrections.py', 'src/api/analysis_generation.py', 'src/api/result_document.py',
    'src/api/result_storage.py', 'tests/test_corrected_publication.py',
    'tests/test_corrected_preparation.py', 'src/scoring/corrected_preparation.py']
def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
before = pins()
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Independent publication closure cannot load inference')
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded
started = time.perf_counter()
try:
    from tests.test_corrected_publication import ready, authorization
    from src.api import analysis_generation
    from src.api.result_document import ResultDocument
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(TEMP)
    previous = (fixture.directory/'.result/current.store').read_bytes()
    native_write = analysis_generation._write_bytes
    mutations = []
    def writing(path, payload, **kwargs):
        result = native_write(path, payload, **kwargs)
        if path.name.endswith('.store.tmp'):
            with path.open('ab') as stream:
                stream.write(b'EXPLICIT_SYNTHETIC_POINTER_TEMP_CORRUPTION')
            mutations.append(path.name)
        return result
    analysis_generation._write_bytes = writing
    token, publication_error = None, None
    try:
        try:
            token = publish_corrected_result(descriptor, fixture.directory,
                authorization=authorization(fixture, descriptor))
        except Exception as error:
            publication_error = type(error).__name__+': '+str(error)
    finally:
        analysis_generation._write_bytes = native_write
    current_unchanged = (fixture.directory/'.result/current.store').read_bytes() == previous
    with ResultDocument(fixture.directory, 'fixture') as document:
        old_readable = document.generation == fixture.snapshot.generation and document.value['analysisRevision'] == 0
    private_unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == fixture.private_hashes[path.name]
                            for path in descriptor.directory.iterdir())
    leftover_delivery_seals = list(fixture.scratch.rglob('corrected_delivery_manifest.json'))
finally:
    builtins.__import__ = native_import
after = pins()
passed = (token is None and publication_error is not None and 'pointer' in publication_error
          and len(mutations) == 1 and current_unchanged and old_readable and private_unchanged
          and not leftover_delivery_seals and before == after and not attempts
          and not {'torch', 'ultralytics'} & set(sys.modules))
receipt = {'status': 'PASS' if passed else 'FAIL',
    'scope': 'ACTUAL_NATIVE_SYNTHETIC_PREPARATION_AND_CAS; POINTER_TEMP_CORRUPTED_AFTER_NATIVE_WRITE_BEFORE_FINAL_CALLBACK',
    'publication_token': token, 'publication_error': publication_error,
    'mutations': mutations, 'current_pointer_unchanged': current_unchanged,
    'old_result_readable': old_readable, 'private_preparation_unchanged': private_unchanged,
    'leftover_delivery_seals': [str(path) for path in leftover_delivery_seals],
    'source_before': before, 'source_after': after, 'source_unchanged': before == after,
    'forbidden_import_attempts': attempts, 'torch_imported': 'torch' in sys.modules,
    'ultralytics_imported': 'ultralytics' in sys.modules, 'seconds': time.perf_counter()-started}
(ART/'pointer_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status', 'publication_error', 'current_pointer_unchanged',
    'old_result_readable', 'private_preparation_unchanged', 'source_unchanged', 'seconds')}))
raise SystemExit(0 if passed else 1)

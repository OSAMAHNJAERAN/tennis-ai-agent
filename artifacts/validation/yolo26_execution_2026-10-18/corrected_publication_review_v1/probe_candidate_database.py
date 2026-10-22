"""Actual isolated generation probe for the pre-callback database seal window."""
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
os.chdir(ROOT)
TEMP = ART/'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
    PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
names = ['src/scoring/corrected_publication.py','src/api/correction_worker.py',
    'src/api/corrections.py','src/api/analysis_generation.py','src/api/result_document.py',
    'src/api/result_storage.py','tests/test_corrected_publication.py',
    'tests/test_corrected_preparation.py','src/scoring/corrected_preparation.py']
def pins():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before = pins()
for name in names:
    target = ART/'sources'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/name,target)
attempts = []
original_import = builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics'}:
        attempts.append(name)
        raise AssertionError('No inference permitted in candidate seal review')
    return original_import(name,*args,**kwargs)
builtins.__import__ = checked
started = time.perf_counter()
try:
    from tests.test_corrected_publication import ready,authorization
    from src.api import analysis_generation
    from src.api.result_document import ResultDocument
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(TEMP)
    marker = (fixture.directory/'.result/current.store').read_bytes()
    native_copy = analysis_generation._copy
    mutations = []
    def copying(source,destination,**kwargs):
        result = native_copy(source,destination,**kwargs)
        if (source.name == 'result.json' and source.parent.suffix == '.files'
                and destination.parent == fixture.directory):
            database = source.parent.with_suffix('.sqlite3')
            with database.open('ab') as stream:
                stream.write(b'EXPLICIT_SYNTHETIC_PRE_FINAL_CALLBACK_DATABASE_DAMAGE')
            mutations.append(str(database.relative_to(TEMP)))
        return result
    analysis_generation._copy = copying
    token, publication_error, read_error = None,None,None
    try:
        try:
            token = publish_corrected_result(descriptor,fixture.directory,
                authorization=authorization(fixture,descriptor))
        except Exception as error:
            publication_error = type(error).__name__+': '+str(error)
    finally:
        analysis_generation._copy = native_copy
    changed = (fixture.directory/'.result/current.store').read_bytes() != marker
    try:
        with ResultDocument(fixture.directory,'fixture') as document:
            document.value['summary']['frames']
    except Exception as error:
        read_error = type(error).__name__+': '+str(error)
finally:
    builtins.__import__ = original_import
after = pins()
receipt = {'probe_executed':True,'scope':'ACTUAL_NATIVE_SYNTHETIC_PREPARATION_AND_FILESYSTEM_CAS; DATABASE_MUTATED_AFTER_MANIFEST_HASH_BEFORE_FINAL_CHECK',
    'publication_token':token,'publication_error':publication_error,'current_pointer_changed':changed,
    'current_result_read_error':read_error,'mutations':mutations,
    'issue_reproduced':bool(token and changed and read_error),'source_before':before,
    'source_after':after,'source_unchanged':before==after,'seconds':time.perf_counter()-started,
    'forbidden_import_attempts':attempts,'torch_imported':'torch' in sys.modules,
    'ultralytics_imported':'ultralytics' in sys.modules}
(ART/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:receipt[key] for key in ('issue_reproduced','publication_error','current_pointer_changed','current_result_read_error','mutations','source_unchanged','seconds')}))
raise SystemExit(0 if before==after and len(mutations)==1 and not attempts else 1)

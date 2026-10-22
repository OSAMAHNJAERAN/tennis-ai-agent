"""Current worker receipt closure and actual post-CAS interrupt handling.

Counter checks use an explicitly synthetic in-memory predictor, bypassing
checkpoint admission only for this metadata seam. Complete publisher probes
use real native synthetic preparation and atomic filesystem commits.
"""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
os.chdir(ROOT)
TEMP = ART/'execution_temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'execution_bootstrap'),
    PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True
names = ['src/scoring/corrected_publication.py','src/api/correction_worker.py',
    'src/api/corrections.py','src/api/analysis_generation.py','src/api/result_document.py',
    'src/api/result_storage.py','tests/test_corrected_publication.py',
    'tests/test_corrected_preparation.py','src/scoring/corrected_preparation.py']
def pins():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before = pins()
attempts = []
original_import = builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics'}:
        attempts.append(name)
        raise AssertionError('No inference permitted in publication closure review')
    return original_import(name,*args,**kwargs)
builtins.__import__ = checked
started = time.perf_counter()
try:
    from tests.test_corrected_publication import ready,authorization
    from src.api.correction_worker import SelectedCorrectionPose,_StrictPoseModel,_crop_options
    from src.api.result_document import ResultDocument
    from src.scoring import corrected_publication as publisher
    fixture, descriptor = ready(TEMP)
    _, manifest, _ = publisher._prepared(descriptor)
    rows = []
    for predictions in (0,2):
        provider = SelectedCorrectionPose(SimpleNamespace(snapshot=fixture.snapshot),TEMP)
        def current():
            if provider.closed: raise ValueError('Explicit closed software metadata provider')
        provider._current = current
        provider.calls = 1
        provider.checkpoint = Path('EXPLICIT_SYNTHETIC_MODEL_IDENTITY_ONLY.pt')
        provider.checkpoint_sha = 'c'*64
        provider.options = _crop_options({'imgsz':640})
        provider.runtime = {'scope':'EXPLICIT_SYNTHETIC_COUNTER_RECEIPT_NOT_MODEL_EXECUTION'}
        provider.architecture = {'scope':'EXPLICIT_SYNTHETIC_COUNTER_RECEIPT_NOT_MODEL_EXECUTION'}
        forwarded = []
        proxy = _StrictPoseModel(SimpleNamespace(predict=lambda image,**kwargs: forwarded.append(image)),
                                 provider.options,current)
        provider.proxy = proxy
        for index in range(predictions): proxy.predict(index)
        record = provider.record()
        provider.close()
        # The receipt is plain JSON and survives producer close. Preserve zero
        # forwards and its distinct, real provider invocation count exactly.
        admitted = publisher._execution(descriptor,manifest,record)
        assert admitted == record and admitted is not record
        assert admitted['pose_provider_calls'] == 1
        assert admitted['pose_prediction_calls'] == len(forwarded) == predictions
        assert admitted['new_vision_inference'] is (predictions>0)
        assert admitted['input_spec']['tracking_policy'] == 'DISABLED_BOUND_SOURCE_CROPS'
        assert admitted['input_spec']['imgsz'] == 640 and admitted['input_spec']['conf'] == .25
        assert proxy.model is None
        rows.append({'actual_software_proxy_predictions':len(forwarded),
            'provider_calls':admitted['pose_provider_calls'],'new_vision_inference':admitted['new_vision_inference'],
            'record_survived_provider_close':True,'publisher_schema_preserved':True})
    interruptions = []
    native_publish = publisher.publish_result
    for exception_class in (KeyboardInterrupt,SystemExit):
        directory = TEMP/exception_class.__name__
        shutil.copytree(fixture.directory,directory)
        committed = []
        def interrupted(*args,**kwargs):
            token = native_publish(*args,**kwargs)
            committed.append(token)
            raise exception_class('EXPLICIT_SYNTHETIC_POST_CAS_INTERRUPT')
        publisher.publish_result = interrupted
        caught,returned = None,None
        try:
            returned = publisher.publish_corrected_result(descriptor,directory,
                authorization=authorization(fixture,descriptor))
        except BaseException as error:
            caught = type(error).__name__
        finally:
            publisher.publish_result = native_publish
        with ResultDocument(directory,'fixture') as document:
            assert document.generation == committed[0]
            assert document.value['analysisRevision'] == 1
        interruptions.append({'interrupt':exception_class.__name__,'caught_interrupt':caught,
            'returned_generation':returned,'actual_committed_generation':committed[0],
            'interrupt_suppressed':caught is None and returned==committed[0],
            'committed_result_readable':True})
finally:
    builtins.__import__ = original_import
after = pins()
receipt = {'scope':'ACTUAL_WORKER_RECORD_PUBLISHER_SCHEMA_SOFTWARE_COUNTER_CLOSURE_AND_NATIVE_POST_CAS_INTERRUPT_PROBE; NO_GPU_INFERENCE',
    'counter_observations':rows,'interrupt_observations':interruptions,
    'source_before':before,'source_after':after,'source_unchanged':before==after,
    'forbidden_import_attempts':attempts,'torch_imported':'torch' in sys.modules,
    'ultralytics_imported':'ultralytics' in sys.modules,'seconds':time.perf_counter()-started}
(ART/'execution_interrupt_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'source_unchanged':before==after,'counter_observations':rows,
                  'interrupt_observations':interruptions,'seconds':receipt['seconds']}))
raise SystemExit(0 if before==after and not attempts else 1)

import builtins, hashlib, json, os, sys, time
from pathlib import Path
root=Path.cwd()
output=root/'artifacts/validation/yolo26_execution_2026-10-18/corrected_shot_sources_checks_v2'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['TENNIS_STORAGE_ROOT']=str(output/'app-storage')
os.environ['TEMP']=os.environ['TMP']=str(output/'temp')
(output/'temp').mkdir(exist_ok=True)
sys.path.insert(0,str(root))
paths=['src/scoring/corrected_shot_sources.py','tests/test_corrected_shot_sources.py','src/scoring/correction_sources.py','src/scoring/corrected_rule_rebuild.py','src/scoring/point_timeline.py','src/scoring/rule_replay_source.py','src/scoring/scoring_engine.py','src/utils/event_analysis_sequence.py','src/utils/bbox_utils.py','src/utils/point_analysis_sequence.py','src/api/result_storage.py','src/api/analysis_generation.py','tests/test_corrected_rule_rebuild.py','tests/test_correction_sources.py']
def pins(): return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
before=pins(); attempts=[]; old_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics'}:
        attempts.append(name)
        raise AssertionError('Forbidden model import: '+name)
    return old_import(name,*args,**kwargs)
builtins.__import__=guarded
import pytest
start=time.perf_counter()
code=pytest.main(['--noconftest','-p','no:cacheprovider','tests/test_corrected_shot_sources.py','-q','--basetemp',str(output/'pytest-temp'),'--junitxml',str(output/'junit.xml')])
elapsed=time.perf_counter()-start
after=pins()
module=sys.modules.get('tests.test_corrected_shot_sources')
receipt={'status':'PASS' if code==0 and before==after and not attempts else 'FAIL','exit_code':code,'elapsed_seconds':elapsed,'source_pins_before':before,'source_pins_after':after,'pins_unchanged':before==after,'forbidden_import_attempts':attempts,'loaded_models':sorted(name for name in ('torch','ultralytics') if name in sys.modules),'capacity':getattr(module,'CAPACITY_OBSERVATIONS',[]),'scope':'SYNTHETIC_NATIVE_SOFTWARE_ONLY; NO_INFERENCE_OR_ACCURACY_QUALIFICATION','published':False}
(output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
raise SystemExit(code if before==after else 3)

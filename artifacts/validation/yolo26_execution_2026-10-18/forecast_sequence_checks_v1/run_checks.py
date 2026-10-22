import builtins,hashlib,json,os,sys,time
from pathlib import Path
root=Path.cwd(); output=root/'artifacts/validation/yolo26_execution_2026-10-18/forecast_sequence_checks_v1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['TENNIS_STORAGE_ROOT']=str(output/'app-storage')
os.environ['TEMP']=os.environ['TMP']=str(output/'temp'); (output/'temp').mkdir(exist_ok=True)
sys.path.insert(0,str(root))
paths=['src/utils/forecast_sequence.py','tests/test_forecast_sequence.py']
def pins(): return {path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}
before=pins(); attempts=[]; original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics','numpy','cv2','pandas'}:
        attempts.append(name); raise AssertionError('Forbidden heavy import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import pytest
start=time.perf_counter()
code=pytest.main(['--noconftest','-p','no:cacheprovider','tests/test_forecast_sequence.py','-q','--basetemp',str(output/'pytest-temp'),'--junitxml',str(output/'junit.xml')])
after=pins(); module=sys.modules.get('tests.test_forecast_sequence')
value={'status':'PASS' if code==0 and before==after and not attempts else 'FAIL','exit_code':code,'elapsed_seconds':time.perf_counter()-start,'source_pins_before':before,'source_pins_after':after,'pins_unchanged':before==after,'forbidden_import_attempts':attempts,'loaded_heavy_runtimes':[name for name in ('torch','ultralytics','numpy','cv2','pandas') if name in sys.modules],'capacity':getattr(module,'CAPACITY_OBSERVATIONS',[]),'scope':'SYNTHETIC_SOFTWARE_STORAGE_ONLY; NO_MODEL_OR_ACCURACY_QUALIFICATION','published':False}
(output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,indent=2)); raise SystemExit(code if before==after else 3)

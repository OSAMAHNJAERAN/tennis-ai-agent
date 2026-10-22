import os,sys,hashlib,json,tempfile,time,importlib.abc
from pathlib import Path
root=Path.cwd();out=root/'artifacts/validation/yolo26_execution_2026-10-16/court175_export_checks_v1';tempfile.tempdir=str(out/'temp');os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';sys.path.insert(0,str(root))
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'PIL','cv2','numpy','torch','ultralytics'}:
            blocked.append(fullname);raise ImportError('Forbidden heavy import:'+fullname)
sys.meta_path.insert(0,Guard());import pytest
sources=['scripts/data/export_yolo26_court_repair_v2.py','scripts/data/consolidate_yolo26_court_source_wave.py','tests/test_yolo26_court_repair_export_v2.py'];digest=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest();before={p:digest(p) for p in sources};start=time.perf_counter()
status=pytest.main(['--noconftest','-q','tests/test_yolo26_court_repair_export_v2.py','tests/test_yolo26_court_source_wave_consolidation.py','--basetemp',str(out/'temp'/'pytest'),f'--junitxml={out/"junit.xml"}'])
receipt={'status':'PASS' if status==0 else 'FAIL','exit_code':int(status),'elapsed_seconds':time.perf_counter()-start,'source_hashes_before':before,'source_hashes_after':{p:digest(p) for p in sources},'forbidden_heavy_import_attempts':blocked,'qualification_evidence':False,'training_performed':False,'actual_images_decoded':False}
with (out/'test_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt));raise SystemExit(status)

"""Frozen owner admission checks: no runtime imports or actual host probes."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import builtins,hashlib,importlib.abc,json,os,shutil,sys,time,traceback
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[4];ART=Path(__file__).resolve().parent
TESTS=['tests/test_resource_quotas.py', 'tests/test_owner_quota_api.py', 'tests/test_product_api.py', 'tests/test_product_ownership_uploads.py']
FORBIDDEN={'torch','ultralytics','torchvision'}
def pins():
    paths=set((ROOT/'src').rglob('*.py'))|set((ROOT/'tests').rglob('*.py'))
    paths.update([ROOT/'pytest.ini',Path(__file__).resolve(),ROOT/'scripts/train/train_yolo26_author_ball_repair.py'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def main():
    if {p.name for p in ART.iterdir()}!={'run_checks.py'}:raise RuntimeError('Fresh output required')
    before=pins();temp=ART/'temp';temp.mkdir();os.chdir(ROOT)
    sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
    os.environ.update(TEMP=str(temp),TMP=str(temp),TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1')
    for name in before:
        target=ART/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    attempts=[];native=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.',1)[0] in FORBIDDEN:
            attempts.append(name);raise AssertionError('Forbidden inference import: '+name)
        return native(name,*args,**kwargs)
    class Tripwire(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.',1)[0] in FORBIDDEN:
                attempts.append(fullname);raise AssertionError('Forbidden inference import: '+fullname)
    tripwire=Tripwire();sys.meta_path.insert(0,tripwire);builtins.__import__=guarded
    calls=[];code=2;fatal=None;clock=time.perf_counter()
    args=['-q','--noconftest','-p','no:cacheprovider','--tb=short','--basetemp',str(temp/'pytest'),'--junitxml',str(ART/'junit.xml'),*TESTS]
    started={'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_before':before,'tests':TESTS,'pytest_arguments':args,'resource_boundary':'Synthetic constant resource_snapshot; actual host/GPU admission remains unreleased by human'}
    (ART/'started.json').write_text(json.dumps(started,indent=2),encoding='utf-8')
    try:
        with (ART/'pytest.txt').open('x',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            from scripts.train import train_yolo26_author_ball_repair as resources
            def synthetic_snapshot(directory,torch=None):
                assert torch is None
                calls.append('synthetic_cpu_snapshot')
                return {'ram':8*1024**3,'disk':8*1024**3,'rss':0,'peak_rss':0,'vram':None}
            resources.resource_snapshot=synthetic_snapshot
            import pytest
            code=int(pytest.main(args))
    except BaseException:fatal=traceback.format_exc()
    finally:builtins.__import__=native;sys.meta_path.remove(tripwire)
    after=pins();counts=None
    if (ART/'junit.xml').is_file():
        suites=list(ET.parse(ART/'junit.xml').getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    loaded=sorted(FORBIDDEN&set(sys.modules))
    passed=code==0 and fatal is None and before==after and not attempts and not loaded and counts and counts['tests'] and not any(counts[k] for k in ('errors','failures','skipped'))
    receipt={'status':'PASS' if passed else 'FAIL','exit_code':code,'fatal':fatal,'counts':counts,'elapsed_seconds':time.perf_counter()-clock,'source_before':before,'source_after':after,'source_stable':before==after,'inference_import_attempts':attempts,'forbidden_runtimes_loaded':loaded,'synthetic_resource_calls':len(calls),'actual_host_ram_gpu_probes':0,'actual_model_inference':False,'tests':TESTS}
    (ART/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in {'source_before','source_after'}},indent=2))
    if not passed:print((ART/'pytest.txt').read_text(encoding='utf-8')[-18000:]);print(fatal or '')
    return 0 if passed else code or 2
if __name__=='__main__':raise SystemExit(main())

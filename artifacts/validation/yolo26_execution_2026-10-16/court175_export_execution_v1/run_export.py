import sys,json,time,importlib.abc,importlib.util
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));OUT=ROOT/'artifacts/validation/yolo26_execution_2026-10-16/court175_export_execution_v1';RELEASE=ROOT/'artifacts/validation/yolo26_execution_2026-10-16/court175_export_release_v2'
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'PIL','cv2','numpy','torch','ultralytics'}:
   blocked.append(fullname);raise ImportError('Forbidden media/model import:'+fullname)
sys.meta_path.insert(0,Guard())
from scripts.data import export_yolo26_court_repair_v2 as module
spec=importlib.util.spec_from_file_location('resource_meter',ROOT/'scripts/data/acquire_yolo26_court_source_wave.py');meter=importlib.util.module_from_spec(spec);spec.loader.exec_module(meter)
start=time.perf_counter();before=meter.current_resources(OUT);expected_source='4094ad486825eb73b415b0305029adf04064ae374ce6b0908ec97df13ef5d345'
if module.digest(Path(module.__file__))!=expected_source:raise ValueError('Root-reviewed exporter changed')
plan=module.prepare_plan();saved=module.read_json(RELEASE/'preflight_plan.json')
if saved!=plan or module.canonical_hash(plan)!='4edd0c0d443c7e203c875f1cda980e908b07080cb7aae46767c99dec44b91c06':raise ValueError('Root-reviewed final plan changed')
approval=module.read_json(RELEASE/'approval_template.json');approval.update(status=module.APPROVED,reviewer='ROOT_AGENT',root_review_note='Root reviewed the complete standalone exporter, tests and exact final175-source preflight. Root checked all229 prior/new source-context reconstruction including rejected links; all four publisher doubles corners define new training-only bbox extent estimates without converting hidden coordinates into visible measurement ground truth; old120 labels reconstruct exactly, prior25 unused development images remain excluded,2421 visible29 null slots and zero TEST. Original FPS/PTS/match/camera-session remain unknown. Permitted scope is private noncommercial academic development, with raw broadcast redistribution uncleared. Root actual visual review attribution is preserved from the source review documents; this approval does not claim independent human review or model qualification and authorizes dataset export only.')
module.validate_approval(approval,plan);approval_path=OUT/'root_approval.json';module.write_json(approval_path,approval);approval_sha=module.digest(approval_path)
result=module.export_dataset(module.DEFAULT_OUTPUT,approval_path,approval_sha)
module.write_json(OUT/'post_export_verification.json',result)
receipt={'status':'COMPLETE_ROOT_AUTHORIZED_NATIVE_COURT175_EXPORT_AND_RECONSTRUCTION_VERIFY_NOT_QUALIFICATION','elapsed_seconds':time.perf_counter()-start,'exporter_sha256':expected_source,'canonical_plan_sha256':module.canonical_hash(plan),'root_approval_path':str(approval_path),'root_approval_sha256':approval_sha,'result':result,'export_receipt_sha256':module.digest(module.DEFAULT_OUTPUT/'export_receipt.json'),'dataset_yaml_sha256':module.digest(module.DEFAULT_OUTPUT/'court/data.yaml'),'frozen_plan_file_sha256':module.digest(module.DEFAULT_OUTPUT/'frozen_export_plan.json'),'resources_before':before,'resources_after':meter.current_resources(OUT),'forbidden_import_attempts':blocked,'training_performed':False,'gpu_used':False,'qualification_evidence':False}
module.write_json(OUT/'receipt.json',receipt);print(json.dumps(receipt,indent=2))

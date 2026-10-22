import sys,importlib.abc,json,hashlib,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root));out=root/'artifacts/validation/yolo26_execution_2026-10-16/court175_export_release_v2'
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'PIL','cv2','numpy','torch','ultralytics'}:
   blocked.append(fullname);raise ImportError('Forbidden media/model import:'+fullname)
sys.meta_path.insert(0,Guard())
from scripts.data import export_yolo26_court_repair_v2 as module
start=time.perf_counter();plan=module.prepare_plan();module.write_json(out/'preflight_plan.json',plan);module.write_json(out/'approval_template.json',module.approval_template(plan));resources=module.guarded_copy_resources(Path(plan['output']),plan['native_image_bytes'])
receipt={'status':'PASS_REAL175_SOURCE_RECONSTRUCTION_AND_NATIVE_LABEL_PREFLIGHT_AWAITING_ROOT_APPROVAL','elapsed_seconds':time.perf_counter()-start,'canonical_plan_sha256':module.canonical_hash(plan),'preflight_file_sha256':module.digest(out/'preflight_plan.json'),'approval_template_sha256':module.digest(out/'approval_template.json'),'exporter_sha256':module.digest(Path(module.__file__)),'counts':plan['counts'],'keypoint_slots':plan['keypoint_slots'],'input_bindings_verified':len(plan['input_bindings']),'native_image_bytes':plan['native_image_bytes'],'resources_for_metadata_byte_copy':resources,'forbidden_import_attempts':blocked,'actual_output_exists':Path(plan['output']).exists(),'dataset_exported':False,'training_performed':False,'qualification_evidence':False}
module.write_json(out/'receipt.json',receipt);print(json.dumps(receipt,indent=2))

from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
root=Path.cwd(); output=root/'artifacts/validation/yolo26_execution_2026-10-18/corrected_forecast_next_seam_v1'
paths=['src/prediction/tennis.py','src/prediction/evaluation.py','src/scoring/match_state.py','src/scoring/match_state_codec.py','src/scoring/point_timeline.py','src/scoring/scoring_engine.py','src/utils/point_analysis_sequence.py','src/utils/scoring_journal.py','src/api/result_storage.py','src/utils/json_array_artifact.py','src/utils/point_timeline_artifact.py','src/api/analysis_generation.py','src/scoring/corrected_rule_rebuild.py','src/scoring/corrected_shot_rebuild.py','src/api/product.py','frontend/src/data/productApi.ts','frontend/src/components/WinProbability.tsx','tests/test_scoring_prediction.py','tests/test_prediction_evaluation.py','tests/test_corrected_shot_rebuild.py','scripts/evaluate/evaluate_publisher_snapshots.py','configs/production/yolo26.yaml']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={path:digest(root/path) for path in paths}
anchors={}
for path in paths:
    anchors[path]=[{'line':index,'symbol':line.strip()} for index,line in enumerate((root/path).read_text(encoding='utf-8-sig').splitlines(),1) if line.startswith(('def ','class ','    def '))]
receipts={}
for name in ['livetennisapi_2026_06_protocol_v3.json','livetennisapi_2026_06_snapshot_import_v3_development_fit.json','livetennisapi_2026_06_snapshot_import_v3_calibration_evaluation.json','livetennisapi_2026_06_snapshot_import_v3_test_evaluation.json']:
    path=root/'data/prediction'/name
    actual=digest(path); expected=path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
    if actual!=expected:raise ValueError('Stored prediction receipt hash differs: '+name)
    receipts[str(path.relative_to(root))]={'sha256':actual,'sidecar_sha256':digest(path.with_suffix(path.suffix+'.sha256')),'sidecar_verified':True}
fit=json.loads((root/'data/prediction/livetennisapi_2026_06_snapshot_import_v3_development_fit.json').read_text())
protocol=json.loads((root/'data/prediction/livetennisapi_2026_06_protocol_v3.json').read_text())
after={path:digest(root/path) for path in paths}
if before!=after:raise ValueError('Reviewed sources changed during read-only mapping')
value={'schema_version':1,'status':'READ_ONLY_IMPLEMENTATION_MAPPING_RETAINED','created_at':datetime.now(timezone.utc).isoformat(),'implementation_complete':False,'independent_accuracy_validated':False,'source_edits':False,'model_inference':False,'published':False,'source_pins_before':before,'source_pins_after':after,'pins_unchanged':True,'reviewed_symbol_lines':anchors,'retained_prediction_artifacts':receipts,'available_fit':{'parameters':fit['parameters'],'status':fit['status'],'calibration_status':fit['calibration_status'],'match_count':fit['match_count'],'snapshot_count':fit['snapshot_count'],'can_qualify_posterior_updating_video_forecasts':False,'protocol_raw_model':protocol['raw_model'],'protocol_scope':protocol['scope'],'publication_qualification':protocol['publication_qualification']},'mapping_sha256':digest(output/'mapping.md'),'mapping_path':str(output/'mapping.md'),'next_action':'Implement fresh corrected causal forecasts with exact raw-state accepted-rule replay, SQLite dedup/history ownership, explicit uncalibrated neutral prior and coupled double-fault/prefix-causality tests.'}
(output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':value['status'],'reviewed_source_count':len(paths),'sidecars_verified':len(receipts),'mapping_sha256':value['mapping_sha256'],'receipt_sha256':digest(output/'receipt.json'),'mapping_path':value['mapping_path']},indent=2))

"""Static AST/hash preparation only; never import or execute the replay helper."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
unit=Path(__file__).resolve().parent
evidence=unit.parent
root=evidence.parents[3]
script=unit/'verify_replay.py'
baseline=evidence/'src/events/event_detector.py'
source=root/'src/events/event_detector.py'
report_path=root/'artifacts/validation/phase6_4_event_recovery_results.json'
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()
def snapshot(paths):
    return {path.relative_to(root).as_posix():{'sha256':digest(path),'bytes':path.stat().st_size}
            for path in sorted(set(paths))}
def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
assert not (unit/'preparation.json').exists() and not (unit/'launch_marker.json').exists() and not (unit/'run').exists()
assert digest(source)=='ccc00b450aa144d59a5fff13ede8a675749dc62fdc500143d3c114af715eff63'
assert digest(baseline)=='078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
assert digest(report_path)=='e3d607adb5cb3bd5b35f7d9077e8ad28f48f094002fc37d097a69fbc86ae1b0e'
assert digest(unit/'preserved/tests/test_phase6_4_recovery_artifacts.py')=='29c5bc953be0e02f23fc880d7432a7cefb4fb49d3b29a2e7e4dbd43606a1ad11'
report=read(report_path)
provenance=report['input_provenance']
historical=[report_path,unit/'preserved_integrity_test.json',unit/'preserved/tests/test_phase6_4_recovery_artifacts.py']
for label,record in provenance.items():
    records=[item for video in record.values() for item in video.values()] if label=='preserved_video_inputs' else [record]
    for item in records:
        path=baseline if label=='event_detector_implementation' else root/item['path']
        assert path.is_file() and digest(path)==item['sha256'] and path.stat().st_size==item['size_bytes'],item['path']
        historical.append(path)
assert set(provenance['preserved_video_inputs'])=={'video_08','video_09','video_10'}
assert report['scientific_split']=='CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED' and report['qualification_evidence'] is False
assert report['readiness_gate']['passed'] is False
assert len(report['ablations'])==8
files=list((root/'src').rglob('*.py'))+[script,baseline,
    root/'scripts/evaluate_phase6_4_event_recovery.py',root/'scripts/audit/measure_candidate_storage.py',
    root/'scripts/evaluate_phase6_4_cross_match.py',root/'configs/phase6_analytics/pipeline.yaml',
    Path(__file__),unit/'preserved_integrity_test.json']
historical += [evidence/'backend'/name for name in ('receipt.json','log.txt','junit.xml','source_bindings.json')]
text=script.read_text(encoding='utf-8')
ast.parse(text,filename=str(script))
for fragment in [
    "evaluator._load_video_inputs(", "evaluator._variant_configs(base_config)",
    "frozen.TennisEventDetector(config=dict(settings)).analyze",
    "evaluator.TennisEventDetector(config=dict(settings)).analyze",
    "len(report['comparisons']) != 21", "historicalHFilteredEventsEqual",
    "ordinaryImportedModuleBindings", "'torch', 'ultralytics', 'cv2', 'onnxruntime'",
    "owned.observed_workload(identity)", "owned.wait_for_retirement()", "monitor.stop_owned(process, owned)",
    "return 1 if code == 0 and not read(UNIT / 'replay_receipt.json')['passed'] else code",
]:
    assert fragment in text,fragment
assert 'evaluator.main(' not in text
preparation={'preparedAt':datetime.now(timezone.utc).isoformat(),
    'status':'PREPARED_NOT_EXECUTED','helperImported':False,'replayRun':False,'inferenceRun':False,
    'script':script.relative_to(root).as_posix(),'scriptSha256':digest(script),
    'launchCommand':'C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe '+script.relative_to(root).as_posix()+' --run',
    'sourceBindings':snapshot(files),'historicalArtifacts':snapshot(historical),
    'scope':'Exact saved-diagnostic analyze compatibility only. No media/models, accuracy or full-match-capacity qualification.',
    'workload':{'variants':['B_HIGH_RECALL_CANDIDATES','C_TEMPORAL_PLAYER_PROXIMITY','D_EVENT_TIME_REFINEMENT',
        'E_EVENT_SPECIFIC_VERIFICATION','F_VISION_DEAD_BALL_GATING','G_CAMERA_MOTION_COMPENSATION','H_FINAL_INTEGRATED'],
        'videoIds':['video_08','video_09','video_10'],'pairComparisons':21,
        'frameCounts':{'video_08':469,'video_09':408,'video_10':1795},'totalSavedFrames':2672,
        'call':'Actual evaluator loader/variant constructors and independent current/frozen analyze instances. No GT passed to analyze.',
        'cameraOffsets':None,'poseSupport':None,
        'notRecomputed':'A_INTEGRATED_PRE_RECOVERY is a retained historical pipeline before-state, not one of evaluator B-H rerunnable settings.'},
    'comparisons':'Every full raw candidate/event/verification-trace/settings/coordinate field, byte-identical default-float JSON with enum values and preserved diagnostic nonfinite behavior. H additionally matches original coverage-filtered saved events; no metric/GT rescoring or historical output rewriting.',
    'modelNoneProof':'Reject torch/ultralytics/cv2/onnxruntime imports and network attempts; hash all ordinary imported module files after actual replay. NumPy/Pandas/YAML imports only for numeric event code.',
    'guards':{'wallSeconds':180,'ownedRssBytes':1000000000,'preflightAvailableRamBytes':800000000,
        'minimumAvailableRamBytes':400000000,'lowRamConsecutiveSamples':5,'sampleSeconds':.25},
    'ownedMonitor':'Exact PID/createTime sample-before-import handshake and remembered-identity retirement with deepest-first scoped termination/Popen fallback. One exclusive attempt; all failures retain outputs.',
    'productionWrites':False,'historicalReportsWritten':False,'rootReviewRequiredBeforeRun':True,
    'expectedReceiptFields':['status','passed','comparisons','sourceBindingsUnchanged',
        'historicalArtifactsUnchanged','preparationUnchanged','actualWorkloadSampledByExactIdentity',
        'allExactOwnedIdentitiesRetiredAtTerminal']}
save(unit/'preparation.json',preparation)
print(json.dumps({'status':preparation['status'],'scriptSha256':digest(script),
                  'preparationSha256':digest(unit/'preparation.json'),'helperImported':False,'replayRun':False}),flush=True)


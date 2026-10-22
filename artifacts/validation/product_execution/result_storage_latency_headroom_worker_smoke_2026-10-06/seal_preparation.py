"""One-shot static preparation seal: stdlib AST/hash reads only, no helper import."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
SCRIPT = ROOT / 'scripts/audit/verify_result_storage_latency_headroom_worker_smoke.py'
FIRST = ROOT / 'scripts/audit/verify_result_storage_latency_worker_smoke.py'
FIRST_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_worker_smoke_2026-10-06'
ORIGIN = ROOT / 'scripts/audit/verify_result_storage_worker_smoke.py'
ORIGIN_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_baseline_2026-10-06'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
GUARD = ROOT / 'scripts/audit/verify_candidate_storage_pipeline_smoke.py'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def snapshot(paths):
    return {path.relative_to(ROOT).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in sorted(set(paths))}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


source = SCRIPT.read_text(encoding='utf-8')
tree = ast.parse(source, filename=str(SCRIPT))
assert digest(FIRST) == '279464db5328b87b0a524b9f669176f9498b609522718ddc4b2cab7873d7fefe'
assert digest(ORIGIN) == '8874064cac5a27ae70a7b0784d5f8eca17b1388f1e80665ab304cd1cab112b82'
assert digest(ORIGIN_UNIT / 'preparation.json') == '63791ac2a3b64c1fe3fcfe6ef62081348d41bea79e9ca68fc42a7b6197c43dfa'
assert digest(ORIGIN_UNIT / 'observer_revision.json') == '135c77b9330213e23f36699d9a367c4b2431ff3fa086820bd6bdbcca1334f434'
first = read(FIRST_UNIT / 'resource_receipt.json')
first_protocol = read(FIRST_UNIT / 'protocol.json')
assert first['status'] == 'NOT_STARTED' and first['passed'] is False
assert first['inferenceChildStarted'] is False and first['samples'] == 0
assert first['minimumSampledAvailableRamBytes'] == 2355306496
assert first['minimumSampledAvailableRamBytes'] < 2500000000
assert first_protocol['scriptSha256'] == digest(FIRST)
assert 'sourceBindingsBefore' not in first_protocol
assert not (FIRST_UNIT / 'run').exists()
for name in ('actual_child_identity.json', 'actual_child_sampled.json', 'child_receipt.json',
             'running_process.json', 'resource_samples.jsonl', 'child_stdout.txt'):
    assert not (FIRST_UNIT / name).exists(), name
assert set(path.name for path in FIRST_UNIT.iterdir()) == {
    'origin.json', 'preparation.json', 'seal_preparation.py', 'launch_marker.json',
    'protocol.json', 'resource_receipt.json'}
assert not any((UNIT / name).exists() for name in
               ('origin.json', 'preparation.json', 'static_binding.json', 'launch_marker.json', 'run'))
required = [
    "UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_headroom_worker_smoke_2026-10-06'",
    "preparation['scriptSha256'] != digest(__file__)",
    "verify_first_attempt() != strict_json(UNIT / 'origin.json')['firstLatencyAttemptBindings']",
    "protocol['firstLatencyAttemptHistoryBefore'] = verify_first_attempt()",
    "('firstLatencyAttemptHistory', 'firstLatencyAttemptHistoryBefore', verify_first_attempt)",
    "protocol['liveProductFileTreeBefore'] = live_product_tree(guard)",
    "('liveProductFileTree', 'liveProductFileTreeBefore', lambda: live_product_tree(guard))",
    "('productionJobStates', 'productionJobStatesBefore', guard.product_jobs)",
    "('originHistoricalArtifacts', 'originHistoricalArtifactsBefore', lambda: origin_history(guard))",
    "constructionObserverStoppedBeforeInference", "sys.setprofile(previous_profile)",
    "process_job(store, claimed, settings)",
    "report['remainingOwnedIdentitiesAfterNormalExit'] = owned.wait_for_retirement()",
    "monitor.stop_owned(process, owned)",
    "argv = [str(CLEAN), str(Path(__file__).resolve()), '--child', '--launch-token', token]",
    "run_child(args.launch_token)",
]
assert all(item in source for item in required)
assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and isinstance(node.value, (ast.Constant, ast.UnaryOp))}
expected_guards = {'SAMPLE_SECONDS': .25, 'PREFLIGHT_AVAILABLE_BYTES': 2500000000,
    'MAXIMUM_OWNED_RSS_BYTES': 3000000000, 'MINIMUM_AVAILABLE_BYTES': 400000000,
    'LOW_RAM_SAMPLES': 5, 'MAXIMUM_WALL_SECONDS': 300}
for name, value in expected_guards.items():
    assert assignments[name] == value
assert source.count('psutil.virtual_memory().available < PREFLIGHT_AVAILABLE_BYTES') == 1
assert "report['availableRamImmediatelyBeforeLaunchBytes'] < PREFLIGHT_AVAILABLE_BYTES" in source
assert digest(ROOT / 'src/api/results.py') == '01d8fb7d04edb0c0d5d980d0c79f71b54c8cc3acea8200c9a486d2b2952f02a2'
assert digest(ROOT / 'src/api/result_storage.py') == 'ac2dd55e91435562cf031aae47b0107cab13eb6a57f2f1acb62405817f97b343'
assert digest(MONITOR) == '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'
now = datetime.now(timezone.utc).isoformat()
first_bindings = snapshot([FIRST] + [path for path in FIRST_UNIT.rglob('*') if path.is_file()])
origin_receipt = {'recordedAt': now, 'helper': ORIGIN.relative_to(ROOT).as_posix(),
    'helperSha256': digest(ORIGIN), 'historicalDirectory': ORIGIN_UNIT.relative_to(ROOT).as_posix(),
    'historicalPreparationSha256': digest(ORIGIN_UNIT / 'preparation.json'),
    'historicalObserverRevisionSha256': digest(ORIGIN_UNIT / 'observer_revision.json'),
    'newHelper': SCRIPT.relative_to(ROOT).as_posix(),
    'reuse': 'Clone of the exact consumed latency helper, preserving original worker/input/normalization/reporting checks. Fresh unit, initial hash gate, no historical observer revision rewrite.',
    'firstLatencyAttemptBindings': first_bindings,
    'firstLatencyAttemptStatus': 'NOT_STARTED', 'firstLatencyAttemptInferenceChildStarted': False,
    'firstLatencyAttemptAvailableRamBytes': 2355306496,
    'retryPolicy': 'Explicit root-reviewed fresh launch only after other workloads finish and independently observed headroom recovers. Both unchanged 2.5 GB preflight checks run again; no automatic retry.',
    'status': 'PREPARED_NOT_EXECUTED', 'inferenceRun': False}
save(UNIT / 'origin.json', origin_receipt)
preparation = {'preparedAt': now, 'status': 'PREPARED_NOT_EXECUTED',
    'scope': 'One actual 30-frame, native-60-FPS worker/model and saved-result compatibility smoke. No independent labels, accuracy, full-match capacity or deployment qualification.',
    'script': SCRIPT.relative_to(ROOT).as_posix(), 'scriptSha256': digest(SCRIPT),
    'originReceiptSha256': digest(UNIT / 'origin.json'), 'sourceAstParsed': True, 'helperImported': False,
    'inferenceRun': False, 'workerInvoked': False, 'rootReviewRequiredBeforeRun': True,
    'launchCommand': 'C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe scripts\\audit\\verify_result_storage_latency_headroom_worker_smoke.py --run',
    'frames': 30, 'nativeFps': 60, 'originalSourceFrames': [390, 420],
    'inputSha256': '12d8eaf3f668738d3082849db92f731a3d1c3bb8124c16ab8e11e80c0c182552',
    'preflightAvailableRamBytes': 2500000000, 'maximumSampledOwnedTreeRssBytes': 3000000000,
    'minimumAvailableRamBytes': 400000000, 'lowRamConsecutiveSamplesToStop': 5,
    'sampleIntervalSeconds': .25, 'maximumChildWallSeconds': 300,
    'firstAttempt': {'status': 'NOT_STARTED', 'inferenceChildStarted': False,
                     'availableRamBytes': 2355306496, 'noRunDirectory': True},
    'headroomObservation': {
        'authority': 'Root message in this task; independently observed read-only 12-second probe. No probe or workload performed by this preparation.',
        'reportedAvailableRamGB': [2.27, 2.29, 2.27, 2.6645, 2.7498],
        'lastTwoAboveUnchangedPreflightThreshold': True, 'unrelatedAppsChangedOrStopped': False,
        'launchCondition': 'Root explicitly dispatches only after other workloads finish; parent independently samples actual available RAM before bindings and immediately before launching the child, requiring >=2.5 GB each time. Recorded probe observations are not a substitute for either check.'},
    'productionWrites': False, 'isolatedValidationJobPublication': True,
    'comparison': 'Frozen original resident results.py/reporting.py use byte-identical new raw artifacts and original claimed job; no volatile values removed; exact strict JSON/CSV/HTML bytes before presentation.',
    'normalizationDimensions': 'Actual native dimensions, scale1 and padding. Original input pixels verified; CRF18 normalized worker pixels independently decoded, not claimed lossless.',
    'liveProductChecks': 'Exactly ten live job states and complete live file/directory tree including SQLite sidecars before/after, read-only. Current API worker poll is active; any persistent file change fails the receipt rather than being ignored.',
    'currentRuntimeReview': {'receipt': 'artifacts/validation/product_execution/result_storage_latency_runtime_verification_2026-10-06.json',
        'passed': True, 'savedJobsAndAnalysisFilesUnchanged': True, 'liveJobs': 10,
        'savedFiles': 319, 'catalogDemos': 3, 'demoExports': 12, 'frontendAssets': 7},
    'historicalChecks': 'Both prior helpers and entire units, original resident baseline and retained media are preserved and bind before/after. firstLatencyAttemptHistoryUnchanged is separately required in the terminal receipt.',
    'launchBindingPolicy': 'All src Python, WASB implementation, model assets/config, source/excerpt, clean interpreter, exact monitor, original/new helpers and immutable preparation/origin receipts bind at launch. Child verifies before work; parent verifies after exact process retirement.',
    'network': 'YOLO_OFFLINE=true, YOLO_AUTOINSTALL=false; socket connect/connect_ex blocked and recorded.',
    'modelEvidence': 'Actual production worker and all configured models; passive construction observer disables profiling before inference. No stubs or model replacements.',
    'ownedProcessGuard': 'Exact actual-interpreter PID/createTime handshake, sampled owned tree, deepest-first stop with Popen fallback, all identities retired on success/error.',
    'retention': 'One exclusive marker consumes this fresh unit on any launch. Preserve all partial logs/artifacts; no automatic retry, guard relaxation or unrelated cleanup.',
    'qualificationEvidence': False, 'productionQualified': False}
save(UNIT / 'preparation.json', preparation)
bindings = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
bindings += [SCRIPT, Path(__file__), MONITOR, GUARD,
    ROOT / 'scripts/audit/verify_registration_storage_pipeline_smoke.py',
    ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml',
    ROOT / 'data/demos/sources.json', ROOT / 'src/api/results.py', ROOT / 'src/api/result_storage.py',
    ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe',
    ROOT / 'models/keypoints_model.pth', ROOT / 'yolo11m.pt',
    ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar', ROOT / 'yolo11s-pose.pt',
    BASELINE / 'receipt.json', BASELINE / 'src/api/results.py', BASELINE / 'src/api/reporting.py',
    UNIT / 'origin.json', UNIT / 'preparation.json',
    ROOT / 'artifacts/validation/product_execution/result_storage_latency_runtime_verification_2026-10-06.json']
bindings += list((ROOT / 'artifacts/research/WASB-SBDT').rglob('*.py'))
current = snapshot(bindings)
save(UNIT / 'static_binding.json', {'sealedAt': now, 'status': 'PREPARED_NOT_EXECUTED',
    'sourceAstParsed': True, 'helperImported': False, 'inferenceRun': False,
    'scriptSha256': digest(SCRIPT), 'preparationSha256': digest(UNIT / 'preparation.json'),
    'originSha256': digest(UNIT / 'origin.json'), 'currentBindings': current,
    'sourceModuleCount': len(list((ROOT / 'src').rglob('*.py'))),
    'testModuleCount': len(list((ROOT / 'tests').rglob('*.py'))),
    'firstLatencyAttemptBindings': first_bindings,
    'originalWorkerHistoryBindings': snapshot([ORIGIN] + [path for path in ORIGIN_UNIT.rglob('*') if path.is_file()]),
    'guardConstantsUnchanged': expected_guards, 'guardRelaxation': False,
    'newRuntimeStateWritten': False, 'oldReceiptsOverwritten': False,
    'dispatchChecked': 'Parent spawns this same new helper path with --child; child only executes when its new exclusive token matches. All old helper/unit paths occur only in read-only binding/history verification.',
    'staticReview': required})
assert snapshot([FIRST] + [path for path in FIRST_UNIT.rglob('*') if path.is_file()]) == first_bindings
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'helperSha256': digest(SCRIPT),
                  'preparationSha256': digest(UNIT / 'preparation.json'),
                  'staticBindingSha256': digest(UNIT / 'static_binding.json'),
                  'helperImported': False, 'inferenceRun': False,
                  'firstAttemptPreserved': True}), flush=True)


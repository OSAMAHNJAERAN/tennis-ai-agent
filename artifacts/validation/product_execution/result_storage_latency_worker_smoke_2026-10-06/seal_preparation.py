"""Parse helper source and write preparation only; never import the helper."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

unit = Path(__file__).resolve().parent
root = unit.parents[3]
script = root / 'scripts/audit/verify_result_storage_latency_worker_smoke.py'
origin = root / 'scripts/audit/verify_result_storage_worker_smoke.py'
origin_unit = root / 'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


source = script.read_text(encoding='utf-8')
tree = ast.parse(source, filename=str(script))
assert digest(origin) == '8874064cac5a27ae70a7b0784d5f8eca17b1388f1e80665ab304cd1cab112b82'
assert digest(origin_unit / 'preparation.json') == '63791ac2a3b64c1fe3fcfe6ef62081348d41bea79e9ca68fc42a7b6197c43dfa'
assert digest(origin_unit / 'observer_revision.json') == '135c77b9330213e23f36699d9a367c4b2431ff3fa086820bd6bdbcca1334f434'
assert "UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_worker_smoke_2026-10-06'" in source
assert "revision = strict_json(UNIT / 'observer_revision.json')" not in source
assert "preparation['scriptSha256'] != digest(__file__)" in source
assert "protocol['liveProductFileTreeBefore'] = live_product_tree(guard)" in source
assert "protocol['originHistoricalArtifactsBefore'] = origin_history(guard)" in source
assert "('liveProductFileTree', 'liveProductFileTreeBefore', lambda: live_product_tree(guard))" in source
assert "constructionObserverStoppedBeforeInference" in source
assert "sys.setprofile(previous_profile)" in source
assert "process_job(store, claimed, settings)" in source
assert "report['remainingOwnedIdentitiesAfterNormalExit'] = owned.wait_for_retirement()" in source
assert "monitor.stop_owned(process, owned)" in source
now = datetime.now(timezone.utc).isoformat()
origin_receipt = {'recordedAt': now, 'helper': origin.relative_to(root).as_posix(),
    'helperSha256': digest(origin), 'historicalDirectory': origin_unit.relative_to(root).as_posix(),
    'historicalPreparationSha256': digest(origin_unit / 'preparation.json'),
    'historicalObserverRevisionSha256': digest(origin_unit / 'observer_revision.json'),
    'newHelper': script.relative_to(root).as_posix(),
    'reuse': 'Cloned genuine worker/input/normalization/reporting/owned-monitor checks. Fresh paths and initial source-hash gate; no fabricated observer revision.',
    'status': 'PREPARED_NOT_EXECUTED', 'inferenceRun': False}
save(unit / 'origin.json', origin_receipt)
preparation = {'preparedAt': now, 'status': 'PREPARED_NOT_EXECUTED',
    'scope': 'One actual 30-frame, native-60-FPS worker/model and saved-result compatibility smoke. No independent labels, accuracy, full-match capacity or deployment qualification.',
    'script': script.relative_to(root).as_posix(), 'scriptSha256': digest(script),
    'originReceiptSha256': digest(unit / 'origin.json'),
    'sourceAstParsed': True, 'helperImported': False,
    'inferenceRun': False, 'workerInvoked': False, 'rootReviewRequiredBeforeRun': True,
    'launchCommand': 'C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe scripts\\audit\\verify_result_storage_latency_worker_smoke.py --run',
    'frames': 30, 'nativeFps': 60, 'originalSourceFrames': [390, 420],
    'inputSha256': '12d8eaf3f668738d3082849db92f731a3d1c3bb8124c16ab8e11e80c0c182552',
    'preflightAvailableRamBytes': 2500000000, 'maximumSampledOwnedTreeRssBytes': 3000000000,
    'minimumAvailableRamBytes': 400000000, 'lowRamConsecutiveSamplesToStop': 5,
    'sampleIntervalSeconds': .25, 'maximumChildWallSeconds': 300,
    'productionWrites': False, 'isolatedValidationJobPublication': True,
    'comparison': 'Frozen original resident results.py/reporting.py use byte-identical new raw artifacts and original claimed job; no volatile values removed; exact strict JSON/CSV/HTML bytes before presentation.',
    'normalizationDimensions': 'Record actual native dimensions, scale1 and padding. Verify original input pixels; CRF18 normalized worker pixels are independently decoded, not claimed lossless.',
    'liveProductChecks': 'Exactly ten live job states and complete live file/directory tree (all bytes/hashes including SQLite sidecars) before/after; read-only. Any change fails the receipt.',
    'historicalChecks': 'Original helper, its entire historical run directory, retained original/30-frame media and original historical pipeline evidence bind before/after; no historical writes.',
    'launchBindingPolicy': 'At authorized launch, hash all src Python modules, full WASB Python implementation, configured model assets, config, retained input/source, clean interpreter, exact monitor, original/new helper and preparation/origin receipts. Child verifies bindings before actual work; parent verifies them after exact process retirement.',
    'network': 'YOLO_OFFLINE=true, YOLO_AUTOINSTALL=false; socket connect/connect_ex fail and are recorded.',
    'modelEvidence': 'Passive construction return observer records actual parameter/device counts, then disables profiling before inference. Actual completed worker and artifacts establish run completion; no model/worker replacements.',
    'ownedProcessGuard': 'Exact actual-interpreter PID/createTime handshake before heavy work; sampled owned redirector/descendants; deepest-first stop with Popen fallback; all known identities retired on success/failure.',
    'retention': 'One exclusive launch marker. Preserve all partial artifacts, logs and receipts; no retry or unrelated cleanup.',
    'qualificationEvidence': False, 'productionQualified': False}
save(unit / 'preparation.json', preparation)
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'scriptSha256': preparation['scriptSha256'],
                  'sourceAstParsed': True, 'helperImported': False, 'inferenceRun': False}), flush=True)

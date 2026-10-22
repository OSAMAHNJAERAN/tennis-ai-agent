"""Read back the tested latency revision, installation, exports and live state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request
import psutil

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

bindings = {}

def bind(path):
    path = Path(path).resolve()
    assert path.is_file()
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    bindings[name] = digest(path)

def checked_binding(path, expected):
    assert digest(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
    bind(path)

def retired(identity):
    try:
        assert psutil.Process(identity['pid']).create_time() != identity['createTimeUnixSeconds']
    except psutil.NoSuchProcess:
        pass

suite_dir = BASE / 'result_storage_latency_backend_2026-10-06'
suite = read(suite_dir / 'receipt.json')
assert suite['passed'] and suite['sourcesUnchanged']
assert suite['counts'] == {'tests': 1278, 'failures': 0, 'errors': 0, 'skipped': 0}
inventory = read(suite_dir / 'source_bindings.json')
for name, expected in inventory.items():
    assert digest(ROOT / name) == expected
    bind(ROOT / name)
for name, expected in suite['artifacts'].items():
    assert digest(suite_dir / name) == expected
    bind(suite_dir / name)
bind(suite_dir / 'receipt.json')

baseline = BASE / 'result_storage_latency_baseline_2026-10-06'
for name, expected in read(baseline / 'receipt.json')['bindings'].items():
    assert digest(baseline / name) == expected['sha256']
    bind(baseline / name)
bind(baseline / 'receipt.json')
measure = BASE / 'result_storage_latency_measurement_2026-10-06'
terminal = read(measure / 'run/report.json')
audit = read(measure / 'audit/terminal_audit.json')
assert terminal['passed'] and audit['passed'] and len(audit['comparisons']) == 2
assert [group['frames'] for group in audit['comparisons']] == [600, 18000]
assert sum(len(group['outputs']) for group in audit['comparisons']) == 10
assert all(group['fiveExactByteComparisonsPassed'] and not group['volatileFieldsExcluded']
           and all(output['exactBytesEqualNow'] for output in group['outputs'].values())
           for group in audit['comparisons'])
for name, expected in audit['retainedFileBindings'].items():
    checked_binding(measure / name, expected)
assert len(audit['runs']) == 4
benchmark_retired = []
for run in audit['runs']:
    assert run['allOwnedIdentitiesRetiredNow'] and run['guardThresholdsNeverTriggered']
    for identity in run['ownershipNow']:
        retired(identity)
        benchmark_retired.append({'pid': identity['pid'], 'createTimeUnixSeconds': identity['createTimeUnixSeconds']})
prepared = read(measure / 'prepared_source_bindings.json')
assert prepared == terminal['sourceBindingsBefore'] == terminal['sourceBindingsAfter']
for name, expected in prepared.items():
    assert digest(ROOT / name) == expected['sha256'] and (ROOT / name).stat().st_size == expected['bytes']
    bind(ROOT / name)
for path in measure.rglob('*'):
    if path.is_file():
        bind(path)

first = BASE / 'result_storage_latency_worker_smoke_2026-10-06'
declined = read(first / 'resource_receipt.json')
assert not declined['passed'] and declined['status'] == 'NOT_STARTED' and not declined['inferenceChildStarted']
assert declined['samples'] == 0 and not (first / 'run').exists()
assert not declined['productionStatePublished'] and declined['monitoredChildWallSeconds'] is None
for path in first.rglob('*'):
    if path.is_file():
        bind(path)
smoke_dir = BASE / 'result_storage_latency_headroom_worker_smoke_2026-10-06'
smoke = read(smoke_dir / 'resource_receipt.json')
for unit in (first, smoke_dir):
    preparation = read(unit / 'preparation.json')
    assert digest(ROOT / preparation['script']) == preparation['scriptSha256']
    assert digest(unit / 'origin.json') == preparation['originReceiptSha256']
    assert preparation['preflightAvailableRamBytes'] == 2500000000
    assert preparation['maximumSampledOwnedTreeRssBytes'] == 3000000000
    assert preparation['minimumAvailableRamBytes'] == 400000000
    assert preparation['lowRamConsecutiveSamplesToStop'] == 5
    assert preparation['maximumChildWallSeconds'] == 300
    bind(ROOT / preparation['script'])
for name, expected in read(smoke_dir / 'origin.json')['firstLatencyAttemptBindings'].items():
    checked_binding(ROOT / name, expected)
worker_verified = smoke['passed']
if worker_verified:
    assert smoke['allOwnedIdentitiesRetiredAfterNormalExit']
    assert smoke['productionJobStatesUnchanged'] and smoke['liveProductFileTreeUnchanged']
    assert smoke['originHistoricalArtifactsUnchanged']
    for name, expected in smoke['sourceBindingsAfter'].items():
        assert digest(ROOT / name) == expected['sha256']
        bind(ROOT / name)
    for identity in smoke['ownedProcessIdentitiesAndPeaks']:
        retired(identity)
    child = read(smoke_dir / 'child_receipt.json')
    assert child['passed'] and not child['networkAttempts']
    assert read(smoke_dir / 'saved_result_comparison.json')['passed']
else:
    assert smoke['status'] == 'NOT_STARTED' and not smoke['inferenceChildStarted']
    assert smoke['samples'] == 0 and not (smoke_dir / 'run').exists()
    assert not smoke['productionStatePublished'] and smoke['monitoredChildWallSeconds'] is None
for path in smoke_dir.rglob('*'):
    if path.is_file():
        bind(path)

documentation_path = BASE / 'result_storage_latency_documentation_2026-10-06.json'
documentation = read(documentation_path)
assert documentation['passed']
assert documentation['currentWorkerPending'] == (not worker_verified)
assert documentation['workerStatus'] == smoke['status']
for inventory_name in ('sourceBindings', 'evidenceBindings'):
    for name, expected in documentation[inventory_name].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        checked_binding(path, expected)
for name, expected in documentation['currentDocumentHashes'].items():
    assert digest(ROOT / name) == expected
    bind(ROOT / name)
bind(documentation_path)
host_path = ROOT / 'artifacts/validation/product_installation/result_storage_latency_host_verification_2026-10-06.json'
host = read(host_path)
assert host['checksPassed'] and len(host['checks']) == 75 and not host['inferenceRun'] and not host['modelsImported']
assert host['manifestSha256'] == digest(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
bind(host_path)
host_wrapper_path = BASE / 'verify_result_storage_latency_installation_2026-10-06.json'
host_wrapper = read(host_wrapper_path)
assert host_wrapper['passed'] and host_wrapper['exitCode'] == 0 and host_wrapper['inputsUnchanged']
assert host_wrapper['checkCount'] == 75 and not host_wrapper['inferenceRun']
assert ROOT / host_wrapper['report'] == host_path and digest(host_path) == host_wrapper['reportSha256']
host_log = BASE / 'verify_result_storage_latency_installation_2026-10-06.log'
assert digest(host_log) == host_wrapper['logSha256']
for name, expected in host_wrapper['inputs'].items():
    assert digest(ROOT / name) == expected
    bind(ROOT / name)
bind(host_log)
bind(host_wrapper_path)

front_root = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'
front, front_seal = read(front_root / 'receipt.json'), read(front_root / 'verification_seal.json')
for name, expected in front['sourceBindings'].items():
    assert digest(ROOT / 'frontend' / name) == digest(front_root / 'frontend' / name) == expected
    bind(ROOT / 'frontend' / name)
    bind(front_root / 'frontend' / name)
for name, expected in front['sharedBindings'].items():
    assert digest(ROOT / name) == digest(front_root / name) == expected
    bind(ROOT / name)
    bind(front_root / name)
for name, expected in front_seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(front_root / 'frontend/dist' / name) == expected
    bind(ROOT / 'frontend/dist' / name)
    bind(front_root / 'frontend/dist' / name)
bind(front_root / 'receipt.json')
bind(front_root / 'verification_seal.json')

runtime_path = BASE / 'result_storage_latency_runtime_verification_2026-10-06.json'
runtime = read(runtime_path)
assert runtime['passed'] and len(runtime['exports']) == 12
before = runtime['snapshotBefore']
with sqlite3.connect((ROOT / 'outputs/product/analyses.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
    connection.row_factory = sqlite3.Row
    jobs = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
assert jobs == before['jobs'] and len(jobs) == 10
files = {path.relative_to(ROOT).as_posix(): digest(path)
         for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
assert files == before['savedAnalysisFiles']
for record in runtime['runtimeProcesses']:
    process = psutil.Process(record['pid'])
    assert process.create_time() == record['createTimeUnixSeconds'] and process.cmdline() == record['cmdline']
    assert Path(process.cwd()).resolve() == ROOT
with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=10) as response:
    health = json.loads(response.read())
    assert response.status == 200 and health['worker_available']
bind(runtime_path)
bind(BASE / 'result_storage_latency_runtime_preflight_2026-10-06.json')
bind(BASE / 'result_storage_latency_idle_stop_2026-10-06.json')
for name in ('result_storage_latency_runtime_verifier_repairs_2026-10-06.json',
             'verify_result_storage_latency_runtime_2026-10-06.py',
             'verify_result_storage_latency_runtime_2026-10-06_initial.py',
             'verify_result_storage_latency_runtime_2026-10-06_wrapper_only.py',
             'verify_result_storage_latency_runtime_PROTOCOL_2026-10-06.md',
             'inspect_result_storage_latency_headroom_2026-10-06.py',
             'inspect_result_storage_latency_headroom_2026-10-06.json',
             'review_result_storage_latency_worker_2026-10-06.py',
             'review_result_storage_latency_worker_2026-10-06.json',
             'result_builder_reads_red_2026-10-06.json', 'result_builder_reads_red_2026-10-06.xml',
             'result_builder_reads_green_2026-10-06.json', 'result_builder_reads_green_2026-10-06.xml'):
    bind(BASE / name)
for name in ('result_reader_snapshot_red_2026-10-06', 'result_reader_snapshot_green_2026-10-06',
             'result_storage_latency_trace_2026-10-06', 'result_storage_read_transaction_probe_2026-10-06'):
    for path in (BASE / name).rglob('*'):
        if path.is_file():
            bind(path)
bind(Path(__file__))
assert all(digest(Path(name) if Path(name).is_absolute() else ROOT / name) == expected
           for name, expected in bindings.items())
receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareEvidenceVerified': True,
           'backendTestsPassed': 1278, 'cleanHostChecksPassed': 75,
           'synthetic600And18000ExactOutputsPassed': True, 'actual30FrameWorkerAndExportsPassed': worker_verified,
           'currentWorkerStatus': smoke['status'], 'currentWorkerVerificationPending': not worker_verified,
           'benchmarkOwnedIdentitiesRetiredAtSeal': benchmark_retired,
           'initialWorkerAttemptPreservedNotStarted': True, 'memoryGuardsUnchanged': True,
           'frontendCurrentAndCopiedInputsUnchanged': len(front['sourceBindings']) + len(front['sharedBindings']),
           'frontendIdenticalProductionFiles': len(front_seal['buildBindings']),
           'savedAnalysisRowsUnchanged': 10, 'savedAnalysisFilesUnchanged': len(files),
           'runtimeProcesses': runtime['runtimeProcesses'], 'health': health, 'bindings': bindings,
           'productionQualified': False, 'accuracyQualified': False, 'fullMatchQualified': False,
           'physicalSpeedAnd3DQualified': False,
           'scope': 'Latency software, synthetic output compatibility and scoped restoration evidence. Current actual-worker verification is pending when its separate flag is false; independent accuracy and long-video work remain.',
           'remainingWork': ('Current-revision actual-worker verification remains pending after two declined memory preflights. '
                             if not worker_verified else '') +
                            'Pipeline pose/racket/player/trajectory storage; browser bounded retrieval; event feature search cost; independent qualification data, labels and physical references, licenses and representative capacity.'}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({key: receipt[key] for key in ('softwareEvidenceVerified', 'backendTestsPassed',
      'cleanHostChecksPassed', 'savedAnalysisRowsUnchanged', 'frontendCurrentAndCopiedInputsUnchanged',
      'frontendIdenticalProductionFiles', 'productionQualified')}))

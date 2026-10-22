"""Final readback of current source, exact outputs, restored host and live state."""
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
    path = Path(path)
    assert path.is_file() and path.resolve().is_relative_to(ROOT)
    bindings[path.relative_to(ROOT).as_posix()] = digest(path)

suite = read(BASE / 'run_result_storage_backend_resumed_2026-10-06.json')
assert suite['passed'] and suite['counts'] == {'tests': 1272, 'failures': 0, 'errors': 0, 'skipped': 0}
inventory = read(BASE / 'result_storage_backend_resumed_sources_2026-10-06.json')
assert all(digest(ROOT / name) == value for name, value in inventory.items())
assert digest(BASE / 'result_storage_backend_RESUMED_2026-10-06.xml') == suite['junitSha256']
assert digest(BASE / 'result_storage_backend_RESUMED_2026-10-06.log') == suite['logSha256']
for name in inventory:
    bind(ROOT / name)
for name in ('result_storage_backend_resumed_sources_2026-10-06.json', 'run_result_storage_backend_resumed_2026-10-06.json',
             'result_storage_backend_RESUMED_2026-10-06.xml', 'result_storage_backend_RESUMED_2026-10-06.log',
             'result_storage_backend_interrupted_2026-10-06.json', 'retire_stalled_result_document_test_2026-10-06.json',
             'document_result_storage_2026-10-06.json', 'correct_result_reader_gate_docs_2026-10-06.json'):
    bind(BASE / name)

initial = read(BASE / 'result_storage_baseline_2026-10-06/receipt.json')
for name, expected in initial['bindings'].items():
    path = BASE / 'result_storage_baseline_2026-10-06' / name
    assert digest(path) == expected['sha256']
    bind(path)
documentation = read(BASE / 'document_result_storage_2026-10-06.json')
correction = read(BASE / 'correct_result_reader_gate_docs_2026-10-06.json')
for name, expected in documentation['bindings'].items():
    if name in correction['previousHashes']:
        historical = BASE / 'result_storage_reader_gate_document_baseline_2026-10-06' / name
        assert digest(historical) == expected
        bind(historical)
    else:
        assert digest(ROOT / name) == expected
        bind(ROOT / name)
assert all(digest(ROOT / name) == value for name, value in correction['currentHashes'].items())
for name in correction['currentHashes']:
    bind(ROOT / name)

measure = BASE / 'result_storage_measurement_2026-10-06'
assert read(measure / 'run/report.json')['passed']
assert read(measure / 'terminal_audit_correction.json')['passed']
for name, expected in read(measure / 'prepared_source_bindings.json').items():
    assert digest(ROOT / name) == expected['sha256'] and (ROOT / name).stat().st_size == expected['bytes']
for path in measure.rglob('*'):
    if path.is_file():
        bind(path)

smoke_dir = BASE / 'result_storage_worker_smoke_2026-10-06'
smoke = read(smoke_dir / 'resource_receipt.json')
assert smoke['passed'] and smoke['allOwnedIdentitiesRetiredAfterNormalExit'] and smoke['productionJobStatesUnchanged']
for name, expected in smoke['sourceBindingsAfter'].items():
    assert digest(ROOT / name) == expected['sha256']
    bind(ROOT / name)
for identity in smoke['ownedProcessIdentitiesAndPeaks']:
    try:
        assert psutil.Process(identity['pid']).create_time() != identity['createTimeUnixSeconds']
    except psutil.NoSuchProcess:
        pass
for path in smoke_dir.rglob('*'):
    if path.is_file():
        bind(path)
child = read(smoke_dir / 'child_receipt.json')
assert child['passed'] and not child['networkAttempts']
assert read(smoke_dir / 'saved_result_comparison.json')['passed']

host_path = ROOT / 'artifacts/validation/product_installation/result_storage_reader_gate_host_verification_2026-10-06.json'
host = read(host_path)
assert host['checksPassed'] and len(host['checks']) == 75 and not host['inferenceRun'] and not host['modelsImported']
assert host['manifestSha256'] == digest(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
bind(host_path)
bind(BASE / 'verify_result_storage_reader_gate_installation_2026-10-06.json')

frontend_root = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'
front = read(frontend_root / 'receipt.json')
front_seal = read(frontend_root / 'verification_seal.json')
for name, expected in front['sourceBindings'].items():
    assert digest(ROOT / 'frontend' / name) == digest(frontend_root / 'frontend' / name) == expected
for name, expected in front['sharedBindings'].items():
    assert digest(ROOT / name) == digest(frontend_root / name) == expected
for name, expected in front_seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(frontend_root / 'frontend/dist' / name) == expected
bind(frontend_root / 'receipt.json')
bind(frontend_root / 'verification_seal.json')
bind(ROOT / 'frontend/dist/assets/index-BWMRwJe_.js')

runtime = read(BASE / 'result_storage_runtime_verification_2026-10-06.json')
assert runtime['passed'] and len(runtime['exports']) == 12
preflight = read(BASE / 'result_storage_runtime_preflight_2026-10-06.json')['snapshot']
with sqlite3.connect((ROOT / 'outputs/product/analyses.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
    connection.row_factory = sqlite3.Row
    current_jobs = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
assert current_jobs == preflight['jobs'] and len(current_jobs) == 10
paths = [path for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()]
current_files = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
assert current_files == preflight['savedAnalysisFiles']
for identity in runtime['runtimeProcesses']:
    process = psutil.Process(identity['pid'])
    assert process.create_time() == identity['create_time'] and process.cmdline() == identity['cmdline']
    assert Path(process.cwd()).resolve() == ROOT
with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=10) as response:
    health = json.loads(response.read())
    assert response.status == 200 and health['status'] == 'ok' and health['worker_available']
for name in ('result_storage_runtime_verification_2026-10-06.json', 'result_storage_runtime_preflight_2026-10-06.json',
             'verify_result_storage_runtime_2026-10-06.py', 'verify_result_storage_runtime_2026-10-06_initial.py',
             'verify_result_storage_runtime_2026-10-06_headers_fixed.py', 'repair_result_storage_runtime_verifier_2026-10-06.json',
             'repair_result_storage_runtime_identity_2026-10-06.py'):
    bind(BASE / name)
bind(Path(__file__))
receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareEvidenceVerified': True,
    'backendTestsPassed': 1272, 'cleanHostChecksPassed': 75,
    'synthetic600And18000ExactOutputsPassed': True, 'actual30FrameWorkerAndExportsPassed': True,
    'frontendCurrentAndCopiedInputsUnchanged': len(front['sourceBindings']) + len(front['sharedBindings']),
    'frontendIdenticalProductionFiles': len(front_seal['buildBindings']),
    'savedAnalysisRowsUnchanged': 10, 'savedAnalysisFilesUnchanged': len(current_files),
    'runtimeProcesses': runtime['runtimeProcesses'], 'health': health,
    'browserObservation': {'url': 'http://127.0.0.1:8000/overview', 'browserId': '2', 'tabId': '1',
        'entryScript': 'index-BWMRwJe_.js', 'djokovicAndSchiavoneDecodedPlaying': True,
        'decoderReadyState': 4, 'observedTimesSeconds': [7.408297, 7.409085],
        'observedDecodedDimensions': [1280, 720], 'thirdGalleryMediaNotObservedPlaying': True,
        'scope': 'Bounded read-only UI observation; no saved screenshot or complete autoplay/latency qualification.',
        'tabMarkedDeliverable': True},
    'verifierFailuresPreserved': ['initial HTTP header case lookup', 'shell-wrapper process count',
                                  'failed CRLF-sensitive repair helper'],
    'initialTerminalAuditAssertionErrorPreserved': True, 'interruptedBackendRunPreservedIncomplete': True,
    'priorLargeRegistrationAndFiveMinuteFailuresRemainUnqualified': True,
    'bindings': bindings, 'newInferenceStartedDuringFinalReadback': False,
    'productionQualified': False, 'accuracyQualified': False, 'fullMatchQualified': False,
    'physicalSpeedAnd3DQualified': False,
    'remainingWork': 'Pipeline player/trajectory storage, browser/frame retrieval, measured result-path latency; independent qualification sources and labels/references, licenses and representative long-video validation.'}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({key: receipt[key] for key in ('softwareEvidenceVerified', 'backendTestsPassed',
     'cleanHostChecksPassed', 'savedAnalysisRowsUnchanged', 'frontendCurrentAndCopiedInputsUnchanged',
     'frontendIdenticalProductionFiles', 'productionQualified')}))

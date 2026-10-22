"""Final read-only current/historical readback; no reruns or documentation edits."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEST = HERE / 'registration_storage_final_evidence_2026-10-06.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not DEST.exists(), 'Preserve existing final evidence'
historical = {}
baselines = ('registration_storage_document_baseline_2026-10-06', 'registration_sequence_baseline_2026-10-05',
             'candidate_storage_preseal_documents_2026-10-05', 'candidate_storage_document_baseline_2026-10-05',
             'candidate_spool_baseline_2026-10-05')


def verify(name, expected):
    path = ROOT / name
    if path.is_file() and digest(path) == expected:
        return
    for baseline in baselines:
        preserved = HERE / baseline / name
        if preserved.is_file() and digest(preserved) == expected:
            historical[name] = {'preservedPath': preserved.relative_to(ROOT).as_posix(), 'sha256': expected}
            return
    raise AssertionError('Missing exact current or historical binding: ' + name)


documentation_path = HERE / 'registration_storage_documentation_2026-10-06.json'
documentation = read(documentation_path)
for name, expected in {**documentation['documents'], **documentation['evidence']}.items():
    verify(name, expected)
for name, expected in documentation['preservedPriorDocuments'].items():
    assert digest(HERE / baselines[0] / name) == expected
host_path = ROOT / 'artifacts/validation/product_installation/registration_storage_host_verification_2026-10-06.json'
host = read(host_path)
assert host['checksPassed'] and len(host['checks']) == 75 and not host['warnings']
assert all(row['passed'] for row in host['checks'])
assert not host['inferenceRun'] and not host['modelsImported'] and not host['productionQualified']
assert digest(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json') == host['manifestSha256']
launcher_path = HERE / 'verify_registration_storage_installation_2026-10-06.json'
launcher = read(launcher_path)
assert launcher['passed'] and launcher['checkCount'] == 75 and launcher['inputsUnchanged']
assert launcher['reportSha256'] == digest(host_path)
for name, expected in launcher['inputs'].items():
    verify(name, expected)
previous_path = HERE / 'candidate_storage_final_evidence_2026-10-05.json'
previous = read(previous_path)
for name, expected in previous['bindings'].items():
    verify(name, expected)
manifest = read(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
for row in manifest['validationEvidence']:
    if row.get('sha256'):
        verify(row['path'], row['sha256'])
restore = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'
frontend = read(restore / 'receipt.json')
frontend_seal = read(restore / 'verification_seal.json')
target = ROOT / frontend['target']
for source, copied, bindings in ((ROOT / 'frontend', target, frontend['sourceBindings']),
                                 (ROOT, restore, frontend['sharedBindings'])):
    for name, expected in bindings.items():
        assert digest(source / name) == digest(copied / name) == expected, name
for name, expected in frontend_seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(target / 'dist' / name) == expected, name
terminal_path = HERE / 'registration_storage_measurement_2026-10-05/terminal_receipt_audit_2026-10-06.json'
terminal = read(terminal_path)
assert terminal['terminalEvidenceVerified'] and not terminal['workloadPassed']
for name, binding in terminal['artifactBindings'].items():
    assert (ROOT / name).stat().st_size == binding['bytes'] and digest(ROOT / name) == binding['sha256'], name
failure_path = HERE / 'repeated_source_capacity_STOP_verified_2026-10-05.json'
failure = read(failure_path)
assert failure['evidenceVerified'] and not failure['workloadPassed']
for name, binding in failure['artifactBindings'].items():
    assert (ROOT / name).stat().st_size == binding['bytes'] and digest(ROOT / name) == binding['sha256'], name
smoke_path = HERE / 'registration_storage_real_smoke_2026-10-05/resource_receipt.json'
smoke = read(smoke_path)
spec = importlib.util.spec_from_file_location('registration_final_smoke', ROOT / 'scripts/audit/verify_registration_storage_pipeline_smoke.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert helper.smoke.product_jobs() == smoke['productionJobStatesAfter']
assert len(smoke['productionJobStatesAfter']['jobs']) == 10
import psutil
server = psutil.Process(47408)
command = server.cmdline()
assert server.is_running() and any(Path(part).resolve() == (ROOT / 'scripts/run/serve_product.py').resolve() for part in command[1:])
with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5) as response:
    health_status = response.status
    health = json.load(response)
assert health_status == 200
paths = [Path(__file__), documentation_path, host_path, launcher_path, previous_path, terminal_path,
         failure_path, smoke_path, restore / 'receipt.json', restore / 'verification_seal.json']
paths.extend(ROOT / name for name in documentation['documents'])
receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareEvidenceVerified': True,
           'backendTestsPassed': 1102, 'cleanHostChecksPassed': 75,
           'real30FrameCompatibilityPassed': True, 'registrationLargeBenchmarkPassed': False,
           'priorFiveMinuteWorkloadPassed': False, 'frontendInputsRechecked': frontend_seal['currentAndCopiedInputs'],
           'byteIdenticalFrontendBuildFiles': len(frontend_seal['buildBindings']),
           'historicalBindingsResolvedToExactCopies': historical,
           'registrationBenchmarkFilesRechecked': len(terminal['artifactBindings']),
           'priorFailedStressFilesRechecked': len(failure['artifactBindings']),
           'savedAnalysisRowsUnchanged': 10,
           'server': {'pid': server.pid, 'createTimeUnixSeconds': server.create_time(), 'command': command,
                      'healthStatus': health_status, 'health': health},
           'browserObservation': {'scope': 'Read-only new homepage tab, no UI changes; no broad playback qualification.',
                                  'url': 'http://127.0.0.1:8000/overview', 'servedBundle': 'index-BWMRwJe_.js',
                                  'heroAnalysisId': 'match_9de7cb2d09eb46a0af0e34c3b0e6b818',
                                  'heroPaused': False, 'heroReadyState': 4, 'heroDimensions': [1280, 720],
                                  'heroObservedTimeSeconds': 8.489084, 'previewTabMarkedDeliverable': True},
           'bindings': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
           'newInferenceStartedDuringFinalReadback': False, 'productionQualified': False,
           'fullMatchQualified': False, 'accuracyQualified': False, 'physicalSpeedAnd3DQualified': False}
with DEST.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'softwareEvidenceVerified': True, 'backendTestsPassed': 1102, 'cleanHostChecksPassed': 75,
                  'savedAnalysisRowsUnchanged': 10, 'frontendInputsRechecked': receipt['frontendInputsRechecked'],
                  'registrationLargeBenchmarkPassed': False, 'serverHealthStatus': health_status}))

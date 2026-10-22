"""One-shot final readback and documentation seal; no inference or retries."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DESTINATION = HERE / 'candidate_storage_final_evidence_2026-10-05.json'
DOC_COPY = HERE / 'candidate_storage_preseal_documents_2026-10-05'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def suite_count(path):
    suites = ET.parse(path).getroot().findall('testsuite')
    assert suites
    assert all(int(suite.get(key, '0')) == 0 for suite in suites for key in ('errors', 'failures', 'skipped'))
    return sum(int(suite.get('tests')) for suite in suites)


assert not DESTINATION.exists() and not DOC_COPY.exists(), 'Preserve existing final evidence'
historical_bindings = {}


def verify_binding(name, expected):
    current = ROOT / name
    if current.is_file() and digest(current) == expected:
        return {'path': name, 'sha256': expected, 'scope': 'current'}
    for baseline in ('candidate_spool_baseline_2026-10-05', 'candidate_storage_document_baseline_2026-10-05'):
        preserved = HERE / baseline / name
        if preserved.is_file() and digest(preserved) == expected:
            value = {'path': preserved.relative_to(ROOT).as_posix(), 'sha256': expected, 'scope': 'exact preserved historical copy'}
            historical_bindings[name] = value
            return value
    raise AssertionError('Missing exact current or preserved binding: ' + name)


manifest_path = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
host_path = ROOT / 'artifacts/validation/product_installation/candidate_storage_host_verification_2026-10-05.json'
host = read(host_path)
assert host['checksPassed'] and len(host['checks']) == 75 and not host['warnings']
assert all(row['passed'] for row in host['checks'])
assert host['manifestSha256'] == digest(manifest_path) == '7b4718de1ca8dbcbce42393e5af9fbbcc234ba20deeda0b69a005884c4fb41e8'
assert not host['inferenceRun'] and not host['modelsImported'] and not host['productionQualified']
installation_launcher = HERE / 'verify_candidate_storage_installation_2026-10-05.json'
installation = read(installation_launcher)
assert installation['passed'] and installation['inputsUnchanged'] and installation['checkCount'] == 75
assert installation['reportSha256'] == digest(host_path)
for name, expected in installation['inputs'].items():
    verify_binding(name, expected)

documentation_path = HERE / 'candidate_storage_documentation_2026-10-05.json'
documentation = read(documentation_path)
for name, expected in {**documentation['documents'], **documentation['evidence']}.items():
    verify_binding(name, expected)
manifest = read(manifest_path)
for row in manifest['validationEvidence']:
    if row.get('sha256'):
        verify_binding(row['path'], row['sha256'])

restore = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'
frontend_seal_path = restore / 'verification_seal.json'
frontend_seal = read(frontend_seal_path)
for name, expected in {**frontend_seal['evidenceBindings'], **frontend_seal['changedSourceBindings']}.items():
    verify_binding(name, expected)
frontend_receipt = read(restore / 'receipt.json')
target = ROOT / frontend_receipt['target']
for source, copied, bindings in ((ROOT / 'frontend', target, frontend_receipt['sourceBindings']),
                                 (ROOT, restore, frontend_receipt['sharedBindings'])):
    for name, expected in bindings.items():
        assert digest(source / name) == digest(copied / name) == expected, name
for name, expected in frontend_seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(target / 'dist' / name) == expected, name

benchmark = HERE / 'candidate_storage_measurement_monitored_2026-10-05'
protocol = read(benchmark / 'protocol.json')
report = read(benchmark / 'report.json')
terminal = read(benchmark / 'terminal_receipt_audit.json')
assert report['passed'] and report['sourceFilesUnchanged'] and len(report['runs']) == 8
assert terminal['passed'] and terminal['allEightActualWorkloadsSampledByExactIdentity']
assert terminal['allRememberedOwnedIdentitiesRetiredAtTerminalAudit']
for name, expected in protocol['sourceFiles'].items():
    if name == 'scripts/audit/measure_candidate_storage.py':
        assert digest(benchmark / 'executed_measure_candidate_storage.py') == expected
    else:
        verify_binding(name, expected)
first_benchmark = HERE / 'candidate_storage_measurement_2026-10-05'
assert digest(first_benchmark / 'executed_measure_candidate_storage.py') == read(first_benchmark / 'protocol.json')['sourceFiles']['scripts/audit/measure_candidate_storage.py']
hardening = HERE / 'candidate_monitor_hardening_2026-10-05'
guard = read(hardening / 'redirector_monitor_regression/receipt.json')
assert guard['passed'] and not guard['remainingOwnedIdentities']
assert guard['sourceSha256'] == digest(ROOT / 'scripts/audit/measure_candidate_storage.py')
assert suite_count(hardening / 'unit_regressions.xml') == 4
assert suite_count(HERE / 'candidate_storage_backend_FINAL_2026-10-05.xml') == 1063

smoke = HERE / 'candidate_storage_real_smoke_2026-10-05'
resource = read(smoke / 'resource_receipt.json')
assert resource['passed'] and resource['status'] == 'COMPLETED'
assert resource['actualWorkloadWasSampled'] and resource['allOwnedIdentitiesRetiredAfterNormalExit']
assert not resource['remainingOwnedIdentitiesAfterNormalExit']
for name, binding in resource['sourceBindingsAfter'].items():
    assert digest(ROOT / name) == binding['sha256'], name
for name, binding in resource['historicalArtifactsAfter'].items():
    assert (ROOT / name).stat().st_size == binding['bytes'] and digest(ROOT / name) == binding['sha256'], name
spec = importlib.util.spec_from_file_location('candidate_final_readback', ROOT / 'scripts/audit/verify_candidate_storage_pipeline_smoke.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert helper.product_jobs() == resource['productionJobStatesAfter']
assert len(resource['productionJobStatesAfter']['jobs']) == 10

failure_path = HERE / 'repeated_source_capacity_STOP_verified_2026-10-05.json'
failure = read(failure_path)
assert failure['evidenceVerified'] and not failure['workloadPassed']
for name, binding in failure['artifactBindings'].items():
    assert (ROOT / name).stat().st_size == binding['bytes'] and digest(ROOT / name) == binding['sha256'], name
previous_seal_path = HERE / 'capacity_stop_final_evidence_2026-10-05.json'
previous_seal = read(previous_seal_path)
for name, expected in previous_seal['bindings'].items():
    verify_binding(name, expected)

# Preserve the exact documents bound by the preceding update before adding the host receipt.
DOC_COPY.mkdir()
for name, expected in documentation['documents'].items():
    destination = DOC_COPY / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write((ROOT / name).read_bytes())
    assert digest(destination) == expected

note = (
    '[The final candidate-storage installation check](../../artifacts/validation/product_installation/candidate_storage_host_verification_2026-10-05.json) '
    f'passed all 75 checks at `{host["verifiedAt"]}`, binding manifest SHA256 `{host["manifestSha256"]}`. '
    'The isolated worker-disabled API returned health/library 200 and invalid-upload 415 without model imports or inference. '
    '[The final evidence readback](../../artifacts/validation/product_execution/candidate_storage_final_evidence_2026-10-05.json) '
    'rechecks the current source and assets, exact historical copies, 124 current/copied frontend inputs, 37 identical build files, '
    'the retained failed stress artifacts and all ten unchanged saved analysis records. '
    'The full goal remains incomplete: model accuracy, representative long-clip capacity and physical speed/3D remain unqualified.'
)
progress = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
text = progress.read_text(encoding='utf-8')
anchor = 'Capacity engineering is now the active unit.'
assert text.count(anchor) == 1
progress.write_text(text.replace(anchor, note + '\n\n' + anchor, 1), encoding='utf-8')
guide = ROOT / 'docs/deployment/PRODUCT_INSTALLATION.md'
text = guide.read_text(encoding='utf-8')
anchor = 'The final October5 [capacity-stop host receipt]'
assert text.count(anchor) == 1
guide.write_text(text.replace(anchor, note + '\n\n' + anchor, 1), encoding='utf-8')
assert digest(manifest_path) == host['manifestSha256']

paths = [manifest_path, host_path, installation_launcher, documentation_path, frontend_seal_path,
         previous_seal_path, failure_path, guide, progress, Path(__file__),
         HERE / 'candidate_storage_backend_FINAL_2026-10-05.xml',
         benchmark / 'report.json', benchmark / 'terminal_receipt_audit.json', benchmark / 'executed_measure_candidate_storage.py',
         first_benchmark / 'terminal_monitor_audit.json', first_benchmark / 'executed_measure_candidate_storage.py',
         hardening / 'unit_regressions.xml', hardening / 'redirector_monitor_regression/receipt.json',
         smoke / 'real_pipeline_smoke.json', smoke / 'artifact_comparison.json', smoke / 'resource_receipt.json']
paths.extend(ROOT / name for name in resource['sourceBindingsAfter'])
paths.extend(DOC_COPY / name for name in documentation['documents'])
paths.extend(ROOT / ('tests/' + name) for name in ('test_ball_candidate_sequence.py', 'test_image_people_streaming.py',
             'test_temporal_candidate_memory.py', 'test_pipeline_candidate_storage.py', 'test_owned_process_monitor.py'))
receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareChecksPassed': True,
           'backendTestsPassed': 1063, 'additionalAuditTestsPassed': 4, 'hostChecksPassed': 75,
           'unchangedPriorFrontendEvidenceFiles': len(frontend_seal['evidenceBindings']),
           'currentAndCopiedFrontendInputs': frontend_seal['currentAndCopiedInputs'],
           'byteIdenticalFrontendBuildFiles': len(frontend_seal['buildBindings']),
           'historicalBindingsResolvedToExactCopies': historical_bindings,
           'retainedStressFilesRechecked': len(failure['artifactBindings']), 'priorFiveMinuteWorkloadPassed': False,
           'historicalSmokeFilesRechecked': len(resource['historicalArtifactsAfter']),
           'savedAnalysisRowsUnchanged': 10, 'real30FrameCompatibilityPassed': True,
           'bindings': {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(set(paths))},
           'productionQualified': False, 'fullMatchQualified': False, 'accuracyQualified': False,
           'newInferenceStartedDuringFinalReadback': False,
           'scope': 'Read-only evidence and preserved-copy checks, followed by installation/progress documentation. No fresh benchmark, model inference or frontend tests.',
           'limitations': ['Candidate measurements are synthetic only.', 'Real pipeline compatibility covers 30 frames only.',
                           'The prior five-minute stress remains failed.', 'Court/player/trajectory/results and API/frontend arrays remain O(N).',
                           'No owner-authorized independent accuracy, speed or 3D qualification footage is available.']}
with DESTINATION.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'softwareChecksPassed': True, 'hostChecksPassed': 75,
                  'currentAndCopiedFrontendInputs': receipt['currentAndCopiedFrontendInputs'],
                  'byteIdenticalFrontendBuildFiles': receipt['byteIdenticalFrontendBuildFiles'],
                  'historicalBindingsResolvedToExactCopies': len(historical_bindings),
                  'savedAnalysisRowsUnchanged': 10, 'receiptSha256': digest(DESTINATION), 'productionQualified': False}))

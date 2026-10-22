"""One-shot registration evidence readback and documentation update.

This preserves failed measurements and historical documentation; no inference,
benchmark retry, result publication, or broad filesystem cleanup is performed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = HERE / 'registration_storage_measurement_2026-10-05'
SMOKE = HERE / 'registration_storage_real_smoke_2026-10-05'
DEST = HERE / 'registration_storage_documentation_2026-10-06.json'
TERMINAL = BENCH / 'terminal_receipt_audit_2026-10-06.json'
COPY = HERE / 'registration_storage_document_baseline_2026-10-06'
DOCS = ('docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
        'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def retired(identity):
    import psutil
    try:
        process = psutil.Process(identity['pid'])
        return abs(process.create_time() - identity['createTimeUnixSeconds']) > .02 or not process.is_running()
    except psutil.NoSuchProcess:
        return True


resume_terminal = '--resume-verified-terminal' in sys.argv
assert not any(path.exists() for path in (DEST, COPY)), 'Preserve existing documentation evidence'
assert TERMINAL.exists() == resume_terminal, 'Terminal evidence must be new, or explicitly reverified without replacement'
protocol, report = read(BENCH / 'protocol.json'), read(BENCH / 'report.json')
assert not report['passed'] and report['error'] == 'RuntimeError: Owned wall-time safeguard'
assert report['sourcesUnchanged'] and report['sourceBindingsAfter'] == protocol['sourceBindingsBefore']
assert len(report['runs']) == 6 and len(report['comparisons']) == 2
assert [row['frames'] for row in report['comparisons']] == [600, 18000]
assert all(row['passed'] and row['allFramesDirectlyComparedToFrozenSyntheticInput'] for row in report['comparisons'])
for name, expected in protocol['sourceBindingsBefore'].items():
    assert digest(ROOT / name) == expected, name
assert digest(BENCH / 'executed_measure_registration_storage.py') == protocol['sourceBindingsBefore']['scripts/audit/measure_registration_storage.py']
cases = []
for case in sorted(path for path in BENCH.iterdir() if path.is_dir()):
    resource = read(case / 'resource_receipt.json')
    identity = read(case / 'actual_identity.json')
    assert resource['ownedIdentities'] and all(retired(row) for row in resource['ownedIdentities'])
    assert any(row['pid'] == identity['pid'] and abs(row['createTimeUnixSeconds'] - identity['createTimeUnixSeconds']) < .02
               and row['sampleCount'] > 0 for row in resource['ownedIdentities'])
    cases.append({'case': case.name, 'passed': resource['passed'], 'samples': resource['samples'],
                  'actualWorkloadSampled': True, 'allRememberedIdentitiesRetired': True})
assert len(cases) == 6
failed = read(BENCH / 'dynamic_432000_new/resource_receipt.json')
assert not failed['passed'] and failed['safeguardStop'] == 'Owned wall-time safeguard'
assert not (BENCH / 'dynamic_432000_new/child_receipt.json').exists()
database, = (BENCH / 'dynamic_432000_new').glob('frame_registration_*.sqlite3')
connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
try:
    connection.execute('PRAGMA query_only=ON')
    rows = connection.execute('SELECT COUNT(*),MIN(frame_index),MAX(frame_index) FROM registration_frames').fetchone()
finally:
    connection.close()
assert rows == (432000, 0, 431999)
benchmark_files = {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                   for path in BENCH.rglob('*') if path.is_file() and path != TERMINAL}
terminal_record = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'terminalEvidenceVerified': True,
                'workloadPassed': False, 'completedPairedComparisons': [600, 18000], 'cases': cases,
                'retainedFailedDatabaseRows': list(rows), 'artifactBindings': benchmark_files,
                'largeNewForwardComparisonCompleted': False, 'largeStaticCaseLaunched': False,
                'maximumConfiguredCaseLaunched': False, 'inferenceRun': False, 'qualificationEvidence': False}
if resume_terminal:
    prior_terminal = read(TERMINAL)
    assert prior_terminal['artifactBindings'] == benchmark_files and prior_terminal['cases'] == cases
    assert prior_terminal['retainedFailedDatabaseRows'] == list(rows) and not prior_terminal['workloadPassed']
else:
    save(TERMINAL, terminal_record)

resource = read(SMOKE / 'resource_receipt.json')
assert resource['passed'] and resource['status'] == 'COMPLETED'
assert resource['actualWorkloadWasSampled'] and resource['allOwnedIdentitiesRetiredAfterNormalExit']
assert not resource['remainingOwnedIdentitiesAfterNormalExit']
assert resource['sourceBindingsUnchanged'] and resource['historicalArtifactsUnchanged'] and resource['productionJobStatesUnchanged']
for name, binding in resource['sourceBindingsAfter'].items():
    assert digest(ROOT / name) == binding['sha256'], name
for name, binding in resource['historicalArtifactsAfter'].items():
    assert (ROOT / name).stat().st_size == binding['bytes'] and digest(ROOT / name) == binding['sha256'], name
comparison = read(SMOKE / 'artifact_comparison.json')
assert comparison['passed'] and len(comparison['jsonFiles']) == 26 and comparison['runConfigBytesEqual']
assert all(row['exactParsedValuesEqual'] for row in comparison['jsonFiles'].values())
assert comparison['annotatedVideosFullyDecoded'] and comparison['annotatedFramesCompared'] == 30
cleanup = read(SMOKE / 'registration_cleanup.json')
assert cleanup['passed'] and not cleanup['unpublishedRegistrationScratchFiles']
assert not any(path.is_file() and path.name.startswith('frame_registration_') for path in (SMOKE / 'run').rglob('*'))
spec = importlib.util.spec_from_file_location('registration_seal_smoke', ROOT / 'scripts/audit/verify_registration_storage_pipeline_smoke.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert helper.smoke.product_jobs() == resource['productionJobStatesAfter']
assert len(resource['productionJobStatesAfter']['jobs']) == 10
smoke = read(SMOKE / 'real_pipeline_smoke.json')
assert smoke['passed'] and not smoke['networkAttempts'] and not smoke['productionQualified'] and not smoke['productStatePublished']

suite = ET.parse(HERE / 'registration_storage_backend_FULL_2026-10-05.xml').getroot().findall('testsuite')
assert suite and all(int(row.get(key, '0')) == 0 for row in suite for key in ('errors', 'failures', 'skipped'))
assert sum(int(row.get('tests')) for row in suite) == 1102
probe_paths = sorted(path for path in HERE.glob('registration_storage_delay_probe*') if path.is_dir())
probes = []
for path in probe_paths:
    child, parent = read(path / 'child_receipt.json'), read(path / 'resource_receipt.json')
    assert parent['probeExecuted'] and parent['actualWorkloadWasSampled'] and not parent['remainingOwnedIdentities']
    assert all(retired(row) for row in parent['ownedIdentities'])
    assert child.get('retainedDatabaseReadOnly', child.get('readOnlyDatabase')) and not child['inferenceRun']
    if child.get('freshAppendFixtureCount'):
        assert child['ownedFixtureRemovedOnSuccess']
    probes.append({'path': path.relative_to(ROOT).as_posix(), 'fastLoop': child['fastLoop'],
                   'freshAppendFixtureCount': child.get('freshAppendFixtureCount')})
assert len(probes) == 10

# Preserve the exact preceding documents before updating the single progress record.
COPY.mkdir()
for name in DOCS:
    destination = COPY / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write((ROOT / name).read_bytes())
    assert digest(destination) == digest(ROOT / name)

note = (
    'Latest registration-storage unit (6 October): dynamic per-frame calibration and anchor transforms now use an exclusively owned SQLite sequence with a 2 MiB cache and commits every 256 records; static cameras share one calibration without a per-frame matrix list or database. Aligned read-only views preserve exact matrix dtype, byte order, buffers, calibration fields, projections, cuts and nullable transforms. Whole-segment participant selection and downstream measurement algorithms remain unchanged. [All 1,102 backend tests](../../artifacts/validation/product_execution/registration_storage_backend_FULL_2026-10-05.xml) passed in 36.18 seconds, including exact-buffer, lifetime, producer/failure and real pipeline integration checks.\n\n'
    '[The synthetic registration benchmark](../../artifacts/validation/product_execution/registration_storage_measurement_2026-10-05/report.json) **failed**: only the 600/18,000-frame old/new pairs completed exact full comparisons. At 18,000 frames actual interpreter peak working set changed 55,316,480 → 47,583,232 bytes, elapsed 0.587 → 1.732 seconds and the owned database occupied 6,836,224 bytes before cleanup. The 432,000-frame new child hit the predeclared 300-second wall guard; sampled actual working set peaked at 47,611,904 bytes, but no completion/forward-comparison receipt exists. Its 432,000 accepted rows and all logs are retained. The 864,000-frame and large static cases were never launched. [Terminal readback](../../artifacts/validation/product_execution/registration_storage_measurement_2026-10-05/terminal_receipt_audit_2026-10-06.json) verifies exact workload sampling, retirement and retained evidence; it does not pass the failed workload.\n\n'
    'Ten bounded diagnostic probes preserve exact executed scripts and window digests. Historical fresh-fill 288k/432k windows were slow while 160k and reopened reads were fast. On 6 October, original → fresh-reader → original connections remained similarly slow and GC took milliseconds; generated-only geometry took 0.09349 seconds before filling and 0.08747 seconds after. The unchanged retained database also became slow on the current host. These observations do not establish an append-count or writer-state defect, a memory leak, or a thermal/scheduling cause. No production performance fix or guard extension was justified. Earlier after-fill probe receipts label `database` with the retained reference path even though their preserved scripts query the fresh fixture; later receipts distinguish both paths. Diagnostic timing thresholds are not product throughput gates.\n\n'
    f'[Fresh offline 30-frame compatibility](../../artifacts/validation/product_execution/registration_storage_real_smoke_2026-10-05/resource_receipt.json) passed with exact 26 JSON values except the two declared runtime fields, unchanged run_config, original source pixels/relative PTS, and 30 fully decoded annotated frames. Pipeline {smoke["metrics"]["total_time_s"]:.3f}s at {smoke["metrics"]["processing_fps"]:.3f} FPS; helper {smoke["elapsedSeconds"]:.3f}s; full monitored child {resource["monitoredChildWallSeconds"]:.3f}s. Actual process peak working set {smoke["processPeakWorkingSetBytes"]:,} bytes, sampled tree RSS {resource["peakSampledOwnedTreeRssBytes"]:,} bytes; PyTorch allocated/reserved peaks {smoke["cudaAllocatorPeakBytes"]["allocated"]:,}/{smoke["cudaAllocatorPeakBytes"]["reserved"]:,} bytes. Registration/candidate scratch files are absent, all owned processes retired, historical files and all ten saved analysis rows are unchanged. Court rejection, unavailable roles/meaningful pose/racket, null physical speed/3D and disabled event authority persist. Registration dependencies omitted from the inherited smoke binding inventory are separately bound by the unchanged benchmark protocol and this seal; the smoke is not whole-workspace integrity evidence.\n\n'
    'The local server was stopped when this continuation began and was restored with the documented localhost command (PID 47408, tool session 3316; recheck actual state before subsequent actions). The unchanged frontend build loads and the homepage hero visibly plays. No new saved inference job was accepted. Next capacity work is bounded player/trajectory/result storage and API/frontend frame retrieval, followed by a newly prescribed workload after useful changes. The failed five-minute and registration benchmarks remain failed; representative full-match capacity, model accuracy and independently measured speed/3D remain unqualified.'
)
progress_path = ROOT / DOCS[0]
progress = progress_path.read_text(encoding='utf-8')
progress = progress.replace('Updated: 2026-10-05, Asia/Singapore.', 'Updated: 2026-10-06, Asia/Singapore.', 1)
anchor = 'Latest candidate-storage unit:'
assert progress.count(anchor) == 1
progress_path.write_text(progress.replace(anchor, note + '\n\n' + anchor, 1), encoding='utf-8')
guide_path = ROOT / DOCS[1]
guide = guide_path.read_text(encoding='utf-8')
anchor = 'The [candidate-storage update]'
assert guide.count(anchor) == 1
guide_path.write_text(guide.replace(anchor,
    'The registration-storage update passes all 1,102 backend tests and the isolated 30-frame compatibility smoke, with successful owned scratch cleanup. Dynamic calibration/transform records use bounded SQLite; static cameras share one constant calibration. The synthetic registration benchmark **failed its 300-second guard at 432,000 frames** after only the 600/18,000-frame comparisons completed; all failure evidence is retained. [Current progress and measured scope](../plans/TENNIS_EXECUTION_PROGRESS.md) distinguish passing software checks from unmet capacity/accuracy gates. Earlier host/document seals remain historical after this update. Player, trajectory, result/API and frontend arrays still require bounded storage and retrieval.\n\n' + anchor, 1), encoding='utf-8')
manifest_path = ROOT / DOCS[2]
manifest = read(manifest_path)
manifest['evidenceUpdatedAt'] = datetime.now(timezone.utc).isoformat()
manifest['productionQualified'] = False
manifest['validation']['registrationStorageEngineering'] = (
    'SOFTWARE_COMPATIBILITY_PASSED_LARGE_SYNTHETIC_MEASUREMENT_FAILED_2026-10-06: exact dynamic registration snapshots with bounded SQLite and constant static camera views; 1102 backend tests and fresh 30-frame clean-runtime compatibility passed. Only 600/18000 synthetic pairs completed. New432000 child stopped at300.294s without final verification; 864000/static cases unlaunched. Closed failed database retained, exact processes retired. No full-match or accuracy qualification.')
manifest['validation']['backendRegistrationStorageChecks'] = 'PASSED: 1102 maintained backend tests in36.18s; production code unchanged since test/benchmark bindings.'
manifest['validation']['modelLoadAndInferenceInCleanEnvironment'] = (
    'PASSED_SHORT_SOFTWARE_COMPATIBILITY_2026-10-06: isolated offline30-frame native60FPS licensed excerpt; exact26JSON values except2 runtime fields, run_config and annotated frame count/PTS/decode preserved. Pipeline20.916s/helper45.591s/fullchild127.178s; all10 saved analyses unchanged, all owned processes retired, candidate and registration scratch absent. Court/roles/pose/racket remain unavailable and physical speed/3D null; no accuracy/capacity/deployment qualification.')
evidence = [
    (HERE / 'registration_storage_backend_FULL_2026-10-05.xml', True, '1102 maintained backend tests; software only.'),
    (BENCH / 'report.json', False, 'Large synthetic benchmark failed wall guard; only600/18000 paired comparisons complete.'),
    (TERMINAL, True, 'Terminal identity/retained-file readback, not workload qualification.'),
    (SMOKE / 'real_pipeline_smoke.json', True, 'Fresh isolated30-frame offline software execution.'),
    (SMOKE / 'artifact_comparison.json', True, 'Exact26JSON values except2 runtime fields; run_config/decode checks.'),
    (SMOKE / 'registration_cleanup.json', True, 'Registration scratch and all prefixed sidecars absent after success.'),
    (SMOKE / 'resource_receipt.json', True, 'Owned-process/source/history/product-row monitoring; bounded scope.')]
for path, passed, scope in evidence:
    manifest['validationEvidence'].append({'path': path.relative_to(ROOT).as_posix(), 'verifiedAt': manifest['evidenceUpdatedAt'],
                                           'scope': scope, 'passed': passed, 'sha256': digest(path),
                                           'historicalReceiptPreserved': True})
manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
paths = [Path(__file__), TERMINAL, ROOT / 'src/court/returning_view_registration.py',
         HERE / 'seal_registration_storage_2026-10-06_initial.py',
         ROOT / 'tests/test_frame_registration_sequence.py', ROOT / 'tests/test_pipeline_registration_storage.py']
paths.extend(path for path, _, _ in evidence)
paths.extend(ROOT / name for name in DOCS)
paths.extend(ROOT / name for name in protocol['sourceBindingsBefore'])
paths.extend(ROOT / name for name in resource['sourceBindingsAfter'])
for probe in probe_paths:
    paths.extend(path for path in probe.iterdir() if path.is_file())
save(DEST, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareEvidenceVerified': True,
            'backendTestsPassed': 1102, 'real30FrameCompatibilityPassed': True,
            'registrationLargeBenchmarkPassed': False, 'priorFiveMinuteWorkloadPassed': False,
            'diagnosticProbeCount': len(probes), 'probes': probes, 'savedAnalysisRowsUnchanged': 10,
            'documents': {name: digest(ROOT / name) for name in DOCS},
            'preservedPriorDocuments': {name: digest(COPY / name) for name in DOCS},
            'evidence': {path.relative_to(ROOT).as_posix(): digest(path) for path in set(paths)},
            'productionQualified': False, 'physicalSpeedAnd3DQualified': False,
            'remainingWork': 'Bound player/trajectory/result/API/frontend storage; independent accuracy and meaningful capacity qualification.'})
print(json.dumps({'softwareEvidenceVerified': True, 'backendTestsPassed': 1102, 'real30FrameCompatibilityPassed': True,
                  'registrationLargeBenchmarkPassed': False, 'diagnosticProbeCount': len(probes), 'savedAnalysisRowsUnchanged': 10}))

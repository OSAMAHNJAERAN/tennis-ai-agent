"""Read back current tested event lookup evidence without running any workload."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request
import psutil

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).parent
RECEIPT = UNIT / 'final_verification.json'
assert not RECEIPT.exists()
bindings = {}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def resolve(name):
    return Path(name) if Path(name).is_absolute() else ROOT / name

def bind(path, expected=None):
    path = Path(path).resolve()
    value = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    if isinstance(expected, dict):
        assert value == expected, str(path)
    elif expected is not None:
        assert value['sha256'] == expected, str(path)
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    if name in bindings:
        assert bindings[name] == value, 'Binding changed during seal: ' + name
    bindings[name] = value

def retired(identity):
    try:
        assert psutil.Process(identity['pid']).create_time() != identity['createTimeUnixSeconds']
    except psutil.NoSuchProcess:
        pass

suite_dir = UNIT / 'backend_final'
suite = read(suite_dir / 'receipt.json')
assert suite['passed'] and suite['sourcesUnchanged']
assert suite['counts'] == {'tests': 1316, 'failures': 0, 'errors': 0, 'skipped': 0}
for name, expected in read(suite_dir / 'source_bindings.json').items():
    bind(ROOT / name, expected)
for name, expected in suite['artifacts'].items():
    bind(suite_dir / name, expected)
bind(suite_dir / 'receipt.json')
failed_suite = read(UNIT / 'backend/receipt.json')
assert not failed_suite['passed'] and failed_suite['counts'] == {'tests': 1316, 'failures': 1, 'errors': 0, 'skipped': 0}
for name, expected in failed_suite['artifacts'].items():
    bind(UNIT / 'backend' / name, expected)
bind(UNIT / 'backend/receipt.json')
red = read(UNIT / 'red_receipt.json')
green = read(UNIT / 'green_run_2026-10-06/green_receipt.json')
assert red['status'] == 'RED' and red['failures'] == 1 and red['tests'] == 33
assert green['status'] == 'GREEN' and green['failures'] == green['errors'] == green['skipped'] == 0 and green['tests'] == 38
bind(ROOT / 'src/events/event_detector.py', green['bindings']['sourceSha256'])
bind(ROOT / 'tests/test_event_timestamp_lookup.py', green['bindings']['testSha256'])
bind(UNIT / 'src/events/event_detector.py', red['bindings']['sourceSha256'])
bind(UNIT / 'red_test_snapshot.py', red['bindings']['testSha256'])
bind(UNIT / 'red.xml', red['xmlSha256'])
bind(UNIT / 'green_run_2026-10-06/green.xml', green['xmlSha256'])
for old, new in zip(red['nearestTimestampMeasurements']['measurements'], green['nearestTimestampMeasurements']['measurements']):
    assert old['frames'] == new['frames'] and old['nearestQueries'] == new['nearestQueries']
    assert old['featureSha256'] == new['featureSha256'] and new['timestampReads'] < old['timestampReads']

first_measurement = UNIT / 'measurement'
first = read(first_measurement / 'run/report.json')
assert not first['passed'] and len(first['runs']) == 2 and not first['comparisons']
assert first['runs'][0]['passed'] and first['runs'][1]['status'] == 'NOT_STARTED'
assert not (first_measurement / 'run/600_new/child_receipt.json').exists()
for identity in first['runs'][0]['ownedProcessIdentitiesAndPeaks']:
    retired(identity)

measurement = UNIT / 'measurement_headroom'
report = read(measurement / 'run/report.json')
assert report['passed'] and report['status'] == 'COMPLETED' and report['sourcesUnchanged']
assert len(report['runs']) == 4 and len(report['comparisons']) == 2
assert [(run['frames'], run['mode']) for run in report['runs']] == [(600, 'old'), (600, 'new'), (18000, 'old'), (18000, 'new')]
assert [comparison['frames'] for comparison in report['comparisons']] == [600, 18000]
first_bindings = read(measurement / 'first_measurement_bindings.json')
assert set(first_bindings) == {path.relative_to(ROOT).as_posix() for path in first_measurement.rglob('*') if path.is_file()}
for name, expected in first_bindings.items():
    bind(ROOT / name, expected)
assert report['firstMeasurementFilesUnchanged']
assert report['firstMeasurementHistoryBefore'] == report['firstMeasurementHistoryAfter']
assert report['firstMeasurementHistoryAfter']['unchanged']
assert report['firstMeasurementHistoryAfter']['fileCount'] == len(first_bindings)
assert report['sourceBindingsBefore'] == report['sourceBindingsAfter']
for name, expected in report['sourceBindingsAfter'].items():
    bind(resolve(name), expected)
for name, expected in report['actualImportedSourceBindingsUnion'].items():
    bind(resolve(name), expected)
for comparison in report['comparisons']:
    assert comparison['passed'] and not comparison['featureFieldExclusions']
    assert set(comparison['outputs']) == {'fixture.json', 'settings.json', 'features.json'}
    for name, values in comparison['outputs'].items():
        assert values['exactBytesEqual'] and values['old'] == values['new']
        for mode in ('old', 'new'):
            bind(measurement / 'run' / f"{comparison['frames']}_{mode}" / name, values[mode])
for run in report['runs']:
    assert run['passed'] and run['actualWorkloadWasSampled'] and run['allOwnedIdentitiesRetiredAtTerminal']
    assert not run['remainingOwnedIdentitiesAtTerminal']
    child_path = measurement / 'run' / f"{run['frames']}_{run['mode']}" / 'child_receipt.json'
    bind(child_path, run['childReceiptBinding'])
    child = read(child_path)
    assert child['passed'] and not child['modelsImported'] and not child['inferenceRun']
    for identity in run['ownedProcessIdentitiesAndPeaks']:
        retired(identity)

replay_dir = UNIT / 'saved_diagnostic_replay_2026-10-06'
replay = read(replay_dir / 'replay_receipt_v3.json')
assert replay['passed'] and replay['sourceBindingsUnchanged'] and replay['historicalArtifactsUnchanged']
assert replay['preparationUnchanged'] and replay['actualWorkloadSampledByExactIdentity']
assert replay['allExactOwnedIdentitiesRetiredAtTerminal'] and len(replay['comparisons']) == 21
assert all(item['exactAnalysisBytes'] for item in replay['comparisons'])
assert sum(item.get('historicalHFilteredEventsEqual', False) for item in replay['comparisons']) == 3
expected_variants = {'B_HIGH_RECALL_CANDIDATES', 'C_TEMPORAL_PLAYER_PROXIMITY', 'D_EVENT_TIME_REFINEMENT',
                     'E_EVENT_SPECIFIC_VERIFICATION', 'F_VISION_DEAD_BALL_GATING',
                     'G_CAMERA_MOTION_COMPENSATION', 'H_FINAL_INTEGRATED'}
assert {(item['variant'], item['video']) for item in replay['comparisons']} == {
    (variant, video) for variant in expected_variants for video in ('video_08', 'video_09', 'video_10')}
for item in replay['comparisons']:
    assert item['frozenSha256'] == item['currentSha256']
    for mode, sha_key in (('frozen', 'frozenSha256'), ('current', 'currentSha256')):
        bind(replay_dir / 'run_v3' / f"{item['variant']}_{item['video']}_{mode}.json",
             {'bytes': item['bytes'], 'sha256': item[sha_key]})
assert replay['consumedV2HistoryUnchanged']
replay_protocol = read(replay_dir / 'protocol_v3.json')
assert replay_protocol['consumedV2HistoryBefore'] == replay['consumedV2HistoryAfter']
consumed_v2 = read(replay_dir / 'v2_history_bindings.json')
assert replay['consumedV2HistoryAfter'] == {key: consumed_v2[key] for key in ('files', 'directories')}
assert set(consumed_v2['files']) == {path.name for path in replay_dir.iterdir() if path.is_file()
                                  and ('_v2' in path.name or path.name == 'prepared_revision_2.json')}
assert consumed_v2['directories'] == ['run_v2'] and (replay_dir / 'run_v2').is_dir()
assert not list((replay_dir / 'run_v2').iterdir())
for name, expected in consumed_v2['files'].items():
    bind(replay_dir / name, expected)
for key in ('sourceBindingsAfter', 'historicalArtifactsAfter'):
    for name, expected in replay[key].items():
        bind(resolve(name), expected)
for identity in replay['ownedProcessIdentitiesAndPeaks']:
    retired(identity)
replay_child = read(replay_dir / 'child_receipt_v3.json')
assert replay_child['passed'] and not replay_child['networkAttempts'] and not replay_child['forbiddenInferenceImports']
assert replay['ordinaryImportedBindingsVerifiedBeforeParentPass'] and replay['ordinaryImportedModuleSourcesUnchangedAtTerminal']
assert replay_child['ordinaryImportedSourcesUnchangedAfterCalculation'] and replay_child['samplingAcknowledgementMatchesExactIdentity']
for key in ('ordinaryImportedModuleBindingsBefore', 'ordinaryImportedModuleBindingsAfter'):
    for name, expected in replay_child[key].items():
        bind(resolve(name), expected)
base_interpreter = replay['baseInterpreterBindingAtTerminal']
bind(Path(base_interpreter['path']), {'bytes': base_interpreter['bytes'], 'sha256': base_interpreter['sha256']})

documentation = read(UNIT / 'documentation_receipt.json')
assert documentation['passed']
# The documentation helper records all checked source/evidence files and the
# exact resulting documents. Assert its recorded bindings, rather than merely
# accepting whatever happens to be on disk at this later check.
for key in ('sourceBindings', 'evidenceBindings'):
    for name, expected in documentation[key].items():
        bind(resolve(name), expected)
correction = read(UNIT / 'installation_correction_receipt.json')
assert correction['passed'] and correction['onlyTwoAssetBindingsChanged']
assert correction['firstDocumentationAndHostEvidencePreservedExact'] and correction['immutableEvidenceUnchanged']
for name, expected in documentation['currentDocumentHashes'].items():
    preserved = correction['firstDocumentationAfterPreservedAs'][name]
    assert {key: preserved[key] for key in ('bytes', 'sha256')} == expected
    bind(ROOT / preserved['path'], expected)
for key in ('sourceBindings', 'evidenceBindings', 'beforeBackupBindings'):
    for name, expected in correction[key].items():
        bind(resolve(name), expected)
for name, expected in correction['currentDocumentHashes'].items():
    bind(ROOT / name, expected)
frontend_restore = read(UNIT / 'frontend_dependency_restore_verification.json')
assert frontend_restore['passed'] and frontend_restore['selectedAndCopiedInputsUnchanged']
assert frontend_restore['selectedAndCopiedBuildsExact'] and frontend_restore['frontendInputsExact'] == 127
assert frontend_restore['frontendTestsPassed'] == 165 and frontend_restore['productionFilesExact'] == 47
for name, expected in frontend_restore['bindings'].items():
    bind(resolve(name), expected)

failed_host = read(UNIT / 'verify_installation.json')
assert not failed_host['passed'] and failed_host['inputsUnchanged'] and failed_host['exitCode'] == 1
bind(ROOT / failed_host['report'], failed_host['reportSha256'])
bind(UNIT / 'verify_installation.log', failed_host['logSha256'])
failed_host_report = read(ROOT / failed_host['report'])
assert len(failed_host_report['checks']) == 75
assert {item['name'] for item in failed_host_report['checks'] if not item['passed']} == {
    'asset:frontend/package.json', 'asset:frontend/pnpm-lock.yaml'}
host_wrapper = read(UNIT / 'verify_installation_final.json')
assert host_wrapper['passed'] and host_wrapper['inputsUnchanged'] and host_wrapper['exitCode'] == 0
assert host_wrapper['checkCount'] == 75 and not host_wrapper['inferenceRun']
for name, expected in host_wrapper['inputs'].items():
    bind(ROOT / name, expected)
host_path = ROOT / host_wrapper['report']
bind(host_path, host_wrapper['reportSha256'])
bind(UNIT / 'verify_installation_final.log', host_wrapper['logSha256'])
host = read(host_path)
assert host['checksPassed'] and len(host['checks']) == 75 and not host['inferenceRun'] and not host['modelsImported']
assert host['manifestSha256'] == digest(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')

runtime = read(UNIT / 'runtime_verification.json')
assert runtime['passed'] and len(runtime['exports']) == 12 and len(runtime['frontend']['assets']) == 7
assert runtime['snapshotBefore'] == runtime['snapshotAfter'] and runtime['savedJobsAndAnalysisFilesUnchanged']
for name, expected in runtime['sourceBindings']['files'].items():
    bind(ROOT / name, expected)
bind(Path(runtime['sourceBindings']['baseInterpreter']['path']), runtime['sourceBindings']['baseInterpreter']['sha256'])
with sqlite3.connect((ROOT / 'outputs/product/analyses.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
files = {path.relative_to(ROOT).as_posix(): digest(path)
         for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
assert rows == runtime['snapshotAfter']['jobs'] and len(rows) == 10
assert files == runtime['snapshotAfter']['savedAnalysisFiles'] and len(files) == 319
for identity in runtime['runtimeProcesses']:
    process = psutil.Process(identity['pid'])
    assert process.create_time() == identity['createTimeUnixSeconds']
    assert process.cmdline() == identity['cmdline'] and Path(process.cwd()).resolve() == ROOT
with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=10) as response:
    health = json.loads(response.read())
    assert response.status == 200 and health['status'] == 'ok' and health['worker_available']

for path in UNIT.rglob('*'):
    if path.is_file() and not any(part in ('tmp', 'bootstrap', '__pycache__') for part in path.relative_to(UNIT).parts):
        bind(path)
assert all({'bytes': resolve(name).stat().st_size, 'sha256': digest(resolve(name))} == value
           for name, value in bindings.items())
result = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'backendTestsPassed': 1316, 'cleanHostChecksPassed': 75, 'focusedRegressionTestsPassed': 38,
          'exactSavedDiagnosticAnalysisPairs': 21, 'syntheticFeatureCounts': [600, 18000],
          'allMonitoredEventValidationProcessesRetiredAtSeal': True, 'savedJobsUnchanged': 10,
          'savedAnalysisFilesUnchanged': 319, 'runtimeProcesses': runtime['runtimeProcesses'],
          'frontendAssetsExact': 7, 'currentRevisionModelInferenceRun': False,
          'currentFrontendTestsPassed': 165, 'currentFrontendInputsExact': 127,
          'currentFrontendProductionFilesExact': 47,
          'productionQualified': False, 'accuracyQualified': False, 'fullMatchQualified': False,
          'physicalSpeedAnd3DQualified': False, 'bindings': bindings,
          'scope': 'Event timestamp lookup software/compatibility and synthetic timing evidence. Historical diagnostic metrics remain historical; no new model or scientific qualification.',
          'remainingWork': 'Repair concurrent frontend regressions: sticky upload header, analyzed previews on first screen, saved history independent of demo catalog. Pose/racket/player/trajectory and browser storage; other event costs; guarded real-worker checks; authorized untouched qualification footage, labels, independent speed/3D references, deployment licensing and representative full-match capacity.'}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({key: result[key] for key in ('passed', 'backendTestsPassed', 'exactSavedDiagnosticAnalysisPairs',
                 'cleanHostChecksPassed', 'savedJobsUnchanged', 'productionQualified')}))

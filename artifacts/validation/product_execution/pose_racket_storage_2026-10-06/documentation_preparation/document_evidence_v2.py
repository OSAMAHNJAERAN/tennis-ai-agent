"""Prepared one-shot documentation publication. Root dispatch is required.

Only --apply writes the three documentation files. Evidence is read without
importing product/model code. Historical content and manifest fields survive.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
ROOT = UNIT.parents[3]
PREPARATION = HERE / 'preparation_v2.json'
RECEIPT = UNIT / 'documentation_receipt.json'
DOCS = ('docs/plans/TENNIS_EXECUTION_PROGRESS.md',
        'docs/deployment/PRODUCT_INSTALLATION.md',
        'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
TRACKED = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size,
                'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def track(path):
    path = Path(path)
    value = binding(path)
    key = name(path)
    require(key not in TRACKED or TRACKED[key] == value, 'Input changed during read: ' + key)
    TRACKED[key] = value
    return path


def load(path):
    return json.loads(track(path).read_text(encoding='utf-8'))


def verify(values, tracked=True):
    for pathname, expected in values.items():
        path = resolve(pathname)
        actual = binding(path)
        require(actual == expected if isinstance(expected, dict) else actual['sha256'] == expected,
                'Evidence/source changed: ' + pathname)
        if tracked:
            track(path)


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def junit(path):
    root = ET.fromstring(track(path).read_bytes())
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    result = {key: sum(int(item.get(key, '0')) for item in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    result['passed'] = result['tests'] - result['failures'] - result['errors'] - result['skipped']
    return result


def software_gate():
    red = load(UNIT / 'pipeline_red/valid_red/receipt.json')
    red_counts = junit(UNIT / 'pipeline_red/valid_red/junit.xml')
    require(red['status'] == 'GENUINE_RED' and red['exitCode'] == 1 and
            red_counts == {'tests': 17, 'failures': 7, 'errors': 0, 'skipped': 0, 'passed': 10},
            'Require the valid behavioral RED, not either preserved setup failure')
    for report, before, after in ((red, 'sourceBefore', 'sourceAfter'),):
        require(report[before] == report[after] and report['allOwnedIdentitiesRetired'], 'RED source/retirement gate')
    # RED is bound to preserved pre-change sources, so do not compare its original
    # pipeline hash to current integrated production.
    storage = load(UNIT / 'storage_focused/receipt.json')
    storage_counts = junit(UNIT / 'storage_focused/junit.xml')
    require(storage['exitCode'] == 0 and storage['sourceUnchanged'] and
            storage['sourceBefore'] == storage['sourceAfter'] and storage_counts['passed'] == 57 and
            not any(storage_counts[key] for key in ('failures', 'errors', 'skipped')), 'Storage focused evidence')
    green = load(UNIT / 'pipeline_green/receipt.json')
    green_counts = junit(UNIT / 'pipeline_green/junit.xml')
    require(green['status'] == 'GREEN' and green['exitCode'] == 0 and green['allOwnedIdentitiesRetired'] and
            green['sourceBindingsUnchanged'] and green['sourceBefore'] == green['sourceAfter'] and
            green_counts['passed'] == 17 and not any(green_counts[key] for key in ('failures', 'errors', 'skipped')),
            'Pipeline GREEN source/count/retirement evidence')
    verify(green['sourceBefore'])
    backend = load(UNIT / 'backend_full/receipt.json')
    counts = junit(UNIT / 'backend_full/junit.xml')
    require(backend['passed'] and backend['status'] == 'PASSED' and backend['exitCode'] == 0 and
            backend['sourcesUnchanged'] and backend['allOwnedIdentitiesRetired'] and
            backend['baseInterpreterAfterMatchesBefore'] and counts == backend['counts'] and
            counts['passed'] == 1390 and not any(counts[key] for key in ('failures', 'errors', 'skipped')),
            'Fresh full backend evidence')
    before = load(UNIT / 'backend_full/source_bindings_before.json')
    after = load(UNIT / 'backend_full/source_bindings_after.json')
    require(before == after, 'Backend complete source inventory differs')
    verify(before['files'])
    for key in ('baseInterpreter', 'actualHelperInterpreter'):
        row = before[key]
        require(binding(Path(row['path'])) == {field: row[field] for field in ('bytes', 'sha256')},
                'Backend interpreter changed')
        track(row['path'])
    contract = load(UNIT / 'measurement/contract_tests/receipt.json')
    contract_counts = junit(UNIT / 'measurement/contract_tests/junit.xml')
    require(contract.get('passed') and contract_counts['passed'] == 8 and
            not any(contract_counts[key] for key in ('failures', 'errors', 'skipped')), 'Measurement helper contract tests')
    return {'validRed': red_counts, 'storageFocused': storage_counts, 'pipelineGreen': green_counts,
            'fullBackend': counts, 'measurementHelperContracts': contract_counts}


def measurement_gate():
    report = load(UNIT / 'measurement/run/report.json')
    require(report['status'] in ('COMPLETED', 'FAILED_PARTIAL', 'NOT_STARTED') and
            report['inferenceRun'] is False and report['modelsImported'] is False and
            report['qualificationEvidence'] is False and report['actualRenderOrMediaRun'] is False,
            'Measurement must be terminal synthetic storage evidence')
    require(report.get('sourcesUnchanged') and report.get('allCompleteFixtureInputsUnchanged') and
            report.get('actualImportedSourcesUnchangedAcrossMeasuredCases'), 'Measurement terminal source/input gates')
    verify(report['sourceBindingsBefore'])
    require(report['sourceBindingsBefore'] == report['sourceBindingsAfter'], 'Measurement sources differ')
    verify(report['inputBindingsBefore'])
    require(report['inputBindingsBefore'] == report['inputBindingsAfter'], 'Measurement full fixtures differ')
    verify(report['actualImportedSourceBindingsUnion'])
    verify(report['retainedFileBindingsAtTerminal'])
    if report['runs']:
        require(report['freshBackendPrerequisiteUnchangedAtTerminal'], 'Backend changed during measurement')
    runs = []
    for row in report['runs']:
        directory = UNIT / 'measurement/run' / (str(row['frames']) + '_' + row['mode'])
        resource = load(directory / 'resource_receipt.json')
        require(row == resource, 'Embedded resource receipt differs')
        require(row['status'] in ('COMPLETED', 'FAILED', 'NOT_STARTED'), 'Nonterminal measured case')
        if row['childStarted']:
            require(row['allOwnedIdentitiesRetiredAtTerminal'] and not row['remainingOwnedIdentitiesAtTerminal'],
                    'Measured case retains an owned process')
        summary = {key: row.get(key) for key in ('frames', 'mode', 'status', 'passed', 'childStarted',
                   'actualWorkloadWasSampled', 'actualOsPeakWorkingSetBytes', 'actualOsPeakProcessCommitBytes',
                   'peakSampledOwnedTreeRssBytes', 'peakSampledOwnedTreePrivateBytes',
                   'monitoredChildWallSeconds', 'minimumSampledAvailableRamBytes', 'safeguardStop', 'error')}
        if row['passed']:
            require(row['status'] == 'COMPLETED' and row['actualWorkloadWasSampled'], 'Measured workload identity not sampled')
            child = load(directory / 'child_receipt.json')
            require(binding(directory / 'child_receipt.json') == row['childReceiptBinding'] and child['passed'] and
                    child['actualWorkloadIdentity'] == row['actualWorkloadIdentity'] and child['ownedScratchCleanupPassed'],
                    'Child output/identity/cleanup evidence')
            for field in ('actualImportedSourceBindingsBefore', 'actualImportedSourceBindingsAfter'):
                verify(child[field])
            for pathname, value in child['outputBindings'].items():
                require(binding(directory / pathname) == value, 'Measured output changed: ' + pathname)
                track(directory / pathname)
            summary['childStages'] = {key: child.get(key) for key in (
                'completeObservationConstructionAndAppendSeconds', 'indexedRenderLikeReadsSeconds',
                'playerMotionSerializationSeconds', 'racketTrackingSerializationSeconds',
                'completeNativeFidelitySerializationSeconds', 'outputHashVerificationSeconds',
                'sequenceContextCleanupSeconds', 'totalChildElapsedSeconds', 'storageDisk')}
        runs.append(summary)
    for pair in report['comparisons']:
        require(pair['passed'] and pair['fieldExclusions'] == [] and
                all(item['exactBytesEqual'] and item['old'] == item['new'] for item in pair['outputs'].values()),
                'A recorded complete pair differs')
    if report['passed']:
        require(report['status'] == 'COMPLETED' and len(runs) == 4 and len(report['comparisons']) == 2 and
                {(row['frames'], row['mode']) for row in runs} == {(n, m) for n in (600, 18000) for m in ('old', 'new')},
                'Completed measurement lacks all four cases')
    else:
        require(report['status'] != 'COMPLETED', 'Failed benchmark labeled completed')
    return {'status': report['status'], 'passed': report['passed'], 'runs': runs,
            'comparisons': report['comparisons'], 'error': report.get('error'),
            'memoryScope': report['memoryScope'], 'timingScope': report['timingScope'],
            'qualificationEvidence': False}


def independent_measurement_gate():
    audit = load(UNIT / 'measurement/terminal_audit.json')
    require(audit['passed'] and audit['status'] == 'PASSED' and
            audit['allAuditBindingsUnchanged'] and audit['fullFileComparisonsIndependentlyPassed'] == 8 and
            audit['nativeRowsAuditedAcrossBothModes'] == 37200 and len(audit['runs']) == 4 and
            len(audit['comparisons']) == 2, 'Independent complete-row/byte/native/type/process audit')
    verify(audit['inputAndEvidenceBindings'])
    return {'passed': True, 'status': 'PASSED', 'fullFileComparisons': 8,
            'nativeRowsAcrossBothModes': 37200, 'receipt': name(UNIT / 'measurement/terminal_audit.json'),
            'receiptBinding': binding(UNIT / 'measurement/terminal_audit.json')}


def worker_gate():
    directory = UNIT / 'worker_smoke_v2'
    report = load(directory / 'resource_receipt.json')
    require(report['status'] in ('COMPLETED', 'NOT_STARTED') and report['productionStatePublished'] is False,
            'Worker requires completed smoke or declared preflight refusal')
    if report['status'] == 'NOT_STARTED':
        require(report['passed'] is False and report['inferenceChildStarted'] is False and report['samples'] == 0 and
                not (directory / 'run').exists() and '2.5 GB' in report.get('notStartedReason', ''),
                'Declined worker must have no child/models/run directory under the original RAM guard')
        return {'status': 'NOT_STARTED', 'passed': False, 'inferenceRun': False,
                'notStartedReason': report['notStartedReason'],
                'availableRamBytes': report.get('availableRamImmediatelyBeforeLaunchBytes', report.get('minimumSampledAvailableRamBytes'))}
    require(report['passed'] and report['inferenceChildStarted'] and report['isolatedValidationJobPublished'] and
            report['actualWorkloadSampledByExactIdentity'] and report['allExactOwnedIdentitiesRetiredAtTerminal'] and
            not report['remainingExactOwnedIdentitiesAtTerminal'], 'Worker publication/exact owned retirement gate')
    for key in ('sourceBindingsUnchanged', 'historicalArtifactsUnchanged', 'productionJobStatesUnchanged',
                'liveProductFileTreeUnchanged', 'originHistoricalArtifactsUnchanged',
                'firstLatencyAttemptHistoryUnchanged', 'secondLatencyAttemptHistoryUnchanged'):
        require(report[key], 'Worker continuity gate: ' + key)
    protocol = load(directory / 'protocol.json')
    verify(protocol['sourceBindingsBefore'])
    comparison = load(directory / 'saved_result_comparison.json')
    require(comparison['passed'] and comparison['comparisonUsesSameNewRawBytes'] and
            comparison['databaseIntegrity'] == 'ok' and comparison['lazyDiskSeriesReader'] and
            comparison['closedOwnedReaderConnection'] and comparison['unpublishedScratchFiles'] == [] and
            comparison['volatileRuntimeFieldsRemoved'] == [] and comparison['qualificationEvidence'] is False and
            all(row['exactBytes'] for row in comparison['comparisons'].values()) and
            comparison['enabledRawPoseRacket']['enabled'] and
            comparison['enabledRawPoseRacket']['rawFrameCounts'] == {'pose_frames': 30, 'racket_frames': 30} and
            all(value == 30 for value in comparison['normalizedSectionCounts'].values()),
            'Enabled complete raw/normalized sections and byte-exact same-artifact result/report comparison')
    return {'status': 'COMPLETED', 'passed': True, 'inferenceRun': True,
            'actualWorkloadIdentity': report['actualWorkloadIdentity'],
            'actualWindowsPeakWorkingSetBytes': report.get('actualWorkload_windowsPeakWorkingSetBytes'),
            'peakSampledOwnedTreeRssBytes': report.get('peakSampledOwnedTreeRssBytes'),
            'comparisons': comparison['comparisons'], 'enabledRawPoseRacket': comparison['enabledRawPoseRacket'],
            'normalizedSectionCounts': comparison['normalizedSectionCounts'], 'qualificationEvidence': False}


def runtime_gate():
    preflight = load(UNIT / 'runtime_preflight.json')
    stopped = load(UNIT / 'idle_stop_receipt.json')
    report = load(UNIT / 'runtime_verification.json')
    require(stopped['passed'] and stopped['acceptedWorkCancelled'] is False and
            stopped['persistentFilesRemoved'] is False and stopped['savedJobsAndAnalysisFilesUnchanged'] and
            stopped['preflightSha256'] == binding(UNIT / 'runtime_preflight.json')['sha256'], 'Idle stop provenance')
    require(report['passed'] and report['runtimeRetainedWithoutRestart'] is False and
            report['savedJobsAndAnalysisFilesUnchanged'] and report['productStateChanged'] is False and
            report['runtimeStoppedOrStartedByThisVerifier'] is False and
            report['preflightSha256'] == binding(UNIT / 'runtime_preflight.json')['sha256'] and
            report['snapshotBefore'] == report['snapshotAfter'] == preflight['snapshot'] and
            report['sourceBindings'] == preflight['sourceBindings'] and report['catalogEntries'] == 3 and
            len(report['exports']) == 12 and report['savedJobs'] == 10 and report['savedAnalysisFiles'] == 319 and
            len(report['frontend']['assets']) == 7, 'Fresh idle reload and existing product compatibility')
    retired = report['previousExactRuntimeIdentitiesRetired']
    require(retired and all(row.get('retired') is True for row in retired), 'Previous exact runtime remains')
    verify(report['sourceBindings']['files'])
    return {'passed': True, 'freshIdleReload': True, 'catalogEntries': report['catalogEntries'],
            'exports': len(report['exports']), 'savedJobs': report['savedJobs'],
            'savedAnalysisFiles': report['savedAnalysisFiles'], 'frontendAssets': len(report['frontend']['assets']),
            'productStateChanged': False, 'inferenceRun': False}


def prepare_documents(evidence):
    originals = {key: track(ROOT / key).read_bytes() for key in DOCS}
    manifest = json.loads(originals[DOCS[2]])
    require(manifest['productionQualified'] is False, 'Manifest production qualification unexpectedly enabled')
    measured = evidence['measurement']
    if measured['passed']:
        metric_lines = []
        for count in (600, 18000):
            pair = [row for row in measured['runs'] if row['frames'] == count]
            metrics = '; '.join(str(row['mode']) + ' actual Windows peak working set ' +
                                str(row['actualOsPeakWorkingSetBytes']) + ' B and peak process commit ' +
                                str(row['actualOsPeakProcessCommitBytes']) + ' B, monitored wall ' +
                                str(row['monitoredChildWallSeconds']) + ' s' for row in pair)
            metric_lines.append(f'{count:,} frames: {metrics}.')
        large = {row['mode']: row for row in measured['runs'] if row['frames'] == 18000}
        stages = ('completeObservationConstructionAndAppendSeconds', 'indexedRenderLikeReadsSeconds',
                  'playerMotionSerializationSeconds', 'racketTrackingSerializationSeconds')
        tradeoff = (' At 18,000 frames, measured old/new construction, indexed reads, pose export and racket export seconds were ' +
                    '; '.join(str(large['old']['childStages'][key]) + '/' + str(large['new']['childStages'][key])
                              for key in stages) + '. The disk-backed path reduced retained memory while these measured operations took longer. '
                    'Its owned database/sidecars totaled ' +
                    str(large['new']['childStages']['storageDisk']['databaseAndSidecarLogicalBytesBeforeCleanup']) +
                    ' logical bytes before successful owned cleanup; this is not allocated disk blocks or total output size. '
                    'The child totals include imports, native fidelity verification and output hashing; stage sums are not the complete workload wall time.')
        measurement_text = ('The one-shot synthetic old/new storage comparison completed both 600- and 18,000-frame pairs. '
                            'Complete native observation records and unmodified player-motion/racket JSON bytes matched, with no excluded fields. ' +
                            ' '.join(metric_lines) + ' These are storage/export workloads without models, media decoding or actual rendering. '
                            'OS process peaks differ from sampled aggregate owned-tree RSS; process commit is not disk usage.' + tradeoff)
    else:
        measurement_text = ('The synthetic storage measurement ended ' + measured['status'] + ': ' + str(measured.get('error')) +
                            '. Only ' + str(len(measured['runs'])) + ' cases and ' + str(len(measured['comparisons'])) +
                            ' complete comparisons are recorded; later cases stopped at the first failure/refusal. '
                            'It does not establish a complete paired performance result.')
    worker = evidence['worker']
    if worker['passed']:
        worker_text = ('A fresh isolated actual worker completed the licensed 30-frame native-60-FPS excerpt with both pose and racket recipes enabled. '
                       'All 30 raw dictionary frames and normalized frames were present; legitimate missing observations remained null. '
                       'The closed committed result reopened lazily and JSON/CSV/HTML matched the frozen resident implementation on the same new raw artifacts exactly. '
                       'The original input remained unchanged. Worker CRF18 H.264 normalization retained native dimensions; normalized pixels are not claimed lossless. '
                       'Physical ball speed/3D, unavailable court coordinates and authoritative events remained withheld. '
                       'This short smoke is compatibility evidence, not accuracy or full-match capacity qualification.')
    else:
        worker_text = ('The fresh worker smoke was NOT_STARTED because the original 2.5 GB available-RAM preflight guard declined launch (' +
                       str(worker['notStartedReason']) + ' Actual available RAM: ' + str(worker['availableRamBytes']) +
                       ' B). No inference child or model work began; no automatic retry was performed. '
                       'The real-worker compatibility smoke remains pending.')
    note = ('Pose/racket record storage update — 2026-10-06\n\n'
            'The production pipeline now writes enabled pose/racket frame dictionaries to exclusively owned SQLite sequences during their existing passes. '
            'Rendering and final raw exporters read bounded snapshots; successful completion removes only owned scratch files, while failure closes and retains diagnostic rows. '
            'Native scalar/NumPy values, integer identities, missing frames, enabled/disabled behavior and raw export bytes are preserved. '
            'Render capture/writer cleanup now covers open/read/draw/write failures and preserves primary errors.\n\n'
            'The valid behavioral RED had 7 failures and 10 passes; the two earlier setup failures remain preserved and are not counted as regression evidence. '
            'The focused utility suite passed 57, integrated pipeline suite passed 17, helper contract suite passed 8, and the fresh full backend passed 1,390 tests. '
            'The software fixtures use deterministic substitutes; they do not test model accuracy.\n\n' + measurement_text + '\n\n' + worker_text + '\n\n'
            'A fresh guarded idle reload retired the prior exact runtime identities. The existing three public examples, 12 JSON/CSV/HTML exports, '
            '10 saved jobs, 319 saved analysis files and seven browser entry assets passed compatibility checks without changing saved product state. '
            'The earlier frontend restoration remains historical and unchanged: 169 tests, 127 source inputs and 47 production files; no new browser-test claim is added.\n\n'
            'This unit bounds only retained pose/racket observations and their late raw export construction. Selected-player records and whole-segment role selection, '
            'motion/trajectory/timestamp arrays and other remaining consumers still grow with video length. Full-match capacity, diverse accuracy, '
            'independent speed/3D references, browser memory and production qualification remain unresolved. Selected configuration, models, licenses and public-source credits are unchanged.\n\n'
            'Evidence: `artifacts/validation/product_execution/pose_racket_storage_2026-10-06/` (valid RED, focused GREEN, backend, '
            'measurement terminal receipts, worker status and fresh runtime verification).\n\n')
    outputs = {}
    for pathname in DOCS[:2]:
        data = originals[pathname]
        newline = b'\r\n' if b'\r\n' in data else b'\n'
        encoded = note.replace('\n', newline.decode('ascii')).encode('utf-8')
        if pathname == DOCS[0]:
            anchor = b'## Resume here' + newline + newline
        else:
            anchor = data.split(newline, 1)[0] + newline + newline
        require(data.count(anchor) == 1, 'Documentation insertion anchor is not unique: ' + pathname)
        outputs[pathname] = data.replace(anchor, anchor + encoded, 1)
        require(outputs[pathname].replace(anchor + encoded, anchor, 1) == data, 'Historical documentation bytes lost')
    current = deepcopy(manifest)
    current['poseRacketStorageEvidenceUpdatedAt'] = stamp()
    current['evidenceUpdatedAt'] = current['poseRacketStorageEvidenceUpdatedAt']
    current['validation']['poseRacketStorageSoftware'] = 'Valid RED 7 failed/10 passed; storage57; pipeline17; fresh full backend1390; bounded pose/racket observation/export scope only.'
    current['validation']['poseRacketStorageSyntheticMeasurement'] = measurement_text
    current['validation']['poseRacketStorageWorker'] = worker_text
    current['validation']['poseRacketStorageRuntime'] = 'Fresh guarded idle reload; 3 demos/12 exports/10 jobs/319 files/7 entry assets unchanged; no inference by runtime verifier.'
    current['validationEvidence'].append(name(RECEIPT))
    for key, value in manifest.items():
        if key not in ('evidenceUpdatedAt', 'validation', 'validationEvidence'):
            require(current[key] == value, 'Prior manifest field changed: ' + key)
    require(all(current['validation'][key] == value for key, value in manifest['validation'].items()) and
            current['validationEvidence'][:-1] == manifest['validationEvidence'], 'Historical validation evidence changed')
    outputs[DOCS[2]] = (json.dumps(current, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    return originals, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--validate-only', action='store_true')
    choices.add_argument('--apply', action='store_true')
    choices.add_argument('--draft-only', action='store_true')
    args = parser.parse_args()
    prepared = load(PREPARATION)
    verify(prepared['immutableInputs'])
    verify(prepared['originalDocuments'])
    before = load(UNIT / 'before.json')
    require(binding(resolve(before['previousFrontendSeal']['path'])) ==
            {key: before['previousFrontendSeal'][key] for key in ('bytes', 'sha256')}, 'Historical frontend final seal changed')
    for pathname in DOCS:
        require(binding(ROOT / pathname) == before['files'][pathname]['before'], 'Docs no longer match unit original')
    evidence = {'software': software_gate(), 'measurement': measurement_gate(),
                'worker': worker_gate(), 'runtime': runtime_gate(),
                'independentMeasurementAudit': independent_measurement_gate(),
                'qualificationEvidence': False, 'productionQualified': False,
                'accuracyQualified': False, 'fullMatchQualified': False, 'physicalSpeedAnd3DQualified': False}
    originals, outputs = prepare_documents(evidence)
    verify(TRACKED, tracked=False)
    if args.draft_only:
        draft = HERE / 'draft_review_v2'
        draft.mkdir(exist_ok=False)
        for pathname, data in outputs.items():
            with (draft / Path(pathname).name).open('xb') as stream:
                stream.write(data)
        save(draft / 'receipt.json', {'preparedAt': stamp(), 'status': 'DRAFT_ONLY',
             'evidence': evidence, 'inputBindings': TRACKED,
             'draftBindings': {Path(key).name: binding(draft / Path(key).name) for key in DOCS},
             'originalDocumentBindings': {key: binding(ROOT / key) for key in DOCS},
             'documentationPublished': False, 'hostVerifierRun': False,
             'modelsImported': False, 'inferenceRun': False, 'qualificationEvidence': False})
        verify(TRACKED, tracked=False)
        print(json.dumps({'passed': True, 'status': 'DRAFT_ONLY', 'draft': name(draft),
                          'documentationPublished': False, 'hostVerifierRun': False}))
        return 0
    if args.validate_only:
        print(json.dumps({'passed': True, 'scope': 'Read-only documentation publication gates',
                          'measurementStatus': evidence['measurement']['status'], 'workerStatus': evidence['worker']['status']}))
        return 0
    require(not RECEIPT.exists() and not (HERE / 'publication').exists(), 'One-shot publication target already consumed')
    publication = HERE / 'publication'
    publication.mkdir(exist_ok=False)
    save(publication / 'launch.json', {'at': stamp(), 'helper': binding(Path(__file__)), 'oneShot': True})
    result = {'startedAt': stamp(), 'status': 'FAILED', 'passed': False, 'evidence': evidence,
              'inputBindings': TRACKED.copy(), 'historicalFrontendPreserved': True,
              'qualificationEvidence': False, 'productionQualified': False, 'documents': {}, 'replaced': []}
    try:
        for index, pathname in enumerate(DOCS):
            (publication / f'{index}.original').write_bytes(originals[pathname])
            with (publication / f'{index}.staged').open('xb') as stream:
                stream.write(outputs[pathname])
            result['documents'][pathname] = {'before': binding(ROOT / pathname),
                'after': binding(publication / f'{index}.staged'),
                'preservedOriginal': name(publication / f'{index}.original')}
        verify(TRACKED, tracked=False)
        # All bytes are prepared/reviewable before any document replacement.
        for index, pathname in enumerate(DOCS):
            require((ROOT / pathname).read_bytes() == originals[pathname], 'Document changed before publication')
            os.replace(publication / f'{index}.staged', ROOT / pathname)
            result['replaced'].append(pathname)
        result['currentDocumentHashes'] = {key: binding(ROOT / key) for key in DOCS}
        require(all(result['currentDocumentHashes'][key] == result['documents'][key]['after'] for key in DOCS),
                'Published document bytes differ')
        verify({key: value for key, value in TRACKED.items() if key not in DOCS}, tracked=False)
        result['inputsUnchangedExceptPublishedDocuments'] = True
        result['status'], result['passed'] = 'COMPLETED', True
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        result['traceback'] = traceback.format_exc()
        result['rollbackErrors'] = []
        for pathname in reversed(result['replaced']):
            try:
                require((ROOT / pathname).read_bytes() == outputs[pathname], 'Concurrent document change; refuse rollback')
                index = DOCS.index(pathname)
                rollback = publication / f'{index}.rollback'
                with rollback.open('xb') as stream:
                    stream.write(originals[pathname])
                os.replace(rollback, ROOT / pathname)
            except BaseException as cleanup:
                result['rollbackErrors'].append(repr(cleanup))
    finally:
        result['completedAt'] = stamp()
        save(RECEIPT, result)
    print(json.dumps({'passed': result['passed'], 'status': result['status'], 'error': result.get('error')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

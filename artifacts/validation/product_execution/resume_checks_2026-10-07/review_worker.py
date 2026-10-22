"""Read-only terminal checks independent of worker execution and golden generation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
WORKER = ROOT / 'artifacts/validation/product_execution/pose_racket_worker_smoke_retry_2026-10-07'

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

output = UNIT / 'worker_review.json'
assert not output.exists()
receipt = load(WORKER / 'resource_receipt.json')
protocol = load(WORKER / 'protocol.json')
review = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
          'attemptStatus': receipt['status'], 'workerPassed': receipt['passed'],
          'scope': 'Independent read-only terminal, retained source, process identity and saved-byte checks. No new inference or accuracy qualification.',
          'qualificationEvidence': False, 'scriptBinding': binding(Path(__file__))}
try:
    assert protocol['preflightAvailableRamBytes'] == 2500000000
    assert protocol['maximumSampledOwnedTreeRssBytes'] == 3000000000
    assert protocol['minimumAvailableRamBytes'] == 400000000
    assert protocol['lowRamConsecutiveSamplesToStop'] == 5
    assert protocol['sampleIntervalSeconds'] == .25
    assert protocol['maximumChildWallSeconds'] == 300
    assert protocol['frames'] == 30 and protocol['nativeFps'] == 60
    assert protocol['launchToken'] == load(WORKER / 'launch_marker.json')['launchToken']
    assert protocol['scriptSha256'] == binding(ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v4.py')['sha256']
    review.update(inferenceChildStarted=receipt['inferenceChildStarted'], samples=receipt['samples'])
    if receipt['inferenceChildStarted']:
        assert receipt['allExactOwnedIdentitiesRetiredAtTerminal']
        assert receipt['remainingExactOwnedIdentitiesAtTerminal'] == []
        sample_rows = [json.loads(line) for line in (WORKER / 'resource_samples.jsonl').read_text().splitlines()]
        assert len(sample_rows) == receipt['samples']
        assert max(row['ownedTreeRssBytes'] for row in sample_rows) == receipt['peakSampledOwnedTreeRssBytes']
        assert min(row['availableRamBytes'] for row in sample_rows) == receipt['minimumSampledAvailableRamBytes']
        identity = load(WORKER / 'actual_child_identity.json')
        assert load(WORKER / 'actual_child_sampled.json')['identity'] == identity
        assert any(any(p['pid'] == identity['pid'] and p['createTimeUnixSeconds'] == identity['createTimeUnixSeconds']
                       for p in row['processes']) for row in sample_rows)
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['createTimeUnixSeconds']
        except psutil.NoSuchProcess:
            pass
    for name, expected in protocol.get('sourceBindingsBefore', {}).items():
        assert binding(ROOT / name) == expected, name
    for key, value in receipt.items():
        if key.endswith('Unchanged'):
            assert value is True, key
    if receipt['passed']:
        assert receipt['status'] == 'COMPLETED' and receipt['inferenceChildStarted']
        assert receipt['actualWorkloadSampledByExactIdentity']
        child = load(WORKER / 'child_receipt.json')
        assert child['passed'] and child['workerInvoked'] and child['inferenceRun']
        assert child['isolatedValidationJobPublished'] and not child['productionStatePublished']
        assert child['networkAttempts'] == []
        comparison = load(WORKER / 'saved_result_comparison.json')
        assert comparison['passed'] and comparison['comparisonUsesSameNewRawBytes']
        assert comparison['volatileRuntimeFieldsRemoved'] == [] and not comparison['presentationApplied']
        assert comparison['databaseIntegrity'] == 'ok' and comparison['closedOwnedReaderConnection']
        assert comparison['unpublishedScratchFiles'] == []
        run = WORKER / 'run'
        for key, expected, actual in [('json', 'golden_result.json', 'reopened_result.json'),
                                      ('csv', 'golden_report.csv', 'reopened_report.csv'),
                                      ('html', 'golden_report.html', 'reopened_report.html')]:
            left, right = binding(run / expected), binding(run / actual)
            assert left == right
            assert comparison['comparisons'][key]['sha256'] == right['sha256']
            assert comparison['comparisons'][key]['bytes'] == right['bytes']
        review['actualWorkerCompletionVerified'] = True
        review['actualSavedResultExactBytesVerified'] = True
        review['enabledRawPoseRacket'] = comparison['enabledRawPoseRacket']
        review['normalizedSectionCounts'] = comparison['normalizedSectionCounts']
    else:
        review['actualWorkerCompletionVerified'] = False
        review['actualSavedResultExactBytesVerified'] = False
        review['failureOrDecline'] = receipt.get('error') or receipt.get('notStartedReason')
    review.update(passed=True, resourceReceiptBinding=binding(WORKER / 'resource_receipt.json'),
                  protocolBinding=binding(WORKER / 'protocol.json'))
except Exception as error:
    review['error'] = repr(error)
finally:
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
print(json.dumps(review, indent=2))
raise SystemExit(0 if review['passed'] else 1)

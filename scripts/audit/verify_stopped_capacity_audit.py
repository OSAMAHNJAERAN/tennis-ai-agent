"""Seal the observed resource-stop evidence without retrying or cleaning inference."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.audit.inspect_repeated_source_capacity import inspect

HERE = ROOT / 'artifacts/validation/product_execution'
STRESS = HERE / 'repeated_source_capacity_uniform_clock_2026-10-05'


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    snapshot = inspect(STRESS)
    protocol = read(STRESS / 'protocol.json')
    resource = read(STRESS / 'resource_receipt.json')
    assert snapshot['sourceBindingsUnchanged']
    assert digest(ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml') == protocol['configurationSha256']
    assert not snapshot['ownerIdentityMatches'] and not snapshot['processes']
    last = snapshot['lastResourceSample']
    retired = []
    for item in last['processes']:
        try:
            process = psutil.Process(item['pid'])
            command = process.cmdline()
            same_owned = (process.create_time() <= datetime.fromisoformat(resource['completedAt']).timestamp()
                          and any(str(STRESS).lower() in token.lower() for token in command))
            assert not same_owned, 'Owned stress descendant is still alive'
            retired.append({'pid': item['pid'], 'state': 'PID_REUSED_BY_DIFFERENT_PROCESS'})
        except psutil.NoSuchProcess:
            retired.append({'pid': item['pid'], 'state': 'NO_SUCH_PROCESS'})
    job = snapshot['job']
    assert job['status'] == 'FAILED' and 'monitoring safeguard' in job['error']
    assert resource['passed'] is False and resource['safeguardStop']
    samples = [json.loads(line) for line in (STRESS / 'resource_samples.jsonl').read_text().splitlines()]
    assert len(samples) == resource['samples'] == snapshot['observedCompletedSamples']
    assert all(left['elapsedSeconds'] < right['elapsedSeconds'] for left, right in zip(samples, samples[1:]))
    assert max(row['ownedTreeRssBytes'] for row in samples) == resource['peakSampledOwnedTreeRssBytes']
    assert max(row['ownedTreePrivateBytes'] for row in samples) == resource['peakSampledOwnedTreePrivateBytes']
    assert min(row['availableRamBytes'] for row in samples) == resource['minimumSampledAvailableRamBytes']
    assert all(row['ownedTreeRssBytes'] <= protocol['maximumSampledOwnedTreeRssBytes'] for row in samples)
    assert last['elapsedSeconds'] < protocol['maximumWallSeconds']
    tail = samples[-protocol['lowRamConsecutiveSamplesToStop']:]
    assert all(row['availableRamBytes'] < protocol['minimumAvailableRamBytes'] for row in tail)
    output = STRESS / 'isolated_product/analyses' / job['id']
    unpublished = ['result.json', 'summary.json', 'annotated_web.mp4', 'annotated.mp4', 'metrics.json']
    assert all(not (output / name).exists() for name in unpublished)
    assert not (STRESS / 'worker_receipt.json').exists()
    assert not (STRESS / 'published_telemetry.csv').exists()
    metadata = read(output / 'media_metadata.json')
    mapping = metadata['timeMapping']
    assert mapping['status'] == 'CAPTURED_FFMPEG_FRAME_MAP'
    assert mapping['frames'] == mapping['uniqueMappedSourceFrames'] == 18000
    assert mapping['duplicatedOutputFrames'] == mapping['droppedSourceFrames'] == 0
    assert mapping['validMeasurementTransitionIntervals'] == [{'startFrame': 1, 'endFrameExclusive': 18000}]
    count = 0
    with (output / 'source_frame_mapping.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            assert record['frame_index'] == record['source_frame_index'] == count
            expected = Fraction(count, 60)
            assert int(record['normalized_pts']) * Fraction(record['normalized_time_base']) == expected
            assert int(record['source_pts']) * Fraction(record['source_time_base']) == expected
            assert record['timestamp_seconds'] == record['source_timestamp_seconds'] == float(expected)
            assert record['validMeasurementTransitionFromPrevious'] is (count != 0)
            count += 1
    assert count == 18000
    source = next(item for item in read(ROOT / 'data/demos/sources.json')['sources'] if item['id'] == 'dimitrov_tiafoe_2024')
    assert digest(ROOT / source['inputPath']) == source['inputSha256']
    credit = read(output / 'media_source.json')
    for key in ('id', 'author', 'sourcePage', 'license', 'licenseUrl', 'derivativeLicense', 'sourceSha256'):
        assert credit[key] == source[key]
    upload = STRESS / 'isolated_product/uploads' / job['id'] / 'original.mp4'
    assert digest(upload) == credit['inputSha256']
    assert 'N/60' in credit['modifications'] and 'repetitions' in credit['modifications']
    assert 'original quantized source PTS are not claimed' in credit['modifications']
    baseline = read(HERE / 'capacity_idle_server_reload_2026-10-05.json')['jobs_after']
    live_db = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(live_db.as_uri() + '?mode=ro', uri=True, timeout=15) as conn:
        current = [list(row) for row in conn.execute('SELECT id,status FROM analyses ORDER BY id')]
    assert current == baseline and all(row[0] != job['id'] for row in current)
    bindings = {str(path.relative_to(ROOT)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                for path in sorted(STRESS.rglob('*')) if path.is_file()}
    report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'evidenceVerified': True,
              'workloadPassed': False, 'qualificationEvidence': False, 'toolSession': 22734, 'toolExitCodeObserved': 1,
              'analysisId': job['id'], 'terminalJob': job, 'resourceReceipt': resource,
              'stopReasonVerified': 'AVAILABLE_SYSTEM_RAM_BELOW_400000000_BYTES_FOR_FIVE_CONSECUTIVE_SAMPLES',
              'stopSamples': tail, 'ownedProcessesRetired': retired,
              'sourceAndConfigurationBindingsUnchanged': True,
              'normalizationOnly': {'mappingRecords': count, 'uniquePreparedInputFrames': count,
                                    'normalizedAndPreparedInputRationalPts': 'Every frame equals N/60',
                                    'scope': 'Prepared synthetic uniform clock; original licensed source PTS and final annotated media are not verified.'},
              'sourceCredit': credit, 'retainedSourceSha256': source['inputSha256'],
              'missingTerminalWorkerReceipt': True, 'unpublishedArtifactsAbsent': unpublished,
              'retainedPartialWriterFiles': [str(path.relative_to(ROOT)) for path in output.glob('.*.tmp')],
              'liveProductJobStatesUnchanged': current, 'artifactBindings': bindings,
              'limitations': ['No completed inference, annotated media, result, report or CSV was produced.',
                              'Parent safeguard force-stopped only its owned workload; no child finally receipt or CUDA allocator peak survived.',
                              'Sampled tree RSS can double-count shared pages and miss brief peaks; private commit is not resident RAM.',
                              'The system RAM decline does not by itself establish a worker memory leak or identify other host allocations.',
                              'Logical file sizes and free-disk changes are different measurements; no cleanup was performed.',
                              'Five-minute and representative full-match capacity remain unqualified; no automatic inference retry was started.']}
    destination = HERE / 'repeated_source_capacity_STOP_verified_2026-10-05.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'evidenceVerified': True, 'workloadPassed': False, 'receipt': str(destination.relative_to(ROOT)),
                      'sha256': digest(destination), 'retainedFiles': len(bindings), 'resourceReceipt': resource,
                      'normalizationRecords': count, 'currentAvailableRamBytes': psutil.virtual_memory().available}))


if __name__ == '__main__':
    main()

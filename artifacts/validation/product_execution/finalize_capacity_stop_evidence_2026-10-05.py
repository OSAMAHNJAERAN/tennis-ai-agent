"""Bind final software checks and the failed capacity run as separate outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


manifest_path = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
host_path = ROOT / 'artifacts/validation/product_installation/capacity_stop_host_verification_2026-10-05.json'
host = read(host_path)
assert host['checksPassed'] is True and len(host['checks']) == 75 and not host['warnings']
assert host['manifestSha256'] == digest(manifest_path) == '34cf4385b5a97953085fae9babb8d20fd67c23e19b1eb49c524e59bb764e9026'
assert not host['inferenceRun'] and not host['modelsImported'] and not host['productionQualified']
restore = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'
seal_path = restore / 'verification_seal.json'
seal = read(seal_path)
for name, expected in {**seal['evidenceBindings'], **seal['changedSourceBindings']}.items():
    assert digest(ROOT / name) == expected, name
receipt = read(restore / 'receipt.json')
target = ROOT / receipt['target']
for source, copied, bindings in ((ROOT / 'frontend', target, receipt['sourceBindings']), (ROOT, restore, receipt['sharedBindings'])):
    for name, expected in bindings.items():
        assert digest(source / name) == digest(copied / name) == expected, name
for name, expected in seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(target / 'dist' / name) == expected, name
failure_path = HERE / 'repeated_source_capacity_STOP_verified_2026-10-05.json'
failure = read(failure_path)
assert failure['evidenceVerified'] and not failure['workloadPassed']
for name, expected in failure['artifactBindings'].items():
    path = ROOT / name
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name

observation = {'observedAt': '2026-10-05T03:05:33.625Z', 'url': 'http://127.0.0.1:8000/examples/djokovic_2012',
               'bundle': 'http://127.0.0.1:8000/assets/index-BWMRwJe_.js', 'replayFrame': 285,
               'video': {'paused': True, 'readyState': 4, 'currentTime': 11.886876, 'width': 1280, 'height': 720,
                         'currentSrc': 'http://127.0.0.1:8000/api/analyses/match_9de7cb2d09eb46a0af0e34c3b0e6b818/media/source?v=18da2027b51a8de0-95580a'},
               'scope': 'Read-only retained browser snapshot after the isolated stress stop; not new playback, GPU or accuracy qualification.',
               'screenshotArtifact': None, 'deliverableRetainedThisTurn': True}
with (HERE / 'capacity_stop_browser_observation_2026-10-05.json').open('x', encoding='utf-8') as stream:
    json.dump(observation, stream, indent=2, allow_nan=False)

guide = ROOT / 'docs/deployment/PRODUCT_INSTALLATION.md'
lines = guide.read_text(encoding='utf-8').splitlines()
index = next(i for i, line in enumerate(lines) if line.startswith('The latest October 5 [capacity-source host receipt]'))
lines[index] = (
    'The final October5 [capacity-stop host receipt](../../artifacts/validation/product_installation/capacity_stop_host_verification_2026-10-05.json) '
    f'passed all75 checks at `{host["verifiedAt"]}`, binding the failure-aware manifest SHA256 `{host["manifestSha256"]}`. '
    f'Receipt SHA256 `{digest(host_path)}`. Fresh worker-disabled isolated storage returned health/library200 and invalid-upload415, '
    'with no model imports or inference. The [pre-stop receipt](../../artifacts/validation/product_installation/capacity_source_restoration_host_verification_2026-10-05.json) '
    'and [live scoped snapshot](../../artifacts/validation/product_execution/capacity_evidence_recheck_2026-10-05.json) remain historical. '
    'The unchanged current frontend restoration still binds124 current/copied inputs,37 byte-identical build files and32 completed evidence files. '
    'The guide’s `.venv-product` path is the environment to create for a new installation; this workspace’s recorded environment is '
    '`artifacts/validation/product_installation/clean_env`. Pytest is intentionally absent from that inference runtime. '
    '[Ten focused worker-publication tests](../../artifacts/validation/product_execution/worker_publication_capacity_2026-10-05.xml) passed separately without inference.'
)
guide.write_text('\n'.join(lines) + '\n', encoding='utf-8')
progress = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
text = progress.read_text(encoding='utf-8')
anchor = '\n\nPlayback rejection and WebGL failure recovery are repaired.'
addition = (
    '\n\n[The final clean-host receipt](../../artifacts/validation/product_installation/capacity_stop_host_verification_2026-10-05.json) '
    f'passed all75 checks at `{host["verifiedAt"]}`, binding the failure-aware manifest `{host["manifestSha256"]}`. '
    'No inference/models were loaded. [The terminal evidence seal](../../artifacts/validation/product_execution/capacity_stop_final_evidence_2026-10-05.json) '
    'rechecks the unchanged32 earlier sealed evidence files,11 changed sources,124 current/copied frontend inputs and37 current/copied builds, '
    'plus all20 retained failed-stress files and the added10-test publication receipt. '
    'Software checks pass; the five-minute workload failed; scientific and full-match qualification remain open.'
)
assert anchor in text
progress.write_text(text.replace(anchor, addition + anchor, 1), encoding='utf-8')
names = [manifest_path, host_path, seal_path, failure_path, guide, progress,
         ROOT / 'tests/test_worker_result_publication.py', HERE / 'worker_publication_capacity_2026-10-05.xml',
         HERE / 'capacity_stop_browser_observation_2026-10-05.json',
         ROOT / 'scripts/audit/inspect_repeated_source_capacity.py', ROOT / 'scripts/audit/verify_stopped_capacity_audit.py',
         Path(__file__).resolve()]
report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'softwareChecksPassed': True,
          'capacityWorkloadPassed': False, 'productionQualified': False, 'qualificationEvidence': False,
          'fullBackendSuitePreviouslyPassed': 974, 'additionalPublicationTestsPassed': 10,
          'frontendTestsPassed': 161, 'hostChecksPassed': 75,
          'unchangedPriorEvidenceFiles': len(seal['evidenceBindings']), 'unchangedPriorChangedSources': len(seal['changedSourceBindings']),
          'currentAndCopiedInputs': seal['currentAndCopiedInputs'], 'byteIdenticalBuildFiles': len(seal['buildBindings']),
          'retainedStressFilesRechecked': len(failure['artifactBindings']),
          'bindings': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in names},
          'limitations': failure['limitations'], 'newInferenceStarted': False, 'partialArtifactsRemoved': False}
destination = HERE / 'capacity_stop_final_evidence_2026-10-05.json'
with destination.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({'verifiedAt': report['verifiedAt'], 'softwareChecksPassed': True, 'capacityWorkloadPassed': False,
                  'receipt': str(destination.relative_to(ROOT)), 'sha256': digest(destination),
                  'hostVerifiedAt': host['verifiedAt'], 'hostReceiptSha256': digest(host_path)}))

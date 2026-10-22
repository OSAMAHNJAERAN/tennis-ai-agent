"""Seal the bounded-storage engineering unit with explicit pending model work."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
FINAL = UNIT / 'final_verification.json'


def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(values):
    for name, expected in values.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        actual = binding(path)
        assert actual == expected if isinstance(expected, dict) else actual['sha256'] == expected, name


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert not FINAL.exists(), 'Seal is exclusive'
    documentation = load(UNIT / 'documentation_receipt.json')
    assert documentation['passed'] and documentation['status'] == 'COMPLETED'
    verify(documentation['currentDocumentHashes'])
    evidence = module_at('_pose_final_document_evidence', UNIT / 'documentation_preparation/document_evidence_v3.py')
    software = evidence.software_gate()
    measurement = evidence.measurement_gate()
    worker = evidence.worker_gate()
    runtime = evidence.runtime_gate()
    independent_audit = evidence.independent_measurement_gate()
    evidence.verify(evidence.TRACKED, tracked=False)
    audit = load(UNIT / 'measurement/terminal_audit.json')
    assert audit['passed'], 'Independent terminal measurement audit must pass'
    host = load(UNIT / 'verify_installation.json')
    assert host['passed'] and host['status'] == 'COMPLETED' and host['inputsUnchangedAtTerminal']
    assert host['allOwnedIdentitiesRetiredAtTerminal'] and not host['remainingOwnedIdentitiesAtTerminal']
    verify(host['inputs'])
    host_report_path = ROOT / host['report']
    assert binding(host_report_path) == host['reportBinding']
    host_report = load(host_report_path)
    assert host_report['checksPassed'] and all(row['passed'] for row in host_report['checks'])
    assert host_report['inferenceRun'] is False and host_report['modelsImported'] == []
    assert host_report['productionQualified'] is False
    previous_seal = ROOT / 'artifacts/validation/product_execution/frontend_entry_recovery_2026-10-06/final_verification.json'
    assert binding(previous_seal)['sha256'] == 'e81b4e627af107b0aba7d706790b390f21c6e40d9ccea8335a951fd7d4a3fae5'
    previous_frontend = load(previous_seal)
    assert previous_frontend['passed'], 'Historical frontend verification must remain available'
    runtime_helper = module_at('_pose_final_live_runtime', UNIT / 'verify_runtime.py')
    live_receipt = load(UNIT / 'runtime_verification.json')
    assert runtime_helper.runtime_processes() == live_receipt['runtime'], 'Exact fresh runtime changed'
    assert runtime_helper.snapshot() == live_receipt['snapshotAfter'], 'Saved jobs or files changed'
    assert runtime_helper.source_bindings() == live_receipt['sourceBindings'], 'Runtime sources or frontend changed'
    worker_receipt = load(UNIT / 'worker_smoke_v2/resource_receipt.json')
    if worker_receipt['status'] == 'NOT_STARTED':
        assert worker_receipt['inferenceChildStarted'] is False and worker_receipt['samples'] == 0
        assert not (UNIT / 'worker_smoke_v2/run').exists()
        assert worker_receipt['minimumSampledAvailableRamBytes'] < 2500000000
    assert not (UNIT / 'worker_smoke/launch_marker.json').exists(), 'Unlaunched original preparation changed'
    production_sources = {
        'src/pipeline/phase6_pipeline.py': 'ce88b62b44613509d5f4217d0d11fe8c13d9136c9f32c8bf97813d3bdc922811',
        'src/utils/identity_frame_sequence.py': 'f2d2118ccbe68493cd191de79c4b674ea2d6e36c7f81bb726bef6ef0deb17f04',
        'tests/test_identity_frame_sequence.py': '149ecfd4eec084c27e70469302413e9cc7d17bbcaced7e0cee8a6fb436e35913',
        'tests/test_pipeline_pose_storage.py': 'c26c9aaf9541c6aecbca01c01b598edf8edfc912d4b1ea27e84c5f9eb55367a0',
        'scripts/audit/verify_pose_racket_storage_worker_smoke.py': '1fee98a45597e5744aa2e22cb4a343b15e929b94bcb36fa23d9d65df5e78734a',
        'scripts/audit/verify_pose_racket_storage_worker_smoke_v2.py': '802963484bd482415a603f9ff1257a72412be137368cde6627ac05ad9185a28d',
    }
    verify(production_sources)
    files = {}
    for path in UNIT.rglob('*'):
        if not path.is_file() or path == FINAL:
            continue
        relative = path.relative_to(UNIT)
        if any(part in ('tmp', 'bootstrap', '__pycache__') for part in relative.parts):
            continue
        files[path.relative_to(ROOT).as_posix()] = binding(path)
    result = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'status': 'PASSED', 'passed': True,
        'scope': 'Bounded pose/racket storage engineering tests, synthetic measurement, clean host and saved-result HTTP compatibility. Full execution prompt remains incomplete.',
        'software': software, 'measurement': measurement, 'worker': worker, 'runtime': runtime,
        'independentMeasurementAudit': independent_audit,
        'measurementIndependentAudit': binding(UNIT / 'measurement/terminal_audit.json'),
        'cleanHostCheckCount': len(host_report['checks']),
        'workerInferenceStatus': worker_receipt['status'],
        'currentRevisionRealWorkerVerified': worker_receipt['passed'],
        'workerPreflightAvailableRamBytes': worker_receipt['minimumSampledAvailableRamBytes'],
        'liveRuntimeProcesses': live_receipt['runtimeProcesses'],
        'savedJobs': live_receipt['savedJobs'], 'savedAnalysisFiles': live_receipt['savedAnalysisFiles'],
        'previousFrontendSeal': {'path': previous_seal.relative_to(ROOT).as_posix(), **binding(previous_seal)},
        'productionSources': production_sources, 'unitFiles': files,
        'qualificationEvidence': False, 'accuracyQualified': False, 'physicalSpeedAnd3DQualified': False,
        'fullMatchQualified': False, 'productionQualified': False, 'fullPromptComplete': False,
        'remaining': ['Current revision real-worker/model check lacks RAM headroom.',
                      'Other resident arrays and whole-match compute/disk capacity remain unqualified.',
                      'Independent recordings, held-out accuracy and physical reference validation remain unavailable.']}
    with FINAL.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'backendTests': software['fullBackend']['passed'],
                      'hostChecks': result['cleanHostCheckCount'], 'workerStatus': result['workerInferenceStatus'],
                      'fullPromptComplete': False}))


if __name__ == '__main__':
    main()

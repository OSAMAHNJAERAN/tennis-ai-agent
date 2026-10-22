"""Seal this bounded worker/runtime milestone and rejected development experiment."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
WORKER = ROOT / 'artifacts/validation/product_execution/pose_racket_worker_smoke_retry_2026-10-07'
MOTION = ROOT / 'artifacts/validation/product_execution/participant_motion_uvy_2026-10-07'
HELPER = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/verify_runtime.py'

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded

assert not (UNIT / 'final_verification.json').exists()
assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
worker, worker_review = load(WORKER / 'resource_receipt.json'), load(UNIT / 'worker_review.json')
runtime_receipt = load(UNIT / 'runtime_verification_v2.json')
preflight = load(UNIT / 'runtime_preflight.json')
motion, motion_resource = load(MOTION / 'run/report.json'), load(MOTION / 'run/resource_receipt.json')
audit, visual = load(MOTION / 'audit/receipt.json'), load(MOTION / 'visual_review.json')
progress = load(UNIT / 'progress_append_receipt.json')
coverage = load(UNIT / 'worker_observation_coverage.json')
assert worker['status'] == 'COMPLETED' and worker['passed']
assert worker_review['passed'] and worker_review['actualWorkerCompletionVerified'] and worker_review['actualSavedResultExactBytesVerified']
assert worker['allExactOwnedIdentitiesRetiredAtTerminal'] and worker['remainingExactOwnedIdentitiesAtTerminal'] == []
assert runtime_receipt['passed'] and runtime_receipt['catalogEntries'] == 3 and len(runtime_receipt['exports']) == 12
assert motion['complete'] and motion_resource['passed'] and motion_resource['allOwnedIdentitiesRetired']
assert audit['passed'] and audit['pooledCounts'] == motion['pooledCounts']
assert visual['decision'] == 'REJECT_MOTION_ONLY_PARTICIPANT_PROMOTION' and visual['reviewedImages'] == 26
assert not motion['productionPromoted'] and not motion['modelInferenceRun']
assert progress['passed'] and progress['originalPrefixPreservedExactly']
assert binding(ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md') == progress['after']
assert binding(UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md') == progress['before']
for relative, expected in visual['imageBindings'].items():
    assert binding(ROOT / relative) == expected
for key, value in worker.items():
    if key.endswith('Unchanged'):
        assert value is True, key

checks = module('_final_resume_runtime', HELPER)
adapter = module('_final_resume_native_console', UNIT / 'hidden_console_runtime.py')
review = load(UNIT / 'hidden_runtime_review.json')
runtime = adapter.runtime_processes(checks, review['runtime'])
assert checks.source_bindings() == preflight['sourceBindings']
assert checks.snapshot() == preflight['snapshot'] == runtime_receipt['snapshotAfter']
status, _, body = checks.request('/api/health')
health = json.loads(body)
assert status == 200 and health['status'] == 'ok' and health['worker_available']
assert adapter.runtime_processes(checks, review['runtime']) == runtime
assert checks.snapshot() == preflight['snapshot']
assert checks.source_bindings() == preflight['sourceBindings']

# Bind stable saved evidence. Live server stdout/stderr remain active logs,
# not immutable final artifacts, and are deliberately excluded from this seal.
paths = [path for directory in (UNIT, WORKER, MOTION) for path in directory.rglob('*')
         if path.is_file() and path.suffix not in ('.pyc',) and path.name not in ('server_stdout.log', 'server_stderr.log')]
paths += [ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v4.py',
          ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md',
          ROOT / 'docs/experiments/WORKER_STORAGE_SMOKE_2026-10-07.md',
          ROOT / 'docs/experiments/PARTICIPANT_MOTION_UVY_2026-10-07.md']
bindings = {path.relative_to(ROOT).as_posix(): binding(path) for path in sorted(set(paths))}
record = {'verifiedAt': datetime.now(timezone.utc).isoformat(),
          'status': 'CURRENT_WORKER_VERIFIED_MOTION_CANDIDATE_REJECTED',
          'boundedVerificationPassed': True, 'actualWorkerPassed': True,
          'actualWorkerFrames': 30, 'workerOutputBytesComparedExactly': ['JSON', 'CSV', 'HTML'],
          'poseRacketObservedCoverage': coverage['countsByCourtRole'],
          'websiteRestoredAndVerified': True, 'currentHealth': health, 'currentRuntime': runtime,
          'existingDemos': 3, 'existingExportsExact': 12,
          'savedJobsUnchanged': 10, 'savedAnalysisFilesUnchanged': 319,
          'originalPostStartFailureRetained': True,
          'motionOriginalTrialPassed': True, 'motionIndependentAuditPassed': True,
          'motionFramesScored': 2230, 'motionProposalMatched': 2479,
          'motionSelectedMatched': 710, 'motionPublisherUnmatched': 1620,
          'motionCandidatePromoted': False, 'qualificationEvidence': False,
          'fullExecutionObjectiveComplete': False,
          'remaining': ['Court-independent accurate active-player recovery and identity validation',
                        'Automatic ball, court, pose, racket and mask accuracy qualification',
                        'Representative untouched supported recordings and independent labels',
                        'Independent physical speed/3D references and validated estimator',
                        'Full-match capacity verification'],
          'evidenceBindings': bindings,
          'scope': 'Successful actual short current-revision worker storage verification, exact existing-result website restoration, and completed audited rejection of a development participant selector. Full original prompt remains active and incomplete.'}
with (UNIT / 'final_verification.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
print(json.dumps({key: record[key] for key in ('status', 'actualWorkerPassed', 'websiteRestoredAndVerified',
                    'motionIndependentAuditPassed', 'motionCandidatePromoted', 'fullExecutionObjectiveComplete')}))

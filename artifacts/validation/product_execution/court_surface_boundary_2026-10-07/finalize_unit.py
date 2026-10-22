"""Seal bounded evidence without converting failed/not-started runs into passes."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
WORKER = ROOT / 'artifacts/validation/product_execution/pose_racket_worker_smoke_2026-10-07'
OUTPUT = UNIT / 'final_verification.json'
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))
def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
def resolve(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path
def verify(values):
    for name, expected in values.items():
        assert binding(resolve(name)) == expected, 'Retained binding changed: ' + name
assert not OUTPUT.exists()
prepared = load(UNIT / 'preparation.json')
verify(prepared['bindings'])
trial = load(UNIT / 'run/resource_receipt.json')
assert trial['status'] == 'FAILED' and trial['exitCode'] == 1 and not trial['passed']
assert trial['allOwnedIdentitiesRetired'] and not trial['remainingOwnedIdentities']
assert not (UNIT / 'run/report.json').exists()
assert trial['sourcesUnchanged'] and trial['savedProductUnchanged']
first = load(UNIT / 'tests/receipt.json')
followup = load(UNIT / 'tests_v2/receipt.json')
assert first['counts'] == {'tests': 35, 'failures': 1, 'errors': 0, 'skipped': 0, 'passed': 34}
assert not first['passed'] and followup['passed']
assert followup['counts'] == {'tests': 36, 'failures': 0, 'errors': 0, 'skipped': 0, 'passed': 36}
for receipt, folder in ((first, 'tests'), (followup, 'tests_v2')):
    verify(receipt['sourceBefore'])
    assert receipt['sourceBefore'] == receipt['sourceAfter'] and receipt['sourcesUnchanged'] and receipt['allOwnedIdentitiesRetired']
    for leaf, expected in receipt['outputs'].items():
        assert binding(UNIT / folder / leaf) == expected
audit = load(UNIT / 'audit/receipt.json')
assert binding(UNIT / 'audit.py')['sha256'] == '36d575878a4729ceb92ca83dffdb9fffdd4d815a612c960cbe41a52f3272fcf9'
assert binding(UNIT / 'audit/receipt.json')['sha256'] == '6d9eacd9aa3c791e003e29b7e5841216129a6d6732111b48a471f354a82409d4'
assert audit['passed'] and not audit['originalRunPassed'] and not audit['loadedModuleInventoryVerified']
assert audit['verifiedExistingCases'] == 64 and audit['validRefinements'] == 47 and audit['initialAbstentions'] == 17
assert audit['labeledCases'] == 40 and audit['independentlyCheckedNativeErrors'] == 2240 and audit['automaticSourceViews'] == 9
assert audit['inputsBefore'] == audit['inputsAfter']
verify(audit['inputsAfter'])
assert audit['gainedVsBoundaryBaseline'] == [] and audit['lostVsBoundaryBaseline'] == [{'case': 27, 'identity': 2}]
assert audit['automaticAdmissions'] == ['dimitrov_tiafoe_2024'] and not audit['spectatorAccepted']
recovery = load(UNIT / 'recovery.json')
assert recovery['pooled'] == audit['pooled'] and not recovery['trialTerminalPassed'] and recovery['experimentRecordsComplete']
assert len(recovery['cases']) == 64 and len(recovery['automatic']) == 9
for name, expected in recovery['evidenceBindings'].items():
    assert binding(UNIT / name) == expected
visual = load(UNIT / 'visual_review.json')
assert binding(UNIT / 'visual_review.json')['sha256'] == '3aa45fcbfacefa49d38795df6f8997024c751cd7e222f873acd96aa5b6771ba0'
assert visual['complete'] and visual['runTerminalStatus'] == 'FAILED'
assert visual['reviewMethod']['allRequestedPanelsViewed'] and visual['reviewMethod']['numberOfChangedPanels'] == 5
assert visual['reviewMethod']['numberOfAutomaticPanels'] == 9 and visual['reviewMethod']['numberOfSpectatorPanels'] == 1
for evidence in visual['artifactBindings']:
    assert binding(resolve(evidence['path'])) == {key: evidence[key] for key in ('bytes', 'sha256')}
worker = load(WORKER / 'resource_receipt.json')
review = load(WORKER / 'terminal_review.json')
assert binding(WORKER / 'terminal_review.json')['sha256'] == '3114660b0445ade1f7e440b018c31a06d025f743003bcf924fa83d288e61c67c'
assert worker['status'] == 'NOT_STARTED' and not worker['passed'] and not worker['inferenceChildStarted']
assert worker['samples'] == 0 and worker['minimumSampledAvailableRamBytes'] == 2435448832
assert worker['minimumSampledAvailableRamBytes'] < 2500000000 and worker['monitoredChildWallSeconds'] is None
assert review['independentReviewPassed'] and not review['actualWorkerCompletionVerified'] and not review['sourceModelMediaProductionChecksExercised']
for evidence in review['currentUnitFilesBeforeThisReview']:
    assert binding(WORKER / evidence['file']) == {key: evidence[key] for key in ('bytes', 'sha256')}
for leaf in review['workloadEvidenceAbsentInCurrentAndPreviousV2']:
    assert not (WORKER / leaf).exists()
origin = load(WORKER / 'origin.json')
for key in ('firstLatencyAttemptBindings', 'secondLatencyAttemptBindings', 'previousPoseRacketAttemptBindings'):
    verify(origin[key])
assert binding(ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v3.py')['sha256'] == '967ddbd5ed2c28d4b64edb13508a6f5b7a81726b70f7d75bebcef398a3a35e11'
runtime = load(UNIT / 'runtime_restore_verification.json')
assert runtime['passed'] and runtime['catalogEntries'] == 3 and len(runtime['exports']) == 12
assert len(runtime['frontend']['assets']) == 7 and runtime['savedJobs'] == 10 and runtime['savedAnalysisFiles'] == 319
for leaf in ('runtime_restore_preflight_wrapper.json', 'runtime_restore_verification_wrapper.json'):
    wrapper = load(UNIT / leaf)
    assert wrapper['passed'] and wrapper['originalHelperUnchanged']
    receipt = 'runtime_restore_preflight.json' if 'preflight' in leaf else 'runtime_restore_verification.json'
    assert binding(UNIT / receipt) == wrapper['receiptBinding']
    assert binding(UNIT / 'restore_runtime_checks.py') == wrapper['wrapperBinding']
helper = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/verify_runtime.py'
assert binding(helper)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
spec = importlib.util.spec_from_file_location('_final_readonly_live_runtime', helper)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
assert checks.runtime_processes() == runtime['runtime']
assert checks.source_bindings() == runtime['sourceBindings']
assert checks.snapshot() == runtime['snapshotAfter']
status, _, body = checks.request('/api/health')
assert status == 200 and json.loads(body)['worker_available']
append = load(UNIT / 'progress_append_receipt.json')
assert append['previousBytesPreservedExactly'] and append['before'] == append['preservedBefore']
assert binding(UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md') == append['before']
assert binding(ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md') == append['after']
assert binding(ROOT / 'docs/experiments/COURT_SURFACE_BOUNDARY_RESULTS_2026-10-07.md')['sha256'] == '22fe25614ac997f6e1be2e8b6af3b803244e6b3fd4ace57ec199521b1f6988e0'
evidence = {path.relative_to(ROOT).as_posix(): binding(path) for directory in (UNIT, WORKER) for path in directory.rglob('*') if path.is_file()}
for path in (ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md', ROOT / 'docs/experiments/COURT_SURFACE_BOUNDARY_RESULTS_2026-10-07.md', ROOT / 'docs/research/ACCURACY_DATA_RECHECK_2026-10-07.md', ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v3.py'):
    evidence[path.relative_to(ROOT).as_posix()] = binding(path)
result = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'status': 'REVIEWED_DEVELOPMENT_CANDIDATE_NOT_PROMOTED',
          'boundedEvidenceAuditPassed': True, 'originalCourtTrialPassed': False, 'originalCourtTrialStatus': 'FAILED_FINALIZATION',
          'recoveredRecordsAudited': True, 'nativeErrorsIndependentlyChecked': 2240,
          'courtSelectionBaselineTp': 369, 'courtSelectionCandidateTp': 368, 'courtCandidatePromoted': False,
          'realWorkerStatus': 'NOT_STARTED', 'realWorkerVerified': False, 'realWorkerInferenceStarted': False,
          'websiteRestoredAndVerified': True, 'savedJobsUnchanged': 10, 'savedAnalysisFilesUnchanged': 319,
          'currentRuntime': runtime['runtime'], 'qualificationEvidence': False, 'fullExecutionObjectiveComplete': False,
          'remaining': ['Real current-revision worker output verification with adequate unchanged resource headroom',
                        'Full-match capacity evidence', 'Untouched independent supported singles recordings and labels',
                        'Automatic player/pose/racket/mask and ball qualification', 'Independent physical speed and 3D references'],
          'trialLimitations': audit['limitations'], 'evidenceBindings': evidence,
          'scope': 'Seal of preserved failed court trial, independent recovered-record/visual audits, genuine zero-work worker memory decline and successful existing-result website restoration. No failed or unstarted run is relabeled passed.'}
with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({key: result[key] for key in ('status', 'boundedEvidenceAuditPassed', 'originalCourtTrialPassed', 'realWorkerStatus', 'websiteRestoredAndVerified', 'fullExecutionObjectiveComplete')}))

"""Narrow evidence-backed label correction; preserve the preceding seals."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROGRESS = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
COPY = HERE / 'registration_storage_correction_baseline_2026-10-06/docs/plans/TENNIS_EXECUTION_PROGRESS.md'
DEST = HERE / 'registration_storage_final_correction_2026-10-06.json'
FINAL = HERE / 'registration_storage_final_evidence_2026-10-06.json'
RESOURCE = HERE / 'registration_storage_real_smoke_2026-10-05/resource_receipt.json'
HELPER = HERE / 'registration_storage_real_smoke_2026-10-05/real_pipeline_smoke.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not COPY.exists() and not DEST.exists(), 'Preserve correction evidence'
prior, resource, helper = read(FINAL), read(RESOURCE), read(HELPER)
for name, expected in prior['bindings'].items():
    assert digest(ROOT / name) == expected, name
assert resource['actualWorkloadWindowsPeakWorkingSetBytes'] == 1825730560
assert helper['processPeakWorkingSetBytes'] == 1779539968
old = 'Actual process peak working set 1,779,539,968 bytes, sampled tree RSS 1,831,260,160 bytes;'
new = ('Helper-stage process peak working set 1,779,539,968 bytes; full workload-interpreter OS peak '
       '1,825,730,560 bytes, sampled tree RSS 1,831,260,160 bytes;')
text = PROGRESS.read_text(encoding='utf-8')
assert text.count(old) == 1
COPY.parent.mkdir(parents=True, exist_ok=False)
with COPY.open('xb') as stream:
    stream.write(PROGRESS.read_bytes())
assert digest(COPY) == prior['bindings'][PROGRESS.relative_to(ROOT).as_posix()]
PROGRESS.write_text(text.replace(old, new, 1), encoding='utf-8')
for name, expected in prior['bindings'].items():
    if ROOT / name != PROGRESS:
        assert digest(ROOT / name) == expected, name
host_path = ROOT / 'artifacts/validation/product_installation/registration_storage_host_verification_2026-10-06.json'
host = read(host_path)
assert host['manifestSha256'] == digest(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'correctionVerified': True,
           'correction': 'Distinguish helper-end OS peak from the larger full-workload interpreter OS peak after comparisons.',
           'helperStagePeakWorkingSetBytes': helper['processPeakWorkingSetBytes'],
           'fullWorkloadInterpreterPeakWorkingSetBytes': resource['actualWorkloadWindowsPeakWorkingSetBytes'],
           'sampledOwnedTreePeakRssBytes': resource['peakSampledOwnedTreeRssBytes'],
           'previousDocumentAndFinalSealsPreserved': True, 'allOtherFinalBindingsUnchanged': True,
           'backendTestsPassed': 1102, 'cleanHostChecksPassed': 75,
           'real30FrameCompatibilityPassed': True, 'registrationLargeBenchmarkPassed': False,
           'priorFiveMinuteWorkloadPassed': False, 'savedAnalysisRowsUnchanged': 10,
           'productionQualified': False, 'fullMatchQualified': False, 'accuracyQualified': False,
           'bindings': {path.relative_to(ROOT).as_posix(): digest(path)
                        for path in (PROGRESS, COPY, FINAL, RESOURCE, HELPER, host_path, Path(__file__))}}
with DEST.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'correctionVerified': True, 'allOtherFinalBindingsUnchanged': True,
                  'fullWorkloadInterpreterPeakWorkingSetBytes': 1825730560, 'manifestUnchanged': True}))

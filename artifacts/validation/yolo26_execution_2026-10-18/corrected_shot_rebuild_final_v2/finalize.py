"""Bind the settled private correction evidence to current source, without inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
REFERENCES = {}
BOUND = {}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(relative):
    path = ROOT / relative
    REFERENCES[relative] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))

def verify_pins(values, *, prefix=''):
    for name, expected in values.items():
        relative = (Path(prefix) / name.replace('\\', '/')).as_posix()
        assert digest(ROOT / relative) == expected, relative
        assert relative not in BOUND or BOUND[relative] == expected, relative
        BOUND[relative] = expected

def stable(document, before, after):
    assert document[before] == document[after]
    verify_pins(document[after])

combined_path = ART.relative_to(ROOT).as_posix() + '/receipt.json'
combined = read(combined_path)
assert combined['exit_code'] == 0 and combined['source_stable']
assert combined['counts'] == dict(tests=256, errors=0, failures=0, skipped=0)
assert not combined['inference_import_attempts'] and not combined['forbidden_inference_loaded']
assert combined['optional_memory_observation_error'] is None
stable(combined, 'source_before', 'source_after')
junit_path = ART / 'junit.xml'
cases = list(ET.parse(junit_path).getroot().iter('testcase'))
assert len(cases) == 256
assert all(case.find('failure') is None and case.find('error') is None
           and case.find('skipped') is None for case in cases)
REFERENCES[junit_path.relative_to(ROOT).as_posix()] = digest(junit_path)

base = 'artifacts/validation/yolo26_execution_2026-10-18/'
adapter = read(base + 'corrected_shot_sources_checks_v4/receipt.json')
assert adapter['status'] == 'PASS' and adapter['exit_code'] == 0
assert not adapter['forbidden_import_attempts'] and not adapter['loaded_models']
stable(adapter, 'source_pins_before', 'source_pins_after')
classifier = read(base + 'shot_classifier_lazy_import_v1/receipt.json')
assert classifier['exit_code'] == 0
stable(classifier, 'source_before', 'source_after')
review = read(base + 'corrected_shot_independent_review_v3/review.json')
assert review['status'] == 'PASS_NO_P1_P2_FOUND' and not review['model_import_attempts']
stable(review, 'reviewed_file_hashes_before', 'reviewed_file_hashes_after')
final_review = read(base + 'corrected_shot_rebuild_final_v2/final_review.json')
assert final_review['status'] == 'PASS' and not final_review['remaining_P1_P2']
assert final_review['receipt_sha256'] == REFERENCES[combined_path]
assert final_review['source_files_and_runner_verified'] == len(combined['source_after'])

frontend = read('artifacts/validation/yolo26_execution_2026-10-17/analysis_generation_frontend_v2/receipt.json')
assert frontend['outcome'] == 'PASS'
assert frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
verify_pins(frontend['sourceBindingsAfter'], prefix='frontend')
admission = read(base + 'gpu_admission_after_corrected_shots_v1.json')
assert admission['training_helpers_unchanged'] and not admission['forbidden_import_attempts']
assert not admission['training_performed'] and not admission['cuda_initialized']
assert admission['training_helpers_expected'] == admission['training_helpers_observed']
verify_pins(admission['training_helpers_observed'])

documents = ['docs/plans/YOLO26_EXECUTION_PROGRESS.md',
    'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md']
for name in documents:
    BOUND[name] = digest(ROOT / name)
    target = ART / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
BOUND[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
record = {'status': 'PASS_CURRENT_SOURCE_BINDINGS',
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'bound_file_count': len(BOUND), 'source_bindings': dict(sorted(BOUND.items())),
    'evidence_sha256': REFERENCES, 'test_count': 256,
    'scope': 'Private fresh corrected physical shots and native analytics; no complete correction publication or independent match qualification'}
with (ART / 'current_source_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
    'completed_this_stage': ['Fresh corrected native physical shot linking',
        'Fresh point/attempt/rally memberships and native match analytics',
        'Strict configured crop-feature branch with source/config binding and bounded continuous window',
        'Actual toy normalized-frame decode, dimension rejection and decoder cleanup'],
    'verification': {'tests': 256, 'runner_seconds': combined['elapsed_seconds'],
        'source_pins_unchanged': len(combined['source_after']),
        'bound_current_files': len(BOUND),
        'parent_peak_working_set_bytes': combined['parent_peak_working_set_bytes'],
        'independent_review': final_review['status'],
        'observations': combined['observations'], 'receipt_sha256': REFERENCES[combined_path]},
    'gpu_host_admission': {'status': admission['status'], 'resources': admission['resources'],
        'observed_local': admission['observed_local'], 'training_performed': False},
    'remaining': ['Fresh bounded causal forecasts and all corrected exports',
        'Fresh annotated and browser rendering, complete generation publication',
        'Durable authenticated correction lifecycle and connected review UI',
        'Selected crop model/runtime identity admission and actual model compatibility',
        'Repaired-data GPU training and independent grouped model/numerical accuracy gates',
        'Actual long-match inference, within-inference recovery, and sustained live video evidence'],
    'published': False, 'independent_accuracy_validated': False,
    'evidence_sha256': REFERENCES}
with (ART / 'delivery_record.json').open('x', encoding='utf-8') as stream:
    json.dump(delivery, stream, indent=2)
print(json.dumps({'status': delivery['status'], 'tests': 256,
    'bound_files': len(BOUND), 'goal_status': delivery['goal_status'],
    'delivery_sha256': digest(ART / 'delivery_record.json')}))

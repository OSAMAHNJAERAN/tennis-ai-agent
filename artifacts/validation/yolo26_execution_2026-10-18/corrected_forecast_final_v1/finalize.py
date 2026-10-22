"""Verify current code/evidence bindings for private causal forecast progress."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
REFERENCES, BOUND = {}, {}

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

base = 'artifacts/validation/yolo26_execution_2026-10-18/'
combined_relative = base + 'corrected_forecast_final_v1/receipt.json'
combined = read(combined_relative)
counts = combined['counts']
assert combined['exit_code'] == 0 and combined['source_stable']
assert counts['tests'] > 250 and not any(counts[key] for key in ('failures', 'errors', 'skipped'))
assert combined['source_before'] == combined['source_after']
assert not combined['inference_import_attempts'] and not combined['forbidden_inference_loaded']
assert combined['optional_memory_observation_error'] is None
verify_pins(combined['source_after'])
inventory = {path.relative_to(ROOT).as_posix() for folder in ('src', 'tests')
             for path in (ROOT / folder).rglob('*.py')}
assert inventory <= {name.replace('\\', '/') for name in combined['source_after']}
for name, expected in combined['source_after'].items():
    assert digest(ART / 'sources' / name) == expected
junit = ART / 'junit.xml'
cases = list(ET.parse(junit).getroot().iter('testcase'))
assert len(cases) == counts['tests']
assert len({(case.attrib['classname'], case.attrib['name']) for case in cases}) == len(cases)
assert all(case.find('failure') is None and case.find('error') is None
           and case.find('skipped') is None for case in cases)
REFERENCES[junit.relative_to(ROOT).as_posix()] = digest(junit)

storage = read(base + 'forecast_sequence_checks_v1/receipt.json')
assert storage['status'] == 'PASS' and storage['source_pins_before'] == storage['source_pins_after']
assert not storage['forbidden_import_attempts'] and not storage['loaded_heavy_runtimes']
verify_pins(storage['source_pins_after'])
sql = read(base + 'prediction_sql_index_checks_v1/receipt.json')
assert sql['exit_code'] == 0 and sql['source_before'] == sql['source_after']
verify_pins(sql['source_after'])
review = read(base + 'corrected_forecast_independent_review_v1/final_review.json')
assert review['status'] == 'PASS_REVIEWED_SCOPE' and not review['remaining_p1_p2']
verify_pins(review['source_pins'])
settled_review = read(base + 'corrected_forecast_final_v1/final_review.json')
assert settled_review['status'] == 'PASS' and not settled_review['remaining_P1_P2']
assert settled_review['receipt_sha256'] == REFERENCES[combined_relative]
frontend = read('artifacts/validation/yolo26_execution_2026-10-17/analysis_generation_frontend_v2/receipt.json')
assert frontend['outcome'] == 'PASS'
assert frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
verify_pins(frontend['sourceBindingsAfter'], prefix='frontend')
admission = read(base + 'gpu_admission_after_corrected_forecasts_v1.json')
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
    'evidence_sha256': REFERENCES, 'test_count': counts['tests'],
    'scope': 'Private native corrected causal statistical forecasts and scratch export; no calibration or whole correction publication'}
with (ART / 'current_source_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
    'completed_this_stage': ['Fresh causal statistical forecasts from new corrected accepted rules and exact raw state',
        'Owned bounded SQLite forecast history and strict completed-point deduplication',
        'Native reviewed service-anchor and terminal outcome joins; pre-award server posterior update',
        'Frozen result/export handles, detached presentation values and private atomic streamed export',
        'Conservative withholding for contradictory terminal scores without inventing prior set detail'],
    'verification': {'tests': counts['tests'], 'runner_seconds': combined['elapsed_seconds'],
        'source_pins_unchanged': len(combined['source_after']), 'bound_current_files': len(BOUND),
        'parent_peak_working_set_bytes': combined['parent_peak_working_set_bytes'],
        'independent_review': settled_review['status'],
        'observations': combined['observations'], 'receipt_sha256': REFERENCES[combined_relative]},
    'gpu_host_admission': {'status': admission['status'], 'resources': admission['resources'],
        'observed_local': admission['observed_local'], 'training_performed': False},
    'remaining': ['All fresh corrected semantic exports/zones and strict result normalization',
        'Fresh annotated/browser rendering and complete-generation correction publication',
        'Durable authenticated correction lifecycle, forecast API/contract, bounded chart and review UI',
        'Selected crop model/runtime identity admission and actual model compatibility',
        'Repaired-data GPU training and independent grouped model/numerical/prediction calibration gates',
        'Actual long-match inference, within-inference recovery, and sustained live video evidence'],
    'published': False, 'calibration_status': 'not_calibrated', 'independent_accuracy_validated': False,
    'evidence_sha256': REFERENCES}
with (ART / 'delivery_record.json').open('x', encoding='utf-8') as stream:
    json.dump(delivery, stream, indent=2)
print(json.dumps({'status': delivery['status'], 'tests': counts['tests'],
    'bound_files': len(BOUND), 'goal_status': delivery['goal_status'],
    'delivery_sha256': digest(ART / 'delivery_record.json')}))

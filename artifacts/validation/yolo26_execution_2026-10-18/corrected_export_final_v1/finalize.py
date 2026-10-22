"""Bind verified current private export progress without claiming completion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
BASE = ART.parent
BOUND, REFERENCES = {}, {}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    REFERENCES[path.relative_to(ROOT).as_posix()] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))

def verify_pins(values, prefix=''):
    for name, expected in values.items():
        relative = (Path(prefix)/name.replace('\\', '/')).as_posix()
        assert digest(ROOT/relative) == expected, relative
        assert relative not in BOUND or BOUND[relative] == expected, relative
        BOUND[relative] = expected

def verify_run(directory):
    receipt = read(directory/'receipt.json')
    assert receipt['exit_code'] == 0 and receipt['source_stable']
    assert receipt['source_before'] == receipt['source_after']
    assert not receipt['inference_import_attempts'] and not receipt['forbidden_inference_loaded']
    assert not any(receipt['counts'][key] for key in ('failures', 'errors', 'skipped'))
    verify_pins(receipt['source_after'])
    for name, expected in receipt['source_after'].items():
        assert digest(directory/'sources'/name) == expected
    junit = directory/'junit.xml'
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases) == receipt['counts']['tests']
    assert len({(case.get('classname'), case.get('name')) for case in cases}) == len(cases)
    assert all(case.find('failure') is None and case.find('error') is None
               and case.find('skipped') is None for case in cases)
    REFERENCES[junit.relative_to(ROOT).as_posix()] = digest(junit)
    REFERENCES[(directory/'pytest.txt').relative_to(ROOT).as_posix()] = digest(directory/'pytest.txt')
    return receipt, cases

combined, _ = verify_run(ART)
capacity, cases = verify_run(BASE/'corrected_export_checks_v3')
assert capacity['counts']['tests'] == 141
capacity_cases = [case for case in cases if case.get('name') ==
    'test_growing_completed_points_bounds_actual_export_history_and_strict_reader_memory']
assert len(capacity_cases) == 1
observations = [value for value in capacity['observations'] if 'completed_point_export_capacity' in value]
assert len(observations) == 1
assert [value['completed_points'] for value in observations[0]['completed_point_export_capacity']] == [8, 128]
inventory = {path.relative_to(ROOT).as_posix() for folder in ('src', 'tests')
             for path in (ROOT/folder).rglob('*.py')}
assert inventory <= {name.replace('\\', '/') for name in combined['source_after']}
frontend = read(ROOT/'artifacts/validation/yolo26_execution_2026-10-17/analysis_generation_frontend_v2/receipt.json')
assert frontend['outcome'] == 'PASS' and frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
verify_pins(frontend['sourceBindingsAfter'], 'frontend')
admission = read(BASE/'gpu_admission_after_corrected_exports_v1.json')
assert admission['training_helpers_unchanged'] and not admission['forbidden_import_attempts']
assert not admission['training_performed'] and not admission['cuda_initialized']
assert admission['training_helpers_expected'] == admission['training_helpers_observed']
verify_pins(admission['training_helpers_observed'])
for name in ('docs/plans/YOLO26_EXECUTION_PROGRESS.md',
             'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md'):
    BOUND[name] = digest(ROOT/name)
    destination = ART/'sources'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, destination)
BOUND[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
verification = {'status': 'PASS_CURRENT_SOURCE_BINDINGS',
    'verified_utc': datetime.now(timezone.utc).isoformat(), 'bound_file_count': len(BOUND),
    'source_bindings': dict(sorted(BOUND.items())), 'evidence_sha256': REFERENCES,
    'test_count': combined['counts']['tests'],
    'scope': 'Private fresh semantic export set and strict consistency/reproduction; no independent accuracy or full correction publication'}
with (ART/'current_source_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(verification, stream, indent=2)
delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
    'completed_this_stage': [
        'Fresh twenty-file private semantic bundle: new rules, points, shots, analytics, landings, forecasts and replay',
        'Exact eight-file native point manifest plus a complete nineteen-artifact private seal written last',
        'Strict point/landing readers and verified unchanged reconstruction of the NEW replay before return',
        'Independent bounded source/forecast copies, distinct successful candidates and no seal on failed candidates',
        'Immutable admitted review command, detached presentation inputs and source/base/revision ownership guards'],
    'verification': {'tests': combined['counts']['tests'],
        'runner_seconds': combined['elapsed_seconds'], 'source_pins_unchanged': len(combined['source_after']),
        'bound_current_files': len(BOUND), 'capacity_observations': observations,
        'parent_peak_working_set_bytes': combined['parent_peak_working_set_bytes'],
        'independent_export_code_review': 'NOT_COMPLETED; delegated reviewer provided initial mapping and audit-authority finding, then hit account usage limit',
        'receipt_sha256': digest(ART/'receipt.json')},
    'gpu_host_admission': {'status': admission['status'], 'resources': admission['resources'],
        'observed_local': admission['observed_local'], 'training_performed': False},
    'remaining': ['Independent review of the new full private exporter',
        'Complete result normalization, forecast HTTP contract and bounded chart/review UI',
        'Fresh annotated/browser rendering, readable report and complete-generation correction publication',
        'Durable authenticated owner-only correction command lifecycle',
        'Selected crop model/runtime identity admission and actual selected-model compatibility',
        'Repaired-data GPU training and independent grouped model/numerical/prediction calibration gates',
        'Actual long-match inference, within-inference recovery, and sustained live video evidence'],
    'published': False, 'independent_accuracy_validated': False, 'evidence_sha256': REFERENCES}
with (ART/'delivery_record.json').open('x', encoding='utf-8') as stream:
    json.dump(delivery, stream, indent=2)
print(json.dumps({'status': delivery['status'], 'goal_status': delivery['goal_status'],
    'tests': combined['counts']['tests'], 'bound_files': len(BOUND),
    'delivery_sha256': digest(ART/'delivery_record.json')}))

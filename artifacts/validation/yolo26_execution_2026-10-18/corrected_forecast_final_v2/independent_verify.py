"""Read existing receipts, source copies and JUnit; do not rerun tests."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path.cwd()
base = root/'artifacts/validation/yolo26_execution_2026-10-18'
output = base/'corrected_forecast_final_v2'
expected_receipt = '8d1077eb88fa106c52b12701d313ad5f1eb08081084fe633fe82b35bed14970f'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
receipt = read(output/'receipt.json')
assert sha(output/'receipt.json') == expected_receipt
assert receipt['exit_code'] == 0 and receipt['source_stable'] is True
assert receipt['counts'] == {'tests':346,'errors':0,'failures':0,'skipped':0}
assert receipt['source_before'] == receipt['source_after']
assert len(receipt['source_after']) == 383
assert not receipt['inference_import_attempts'] and not receipt['forbidden_inference_loaded']
for name,digest in receipt['source_after'].items():
    assert sha(root/name) == digest, ('Current source differs',name)
    assert sha(output/'sources'/name) == digest, ('Retained source differs',name)

def junit(path):
    tree = ET.parse(path).getroot()
    cases = tree.findall('.//testcase')
    ids = [(case.get('classname'),case.get('name')) for case in cases]
    assert len(ids) == len(set(ids)), ('Duplicate JUnit case IDs',path)
    return tree,cases
tree,cases = junit(output/'junit.xml')
assert len(cases) == 346
assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
assert all(int(suite.get('failures','0')) == int(suite.get('errors','0')) == int(suite.get('skipped','0')) == 0
           for suite in tree.findall('testsuite'))
log = (output/'pytest.txt').read_text(encoding='utf-8')
assert '346 passed, 7 deselected, 1 warning' in log
assert 'FAILED ' not in log and 'ERROR ' not in log

prior_review_path = base/'corrected_forecast_independent_review_v1/final_review.json'
prior_review = read(prior_review_path)
assert prior_review['status'] == 'PASS_REVIEWED_SCOPE' and not prior_review['remaining_p1_p2']
assert len(prior_review['source_pins']) == 4
for name,digest in prior_review['source_pins'].items():
    assert sha(root/name) == digest

previous_dir = base/'corrected_forecast_final_v1'
previous = read(previous_dir/'receipt.json')
assert previous['counts'] == {'tests':345,'errors':0,'failures':1,'skipped':0}
assert previous['source_before'] == previous['source_after']
assert not previous['inference_import_attempts'] and not previous['forbidden_inference_loaded']
previous_tree,previous_cases = junit(previous_dir/'junit.xml')
failures = [case for case in previous_cases if case.find('failure') is not None]
assert len(failures) == 1 and failures[0].get('name') == 'test_profiles_do_not_merge_names_and_unconfirmed_motion_is_excluded'
capacity_name = 'test_growing_completed_points_bounds_forecast_history_and_deduplication_in_python'
capacity_cases = [case for case in previous_cases if case.get('name') == capacity_name]
assert len(capacity_cases) == 1 and not list(capacity_cases[0])
capacity_names = ['src/scoring/corrected_forecasts.py','src/utils/forecast_sequence.py',
                  'src/prediction/tennis.py','tests/test_corrected_forecasts.py']
capacity_pins = {}
for name in capacity_names:
    key = str(Path(name))
    digest = previous['source_after'][key]
    assert digest == sha(root/name) == sha(previous_dir/'sources'/name)
    capacity_pins[name] = digest
capacity_records = [row for row in previous['observations'] if 'completed_point_forecast_capacity' in row]
assert len(capacity_records) == 1
capacity = capacity_records[0]
small,large = capacity['completed_point_forecast_capacity']
assert small['completed_points'] == small['deduplicated_point_ids'] == 8
assert large['completed_points'] == large['deduplicated_point_ids'] == 128
assert (small['history_rows'],large['history_rows']) == (49,769)
assert large['retained_python_bytes_inside_context']-small['retained_python_bytes_inside_context'] == 11860
assert large['peak_python_bytes']-small['peak_python_bytes'] < 2*1024*1024

# The complete previous run remains a failed run. Its capacity case is reused
# only because that exact passing case and all its source dependencies match.
shared_source_changes = [name for name,digest in previous['source_after'].items()
    if name in receipt['source_after'] and receipt['source_after'][name] != digest]
assert shared_source_changes == [str(Path('tests/test_product_matches.py'))]
fixture_name = str(Path('tests/test_product_matches.py'))
old_fixture = (previous_dir/'sources'/fixture_name).read_text(encoding='utf-8')
new_fixture = (root/fixture_name).read_text(encoding='utf-8')
diff = ''.join(difflib.unified_diff(old_fixture.splitlines(keepends=True),new_fixture.splitlines(keepends=True),
    fromfile='final_v1/test_product_matches.py',tofile='current/test_product_matches.py'))
assert "'id': 'match_test'" in diff
assert 'test_confirmed_profile_role_cannot_admit_missing_or_wrong_result_identity' in diff

profile_dir = base/'product_motion_identity_checks_v1'
profile = read(profile_dir/'receipt.json')
assert profile['pytest_exit_code'] == 0
assert profile['junit'] == {'tests':11,'failures':0,'errors':0,'skipped':0}
assert profile['source_before'] == profile['source_after'] and profile['source_pins_stable'] is True
assert not profile['model_import_attempts'] and not profile['forbidden_loaded']
for name,digest in profile['source_after'].items():
    assert sha(root/name) == digest
profile_tree,profile_cases = junit(profile_dir/'junit.xml')
assert len(profile_cases) == 11 and all(not list(case) for case in profile_cases)
negative_cases = [case for case in profile_cases
    if case.get('name','').startswith('test_confirmed_profile_role_cannot_admit_missing_or_wrong_result_identity[')]
assert len(negative_cases) == 2

review = {'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
    'reviewer':'/root/forecast_independent_review',
    'authorship':'Read-only independent review; no source/test edits and no tests rerun.',
    'receipt_sha256':expected_receipt,'remaining_P1_P2':[],
    'verified_current_source_pins':383,'verified_retained_source_copies':383,
    'verified_unique_passing_junit_cases':346,'deselected_cases':7,
    'runner_elapsed_seconds':receipt['elapsed_seconds'],
    'parent_peak_working_set_bytes':receipt['parent_peak_working_set_bytes'],
    'inference_import_attempts':receipt['inference_import_attempts'],
    'forbidden_inference_loaded':receipt['forbidden_inference_loaded'],
    'current_prior_independent_review_sha256':sha(prior_review_path),
    'prior_independent_review_source_pins':prior_review['source_pins'],
    'closed_findings':prior_review['closed_findings'],
    'capacity':{'receipt_sha256':sha(previous_dir/'receipt.json'),
        'source_pins':capacity_pins,'exact_junit_case':capacity_name,
        'case_passed':True,'whole_previous_run_failed':True,
        'unrelated_previous_failure':failures[0].get('name'),
        'measurements':capacity['completed_point_forecast_capacity'],
        'exclusions':capacity['excludes'],'retained_python_growth_bytes':11860},
    'profile_fixture_repair':{'only_shared_source_change':shared_source_changes,
        'application_sources_unchanged':True,
        'focus_receipt_sha256':sha(profile_dir/'receipt.json'),
        'focus_unique_passing_cases':11,'negative_identity_controls':2,
        'change':'Positive legacy fixture now supplies matching summary.id; missing and foreign identity remain rejected.'},
    'artifact_sha256':{str(path.relative_to(root)):sha(path) for path in
        (output/'junit.xml',output/'pytest.txt',previous_dir/'junit.xml',profile_dir/'junit.xml')},
    'limits':['Synthetic native rule/shot/forecast chain and two toy crop videos; no inference or independent tennis accuracy qualification.',
              'Forecasts remain explicitly not calibrated; independent prediction calibration/evaluation remains required.',
              'Capacity reuses one passing current-source case from a prior run with one unrelated fixture failure; it does not relabel the whole prior run as passing.',
              'Capacity excludes fixture/rule/shot setup, SQLite native memory and numerical library allocations; no long-match or live capacity qualification.',
              'Private scratch correction only; authenticated owner workflow, full corrected exports/rendering and generation compare-and-swap publication remain required.',
              'The terminal-state safeguard applies to the private native forecast stage; the legacy manual/global invariant API is unchanged.'],
    'published':False,'goal_complete':False}
(output/'final_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'current_pins':383,'retained_copies':383,
    'unique_passing_cases':346,'reused_capacity_case_passed':True,
    'remaining_P1_P2':[],'final_review_sha256':sha(output/'final_review.json')},indent=2))

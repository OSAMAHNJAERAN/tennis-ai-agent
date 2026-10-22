"""Independent read-only receipt/JUnit/current-source verification; no rerun."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / 'corrected_shot_independent_review_v3' / 'review.json'
EXPECTED_RECEIPT = 'e5591c633e771308fdbcfb3f1658e9f7b3090f17dd630f4add838341ec6d3f36'
EXPECTED_REVIEW = 'c92c0b3bfd208183cce582c667273a7432fd932fb52c13e9188a0b1b9329a257'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt_path = HERE / 'receipt.json'
assert digest(receipt_path) == EXPECTED_RECEIPT
assert digest(REVIEW) == EXPECTED_REVIEW
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
review = json.loads(REVIEW.read_text(encoding='utf-8'))
assert receipt['exit_code'] == 0 and receipt['source_stable'] is True
assert receipt['source_before'] == receipt['source_after']
assert len(receipt['source_after']) == 378
assert receipt['inference_import_attempts'] == []
assert receipt['forbidden_inference_loaded'] == []
assert receipt['published'] is False
assert receipt['optional_memory_observation_error'] is None
assert receipt['parent_peak_working_set_bytes'] == 231391232
assert review['findings'] == [] and review['source_pins_stable'] is True
assert review['reviewed_file_hashes_before'] == review['reviewed_file_hashes_after']
assert len(review['reviewed_file_hashes_after']) == 15
assert len(review['deterministic_edge_probes']) == 10
assert all(probe['pass'] is True for probe in review['deterministic_edge_probes'])

current = {name: digest(ROOT / name) for name in receipt['source_after']}
assert current == receipt['source_after']
assert all(digest(HERE / 'sources' / name) == sha
           for name, sha in receipt['source_after'].items())
source_inventory = {str(path.relative_to(ROOT))
                    for folder in ('src', 'tests')
                    for path in (ROOT / folder).rglob('*.py')}
expected_inventory = set(receipt['source_after']) - {str((HERE / 'run_checks.py').relative_to(ROOT))}
assert source_inventory == expected_inventory
assert all(digest(ROOT / name) == sha
           for name, sha in review['reviewed_file_hashes_after'].items())

junit_path = HERE / 'junit.xml'
junit = ET.parse(junit_path).getroot()
suites = list(junit.iter('testsuite'))
counts = {name: sum(int(suite.get(name, 0)) for suite in suites)
          for name in ('tests', 'errors', 'failures', 'skipped')}
assert counts == receipt['counts'] == {'tests': 256, 'errors': 0, 'failures': 0, 'skipped': 0}
cases = list(junit.iter('testcase'))
assert len(cases) == 256
assert not list(junit.iter('failure')) and not list(junit.iter('error')) and not list(junit.iter('skipped'))
identities = [(case.get('classname'), case.get('name')) for case in cases]
assert len(set(identities)) == len(identities)
module_counts = Counter(case.get('classname') for case in cases)
assert set(module_counts) == {name[:-3].replace('/', '.') for name in receipt['selected_test_modules']}
for name in (
    'test_corrected_double_fault_rebuilds_native_shots_memberships_and_analytics',
    'test_missing_required_pose_provider_refuses_without_silent_geometry_fallback',
    'test_corrected_shot_pipeline_keeps_owned_buffers_bounded_as_frames_and_shots_grow',
    'test_live_pose_callback_decodes_actual_sealed_frames_with_bounded_cache_and_cleanup',
    'test_actual_crop_decode_rejects_detection_pixel_dimensions_from_another_source',
):
    assert ('tests.test_corrected_shot_rebuild', name) in identities
for where in ('entry', 'crop_provider', 'consumer_exit'):
    name = 'test_current_generation_advance_rejects_obsolete_fresh_shot_work[' + where + ']'
    assert ('tests.test_corrected_shot_rebuild', name) in identities
log_path = HERE / 'pytest.txt'
log = log_path.read_text(encoding='utf-8')
assert re.search(r'256 passed, 9 deselected in [\d.]+s', log)

observations = receipt['observations']
capacity = next(item['corrected_shot_capacity'] for item in observations if 'corrected_shot_capacity' in item)
assert [(item['frame_count'], item['shot_count'], item['accepted_input_count'])
        for item in capacity] == [(257, 33, 34), (4097, 513, 514)]
assert [item['peak_bytes_including_cleanup'] for item in capacity] == [2258631, 2215912]
media = next(item['actual_synthetic_media'] for item in observations if 'actual_synthetic_media' in item)
assert media == {'codec': 'mp4v', 'frame_count': 12, 'fps': 30.0,
    'width': 1280, 'height': 960, 'crop_global_frame_indices_read': list(range(5, 12)),
    'maximum_cache_frames': 8, 'decoder_closed': True, 'selected_model_execution': False}

# Recheck after reading all retained evidence, without modifying source/test files.
assert current == {name: digest(ROOT / name) for name in receipt['source_after']}
assert digest(receipt_path) == EXPECTED_RECEIPT and digest(REVIEW) == EXPECTED_REVIEW
final = {
    'status': 'PASS',
    'remaining_P1_P2': [],
    'reviewer': '/root/classifier_lazy_import',
    'authorship_disclosure': review['authorship_disclosure'],
    'scope': 'INDEPENDENT_READ_ONLY_FINAL_RECEIPT_JUNIT_CURRENT_378_PINS_AND_PRIOR_15_PIN_SOURCE_REVIEW; NO_TEST_RERUN',
    'receipt_sha256': EXPECTED_RECEIPT,
    'junit_sha256': digest(junit_path),
    'pytest_log_sha256': digest(log_path),
    'source_review_path': str(REVIEW.relative_to(ROOT)),
    'source_review_sha256': EXPECTED_REVIEW,
    'source_files_and_runner_verified': len(current),
    'current_source_inventory_verified': len(source_inventory),
    'retained_source_copies_verified': len(current),
    'prior_review_current_pins_verified': len(review['reviewed_file_hashes_after']),
    'source_hash_mismatches': [],
    'junit_counts': counts,
    'unique_passed_testcases': len(identities),
    'testcases_by_module': dict(module_counts),
    'deselected_cases_from_log': 9,
    'runner_elapsed_seconds': receipt['elapsed_seconds'],
    'model_import_attempts': [],
    'model_runtimes_loaded': [],
    'parent_peak_working_set_bytes': receipt['parent_peak_working_set_bytes'],
    'corrected_shot_capacity': capacity,
    'actual_synthetic_media': media,
    'source_review_findings': review['findings'],
    'source_review_edge_probes': len(review['deterministic_edge_probes']),
    'previous_attempts_disclosed': [receipt['previous_attempt'], review['previous_probes']],
    'limitations': [
        'Synthetic source measurements/review bindings and trusted feature callback; no selected model execution or independent physical/vision accuracy qualification.',
        'The two toy video cases exercise actual decode and metadata refusal; they do not qualify tennis video inference.',
        'Capacity observations exclude fixture creation, numerical imports, and native SQLite/NumPy allocations outside tracemalloc.',
        'Whole correction publication, causal forecasts, updated exports/rendering, and authenticated correction workflow remain unfinished.',
        'Long match and genuine live-video acceptance remain outside this verification scope.',
    ],
    'published': False,
}
with (HERE / 'final_review.json').open('x', encoding='utf-8') as stream:
    json.dump(final, stream, indent=2)
print(json.dumps({key: final[key] for key in (
    'status', 'remaining_P1_P2', 'source_files_and_runner_verified',
    'prior_review_current_pins_verified', 'junit_counts', 'deselected_cases_from_log',
    'runner_elapsed_seconds', 'actual_synthetic_media')}))
print('final_review_sha256=' + digest(HERE / 'final_review.json'))

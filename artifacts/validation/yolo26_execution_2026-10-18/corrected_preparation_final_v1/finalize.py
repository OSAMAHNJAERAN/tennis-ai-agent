"""Freeze current private preparation evidence; preserve the full active goal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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

combined = read(ART/'receipt.json')
assert combined['exit_code'] == 0 and combined['source_stable']
assert combined['source_before'] == combined['source_after']
assert not combined['inference_import_attempts'] and not combined['forbidden_inference_loaded']
assert combined['optional_memory_observation_error'] is None
assert all(combined['counts'][key] == 0 for key in ('errors', 'failures', 'skipped'))
verify_pins(combined['source_after'])
inventory = {path.relative_to(ROOT).as_posix() for folder in ('src', 'tests')
             for path in (ROOT/folder).rglob('*.py')}
assert inventory <= {name.replace('\\', '/') for name in combined['source_after']}
for name, expected in combined['source_after'].items():
    assert digest(ART/'sources'/name) == expected, name
cases = list(ET.parse(ART/'junit.xml').getroot().iter('testcase'))
assert len(cases) == combined['counts']['tests']
assert len({(case.get('classname'), case.get('name')) for case in cases}) == len(cases)
assert all(case.find('failure') is None and case.find('error') is None
           and case.find('skipped') is None for case in cases)
assert len([case for case in cases if case.get('classname') == 'tests.test_corrected_preparation']) == 26
for name in (
    'test_preparation_owned_python_buffers_stay_bounded_as_real_video_frames_and_history_grow',
    'test_owned_presentation_and_report_bound_python_retention_as_completed_points_grow',
    'test_growing_completed_points_bounds_actual_export_history_and_strict_reader_memory',
    'test_actual_private_render_keeps_python_frame_buffers_bounded_as_duration_grows',
    'test_optional_sealed_observations_keep_exact_bytes_and_derivative_credit_is_fresh',
):
    assert len([case for case in cases if case.get('name') == name]) == 1, name
for name in ('junit.xml', 'pytest.txt'):
    REFERENCES[(ART/name).relative_to(ROOT).as_posix()] = digest(ART/name)

preparation_capacity = next(value for value in combined['preparation_observations']
    if 'private_preparation_capacity' in value)
assert [row['normalized_video_frames'] for row in preparation_capacity['private_preparation_capacity']] == [12, 272]
assert [row['completed_point_samples'] for row in preparation_capacity['private_preparation_capacity']] == [1, 1]
forecast_capacity = next(value for value in combined['forecast_presentation_observations']
    if 'actual_owned_forecast_presentation_capacity' in value)
assert [row['completed_points'] for row in forecast_capacity['actual_owned_forecast_presentation_capacity']] == [8, 128]
export_capacity = next(value for value in combined['export_observations']
    if 'completed_point_export_capacity' in value)
assert [row['completed_points'] for row in export_capacity['completed_point_export_capacity']] == [8, 128]
render_capacity = combined['render_capacity_observations'][0]
assert [row['frames'] for row in render_capacity['actual_corrected_render_frame_capacity']] == [12, 128]

review = read(BASE/'corrected_preparation_review_final_v1/final_review.json')
assert review['status'] == 'PASS' and not review['remaining_P1_P2']
assert review['receipt_sha256'] == digest(ART/'receipt.json')
assert review['combined_counts'] == combined['counts']
verify_pins(review['source_sha256'])
frontend = read(ROOT/'artifacts/validation/yolo26_execution_2026-10-17/analysis_generation_frontend_v2/receipt.json')
assert frontend['outcome'] == 'PASS' and frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
verify_pins(frontend['sourceBindingsAfter'], 'frontend')
admission = read(BASE/'gpu_admission_after_corrected_preparation_v1.json')
assert admission['training_helpers_unchanged'] and not admission['forbidden_import_attempts']
assert not admission['training_performed'] and not admission['cuda_initialized']
assert admission['training_helpers_expected'] == admission['training_helpers_observed']
verify_pins(admission['training_helpers_observed'])
admission_runner = BASE/'run_gpu_admission_after_corrected_preparation_v1.py'
assert digest(admission_runner) == admission['runner_sha256']
BOUND[admission_runner.relative_to(ROOT).as_posix()] = digest(admission_runner)
mapping = BASE/'authenticated_correction_handoff_v1/MAPPING.md'
mapping_pins = dict(re.findall(r'\| ([^|]+?) \| ([0-9a-f]{64}) \|', mapping.read_text(encoding='utf-8')))
assert len(mapping_pins) == 29
verify_pins(mapping_pins)
REFERENCES[mapping.relative_to(ROOT).as_posix()] = digest(mapping)

preview = read(ART/'report_preview/receipt.json')
visual = read(ART/'report_preview/visual_review.json')
assert preview['synthetic_source'] and preview['private_unpublished_correction']
assert preview['browser_parent_exit_code'] == 0
assert visual['status'] == 'PASS_LOCAL_SYNTHETIC_REPORT_LAYOUT'
assert visual['screenshot_sha256'] == preview['screenshot_sha256']
for name, expected in (('report_path', preview['report_sha256']),
                       ('screenshot_path', preview['screenshot_sha256'])):
    path = Path(preview[name])
    assert digest(path) == expected
    REFERENCES[path.relative_to(ROOT).as_posix()] = expected
assert visual['remaining_owned_browser_processes'] == 0
tail = read(ART/'report_preview/tail_receipt_v2.json')
assert tail['browser_parent_exit_code'] == 0 and tail['source_report_sha256'] == preview['report_sha256']
assert tail['screenshot_sha256'] == visual['tail_screenshot_sha256']
tail_source = Path(preview['report_path']).read_bytes()
extraction = tail['qa_extraction']
tail_qa = ART/'report_preview/tail_content_report.html'
assert tail_qa.read_bytes() == (tail_source[:extraction['prefix_end_byte']]
                              +tail_source[extraction['tail_start_byte']:])
assert digest(tail_qa) == tail['qa_copy_sha256']
for path in (tail_qa, Path(tail['screenshot_path'])):
    REFERENCES[path.relative_to(ROOT).as_posix()] = digest(path)
assert digest(Path(tail['screenshot_path'])) == tail['screenshot_sha256']

for name in ('docs/plans/YOLO26_EXECUTION_PROGRESS.md',
             'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md'):
    BOUND[name] = digest(ROOT/name)
    target = ART/'sources'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, target)
for path in (Path(__file__), ART/'preview_report.py', ART/'preview_report_tail.py',
             ART/'preview_report_tail_v2.py'):
    BOUND[path.relative_to(ROOT).as_posix()] = digest(path)

verification = {'status': 'PASS_CURRENT_SOURCE_BINDINGS',
    'verified_utc': datetime.now(timezone.utc).isoformat(), 'bound_file_count': len(BOUND),
    'source_bindings': dict(sorted(BOUND.items())), 'evidence_sha256': REFERENCES,
    'test_count': combined['counts']['tests'],
    'scope': 'Complete private corrected result/report and native chain over synthetic observations; surrounding API/storage regressions. Independent vision/prediction, actual long matches and live operation remain unqualified.'}
delivery = {'status': 'PROGRESS', 'previous_goal_turn_classification': 'PROGRESS',
    'goal_status': 'ACTIVE_INCOMPLETE',
    'completed_this_stage': [
        'Complete 49/54-file private corrected observations/video/result/report preparation',
        'Strict normalized point/landing evidence, owned causal native forecast history and bounded gap-preserving report curve',
        'Retained original inference provenance and analysis metadata without current checkpoint lookup',
        'Independent copy/source mutation closures, streamed report hashes, lazy exposed/stored result lifetime checks',
        'Consistent fresh derivative credit in JSON/HTML/CSV and local report layout review',
    ],
    'verification': {'tests': combined['counts']['tests'],
        'runner_seconds': combined['elapsed_seconds'], 'source_pins_unchanged': len(combined['source_after']),
        'bound_current_files': len(BOUND),
        'parent_peak_working_set_bytes': combined['parent_peak_working_set_bytes'],
        'preparation_capacity': preparation_capacity, 'forecast_presentation_capacity': forecast_capacity,
        'export_capacity': export_capacity, 'render_capacity': render_capacity,
        'receipt_sha256': digest(ART/'receipt.json'),
        'independent_review': str((BASE/'corrected_preparation_review_final_v1/final_review.json').relative_to(ROOT))},
    'training_admission': {key: admission[key] for key in ('observed_local', 'status', 'resources',
        'before_gpu_ram_floor_bytes', 'training_helpers_unchanged', 'training_performed')},
    'next_connected_seam': str(mapping.relative_to(ROOT)),
    'remaining_master_requirements': [
        'Durable owner-authenticated uploaded correction commands, worker, recovery/CAS and frontend review/forecast integration',
        'Repaired YOLO26 GPU stages after fixed resource admission and current full execution release; no private pilot promotion',
        'Untouched grouped independent detector/identity/ground/event/speed/point-winner/line-call/prediction evidence and gates',
        'Actual 10/30-minute and 2–3-hour inference, within-inference crash/resume and complete-match consistency',
        'Genuine live ingest with sustained FPS, coverage and measured P95 latency under stated capacity',
        'Qualified connected upload-to-report experience and deployment/user/operating acceptance',
    ], 'models_promoted': False, 'private_correction_published': False,
    'independent_accuracy_validated': False, 'master_goal_complete': False}
for name, value in (('current_source_verification.json', verification), ('delivery_record.json', delivery)):
    with (ART/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
print(json.dumps({'status': 'PROGRESS', 'tests': combined['counts']['tests'],
    'bound_current_files': len(BOUND), 'delivery_sha256': digest(ART/'delivery_record.json'),
    'goal_status': 'ACTIVE_INCOMPLETE'}, indent=2))

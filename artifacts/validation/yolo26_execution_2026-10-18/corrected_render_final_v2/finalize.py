"""Verify frozen current correction/export/render evidence; master stays active."""
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
assert combined['counts'] == {'tests': 532, 'failures': 0, 'errors': 0, 'skipped': 0}
verify_pins(combined['source_after'])
inventory = {path.relative_to(ROOT).as_posix() for folder in ('src', 'tests')
             for path in (ROOT/folder).rglob('*.py')}
assert inventory <= {name.replace('\\', '/') for name in combined['source_after']}
for name, expected in combined['source_after'].items():
    assert digest(ART/'sources'/name) == expected
cases = list(ET.parse(ART/'junit.xml').getroot().iter('testcase'))
assert len(cases) == 532
assert len({(case.get('classname'), case.get('name')) for case in cases}) == len(cases)
assert all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in cases)
for name in ('junit.xml', 'pytest.txt'):
    REFERENCES[(ART/name).relative_to(ROOT).as_posix()] = digest(ART/name)
for name in ('test_growing_completed_points_bounds_actual_export_history_and_strict_reader_memory',
             'test_actual_private_render_keeps_python_frame_buffers_bounded_as_duration_grows',
             'test_one_horizon_crossing_display_frame_withholds_geometry_then_recovers'):
    assert len([case for case in cases if case.get('name') == name]) == 1, name
render_capacity = combined['render_capacity_observations']
assert len(render_capacity) == 1
measurements = render_capacity[0]['actual_corrected_render_frame_capacity']
assert [row['frames'] for row in measurements] == [12, 128]
assert all([check['file'] for check in row['actual_full_decode_checks']] ==
           ['source.mp4', 'annotated.mp4', 'annotated_web.mp4'] for row in measurements)
export_capacity = [value for value in combined['observations'] if 'completed_point_export_capacity' in value]
assert len(export_capacity) == 1
assert [row['completed_points'] for row in export_capacity[0]['completed_point_export_capacity']] == [8, 128]
review = read(BASE/'corrected_render_review_final_v2/final_review.json')
assert not review['remaining_P1_P2']
assert review['receipt_sha256'] == digest(ART/'receipt.json')
verify_pins(review['source_sha256'])
export_review = read(BASE/'corrected_export_review_closure_v1/final_review.json')
assert export_review['status'] == 'PASS_WITH_CLOSED_P2' and not export_review['remaining_P1_P2']
verify_pins(export_review['source_sha256'])
frontend = read(ROOT/'artifacts/validation/yolo26_execution_2026-10-17/analysis_generation_frontend_v2/receipt.json')
assert frontend['outcome'] == 'PASS' and frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
verify_pins(frontend['sourceBindingsAfter'], 'frontend')
admission = read(BASE/'gpu_admission_after_corrected_render_v1.json')
assert admission['training_helpers_unchanged'] and not admission['forbidden_import_attempts']
assert not admission['training_performed'] and not admission['cuda_initialized']
assert admission['training_helpers_expected'] == admission['training_helpers_observed']
verify_pins(admission['training_helpers_observed'])
admission_runner = BASE/'run_gpu_admission_after_corrected_render_v1.py'
assert digest(admission_runner) == admission['runner_sha256']
BOUND[admission_runner.relative_to(ROOT).as_posix()] = digest(admission_runner)
mapping = BASE/'corrected_prepared_result_map_v1/MAPPING.md'
mapping_pins = dict(re.findall(r'\| ([^|]+?) \| ([0-9a-f]{64}) \|', mapping.read_text(encoding='utf-8')))
assert len(mapping_pins) == 31
verify_pins(mapping_pins)
REFERENCES[mapping.relative_to(ROOT).as_posix()] = digest(mapping)
preview = read(ART/'previews/receipt.json')
assert preview['decoded_frames'] == 12 and preview['synthetic_source'] is True
for path in preview['previews']:
    REFERENCES[Path(path).relative_to(ROOT).as_posix()] = digest(Path(path))
for name in ('docs/plans/YOLO26_EXECUTION_PROGRESS.md',
             'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md'):
    BOUND[name] = digest(ROOT/name)
    destination = ART/'sources'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, destination)
for path in (Path(__file__), ART/'preview.py'):
    BOUND[path.relative_to(ROOT).as_posix()] = digest(path)
verification = {'status': 'PASS_CURRENT_SOURCE_BINDINGS',
    'verified_utc': datetime.now(timezone.utc).isoformat(), 'bound_file_count': len(BOUND),
    'source_bindings': dict(sorted(BOUND.items())), 'evidence_sha256': REFERENCES,
    'test_count': 532, 'scope': 'Private fresh correction exports and actual CPU video rendering; synthetic source evidence, no independent match accuracy'}
with (ART/'current_source_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(verification, stream, indent=2)
delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
    'completed_this_stage': ['New source-aligned native/browser annotated video from corrected native owners',
        'Hash-bound export input, streamed live-event consistency and mutation-generation pin',
        'Causal score/uncalibrated prediction display and fresh current-frame landing polygons',
        'Source/camera/identity barriers and frame-local projective geometry recovery',
        'Full source/native/browser decode, independent closed-findings review and bounded render/export growth'],
    'verification': {'tests': 532, 'runner_seconds': combined['elapsed_seconds'],
        'source_pins_unchanged': len(combined['source_after']), 'bound_current_files': len(BOUND),
        'parent_peak_working_set_bytes': combined['parent_peak_working_set_bytes'],
        'render_capacity_observations': render_capacity, 'export_capacity_observations': export_capacity,
        'independent_review': review['status'], 'receipt_sha256': digest(ART/'receipt.json')},
    'gpu_host_admission': {'status': admission['status'], 'resources': admission['resources'],
        'observed_local': admission['observed_local'], 'training_performed': False},
    'remaining': ['Complete normalized result/report and forecast HTTP/contract/chart',
        'Durable authenticated owner-only correction commands and complete-generation compare-and-swap publication',
        'Selected crop model/runtime identity admission and actual selected-model compatibility',
        'Repaired-data GPU training and independent grouped model/numerical/prediction calibration gates',
        'Actual long-match inference, within-inference recovery, and sustained genuine live video evidence'],
    'published': False, 'independent_accuracy_validated': False, 'evidence_sha256': REFERENCES}
with (ART/'delivery_record.json').open('x', encoding='utf-8') as stream:
    json.dump(delivery, stream, indent=2)
print(json.dumps({'status': delivery['status'], 'goal_status': delivery['goal_status'],
    'tests': 532, 'bound_files': len(BOUND), 'delivery_sha256': digest(ART/'delivery_record.json')}))

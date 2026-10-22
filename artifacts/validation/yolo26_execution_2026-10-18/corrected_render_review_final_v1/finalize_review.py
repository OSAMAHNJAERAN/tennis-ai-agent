"""Retain an independent current-source private-render review."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def receipt(folder):
    path = ART.parent / folder / 'receipt.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    return value, {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
event_old, event_old_ref = receipt('corrected_render_review_v3')
event_fixed, event_fixed_ref = receipt('corrected_render_review_closure_v1')
restored, restored_ref = receipt('corrected_render_review_midrestore_v1')
horizon_old, horizon_old_ref = receipt('corrected_render_review_horizon_v1')
horizon, horizon_ref = receipt('corrected_render_review_geometry_closure_v1')
overflow_old, overflow_old_ref = receipt('corrected_render_review_projection_overflow_v1')
overflow, overflow_ref = receipt('corrected_render_review_overflow_closure_v1')
assert event_old['source_stable'] and event_old['result']['render_accepted']
assert event_old['result']['substituted_unknown_badge_drawn']
assert event_fixed['source_stable'] and not event_fixed['result']['render_accepted']
assert restored['source_stable'] and not restored['result']['render_accepted']
assert restored['result']['callback_mutation'][0]['event_restored']
assert restored['result']['handles_closed'] and not restored['result']['render_seals']
assert horizon_old['source_stable'] and not horizon_old['result']['render_accepted']
for value, reason in ((horizon, 'PROJECTIVE_HORIZON'), (overflow, 'PROJECTED_COORDINATE_RANGE')):
    assert value['source_stable']
    result = value['result']
    assert result['render_accepted']
    assert result['independent_decoded_frame_counts'] == {
        'source.mp4': 12, 'annotated.mp4': 12, 'annotated_web.mp4': 12}
    assert result['geometry_withheld_frame_count'] == 1
    assert result['geometry_withheld_reason_counts'] == {reason: 1}
    assert 9 not in result['court_drawn_frames'] and 10 in result['court_drawn_frames']
    assert result['camera_flag_after_render'] is True and result['committed_base_unchanged']
    assert any(frame == 9 and reason in caption for frame, caption in result['unavailable_captions'])
    assert not value['forbidden_import_attempts'] and not value['forbidden_imports_loaded']
    assert all(digest(ROOT / name) == expected for name, expected in value['source_after'].items())

paths = ['src/scoring/corrected_rendering.py', 'src/scoring/corrected_exports.py',
    'src/scoring/corrected_landing_exports.py', 'src/scoring/corrected_rule_rebuild.py',
    'src/scoring/corrected_forecasts.py', 'src/scoring/corrected_shot_rebuild.py',
    'src/scoring/corrected_shot_sources.py', 'src/scoring/correction_sources.py',
    'src/utils/event_analysis_sequence.py', 'src/visualization/landing_zones.py',
    'src/visualization/video_annotator.py', 'src/visualization/mini_court.py',
    'tests/test_corrected_rendering.py']
pins = {name: digest(ROOT / name) for name in paths}
report = {
    'status': 'PASS_WITH_CLOSED_FINDINGS',
    'remaining_P1_P2': [],
    'source_sha256': pins,
    'independence': 'Reviewer changed no implementation or test files in this stage. Only isolated read-only-review harnesses, synthetic fixture outputs and receipts were written.',
    'closed_findings': [
        {'priority': 1, 'title': 'Renderer imported an absent parser alias',
         'proof': 'Initial independent invocation raised ImportError before rendering.',
         'baseline': receipt('corrected_render_review_v1')[1],
         'fix': 'Use the existing _ephemeral_storage adapter through the intended alias.',
         'closure': [horizon_ref, overflow_ref]},
        {'priority': 1, 'title': 'Small parsed landing arrays failed reader cleanup',
         'proof': 'The native parser keeps small arrays as lists; closing(iter(list)) raised AttributeError for default empty landing records.',
         'baseline': receipt('corrected_render_review_v2')[1],
         'fix': 'The private _reader context closes only closeable iterators and retains cleanup failures correctly.',
         'closure': [horizon_ref, overflow_ref]},
        {'priority': 2, 'title': 'Writable live events could render semantics different from the sealed export',
         'proof': 'A normal indexed event-type write changed sealed BOUNCE to a rendered UNKNOWN_EVENT while both actual video codecs and final seal succeeded.',
         'baseline': event_old_ref,
         'fix': 'Stream-compare live physical events to pinned match_events and capture the owner change token before comparison; retain that exact token before and after every frame and final checks.',
         'closure': [event_fixed_ref, restored_ref],
         'restore_equal_case': 'Normal changed-and-restored writes advanced generation 6 to 8 and were rejected with closed source/writer handles, no final seal and unchanged exports/base.'},
        {'priority': 2, 'title': 'One unusable display projection aborted an otherwise recoverable video',
         'proof': 'A finite invertible frame-9 matrix crossed the court horizon; another same-hemisphere matrix exceeded OpenCV coordinate representation. Semantic exports had succeeded before the renderer aborted.',
         'baseline': [horizon_old_ref, overflow_old_ref],
         'baseline_limitation': 'The overflow baseline application pins were stable but test_corrected_rendering.py changed during the run; that receipt correctly reports source_stable=False. Both final closure receipts are fully source-stable.',
         'fix': 'Admit whole-court and retained landmark display projection separately from source qualification, withhold only known unusable display geometry and reset continuity; retain source matrices and accepted rules.',
         'closure': [horizon_ref, overflow_ref],
         'executed_closure': 'Both cases fully decode source/native/browser at 12/12/12, withhold frame 9 with an explicit reason, and restore court drawing on frame 10.'},
    ],
    'reviewed_boundaries': [
        'Native frozen export manifest digest plus exact inventory and every artifact hash before and after rendering.',
        'Source-only native rendering and fresh H264 browser encoding; full source/native/browser decode count, dimensions and normalized frame rate.',
        'Current native frame-state HUD, accepted line calls, dead-ball suppression and prefix-only forecast history without future-event badges.',
        'Current display-frame landing reprojection plus camera/identity/source barriers and bounded trail/history storage.',
        'Detached actual engine initial state rather than mutable timeline presentation fallback.',
        'No renderer use of a fourth model or unsupported airborne/player ground reconstruction.',
        'Final video hashes bound before and after full decoding and before returning a private render seal.',
    ],
    'limitations': [
        'The parent current-source combined integration run is separate evidence and was still running when this independent review was finalized.',
        'All footage and geometry in these probes are explicitly synthetic. No independent detector, calibration, speed, prediction, long-match or live qualification is established.',
        'Rendering remains private; report construction, authenticated correction lifecycle, complete-generation publication and product integration remain outside this helper.',
    ],
    'published': False,
}
(ART / 'final_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': report['status'], 'remaining_P1_P2': [],
                  'report_sha256': digest(ART / 'final_review.json')}, indent=2))

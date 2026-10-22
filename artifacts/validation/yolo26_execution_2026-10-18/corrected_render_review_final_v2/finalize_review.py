"""Verify the settled combined receipt and final HUD-only display change."""
import difflib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
EXECUTION = ART.parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
receipt_path = EXECUTION / 'corrected_render_final_v2/receipt.json'
receipt = read(receipt_path)
assert receipt['exit_code'] == 0 and receipt['source_stable']
assert receipt['counts'] == {'tests': 532, 'errors': 0, 'failures': 0, 'skipped': 0}
assert receipt['source_before'] == receipt['source_after']
assert not receipt['inference_import_attempts'] and not receipt['forbidden_inference_loaded']
for name, expected in receipt['source_after'].items():
    assert digest(ROOT / name) == expected, name
suites = ET.parse(receipt_path.with_name('junit.xml')).getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
          for key in ('tests', 'errors', 'failures', 'skipped')}
assert counts == receipt['counts']
earlier_path = EXECUTION / 'corrected_render_review_final_v1/final_review.json'
assert digest(earlier_path) == '6a4197eb28f422832f773eff8e5ec571330c72b9d7ef03049865711072d97461'
earlier = read(earlier_path)
assert earlier['remaining_P1_P2'] == []
label_path = EXECUTION / 'corrected_render_label_review_v1/receipt.json'
label = read(label_path)
assert label['source_stable'] and not label['forbidden_import_attempts']
assert not label['forbidden_imports_loaded']
for name, expected in label['source_after'].items():
    assert digest(ROOT / name) == expected
assert label['result']['all_actual_labels_fit']
assert label['result']['all_point_state_labels_fit']
assert label['result']['label_clearance_min'] >= 100
assert label['result']['native_metadata'] == label['result']['browser_metadata'] == {
    'width': 1280, 'height': 960, 'fps': 30., 'frames': 12}
assert label['result']['sealed_exports_unchanged'] and label['result']['committed_base_unchanged']
previous_source = EXECUTION / 'corrected_render_final_v1/sources/src/scoring/corrected_rendering.py'
current_source = ROOT / 'src/scoring/corrected_rendering.py'
diff = ''.join(difflib.unified_diff(previous_source.read_text(encoding='utf-8').splitlines(True),
    current_source.read_text(encoding='utf-8').splitlines(True),
    fromfile='settled-before-final-HUD-label', tofile='current-final-HUD-label'))
assert 'state_label' in diff and 'getTextSize' in diff
(ART / 'final_hud_change.diff').write_text(diff, encoding='utf-8')
paths = sorted(set(earlier['source_sha256']) | set(label['source_after']))
pins = {name: digest(ROOT / name) for name in paths}
report = {
    'status': 'PASS', 'remaining_P1_P2': [],
    'source_sha256': pins,
    'receipt_sha256': digest(receipt_path),
    'receipt': str(receipt_path.relative_to(ROOT)),
    'combined_counts': counts,
    'combined_current_source_pin_count': len(receipt['source_after']),
    'prior_full_review': {'path': str(earlier_path.relative_to(ROOT)), 'sha256': digest(earlier_path)},
    'last_change_review': {
        'scope': 'Final readable native service/state label with measured width fitting before the mini-court.',
        'comparison': 'Read-only diff against corrected_render_final_v1 retained source; only the HUD label block changed.',
        'diff': 'final_hud_change.diff', 'diff_sha256': digest(ART / 'final_hud_change.diff'),
        'actual_label_probe_receipt': {'path': str(label_path.relative_to(ROOT)), 'sha256': digest(label_path)},
        'actual_text_minimum_mini_court_clearance_px': label['result']['label_clearance_min'],
        'all_declared_point_state_labels_fit_1280px': label['result']['all_point_state_labels_fit'],
        'actual_native_and_browser_decoded_frames': 12,
        'visual_inspection': 'Viewed local frame4.png: full service/state label remains readable left of the mini-court; corrected receiver score 30 and fault banner remain readable.',
        'frame4_path': label['result']['frame4_path'],
        'frame4_sha256': digest(Path(label['result']['frame4_path'])),
    },
    'closed_findings': earlier['closed_findings'],
    'evidence_limits': [
        'Earlier defect proofs and closure probes retain the source hashes they actually executed; the final HUD change is separately checked above.',
        'The 532-test current combined run includes actual native/browser toy-codec checks and separately scoped capacity checks; it proves software integrity, not independent tennis accuracy.',
        'All retained review footage is synthetic. No detector, speed, calibration, prediction calibration, complete-match or sustained live gate is established.',
        'Private export/render helpers do not implement authenticated correction publication, normalized reports or UI integration.',
    ],
    'independence': 'Reviewer changed no implementation/test sources; only private review harnesses and evidence artifacts were created.',
    'published': False,
}
(ART / 'final_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': 'PASS', 'counts': counts, 'source_pin_count': len(pins),
    'receipt_sha256': report['receipt_sha256'],
    'review_sha256': digest(ART / 'final_review.json')}, indent=2))

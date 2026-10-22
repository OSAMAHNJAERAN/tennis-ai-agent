"""Bind the independent review findings and explicit post-fix probes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
baseline = ART.parent / 'corrected_export_review_v1/receipt.json'
closure = ART / 'receipt.json'
old = json.loads(baseline.read_text(encoding='utf-8'))
fixed = json.loads(closure.read_text(encoding='utf-8'))
assert old['source_stable'] and fixed['source_stable']
assert all(row.get('export_accepted') is True for row in old['results'])
assert all(row.get('export_accepted') is False for row in fixed['results'])
assert all(row.get('committed_base_unchanged') is True for row in fixed['results'])
assert not fixed['forbidden_import_attempts'] and not fixed['forbidden_imports_loaded']
paths = ['src/scoring/corrected_exports.py', 'src/scoring/corrected_landing_exports.py',
         'src/scoring/corrected_rule_rebuild.py', 'src/scoring/corrected_forecasts.py']
pins = {name: digest(ROOT / name) for name in paths}
assert all(fixed['source_after'][name.replace('/', '\\')] == value
           for name, value in pins.items())
report = {
    'status': 'PASS_WITH_CLOSED_P2',
    'remaining_P1_P2': [],
    'source_sha256': pins,
    'reviewer_contribution': 'Read-only implementation review and isolated probe artifacts; reviewer changed no application source or tests in this stage.',
    'closed_findings': [{
        'priority': 2,
        'title': 'Writable physical event snapshots could contradict accepted source evidence in a final-sealed export',
        'original_proof': 'Normal indexed writes injected a nonbounce (999,999)m court position, confidence .001 in place of .99, or an unsupported trajectory state. Each produced a final seal while native replay retained original evidence.',
        'fix': 'Strict canonical confidence/trajectory source joins and mandatory unavailable nonbounce court position before writing physical events.',
        'closure': 'All three exact prior commands are now rejected before any final export seal; immutable base remains unchanged.',
        'baseline_receipt': {'path': str(baseline.relative_to(ROOT)), 'sha256': digest(baseline)},
        'closure_receipt': {'path': str(closure.relative_to(ROOT)), 'sha256': digest(closure)},
    }],
    'other_reviewed_boundaries': [
        'Unique newly created private stage and exact 19-artifact inventory; final manifest written last and removed on caught failure.',
        'Independent bounded source copy without hardlinks; no generation publication from this coordinator.',
        'Strict owned ResultStorage point reader plus unchanged reproduction of the newly generated lossless replay.',
        'Fresh native analytics reducer, detached admitted command and frozen forecast metadata instead of mutable consumer audits.',
        'New frozen manifest_sha256 binds the coordinator return for the subsequent renderer to verify.',
    ],
    'limitations': [
        'Full 489-test frozen receipt predates the closed finding and extra manifest binding; parent must run current-source integration after the intentional fixes.',
        'Renderer before/after manifest hash admission was planned but its source did not yet exist during this review; no renderer claim.',
        'Synthetic software consistency only; no vision, speed, calibration, live, long-match or independent accuracy qualification.',
    ],
    'published': False,
}
(ART / 'final_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': report['status'], 'remaining_P1_P2': [],
                  'report_sha256': digest(ART / 'final_review.json')}, indent=2))

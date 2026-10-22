"""Bind the independent preparation review to the settled coupled run."""
import ast
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
def reference(folder):
    path = EXECUTION / folder / 'receipt.json'
    return read(path), {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}

baseline, baseline_ref = reference('corrected_preparation_review_v1')
copy_closure, copy_ref = reference('corrected_preparation_review_copy_closure_v1')
transient, transient_ref = reference('corrected_preparation_review_transient_v1')
assert baseline['source_stable'] and baseline['result']['preparation_successful_exit']
assert baseline['result']['candidate_seal_exists']
assert baseline['result']['copied_points'] == 'FORGED 999'
assert baseline['result']['trusted_export_points'] == '30'
for value in (copy_closure, transient):
    assert value['source_stable'] and not value['result']['preparation_successful_exit']
    assert not value['result']['final_seals']
    assert value['result']['committed_base_unchanged']
    assert value['result']['export_unchanged'] and value['result']['render_unchanged']
    assert not value['forbidden_import_attempts'] and not value['forbidden_imports_loaded']

# The proof source was retained by the separately executed focused runner.
# Verify the complete current module differs only by the reviewed derivative
# description assignment; all copy and report guards remain identical AST.
closure_source = EXECUTION / 'corrected_preparation_checks_v2/sources/src/scoring/corrected_preparation.py'
assert digest(closure_source) == copy_closure['source_after']['src/scoring/corrected_preparation.py']
old_tree = ast.parse(closure_source.read_text(encoding='utf-8'))
current_tree = ast.parse((ROOT / 'src/scoring/corrected_preparation.py').read_text(encoding='utf-8'))
new_assignment = ast.dump(ast.parse("result['mediaSource'] = {**source, 'modifications': modifications}").body[0], include_attributes=False)
class RemoveReviewedDescription(ast.NodeTransformer):
    count = 0
    def visit_Assign(self, node):
        if ast.dump(node, include_attributes=False) == new_assignment:
            self.count += 1
            return None
        return self.generic_visit(node)
normalizer = RemoveReviewedDescription()
normalized_current = normalizer.visit(current_tree)
assert normalizer.count == 1
assert ast.dump(old_tree, include_attributes=False) == ast.dump(normalized_current, include_attributes=False)

receipt_path = EXECUTION / 'corrected_preparation_final_v1/receipt.json'
receipt = read(receipt_path)
assert receipt['exit_code'] == 0 and receipt['source_stable']
assert receipt['source_before'] == receipt['source_after']
assert not receipt['inference_import_attempts'] and not receipt['forbidden_inference_loaded']
assert all(receipt['counts'][key] == 0 for key in ('errors', 'failures', 'skipped'))
selected = receipt.get('selected_test_modules', receipt.get('tests', []))
for name in ('tests/test_corrected_preparation.py', 'tests/test_corrected_forecast_presentation.py',
             'tests/test_corrected_rendering.py', 'tests/test_corrected_exports.py',
             'tests/test_bounded_reporting.py', 'tests/test_worker_result_publication.py'):
    assert name in selected, (name, selected)
for name, expected in receipt['source_after'].items():
    assert digest(ROOT / name) == expected, name
suites = ET.parse(receipt_path.with_name('junit.xml')).getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
          for key in ('tests', 'errors', 'failures', 'skipped')}
assert counts == receipt['counts']
optional, optional_ref = reference('corrected_preparation_optional_v1')
assert optional['source_stable'] and optional['counts'] == {
    'tests': 1, 'errors': 0, 'failures': 0, 'skipped': 0}
paths = ['src/scoring/corrected_preparation.py', 'src/scoring/corrected_forecast_presentation.py',
    'src/scoring/corrected_rendering.py', 'src/scoring/corrected_exports.py',
    'src/scoring/corrected_landing_exports.py', 'src/api/results.py', 'src/api/reporting.py',
    'src/api/analysis_generation.py', 'src/api/result_storage.py', 'src/api/result_document.py',
    'src/api/point_timeline_contract.py', 'src/api/landing_zone_contract.py',
    'tests/test_corrected_preparation.py', 'tests/test_corrected_forecast_presentation.py']
report = {
    'status': 'PASS', 'remaining_P1_P2': [],
    'source_sha256': {name: digest(ROOT / name) for name in paths},
    'receipt': str(receipt_path.relative_to(ROOT)), 'receipt_sha256': digest(receipt_path),
    'combined_counts': counts, 'combined_current_source_pin_count': len(receipt['source_after']),
    'reviewer_independence': 'No application or test source edits by reviewer; only isolated review harnesses, synthetic outputs and receipts were created.',
    'closed_findings': [{
        'priority': 2, 'title': 'Copied semantic artifacts could change during report production and be rebound into a valid preparation seal',
        'baseline': baseline_ref,
        'executed_proof': 'The first HTML producer callback changed only copied match_state.json from receiver 30 to FORGED 999. Native state, trusted exports, reports and base stayed unchanged, but the preparation entered, exited successfully and retained a final seal.',
        'fix': 'Expected copy records come from the frozen native export/render producer manifests and original generation inventory; retain and check every copy before parsing, after each report writer, before sealing and through context exit. Hash generated report bytes as streamed before later writers execute.',
        'closure': copy_ref,
        'transient_producer_guard': transient_ref,
        'closure_result': 'Exact staged-copy mutation is rejected without a seal. A producer metrics mutation after entry guards is rejected before copying against its frozen record, even when cleanup restores all original bytes. Export/render/base bytes remain unchanged.',
        'evidence_note': 'Closure receipts retain their original source pins; the only subsequent preparation addition updates new derivative modifications in the normalized mediaSource, covered by the current coupled run and optional-credit fixture.',
    }],
    'reviewed_boundaries': [
        'Complete immutable observation inventory, exactly one sealed original extension, and independent copied assets; no mutable analysis-root fallback.',
        'Current owned rule/shot/forecast/export/render bindings and frozen manifest records; no current-generation pointer publication.',
        'Captured bounded inference provenance from the sealed base result, with explicit unknown fallback and no lookup of current checkpoint files.',
        'Original display filename, label and analysis timestamp from the sealed summary; caller job values supply neither provenance nor authorization.',
        'Strict native point and landing readers before normalization, followed by strict normalized source/revision/content checks.',
        'Fresh causal native forecasts reverified and detached into the one result-storage owner; both exposed result and stored root hashes checked against sealed report JSON.',
        'Atomic streamed escaped HTML/JSON/CSV, artifact hashes pinned before later producers, fresh derivative credit plus unchanged original attribution.',
        'Consumer/write failures remove only the pending preparation seal, close owned storage and preserve earlier complete private bundles and the immutable base.',
    ],
    'optional_credit_fixture': optional_ref,
    'historical_closure_source_equivalence': {
        'retained_source': str(closure_source.relative_to(ROOT)),
        'sha256': digest(closure_source),
        'current_ast_difference': "Only result['mediaSource'] = {**source, 'modifications': modifications}",
        'all_copy_report_and_owner_guard_ast_unchanged': True,
    },
    'future_integration_note': 'A future authenticated publisher should open a separate result-storage owner from sealed report JSON or use an explicit commit protocol. Publishing this context-owned storage closes it and would violate the preparation exit guard.',
    'limitations': [
        'Synthetic toy imagery, coordinates, identity reviews and retained model hashes; these tests establish software consistency, not independent tennis accuracy.',
        'Private preparation does not supply owner-authenticated correction commands, queue integration, compare-and-swap publication or UI review editing.',
        'Complete-match/live operation, model training, detector/geometry/speed/scoring qualification and calibrated forecast accuracy remain separate goal work.',
    ],
    'published': False,
}
(ART / 'final_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': 'PASS', 'counts': counts,
    'source_pin_count': len(report['source_sha256']),
    'receipt_sha256': report['receipt_sha256'],
    'review_sha256': digest(ART / 'final_review.json')}, indent=2))

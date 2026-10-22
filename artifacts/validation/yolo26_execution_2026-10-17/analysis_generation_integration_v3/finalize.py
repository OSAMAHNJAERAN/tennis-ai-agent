"""Bind existing scoped evidence to current files; no tests/models are run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
BASE = RUN.parent
ROOT = RUN.parents[3]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path):
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': digest(path)}


def main():
    targets = [RUN / 'current_source_verification.json', RUN / 'delivery_record.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve completed verification/delivery evidence')
    backend = read(RUN / 'receipt.json')
    frontend = read(BASE / 'analysis_generation_frontend_v2/receipt.json')
    frontend_first = read(BASE / 'analysis_generation_frontend_v1/receipt.json')
    pure = read(BASE / 'analysis_generation_pure_v3/receipt.json')
    capacity = read(BASE / 'analysis_generation_copy_capacity_v1/receipt.json')
    review = read(RUN / 'final_review.json')
    admission = read(BASE / 'gpu_admission_after_generation_v1.json')
    assert backend['status'] == 'PASS' and backend['counts'] == {
        'passed': 283, 'failed': 0, 'errors': 0, 'skipped': 0}
    assert frontend['outcome'] == frontend_first['outcome'] == 'PASS'
    assert frontend['sourcesUnchanged'] and frontend['productionUnchangedFromV1']
    assert frontend['tests']['passed'] == 48 and frontend_first['tests']['passed'] == 52
    assert pure['exit_code'] == 0 and pure['test_counts'] == {'passed': 51}
    assert capacity['status'] == 'PASS' and capacity['sources_unchanged']
    assert review['status'] == 'PASS' and review['remaining_actionable_p1_p2'] == []
    assert admission['training_helpers_unchanged'] and not admission['training_performed']
    assert backend['source_before'] == backend['source_after']
    assert frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
    expected = {}
    def add(bindings, prefix=ROOT):
        for name, checksum in bindings.items():
            path = prefix / name
            relative = str(path.relative_to(ROOT)).replace('\\', '/')
            if relative in expected:
                assert expected[relative] == checksum, ('Conflicting evidence', relative)
            expected[relative] = checksum
    add(backend['source_after'])
    add(frontend['sourceBindingsAfter'], ROOT / 'frontend')
    add(pure['source_after'])
    add(capacity['source_after'])
    add(review['source_pins'])
    add(admission['training_helpers_observed'])
    observed = {name: digest(ROOT / name) for name in expected}
    assert observed == expected, [name for name in expected if observed[name] != expected[name]]
    documents = ['docs/plans/YOLO26_EXECUTION_PROGRESS.md',
                 'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md']
    verification = {'status': 'PASS', 'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Current hashes equal the final backend/frontend/pure/capacity/review and frozen-training-helper bindings; this is not another test run or inference release.',
        'bound_file_count': len(expected), 'expected': expected, 'observed': observed,
        'documents': {name: digest(ROOT / name) for name in documents},
        'runner_sha256': digest(Path(__file__))}
    targets[0].write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    artifacts = [RUN / 'receipt.json', RUN / 'process_restart_evidence.json', RUN / 'final_review.json',
        BASE / 'analysis_generation_frontend_v1/receipt.json',
        BASE / 'analysis_generation_frontend_v2/receipt.json',
        BASE / 'analysis_generation_pure_v3/receipt.json',
        BASE / 'generation_http_checks_v1/post_review_v1/receipt.json',
        BASE / 'generation_owner_cache_review_v1/receipt.json',
        BASE / 'analysis_generation_copy_capacity_v1/receipt.json',
        BASE / 'gpu_admission_after_generation_v1.json', targets[0]]
    delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
        'previous_goal_turn_classification': 'PROGRESS',
        'implemented': ['Immutable complete-generation publication after native worker finalization',
            'Committed historical visibility and strict matching revision metadata',
            'Generation-bound replay/JSON/HTML/CSV/original/annotated/timing readers',
            'Explicit public generation selection, revocation and private newer owner versions',
            'Authenticated private no-store caching and curated Cookie variation',
            'Frontend generation replay/export/preview binding'],
        'verification': {'backend_passed': 283, 'backend_pins': 53,
            'frontend_first_passed': 52, 'frontend_followup_passed': 48,
            'counts_overlap_do_not_sum': True, 'current_source_verification': reference(targets[0]),
            'independent_publication_review': 'Four independently reproduced P2 issues repaired; no remaining actionable P1/P2 in the reviewed scope.'},
        'scope_limits': ['Worker/HTTP inference and media fixtures synthetic; existing demo test decodes only three 100x100 toy frames',
            'Copy/hash capacity uses nonplayable 8/64 MiB inputs; excludes model/decoder/full-match/live behavior',
            'Windows process crash verified; no storage/power-loss recovery guarantee',
            'Copies/history/failed diagnostics increase retained disk bytes; no automatic retention policy'],
        'remaining_master_requirements': ['Connected uploaded correction and audit commands',
            'Context-dependent event/call, score/prediction and analytics recomputation with fresh annotated media',
            'Repaired-data GPU stages after unchanged resource admission and a fresh full source/runtime release',
            'Untouched grouped independent vision/geometry/event/speed/outcome/prediction qualification',
            'Actual 10/30 minute and two-to-three-hour matches, within-inference crash resume and sustained live throughput/latency'],
        'resource_observation': reference(BASE / 'gpu_admission_after_generation_v1.json'),
        'new_actual_model_inference': False, 'new_training_performed': False,
        'independent_accuracy_validated': False, 'goal_complete': False,
        'evidence': [reference(path) for path in artifacts]}
    targets[1].write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': delivery['status'], 'goal': delivery['goal_status'],
                      'current_bound_files': len(expected), 'backend_passed': 283,
                      'delivery_sha256': digest(targets[1])}))


if __name__ == '__main__':
    main()

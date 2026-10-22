"""Verify retained corrected-rule evidence against current files, without runtime imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
BASE = RUN.parent
ROOT = RUN.parents[3]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ref(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def main():
    current_path, delivery_path = (RUN / name for name in
                                   ('current_source_verification.json', 'delivery_record.json'))
    if current_path.exists() or delivery_path.exists():
        raise FileExistsError('Retain previous verification/delivery evidence')
    combined = read(RUN / 'receipt.json')
    decoder = read(BASE / 'rule_replay_source_checks_v3/receipt.json')
    adapters = read(BASE / 'correction_sources_checks_v6/receipt.json')
    core = read(BASE / 'corrected_rule_rebuild_checks_v4/receipt.json')
    frontend = read(BASE / 'analysis_generation_frontend_v2/receipt.json')
    review = read(RUN / 'final_review.json')
    admission_path = BASE.parent / 'yolo26_execution_2026-10-18/gpu_admission_after_corrected_rules_v1.json'
    admission = read(admission_path)
    assert combined['status'] == decoder['status'] == review['status'] == 'PASS'
    assert combined['counts'] == {'passed': 780, 'failed': 0, 'errors': 0, 'skipped': 0}
    assert decoder['counts']['passed'] == 186 and not decoder['forbidden_modules_present']
    assert adapters['exit_code'] == core['exit_code'] == 0
    assert adapters['source_pins_stable'] and core['source_stable']
    assert core['counts'] == {'tests': 41, 'errors': 0, 'failures': 0, 'skipped': 0}
    assert frontend['outcome'] == 'PASS' and frontend['sourcesUnchanged']
    assert review['remaining_actionable_p1_p2'] == [] and not review['overall_master_goal_complete']
    assert admission['training_helpers_unchanged'] and not admission['training_performed']
    expected = {}
    def add(bindings, prefix=ROOT):
        for name, checksum in bindings.items():
            path = prefix / name
            relative = path.relative_to(ROOT).as_posix()
            assert relative not in expected or expected[relative] == checksum, relative
            expected[relative] = checksum
    for receipt in (combined, decoder, adapters, core):
        assert receipt['source_before'] == receipt['source_after']
        add(receipt['source_after'])
    assert frontend['sourceBindingsBefore'] == frontend['sourceBindingsAfter']
    add(frontend['sourceBindingsAfter'], ROOT / 'frontend')
    add(review['current_source_and_test_sha256'])
    add(admission['training_helpers_observed'])
    for evidence in review['evidence']:
        assert digest(ROOT / evidence['receipt']) == evidence['receipt_sha256']
    closure = review['closure_probe']
    assert digest(ROOT / closure['receipt']) == closure['receipt_sha256']
    observed = {name: digest(ROOT / name) for name in expected}
    assert expected == observed, [name for name in expected if expected[name] != observed[name]]
    documents = ['docs/plans/YOLO26_EXECUTION_PROGRESS.md',
                 'docs/plans/ANALYTICS_CAPACITY_SEAM_2026-10-16.md', 'docs/PRODUCT_WORKFLOW.md']
    for name in documents:
        destination = RUN / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(), destination
        shutil.copy2(ROOT / name, destination)
    verification = {'status': 'PASS', 'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Current files equal retained combined/focused/frontend/review and frozen helper bindings; no new tests, model execution or GPU release.',
        'bound_file_count': len(expected), 'expected': expected, 'observed': observed,
        'documents': {name: digest(ROOT / name) for name in documents},
        'runner_sha256': digest(Path(__file__))}
    current_path.write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    evidence_paths = [RUN / 'receipt.json', RUN / 'junit.xml', RUN / 'final_review.json',
        RUN / 'process_restart_evidence.json', BASE / 'rule_replay_source_checks_v3/receipt.json',
        BASE / 'correction_sources_checks_v6/receipt.json',
        BASE / 'corrected_rule_rebuild_checks_v4/receipt.json', ROOT / closure['receipt'],
        admission_path, current_path]
    delivery = {'status': 'PROGRESS', 'goal_status': 'ACTIVE_INCOMPLETE',
        'previous_goal_turn_classification': 'PROGRESS',
        'implemented': ['Shared bounded strict original replay source decoding with unchanged reproduction semantics',
            'Sealed unrounded ball/current-camera sources and optical/confidence/continuity/projective-view masks',
            'Private source-bound initial-state/semantic amendment processing with fresh actual context-dependent line calls',
            'Rule revision zero inside new analysis revision with stale-current checks and no recorded-call reuse'],
        'decisive_case': 'Synthetic initial second-serve correction0-0/deuce to0-15/ad; identical observations now FAR_AD/SERVE_FAULT/DOUBLE_FAULT; one receiver point, final0-30. Actual numerical engine/refiner, no vision accuracy.',
        'verification': {'combined_passed': 780, 'combined_deselected_capacity_cases': 4,
            'combined_source_pins': 372, 'decoder_passed': 186, 'adapter_passed': 55,
            'core_passed': 41, 'counts_overlap_do_not_sum': True,
            'current_source_verification': ref(current_path)},
        'review': {'status': review['status'], 'remaining_actionable_p1_p2': [],
            'authorship_disclosure': review['authorship_disclosure']},
        'remaining': ['Fresh physical-shot relinking/classification and attachment',
            'Derived analytics and causal forecasts', 'Fresh source-based annotated/browser rendering',
            'Complete corrected generation and durable authorized command/audit/UI lifecycle',
            'Qualified repaired-data GPU stages and independent grouped accuracy',
            'Actual long-match, within-inference recovery and sustained live capacity'],
        'scope_limits': ['Synthetic measured observations; no actual match/model inference',
            'Private pending timeline; no correction endpoint or publication enabled',
            'Capacity tracing excludes fixtures and native SQLite/NumPy allocations',
            'Primitive review is not independent authorship review of all shared decoder code'],
        'new_training_performed': False, 'independent_accuracy_validated': False,
        'goal_complete': False, 'evidence': [ref(path) for path in evidence_paths]}
    delivery_path.write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': delivery['status'], 'goal_status': delivery['goal_status'],
        'bound_files': len(expected), 'combined_passed': 780, 'delivery_sha256': digest(delivery_path)}))


if __name__ == '__main__':
    main()

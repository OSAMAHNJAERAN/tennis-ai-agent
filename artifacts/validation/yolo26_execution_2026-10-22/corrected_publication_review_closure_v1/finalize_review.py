"""Bind independent review findings to exact current production closures."""
import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
def read(name):
    return json.loads((ART/name).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
database = read('receipt.json')
pointer = read('pointer_receipt.json')
execution = read('execution_interrupt_receipt.json')
records = [database, pointer, execution]
pins = database['source_after']
assert len(pins) == 9
assert all(record['source_unchanged'] and record['source_before'] == pins == record['source_after']
           and not record['forbidden_import_attempts'] and not record['torch_imported']
           and not record['ultralytics_imported'] for record in records)
assert all(sha(ROOT/name) == digest for name, digest in pins.items())
assert database['closure_passed'] and not database['issue_reproduced']
assert database['publication_token'] is None and not database['current_pointer_changed']
assert database['current_result_read_error'] is None and len(database['mutations']) == 1
assert 'database changed from trusted closed producer bytes' in database['publication_error']
assert pointer['status'] == 'PASS' and pointer['publication_token'] is None
assert pointer['current_pointer_unchanged'] and pointer['old_result_readable']
assert pointer['private_preparation_unchanged'] and not pointer['leftover_delivery_seals']
assert execution['closure_passed']
assert [(row['actual_software_proxy_predictions'], row['provider_calls'], row['new_vision_inference'])
        for row in execution['counter_observations']] == [(0, 1, False), (2, 1, True)]
assert all(row['record_survived_provider_close'] and row['publisher_schema_preserved']
           for row in execution['counter_observations'])
assert [row['interrupt'] for row in execution['interrupt_observations']] == ['KeyboardInterrupt', 'SystemExit']
assert all(row['caught_interrupt'] == row['interrupt'] and row['returned_generation'] is None
           and not row['interrupt_suppressed'] and row['committed_result_readable']
           for row in execution['interrupt_observations'])
receipt_names = ['receipt.json', 'pointer_receipt.json', 'execution_interrupt_receipt.json']
evidence_names = [*receipt_names, 'verify_database_gap.py', 'verify_pointer_gap.py',
                  'verify_execution_interrupts.py', Path(__file__).name]
result = {'status': 'PASS_CURRENT_BACKEND_FINDING_CLOSURE',
    'review_scope': 'READ_ONLY CORRECTION WORKER/STORE/PUBLISHER REVIEW AND ACTUAL SYNTHETIC NATIVE CAS FAULT PROBES',
    'current_source_pins': pins, 'evidence_sha256': {name: sha(ART/name) for name in evidence_names},
    'closed_findings': [
        {'priority': 'P1', 'issue': 'Candidate DB mutation after native manifest hash and before final callback was learned as a trusted baseline.',
         'proof': 'Original mutation probe now rejects trusted closed producer byte mismatch; current pointer and old readable result remain intact.'},
        {'priority': 'P1', 'issue': 'Candidate pointer temporary could be corrupted before final callback and committed.',
         'proof': 'Actual native pointer writer corruption is rejected against reconstructed canonical generation pointer before commit.'},
        {'priority': 'P2', 'issue': 'Committed KeyboardInterrupt/SystemExit were swallowed as ordinary lost acknowledgements.',
         'proof': 'Both actual post-CAS interrupts propagate while committed generations remain readable.'},
        {'priority': 'P2', 'issue': 'Crop inputs diverged from pinned predict defaults and provider invocations could be reported as actual inference.',
         'proof': 'Current proxy uses checkpoint-derived crop options; detached records preserve zero/two actual software proxy calls separately from one provider call.'}],
    'remaining_p1_p2_in_reviewed_backend_scope': [],
    'frontend_scope': 'Historical completed-view and multiline-review findings are assigned to frontend owner; additional foreign-base active-command historical autojump was reported and remains outside this backend closure.',
    'qualifications': {'actual_checkpoint_or_cuda_execution': False, 'independent_vision_accuracy': False,
        'real_long_match': False, 'genuine_live_video': False, 'frontend_browser_visuals': False},
    'goal_status': 'ACTIVE_INCOMPLETE'}
(ART/'final_review.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'source_pins': len(pins),
                  'final_review_sha256': sha(ART/'final_review.json')}))

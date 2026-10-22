"""Retain source-bound conclusions after the three exact review probes close."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path.cwd()
base = Path('artifacts/validation/yolo26_execution_2026-10-18')
output = root/base/'corrected_forecast_independent_review_v1'
sources = ['src/scoring/corrected_forecasts.py','src/utils/forecast_sequence.py',
           'src/prediction/tennis.py','src/scoring/scoring_rules.py']
pins = {path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources}
receipts = [
    'corrected_forecast_independent_review_v1/receipt.json',
    'corrected_forecast_independent_closure_v2/receipt.json',
    'corrected_forecast_terminal_state_review_v1/receipt.json',
    'corrected_forecast_terminal_state_closure_v1/receipt.json',
]
evidence = {}
for relative in receipts:
    path = root/base/relative
    data = path.read_bytes()
    document = json.loads(data)
    evidence[str(base/relative)] = {'sha256':hashlib.sha256(data).hexdigest(),
                                  'status':document['status']}
    if 'closure' in relative:
        assert document['status'] == 'PASS'
        assert document['source_pins_before'] == document['source_pins_after']
        assert not document['forbidden_import_attempts'] and not document['loaded_models']
        for name,pin in document['source_pins_after'].items():
            if name in pins:
                assert pin == pins[name]
review = {'status':'PASS_REVIEWED_SCOPE', 'created_utc':datetime.now(timezone.utc).isoformat(),
    'reviewer_task':'/root/forecast_independent_review',
    'authorship':'Reviewer did not edit implementation or tests; authored retained private probes only.',
    'scope':['corrected native causal forecast driver and frozen result/export handles',
             'owned bounded forecast sequence and optional caller-owned predictor point index',
             'raw state, source timing, reviewed service anchor and point-terminal joins',
             'unknown/authority-off/calibration withholding, streaming and failure ownership'],
    'source_pins':pins,'remaining_p1_p2':[],
    'closed_findings':[
        {'priority':'P2','issue':'Mutable audit/current presentation aliases could forge export provenance and calibration claims.',
         'fix':'Frozen owned result fields; fresh detached current/audit values; native bound export handles.',
         'exact_reproduction_and_closure_retained':True},
        {'priority':'P2','issue':'Replacing public history and current could export one forged row instead of the sealed seven-row history.',
         'fix':'Public result handle replacement raises FrozenInstanceError; exporter requires the owning history handle.',
         'exact_reproduction_and_closure_retained':True},
        {'priority':'P2','issue':'Contradictory reviewed terminal set counts passed existing shared invariants and produced a false active-match forecast.',
         'fix':'Private native terminal context validation withholds all affected snapshots and posterior updates; original raw state hash retained.',
         'reason_code':'INVALID_TERMINAL_SCORE',
         'limitation':'Existing manual/global scoring invariant API was not changed; this closure applies to the new private native forecast path.',
         'exact_reproduction_and_closure_retained':True}],
    'evidence':evidence,
    'limitations':['Private scratch artifact only; future whole-generation correction publisher still requires owner authorization and compare-and-swap publication.',
                   'No independent calibration, match accuracy, inference, long-video or live qualification follows from these synthetic software probes.',
                   'Native Python private-field mutation, raw SQL mutation and malicious monkeypatching are outside the public handle integrity contract.'],
    'published':False,'goal_complete':False}
(output/'final_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'remaining_p1_p2':review['remaining_p1_p2'],
    'closed_findings':len(review['closed_findings']),
    'final_review_sha256':hashlib.sha256((output/'final_review.json').read_bytes()).hexdigest()},indent=2))

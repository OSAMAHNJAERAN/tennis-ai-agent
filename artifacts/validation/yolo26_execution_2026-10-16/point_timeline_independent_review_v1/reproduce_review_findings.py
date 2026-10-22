"""Independent read-only review probes of the current native timeline.

Small synthetic rule/context fixtures do not qualify perception or identity.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'ultralytics', 'numpy', 'cv2', 'PIL'):
            raise RuntimeError('Independent review refuses model/image import: '+fullname)
sys.meta_path.insert(0, RefuseModels())

import pytest
from tests import test_point_timeline as fixtures
from src.scoring.point_timeline import (PointTimeline, ReviewedRuleSignal, candidate_event_context,
                                       process_timeline_event, merge_rule_inputs)
from src.utils.point_analysis_sequence import PointAnalysisSequence

OUT = Path(__file__).resolve().parent
paths = [ROOT/'src/scoring/point_timeline.py',ROOT/'src/utils/point_analysis_sequence.py',
         ROOT/'src/scoring/scoring_engine.py',ROOT/'tests/test_point_timeline.py']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path.relative_to(ROOT)):sha(path) for path in paths}
results = {}
with pytest.MonkeyPatch.context() as patch:
    api = fixtures.api.__wrapped__(patch)
    # A new current confirmed identity revision appears after legal service.
    # No explicit point revision/correction/re-anchoring signal is supplied.
    source = fixtures.frames()
    for row in source[2:]:
        row[1]['revision'] = 1
    def contexts(ev, engine):
        old = fixtures.binding_context(ev, source, engine).binding
        current = replace(old, identity_revision=source[ev.frame_index][1]['revision'],
                          calibration_revision='revised-calibration' if ev.frame_index>=2 else old.calibration_revision)
        return candidate_event_context(ev, source, lambda *args: current,
            analysis_id='fixture',source_fingerprint='synthetic-source-v1')
    directory = OUT/'mid_rally_revision'; directory.mkdir(exist_ok=False)
    with PointAnalysisSequence(directory,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        events=[fixtures.event(1,'SERVE_CONTACT',1),fixtures.event(2,'BOUNCE'),
                fixtures.event(3,'PLAYER_2_HIT',2),fixtures.event(4,'BOUNCE')]
        engine,timeline,calls=fixtures.drive(api,owner,events,source,
            decisions={2:'SERVE_IN',4:'OUT'},contexts=contexts)
        transitions=list(owner.rows('transitions'))
        results['mid_rally_identity_and_calibration_revision_change']={
            'completed_points':len(owner.rows('points',status='COMPLETED')),
            'score_confirmed':engine.state.score_confirmed,
            'event3_rule_type':transitions[2]['rule_event_type'],
            'start_identity_revision':transitions[0]['context']['binding']['identity_revision'],
            'return_identity_revision':transitions[2]['context']['binding']['identity_revision'],
            'start_calibration_revision':transitions[0]['context']['binding']['calibration_revision'],
            'return_calibration_revision':transitions[2]['context']['binding']['calibration_revision'],
            'expected':'REVIEW_REQUIRED_UNTIL_EXPLICIT_REANCHORED_REVISION'}
    source=fixtures.frames(); ev=fixtures.event(1,'SERVE_CONTACT',1)
    directory=OUT/'duplicate_role';directory.mkdir(exist_ok=False)
    with PointAnalysisSequence(directory,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,calls=fixtures.drive(api,owner,[ev],source)
        replacement=fixtures.event(1,'SERVE_CONTACT',2)
        replacement.timestamp_s=.9
        replacement.confidence=.1
        error=None
        try:
            _,outcome,_=process_timeline_event(replacement,fixtures.binding_context(replacement,source,engine),
                engine,timeline,calls,ball_points=(),homography_matrix=None,frames=(),geometry_available=True)
            observed=outcome.outcome_type.value
        except ValueError as value:
            error=str(value);observed=None
        results['duplicate_id_with_changed_participant_timestamp_confidence']={
            'rejected':error is not None,'error':error,'returned_outcome':observed,
            'expected':'REJECT_CONFLICTING_EVENT_PAYLOAD_OR_REQUIRE_NEW_REVISION'}
    directory=OUT/'owner_context_mismatch';directory.mkdir(exist_ok=False)
    with PointAnalysisSequence(directory,'different-analysis',source_fingerprint='different-source') as owner:
        engine,timeline,calls=fixtures.drive(api,owner,[ev],source)
        transition=next(iter(owner.rows('transitions')))
        results['owner_context_analysis_and_source_mismatch']={
            'owner_analysis':owner.analysis_id,'binding_analysis':transition['context']['binding']['analysis_id'],
            'owner_source':owner.source_fingerprint,'binding_source':transition['context']['binding']['source_fingerprint'],
            'outcome':transition['outcome_type'],'attempt_count':len(owner.rows('attempts')),
            'expected':'REJECT_OR_WITHHOLD_CONTEXT_NOT_BOUND_TO_THE_OWNER'}
    engine=api.TennisScoringEngine()
    serve=fixtures.event(1,'SERVE_CONTACT',1)
    bounce=fixtures.event(3,'BOUNCE');bounce.frame_index=1;bounce.timestamp_s=.3
    context=fixtures.binding_context(bounce,source,engine)
    signal=ReviewedRuleSignal(2,1,.2,'SERVICE_LET','explicit-synthetic-earlier-let',context)
    merged=list(merge_rule_inputs([serve,bounce],[signal]))
    results['same_frame_reviewed_let_preceding_bounce']={
        'actual_event_ids':[item.event_id for item in merged],
        'declared_timestamps':[item.timestamp_s for item in merged],
        'expected':'PRESERVE_EXPLICIT_ORDER_OR_ABSTAIN_AMBIGUOUS_SAME_FRAME_SIGNALS'}
after = {str(path.relative_to(ROOT)):sha(path) for path in paths}
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'findings':results,
    'source_bindings_before':before,'source_bindings_after':after,
    'sources_changed_during_probe':[key for key in before if before[key]!=after[key]],
    'torch_imported':'torch' in sys.modules,'ultralytics_imported':'ultralytics' in sys.modules,
    'model_used':False,'media_used':False,'training_performed':False,'production_source_edited':False,
    'qualification_evidence':False,'script_sha256':sha(Path(__file__))}
with (OUT/'receipt.json').open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'receipt':str(OUT/'receipt.json'),'findings':results,
                  'sources_changed_during_probe':receipt['sources_changed_during_probe']}))

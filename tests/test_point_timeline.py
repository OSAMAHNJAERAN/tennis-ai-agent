"""Real rules, reducer, disk ledger and exporters; all evidence is synthetic."""
from __future__ import annotations
from dataclasses import replace
from contextlib import ExitStack
import ast
import hashlib
import os
import time
import gc
import tracemalloc
import weakref
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_scoring_service_phase import actual_classes
from src.scoring.point_timeline import (PointTimeline, ReviewedParticipantBinding, ReviewedRuleSignal,
    candidate_event_context, process_timeline_event, source_duration, context_record, merge_rule_inputs)
from src.scoring.point_timeline import native_match_analytics
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.shot_analysis.shot_types import (ShotEventEvidence, ShotType, ShotDirection,
    ShotClassificationSource, ShotOutcome)
from src.utils.analytics_json_artifact import write_membership_analytics
from src.utils.analytics_json_artifact import write_match_analytics, native_match_analytics_summary
from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.point_timeline_artifact import publish_point_timeline_manifest, POINT_TIMELINE_FILES
from src.scoring.rule_replay import write_rule_replay, RULE_REPLAY_NAME


@pytest.fixture
def api(monkeypatch):
    actual = actual_classes()
    for name in ('src.scoring.match_state', 'src.scoring.point_outcome_resolver', 'src.line_calling.line_call_engine'):
        module = ModuleType(name)
        module.__dict__.update(vars(actual))
        monkeypatch.setitem(sys.modules, name, module)
    return actual


def frames(count=32):
    pts = [0, 200, 390, 600, 917, 1200, 1347, 1887, 2100, 2350, 2500, 2800, 3100]
    return [{0: {'analysis_id':'fixture', 'source_pts': str(10**18 + (pts[i] if i < len(pts) else 3100+(i-12)*250)),
                 'source_time_base': '1/1000', 'source_frame_index': i, 'segment_id': 'camera-0',
                 'analysis_allowed': True, 'review_evidence_id': 'synthetic-normal-speed-review',
                 'measurement_transition_allowed': i > 0, 'anchor_frame_index': 0},
             1: {'analysis_id':'fixture', 'epoch': 0, 'revision': 0, 'observations': [
                 {'track_key': 'track-A', 'player_id': 'A', 'state': 'CONFIRMED'},
                 {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}} for i in range(count)]


def event(ident, kind, player=None):
    return SimpleNamespace(event_id=ident, frame_index=ident-1, timestamp_s=ident/10,
        event_type=SimpleNamespace(value=kind), player_id=player, confidence=.99,
        ball_position_px=None, court_position_m=None, trajectory_state='SYNTHETIC_ONLY',
        evidence='Explicit rule fixture; no video or accuracy reference')


def binding_context(ev, source, engine, *, swapped=False):
    role = ev.player_id or engine.state.server_id
    participant = 3-role if swapped else role
    identity = 'A' if participant == 1 else 'B'
    interval = source[ev.frame_index][0]
    binding = ReviewedParticipantBinding(ev.frame_index, interval['segment_id'], role, participant,
        identity, 'track-'+identity, 'NEAR_COURT' if participant == 1 else 'FAR_COURT',
        0, 'synthetic-qualified-calibration', 0, 'synthetic-end-proof', analysis_id='fixture',
        source_fingerprint='synthetic-source-v1', source_frame_index=interval['source_frame_index'],
        source_pts=interval['source_pts'], source_time_base=interval['source_time_base'], identity_epoch=0,
        participant_mapping_ref='synthetic-reviewed-participant-map', geometry_qualification_ref='synthetic-geometry-proof')
    return candidate_event_context(ev, source, lambda *args: binding,
        analysis_id='fixture', source_fingerprint='synthetic-source-v1')


class SuppliedCalls:
    def __init__(self, api, decisions):
        self.api, self.decisions, self.requests = api, decisions, []
    def evaluate_bounce(self, **kwargs):
        self.requests.append((kwargs['context'].value, getattr(kwargs['target_service_box'], 'value', None)))
        return self.api.LineCallEvidence(kwargs['event_id'], kwargs['bounce_frame'],
            self.api.LineCallDecision(self.decisions[kwargs['event_id']]), kwargs['context'],
            self.api.CourtLineType.NONE, (3., 6.), (3., 6.), 'DETECTED', 100., 'SYNTHETIC_ONLY',
            1., 100., 1., 'SYNTHETIC_ONLY', .99, 'Explicit rule fixture, not geometric GT', 'SYNTHETIC_ONLY')


def drive(api, owner, events, source, engine=None, decisions=None, contexts=None):
    engine = engine or api.TennisScoringEngine()
    timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
    caller = SuppliedCalls(api, decisions or {})
    for ev in events:
        context = contexts(ev, engine) if contexts else binding_context(ev, source, engine)
        process_timeline_event(ev, context, engine, timeline, caller,
            ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
    return engine, timeline, caller


def shot(ident, kind, player):
    return ShotEventEvidence(ident, ident, ident-1, ident/10, player,
        ShotType.SERVE if kind == 'SERVE_CONTACT' else ShotType.UNKNOWN, .9,
        ShotClassificationSource.EVENT_PASSTHROUGH, ShotDirection.UNKNOWN, 0., 'SYNTHETIC_ONLY')


def primary_events():
    return [event(i, kind, player) for i, kind, player in (
        (1, 'SERVE_CONTACT', 1), (2, 'BOUNCE', None), (3, 'PLAYER_2_HIT', 2), (4, 'BOUNCE', None),
        (5, 'SERVE_CONTACT', 1), (6, 'BOUNCE', None), (7, 'PLAYER_2_HIT', 2), (8, 'BOUNCE', None),
        (9, 'SERVE_CONTACT', 2), (10, 'BOUNCE', None), (11, 'BOUNCE', None),
        (12, 'SERVE_CONTACT', 2), (13, 'BOUNCE', None))]


def test_two_points_keep_prepoint_server_attempts_and_exact_terminal_clock(api, tmp_path):
    source = frames()
    engine = api.TennisScoringEngine()
    engine.state.points_p1 = 3
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint='synthetic-source-v1') as owner:
        engine, timeline, calls = drive(api, owner, primary_events(), source, engine,
            {2:'SERVE_FAULT', 4:'IN', 6:'SERVE_IN', 8:'OUT', 10:'SERVE_FAULT', 11:'IN', 13:'SERVE_FAULT'})
        shots = [shot(e.event_id, e.event_type.value, e.player_id) for e in primary_events() if e.player_id]
        timeline.attach_shots(shots)
        timeline.finish()
        points = list(owner.rows('points'))
        rallies = list(owner.rows('rallies'))
        attempts = list(owner.rows('attempts'))
        assert len(points) == 2 and [p['status'] for p in points] == ['COMPLETED', 'COMPLETED']
        assert [(p['server_id'],p['game_number'],p['winner_id']) for p in points] == [(1,1,1),(2,2,1)]
        assert [(a['server_id'],a['official_attempt'],a['status']) for a in attempts] == [
            (1,1,'FIRST_SERVE_FAULT'),(1,2,'SERVE_IN_PLAY'),(2,1,'FIRST_SERVE_FAULT'),(2,2,'DOUBLE_FAULT')]
        assert len(rallies) == 1 and list(rallies[0]['shot_ids']) == [5,7]
        assert rallies[0]['duration_s'] == .970 and rallies[0]['total_strokes_including_serve'] == 2
        assert [s.outcome for s in shots if s.match_event_id in (1,9,12)] == [ShotOutcome.FAULT]*3
        assert [row['event_id'] for row in owner.member_rows() if row['play_scope']=='LEGAL_RALLY'] == [5,7]
        assert calls.requests[0][0] == 'SERVE' and calls.requests[2][0] == 'SERVE'
        assert owner.frame_state(0, timeline.initial_state)['player_1']['games'] == 0
        assert owner.frame_state(7, timeline.initial_state)['player_1']['games'] == 1
        target = tmp_path/'points.json'
        write_membership_analytics(target,'points',owner.rows('points'),lambda value:value)
        assert len(json.loads(target.read_text())['points']) == 2
        view = owner.rows('points')
        reader = iter(view); next(reader)
    assert not list(tmp_path.glob('point_analysis_*.sqlite3*'))
    with pytest.raises(RuntimeError): len(view)


@pytest.mark.parametrize('second', [False,True])
def test_reviewed_service_let_is_same_slot_new_execution_and_not_live_member(api,tmp_path,second):
    source = frames()
    engine = api.TennisScoringEngine()
    events = [event(1,'SERVE_CONTACT',1),event(2,'BOUNCE')]
    decisions = {2:'SERVE_FAULT'} if second else {}
    if not second: events = []
    start = 3 if second else 1
    events.append(event(start,'SERVE_CONTACT',1))
    stub = event(start+1,'UNKNOWN')
    context = binding_context(stub,source,engine)
    events.append(ReviewedRuleSignal(start+1,start,start/10,'SERVICE_LET','synthetic-reviewed-let',context))
    events += [event(start+2,'SERVE_CONTACT',1),event(start+3,'BOUNCE'),
               event(start+4,'PLAYER_2_HIT',2),event(start+5,'BOUNCE')]
    decisions.update({start+3:'SERVE_IN',start+5:'OUT'})
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,_ = drive(api,owner,events,source,engine,decisions,
            contexts=lambda ev, eng: ev.context if isinstance(ev,ReviewedRuleSignal) else binding_context(ev,source,eng))
        shots=[shot(e.event_id,e.event_type.value,e.player_id) for e in events if not isinstance(e,ReviewedRuleSignal) and e.player_id]
        timeline.attach_shots(shots); timeline.finish()
        attempts=list(owner.rows('attempts'))
        replayed=attempts[1:] if second else attempts
        assert [(a['official_attempt'],a['replay_index'],a['status']) for a in replayed] == [
            (2 if second else 1,0,'SERVICE_LET'),(2 if second else 1,1,'SERVE_IN_PLAY')]
        assert len(owner.rows('points',status='COMPLETED'))==1
        assert [m['event_id'] for m in owner.member_rows() if m['play_scope']=='LEGAL_RALLY']==[start+2,start+4]


def test_reviewed_signal_after_closed_point_keeps_explicit_reference(api,tmp_path):
    source=frames(); engine=api.TennisScoringEngine()
    context=binding_context(event(5,'UNKNOWN'),source,engine)
    signal=ReviewedRuleSignal(5,4,.5,'SERVICE_LET','synthetic-post-point-review',context)
    events=[event(1,'SERVE_CONTACT',1),event(2,'BOUNCE'),event(3,'SERVE_CONTACT',1),event(4,'BOUNCE'),signal]
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,_=drive(api,owner,events,source,engine,{2:'SERVE_FAULT',4:'SERVE_FAULT'},
            contexts=lambda ev,eng: ev.context if isinstance(ev,ReviewedRuleSignal) else binding_context(ev,source,eng))
        timeline.finish()
        transition=owner.event(5)
        assert transition['outcome_type']=='DEAD_BALL_IGNORED'
        assert transition['point_uid'] is None
        assert transition['physical_event_type']=='SERVICE_LET'
        assert transition['reviewed_rule_evidence_ref']=='synthetic-post-point-review'


def test_unknown_identity_or_midclip_entry_cannot_award_point(api,tmp_path):
    source=frames()
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,calls=drive(api,owner,[event(1,'PLAYER_2_HIT',2),event(2,'BOUNCE')],source,
            decisions={},contexts=lambda ev,eng:candidate_event_context(ev,source))
        timeline.finish()
        assert engine.state.score_confirmed is False
        assert len(owner.rows('points',status='COMPLETED'))==0
        assert len(owner.rows('rallies'))==0 and calls.requests==[]


@pytest.mark.parametrize('mutation', ['cut','reversed','missing','unreviewed','reanchor'])
def test_every_intervening_source_edge_is_required(api,tmp_path,mutation):
    source=frames()
    original=source[2][0]
    if mutation=='cut': original['measurement_transition_allowed']=False
    if mutation=='reversed': original['source_pts']=source[1][0]['source_pts']
    if mutation=='missing': original['source_pts']=None
    if mutation=='unreviewed': original['review_evidence_id']=None
    if mutation=='reanchor': original['anchor_frame_index']=2
    start=context_record(binding_context(event(1,'SERVE_CONTACT',1),source,api.TennisScoringEngine()))
    end=context_record(binding_context(event(5,'BOUNCE'),source,api.TennisScoringEngine()))
    assert source_duration(start,end,source)[0] is None


def test_different_source_fingerprints_and_revisions_have_distinct_uids(api,tmp_path):
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='one') as a, PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='two') as b:
        assert PointTimeline(a,{}).uid('point','anchor') != PointTimeline(b,{}).uid('point','anchor')


def test_unordered_review_signals_fail_instead_of_becoming_detected_lets(api):
    source=frames(); engine=api.TennisScoringEngine()
    ev=event(3,'UNKNOWN'); context=binding_context(ev,source,engine)
    signal=ReviewedRuleSignal(20,2,.3,'SERVICE_LET','explicit-reviewed-fixture',context)
    with pytest.raises(ValueError):
        list(merge_rule_inputs([event(5,'BOUNCE'),event(1,'BOUNCE')],[signal]))


@pytest.mark.parametrize('authority',[True,False])
def test_exact_native_phase6_scoring_aggregation_and_five_exports(api,tmp_path,authority):
    _native_phase6_export_fixture(api,tmp_path,authority)


def test_exact_native_phase6_reviewed_service_let_exports(api,tmp_path):
    _native_phase6_export_fixture(api,tmp_path,True,reviewed_let=True)


def _native_phase6_export_fixture(api,tmp_path,authority,*,reviewed_let=False):
    path=Path(__file__).resolve().parents[1]/'src/pipeline/phase6_pipeline.py'
    source=path.read_text()
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Phase6Pipeline')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_with_frame_exports')
    nodes=run.body
    start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.AnnAssign) and getattr(n.target,'id',None)=='line_call_evidences')
    stop=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='authoritative_events' for t in n.targets))
    step10start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='final_st' for t in n.targets))
    step10end=next(i for i,n in enumerate(nodes[step10start:],step10start) if isinstance(n,ast.Expr) and 'Generated' in ast.unparse(n))
    source_file=tmp_path/'synthetic_source_identity.bin'
    source_file.write_bytes(b'Synthetic source identity fixture, not a video or accuracy reference.')
    source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest()
    initial=api.MatchState(match_id='fixture',points_p1=3)
    engine=api.TennisScoringEngine(initial_state=initial)
    intervals=frames()
    inputs=primary_events()
    caller=SuppliedCalls(api,{2:'SERVE_FAULT',4:'IN',6:'SERVE_IN',8:'OUT',10:'SERVE_FAULT',11:'IN',13:'SERVE_FAULT'})
    reviewed_signals=()
    if reviewed_let:
        inputs=[event(1,'SERVE_CONTACT',1),event(3,'SERVE_CONTACT',1),event(4,'BOUNCE'),
                event(5,'PLAYER_2_HIT',2),event(6,'BOUNCE')]
        caller=SuppliedCalls(api,{4:'SERVE_IN',6:'OUT'})
        context=binding_context(event(2,'UNKNOWN'),intervals,engine)
        context=replace(context,binding=replace(context.binding,source_fingerprint=source_hash))
        reviewed_signals=(ReviewedRuleSignal(2,1,.2,'SERVICE_LET','synthetic-reviewed-let',context),)
    def provider(ev,*args):
        return replace(binding_context(ev,intervals,namespace['self'].scoring_engine).binding,
                       source_fingerprint=source_hash)
    class Shots(list):
        def seal(self): self.sealed=True
    shots=Shots(shot(e.event_id,e.event_type.value,e.player_id) for e in inputs if e.player_id) if authority else Shots()
    namespace=dict(vars(api),PointAnalysisSequence=PointAnalysisSequence,PointTimeline=PointTimeline,
        ReviewedRuleSignal=ReviewedRuleSignal,candidate_event_context=candidate_event_context,
        merge_rule_inputs=merge_rule_inputs,process_timeline_event=process_timeline_event,
        native_match_analytics=native_match_analytics,hashlib=hashlib,time=time,os=os,
        publish_point_timeline_manifest=publish_point_timeline_manifest,
        write_rule_replay=write_rule_replay,RULE_REPLAY_NAME=RULE_REPLAY_NAME,
        disk_players=True,authoritative_enabled=authority,input_video_path=str(source_file),
        output_dir=str(tmp_path),candidate_frames=intervals,frames=(),ball_points=(),homography_matrix=None,
        is_h_valid=True,detected_events=inputs if authority else [],
        event_output=SimpleNamespace(events_by_frame=lambda:iter(inputs)),
        self=SimpleNamespace(scoring_engine=engine,line_call_engine=caller,
            reviewed_participant_binding_provider=provider,reviewed_rule_signals=reviewed_signals,
            config={'match_configuration':{'match_id':'fixture'}}),
        t0=time.time(),shot_evidences=shots,shot_output=shots,p1_motion={'distance_estimator':'SYNTHETIC_ONLY'},
        _sanitize=lambda x:x,write_membership_analytics=write_membership_analytics,
        write_match_analytics=write_match_analytics,write_json_array_artifact=write_json_array_artifact)
    with ExitStack() as stack:
        namespace['export_stack']=stack
        physical_export = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='event_export_records')
        namespace['sanitize_json_value'] = lambda value:value
        exec(compile(ast.Module(body=[physical_export],type_ignores=[]),str(path),'exec'),namespace)
        exec(compile(ast.Module(body=nodes[start:stop+1],type_ignores=[]),str(path),'exec'),namespace)
        exec(compile(ast.Module(body=nodes[step10start:step10end],type_ignores=[]),str(path),'exec'),namespace)
        exports=[n for n in nodes if isinstance(n,(ast.Expr,ast.If)) and any(name in ast.unparse(n) for name in
            ("'rallies.json'","'point_analytics.json'","'match_analytics.json'","'serve_executions.json'","'scoring_events.json'","'match_events.json'"))]
        assert len(exports)==6
        exec(compile(ast.Module(body=exports,type_ignores=[]),str(path),'exec'),namespace)
        history_exports = [n for n in nodes if isinstance(n,(ast.With,ast.Expr)) and
            ('"match_state.json"' in ast.unparse(n) or "'match_state.json'" in ast.unparse(n)
             or 'self.scoring_engine.save_history_json' in ast.unparse(n))]
        assert len(history_exports) == 2
        exec(compile(ast.Module(body=history_exports,type_ignores=[]),str(path),'exec'),namespace)
        publication = [n for n in nodes if isinstance(n,ast.If) and
                       'publish_point_timeline_manifest(' in ast.unparse(n)]
        assert len(publication) == 1
        exec(compile(ast.Module(body=publication,type_ignores=[]),str(path),'exec'),namespace)
        manifest = json.loads((tmp_path/'point_timeline_manifest.json').read_text())
        replay = json.loads((tmp_path/RULE_REPLAY_NAME).read_text())
        assert replay['authoritative_events_enabled'] is authority
        assert replay['source_frame_count'] == len(intervals)
        assert len(replay['source_frames']) == len(intervals)
        assert replay['input_count'] == len(replay['inputs']) == ((6 if reviewed_let else len(inputs)) if authority else 0)
        assert replay['source_sha256'] == source_hash
        assert replay['initial_state']['fields']['points_p1'] == 3
        assert replay['independent_accuracy_validated'] is False
        physical = json.loads((tmp_path/'match_events.json').read_text())
        assert physical['events_count'] == len(physical['events']) == (len(inputs) if authority else 0)
        assert [row['event_id'] for row in physical['events']] == ([e.event_id for e in inputs] if authority else [])
        assert manifest['analysis_id'] == 'fixture' and manifest['analysis_revision'] == 0
        assert manifest['source_sha256'] == source_hash and manifest['scoring_revision'] == 0
        assert manifest['authoritative_events_enabled'] is authority
        assert manifest['independent_accuracy_validated'] is False
        assert set(manifest['artifacts']) == set(POINT_TIMELINE_FILES)
        for name, binding in manifest['artifacts'].items():
            assert binding['sha256'] == hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
        points=json.loads((tmp_path/'point_analytics.json').read_text())
        match=json.loads((tmp_path/'match_analytics.json').read_text())
        assert points['schema_version']=='2.0'
        assert 'SCORING_HISTORY_INPUTS_AND_CORRECTION_HISTORY' not in match['remaining_capacity_limits']
        assert 'WHOLE_ANALYSIS_CORRECTION_PUBLICATION' in match['remaining_capacity_limits']
        assert len(points['points'])==((1 if reviewed_let else 2) if authority else 0)
        assert match['overview']['total_points_played']==((1 if reviewed_let else 2) if authority else None)
        if reviewed_let:
            attempts=json.loads((tmp_path/'serve_executions.json').read_text())['serve_executions']
            assert [(a['official_attempt'],a['replay_index'],a['status']) for a in attempts]==[
                (1,0,'SERVICE_LET'),(1,1,'SERVE_IN_PLAY')]
            transitions=json.loads((tmp_path/'scoring_events.json').read_text())['scoring_events']
            reviewed=[row for row in transitions if row['reviewed_rule_evidence_ref'] is not None]
            assert len(transitions)==6 and len(reviewed)==1
            assert reviewed[0]['event_id']==2 and reviewed[0]['physical_event_type']=='SERVICE_LET'
            assert reviewed[0]['reviewed_rule_evidence_ref']=='synthetic-reviewed-let'
            assert match['overview']['effective_play_seconds']==.810
        elif authority:
            assert match['serve_analytics']['by_player']['2']['double_faults']==1
            assert match['overview']['effective_play_seconds']==.970
        else:
            assert caller.requests==[] and match['score_summary'] is None
        summary=native_match_analytics_summary(namespace['match_analytics'])
    json.dumps(summary,allow_nan=False)
    assert namespace['self'].scoring_engine is engine
    assert not list(tmp_path.glob('point_analysis_*.sqlite3*'))
    assert not list(tmp_path.glob('scoring_journal_*.sqlite3*'))


def test_swapped_physical_roles_translate_scoring_tokens_and_unknown_serve_never_defaults(api,tmp_path):
    source=frames(); engine=api.TennisScoringEngine(initial_server_id=2)
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        timeline=PointTimeline(owner,engine.state.to_dict(),source_frames=source)
        ev=event(1,'SERVE_CONTACT',1)
        context=binding_context(ev,source,engine,swapped=True)
        assert timeline.scoring_input(ev,context,engine.state.to_dict())==('SERVE_CONTACT',2)
        returned=event(2,'PLAYER_2_HIT',2)
        assert timeline.scoring_input(returned,binding_context(returned,source,engine,swapped=True),engine.state.to_dict())==('PLAYER_1_HIT',1)
        assert timeline.scoring_input(event(1,'SERVE_CONTACT'),context,engine.state.to_dict())==('CONSEQUENTIAL_GAP',None)


@pytest.mark.parametrize('field,value',[('source_pts','1'),('source_fingerprint','other'),
    ('identity_revision',1),('anchor_frame_index',1),('geometry_qualification_ref','')])
def test_stale_reviewed_identity_binding_abstains(api,field,value):
    source=frames(); engine=api.TennisScoringEngine(); ev=event(2,'SERVE_CONTACT',1)
    binding=replace(binding_context(ev,source,engine).binding,**{field:value})
    context=candidate_event_context(ev,source,lambda *args:binding,analysis_id='fixture',source_fingerprint='synthetic-source-v1')
    assert not context.binding_current


def test_duplicate_transition_is_idempotent_but_conflicting_source_id_fails(api,tmp_path):
    source=frames(); ev=event(1,'SERVE_CONTACT',1)
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,caller=drive(api,owner,[ev],source)
        version=engine.state.state_version
        context=binding_context(ev,source,engine)
        process_timeline_event(ev,context,engine,timeline,caller,ball_points=(),homography_matrix=None,frames=(),geometry_available=True)
        assert engine.state.state_version==version and len(owner.rows('transitions'))==1
        ev.frame_index=1
        with pytest.raises(ValueError,match='different source evidence'):
            process_timeline_event(ev,binding_context(ev,source,engine),engine,timeline,caller,
                ball_points=(),homography_matrix=None,frames=(),geometry_available=True)


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_export_failure_and_cancel_preserve_published_file_and_close_readers(api,tmp_path,failure):
    destination=tmp_path/'published.json'; destination.write_bytes(b'preserved revision')
    with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
        engine,timeline,_=drive(api,owner,[event(1,'SERVE_CONTACT',1)],frames())
        timeline.attach_shots([shot(1,'SERVE_CONTACT',1)]); timeline.finish()
        def interrupt(value):
            if isinstance(value,int): raise failure('explicit synthetic export interruption')
            return value
        with pytest.raises(failure):
            write_membership_analytics(destination,'points',owner.rows('points'),interrupt)
        assert destination.read_bytes()==b'preserved revision'
        assert owner.readers==set() and not list(tmp_path.glob('.published.json.*.tmp'))
        with pytest.raises(ValueError,match='sealed'):
            owner.put('points','new',{})
    assert not list(tmp_path.glob('point_analysis_*.sqlite3*'))


@pytest.mark.parametrize('count',[257,1025,4097])
def test_indexed_semantic_ledger_scales_completed_points_without_live_membership_arrays(api,tmp_path,count):
    # This measures the new semantic owner/reducer, not the deferred Engine
    # history, media inference or end-to-end capacity. Independent typed rule
    # fixtures repeat a completed point; they are not a long-match validation.
    class SourceFrames:
        def __getitem__(self,index):
            if index<0 or index>=count*4: raise IndexError(index)
            return {0:dict(analysis_id='fixture',source_pts=str(10**18+index*250),source_time_base='1/1000',
                source_frame_index=index,segment_id='camera-0',analysis_allowed=True,
                review_evidence_id='synthetic-normal-speed-review',measurement_transition_allowed=index>0,anchor_frame_index=0),
                1:dict(analysis_id='fixture',epoch=0,revision=0,observations=[
                    dict(track_key='track-A',player_id='A',state='CONFIRMED'),
                    dict(track_key='track-B',player_id='B',state='CONFIRMED')])}
    class LazyShots(Sequence):
        def __init__(self): self.live=weakref.WeakSet(); self.peak=0
        def __len__(self): return count*2
        def __getitem__(self,index):
            if not 0<=index<len(self): raise IndexError(index)
            point,returning=divmod(index,2)
            ident=point*4+(3 if returning else 1)
            value=shot(ident,'PLAYER_2_HIT' if returning else 'SERVE_CONTACT',2 if returning else 1)
            key=id(value); ref=weakref.ref(value,lambda ref:self.refs.pop(key,None))
            self.refs[key]=ref; self.peak=max(self.peak,len(self.refs))
            return value
        def __setitem__(self,index,value): pass  # No changed physical fields in this fixture.
        refs={}
    source=SourceFrames(); shots=LazyShots(); shots.refs={}
    engine=api.TennisScoringEngine()
    pre=engine.state.to_dict()
    stages=[('SERVE_CONTACT',1,'SERVE_IN_PLAY',False,None),('BOUNCE',None,'SERVE_IN_PLAY',False,None),
            ('PLAYER_2_HIT',2,'RALLY_CONTINUES',False,None),('BOUNCE',None,'POINT_WON_PLAYER_1',True,1)]
    tracemalloc.start()
    try:
        with PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as owner:
            timeline=PointTimeline(owner,pre,source_frames=source)
            for point in range(count):
                for offset,(kind,player,result,ending,winner) in enumerate(stages):
                    ident=point*4+offset+1; ev=event(ident,kind,player)
                    context=binding_context(ev,source,engine)
                    outcome=api.PointOutcome(api.PointOutcomeType(result),winner,'SYNTHETIC_ONLY',ending,.99,ident)
                    timeline.observe(ev,context,pre,outcome,pre,rule_type=kind,rule_player=player)
            timeline.attach_shots(shots); timeline.finish()
            assert len(owner.rows('points',status='COMPLETED'))==count
            assert len(owner.rows('rallies',status='COMPLETED'))==count
            analytics=native_match_analytics(owner,shots,engine.state,authoritative=True)
            write_membership_analytics(tmp_path/'points.json','points',owner.rows('points'),lambda x:x,schema_version='2.0')
            write_match_analytics(tmp_path/'match.json',analytics,lambda x:x)
            assert shots.peak<=3
            assert not isinstance(analytics['ball_landing_heatmap'],list)
            assert owner.readers==set()
            assert owner.pending==0
            current,peak=tracemalloc.get_traced_memory()
            (tmp_path/'capacity.json').write_text(json.dumps(dict(completed_points=count,source_events=count*4,
                shot_memberships=count*2,peak_live_shots=shots.peak,traced_current_bytes=current,traced_peak_bytes=peak,
                excludes_scoring_history_and_inference=True)))
            assert peak<4*1024*1024
    finally:
        tracemalloc.stop()
    gc.collect()
    assert not shots.refs


def test_corrected_rule_revision_requires_new_derived_timeline_even_for_a_retry(api, tmp_path):
    from src.utils.scoring_journal import ScoringJournal
    source = frames()
    with ScoringJournal(tmp_path) as rules, PointAnalysisSequence(
            tmp_path, 'fixture', source_fingerprint='synthetic-source-v1') as ledger:
        engine = api.TennisScoringEngine(journal=rules.revision())
        timeline = PointTimeline(ledger, engine.state.to_dict(), source_frames=source)
        ev = event(1, 'SERVE_CONTACT', 1)
        caller = SuppliedCalls(api, {})
        context = binding_context(ev, source, engine)
        process_timeline_event(ev, context, engine, timeline, caller,
            ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        engine.correct_event(1, reason='Explicit synthetic confidence review', expected_revision=0, confidence=.8)
        with pytest.raises(ValueError, match='rebuilt analysis timeline'):
            process_timeline_event(ev, context, engine, timeline, caller,
                ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert len(ledger.rows('transitions')) == 1 and engine.revision == 1

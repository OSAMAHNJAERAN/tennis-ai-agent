"""Actual typed event/shot SQLite storage joined to the rule timeline.

NumPy is imported by the existing shot codec. No model, image decode or video
is used; supplied event/line decisions are explicit synthetic rule fixtures.
"""
import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
import sys
from typing import Any, Dict, Optional, Tuple

from tests.test_point_timeline import api, frames, binding_context, SuppliedCalls, primary_events, shot
from src.scoring.point_timeline import PointTimeline, process_timeline_event, native_match_analytics
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.shot_evidence_sequence import ShotEvidenceSequence
from src.utils.analytics_json_artifact import write_match_analytics


def test_actual_event_and_shot_owners_keep_semantic_memberships_and_cleanup(api,monkeypatch,tmp_path):
    module=ModuleType('src.events.event_detector')
    module.__dict__.update(Enum=Enum,dataclass=dataclass,field=field,Any=Any,Dict=Dict,Optional=Optional,Tuple=Tuple)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    path=Path(__file__).resolve().parents[1]/'src/events/event_detector.py'
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef)
           and node.name in ('EventType','TennisEvent','EventCandidate')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
    source=frames(); engine=api.TennisScoringEngine(); engine.state.points_p1=3
    caller=SuppliedCalls(api,{2:'SERVE_FAULT',4:'IN',6:'SERVE_IN',8:'OUT',10:'SERVE_FAULT',11:'IN',13:'SERVE_FAULT'})
    with EventAnalysisSequence(tmp_path) as event_owner, ShotEvidenceSequence(tmp_path) as shots, PointAnalysisSequence(tmp_path,'fixture',source_fingerprint='synthetic-source-v1') as ledger:
        for value in primary_events():
            event_owner.events.append(module.TennisEvent(value.event_id,module.EventType(value.event_type.value),
                value.frame_index,value.timestamp_s,value.player_id,(100.,200.),None,.99,'DETECTED'))
            if value.player_id:
                shots.append(shot(value.event_id,value.event_type.value,value.player_id))
        timeline=PointTimeline(ledger,engine.state.to_dict(),source_frames=source)
        for value in event_owner.events_by_frame():
            process_timeline_event(value,binding_context(value,source,engine),engine,timeline,caller,
                ball_points=(),homography_matrix=None,frames=(),geometry_available=True)
        timeline.attach_shots(shots); shots.seal(); timeline.finish()
        assert [row['event_id'] for row in ledger.member_rows() if row['play_scope']=='LEGAL_RALLY']==[5,7]
        assert len(ledger.rows('points',status='COMPLETED'))==2
        analytics=native_match_analytics(ledger,shots,engine.state,authoritative=True)
        write_match_analytics(tmp_path/'match.json',analytics,lambda x:x)
        assert analytics['serve_analytics']['by_player']['2']['double_faults']==1
        assert ledger.readers==set() and shots._readers==set()
    assert not list(tmp_path.glob('point_analysis_*.sqlite3*'))
    assert not list(tmp_path.glob('shot_evidence_*.sqlite3*'))
    assert not list(tmp_path.glob('event_analysis_*.sqlite3*'))

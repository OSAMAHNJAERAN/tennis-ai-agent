"""Read-only semantic RED: actual scoring, linker and Phase6 source blocks.

All contacts, decisions, identity/end confirmations and timing are synthetic
fixtures. No geometric accuracy, event accuracy or production authority follows.
"""
import ast
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'ultralytics'):
            raise RuntimeError('Read-only semantic probe refuses model runtime: ' + fullname)


sys.meta_path.insert(0, RefuseModels())
from src.scoring.scoring_engine import TennisScoringEngine
from src.scoring.point_outcome_resolver import PointOutcomeType
from src.scoring.match_state import MatchState, BallPlayState, ServerCourtEnd
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision, LineCallContext
from src.line_calling.line_geometry import CourtLineType
from src.events.event_detector import TennisEvent, EventType
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.shot_evidence_sequence import ShotEvidenceSequence


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


capacity = load('actual_point_capacity_helpers', ROOT / 'tests/test_analytics_capacity_seam.py')
linking = load('actual_point_linker_helpers', ROOT / 'tests/test_shot_evidence_sequence.py')
BASE = Path(__file__).resolve().parent
PATHS = [ROOT / path for path in (
    'src/pipeline/phase6_pipeline.py', 'src/scoring/scoring_engine.py',
    'src/scoring/point_outcome_resolver.py', 'src/scoring/scoring_rules.py', 'src/scoring/event_history.py',
    'src/scoring/match_state.py', 'src/analytics/rally_analyzer.py', 'src/analytics/serve_analyzer.py',
    'src/analytics/match_analytics.py', 'src/analytics/shot_statistics.py',
    'src/shot_analysis/shot_linker.py', 'src/shot_analysis/shot_types.py',
    'src/events/event_detector.py', 'src/utils/shot_evidence_sequence.py')]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuleSignal(Enum):
    SERVICE_LET = 'SERVICE_LET'
    CONSEQUENTIAL_GAP = 'CONSEQUENTIAL_GAP'


ORIGIN = 10**18


def event(ident, frame, milliseconds, kind, player=None):
    if isinstance(kind, str):
        kind = EventType(kind)
    return TennisEvent(ident, kind, frame, float(Fraction(milliseconds, 1000)), player,
        (500., 400.), (3., 6.) if player is None else (3., 19.), .99, 'DETECTED',
        evidence={'fixture_only': True, 'source_pts': str(ORIGIN+milliseconds),
            'source_time_base': '1/1000', 'source_frame_index': frame,
            'camera_segment_id': 'synthetic-eligible-camera-0',
            'measurement_transition_allowed': True})


def line_call(ev, decision, context):
    return LineCallEvidence(ev.event_id, ev.frame_index, LineCallDecision(decision), LineCallContext(context),
        CourtLineType.NEAR_SERVICE_LINE, ev.ball_position_px, ev.court_position_m, 'DETECTED',
        20., 'SYNTHETIC_PATCH', 1., 19., 0., 'SYNTHETIC_ONLY', .99,
        'EXPLICIT_SYNTHETIC_VERIFIED_RULE_FIXTURE_NOT_GEOMETRIC_GT', 'SYNTHETIC_ONLY')


class ObservedScoringEngine(TennisScoringEngine):
    def __init__(self, state):
        super().__init__(initial_state=state)
        self.observed_transitions = []

    def process_match_event(self, **kwargs):
        previous = copy.deepcopy(self.get_current_state().to_dict())
        resolver_before = {'ball_play_state': self.resolver.ball_play_state.value,
                           'serve_attempt': self.resolver.serve_attempt,
                           'current_server_id': self.resolver.current_server_id}
        state, outcome = super().process_match_event(**kwargs)
        self.observed_transitions.append({'event_id': kwargs['event_id'], 'event_type': kwargs['event_type'],
            'previous_state': previous, 'new_state': copy.deepcopy(state.to_dict()),
            'resolver_before': resolver_before, 'resolver_after': {
                'ball_play_state': self.resolver.ball_play_state.value,
                'serve_attempt': self.resolver.serve_attempt,
                'current_server_id': self.resolver.current_server_id},
            'outcome': outcome.to_dict()})
        return state, outcome


def actual_step7():
    path = ROOT / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.AnnAssign)
                 and isinstance(node.target, ast.Name) and node.target.id == 'line_call_evidences')
    end = next(i for i, node in enumerate(run.body[start:], start)
               if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'ev')
    block = run.body[start:end+1]
    bounce = next(node for node in block[-1].body if isinstance(node, ast.If)
                  and 'EventType.BOUNCE' in ast.unparse(node.test))
    end_guess = [node for node in bounce.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in ('server_end', 'expected_box')
                         for target in node.targets)]
    assert len(end_guess) == 2
    return path, compile(ast.Module(body=block, type_ignores=[]), str(path), 'exec'), end_guess


def cases():
    ordinary = [
        event(1, 0, 0, 'SERVE_CONTACT', 1), event(2, 4, 181, 'BOUNCE'),
        event(3, 5, 225, 'PLAYER_2_HIT', 2), event(4, 6, 280, 'BOUNCE'),
        event(5, 20, 917, 'SERVE_CONTACT', 1), event(6, 24, 1103, 'BOUNCE'),
        event(7, 29, 1349, 'PLAYER_2_HIT', 2), event(8, 40, 1887, 'BOUNCE'),
        event(9, 185, 8501, 'SERVE_CONTACT', 2), event(10, 194, 8905, 'BOUNCE'),
        event(11, 195, 8981, 'BOUNCE'), event(12, 200, 9207, 'SERVE_CONTACT', 2),
        event(13, 209, 9619, 'BOUNCE')]
    yield {'name': 'two_completed_points_first_fault_second_in_game_change_then_double_fault',
        'events': ordinary, 'state': MatchState(match_id='synthetic-two-point', points_p1=3, points_p2=1),
        'decisions': {2: ('SERVE_FAULT', 'SERVE'), 4: ('IN', 'RALLY'), 6: ('SERVE_IN', 'SERVE'),
                      8: ('OUT', 'RALLY'), 10: ('SERVE_FAULT', 'SERVE'), 11: ('IN', 'RALLY'),
                      13: ('SERVE_FAULT', 'SERVE')},
        'expected': {'completed_points': 2, 'qualified_live_rallies': 1,
            'points': [
                {'anchor_event_id': 1, 'terminal_event_id': 8, 'winner_id': 1, 'server_id': 1,
                 'set_number': 1, 'game_number': 1, 'live_rally_contact_event_ids': [5, 7],
                 'attempts': [{'contact_event_id': 1, 'official_attempt': 1, 'replay_index': 0,
                               'result_event_id': 2, 'result': 'FIRST_SERVE_FAULT'},
                              {'contact_event_id': 5, 'official_attempt': 2, 'replay_index': 0,
                               'result_event_id': 6, 'result': 'SERVE_IN_PLAY'}],
                 'live_rally_source_elapsed_s': .970},
                {'anchor_event_id': 9, 'terminal_event_id': 13, 'winner_id': 1, 'server_id': 2,
                 'set_number': 1, 'game_number': 2, 'live_rally_contact_event_ids': [],
                 'attempts': [{'contact_event_id': 9, 'official_attempt': 1, 'replay_index': 0,
                               'result_event_id': 10, 'result': 'FIRST_SERVE_FAULT'},
                              {'contact_event_id': 12, 'official_attempt': 2, 'replay_index': 0,
                               'result_event_id': 13, 'result': 'DOUBLE_FAULT'}],
                 'live_rally_source_elapsed_s': None}],
            'dead_post_fault_event_ids': [3, 4, 11],
            'per_server_serves': {'1': {'first_attempts': 1, 'second_attempts': 1, 'first_faults': 1,
                                       'second_in': 1, 'double_faults': 0},
                                  '2': {'first_attempts': 1, 'second_attempts': 1, 'first_faults': 1,
                                       'second_in': 0, 'double_faults': 1}}}}
    yield {'name': 'let_then_replayed_first_serve_then_completed_point',
        'events': [event(1, 0, 0, 'SERVE_CONTACT', 1), event(2, 1, 23, RuleSignal.SERVICE_LET),
                   event(3, 2, 71, 'BOUNCE'), event(4, 5, 311, 'SERVE_CONTACT', 1),
                   event(5, 8, 483, 'BOUNCE'), event(6, 11, 743, 'PLAYER_2_HIT', 2),
                   event(7, 15, 991, 'BOUNCE')], 'state': MatchState(match_id='synthetic-let'),
        'decisions': {3: ('IN', 'RALLY'), 5: ('SERVE_IN', 'SERVE'), 7: ('OUT', 'RALLY')},
        'expected': {'completed_points': 1, 'qualified_live_rallies': 1,
            'point_anchor_event_id': 1, 'live_rally_contact_event_ids': [4, 6],
            'official_first_attempts': 1, 'first_serve_contact_executions': 2, 'service_lets': 1,
            'attempts': [{'contact_event_id': 1, 'official_attempt': 1, 'replay_index': 0,
                          'result_event_id': 2, 'result': 'SERVICE_LET'},
                         {'contact_event_id': 4, 'official_attempt': 1, 'replay_index': 1,
                          'result_event_id': 5, 'result': 'SERVE_IN_PLAY'}],
            'nonplay_post_let_event_ids': [3], 'live_rally_source_elapsed_s': .680}}
    yield {'name': 'consequential_gap_then_review_pause_no_fabricated_new_point',
        'events': [event(1, 0, 0, 'SERVE_CONTACT', 1), event(2, 5, 201, 'BOUNCE'),
                   event(3, 11, 677, 'PLAYER_2_HIT', 2), event(4, 12, 701, RuleSignal.CONSEQUENTIAL_GAP),
                   event(5, 19, 1193, 'BOUNCE'), event(6, 150, 20001, 'SERVE_CONTACT', 1),
                   event(7, 160, 20499, 'BOUNCE')], 'state': MatchState(match_id='synthetic-gap'),
        'decisions': {2: ('SERVE_IN', 'SERVE'), 5: ('OUT', 'RALLY'), 7: ('SERVE_FAULT', 'SERVE')},
        'expected': {'completed_points': 0, 'point_status': 'REVIEW_REQUIRED', 'winner_id': None,
            'point_anchor_event_id': 1, 'eligible_pre_gap_contact_event_ids': [1, 3],
            'review_signal_event_id': 4, 'post_pause_event_ids_without_new_point_authority': [5, 6, 7],
            'duration_s': None, 'duration_reason': 'CONSEQUENTIAL_GAP_PREVENTS_COMPLETE_RALLY_TIMING'}}
    yield {'name': 'midclip_without_initial_serve_or_confirmed_initial_score',
        'events': [event(1, 0, 0, 'PLAYER_1_HIT', 1), event(2, 8, 710, 'BOUNCE'),
                   event(3, 12, 1230, 'PLAYER_2_HIT', 2), event(4, 14, 1380, 'BOUNCE')],
        'state': MatchState(match_id='synthetic-partial', initial_state_confirmed=False,
                            score_confirmed=False, score_source='UNAVAILABLE'),
        'decisions': {2: ('IN', 'RALLY'), 4: ('OUT', 'RALLY')},
        'expected': {'completed_points': 0, 'point_status': 'PARTIAL_UNANCHORED_REVIEW_REQUIRED',
            'point_anchor_event_id': None, 'winner_id': None, 'serve_attempt': None,
            'partial_contact_event_ids': [1, 3], 'duration_s': None}}


def run_case(fixture, *, authority=True):
    directory = BASE / (fixture['name'] + ('' if authority else '_authority_off'))
    directory.mkdir(exist_ok=False)
    engine = ObservedScoringEngine(fixture['state'])
    events = fixture['events']
    requested_calls = []
    decisions = {ev.event_id: line_call(ev, *fixture['decisions'][ev.event_id])
                 for ev in events if ev.event_id in fixture['decisions']}
    class FixtureLineCalls:
        def evaluate_bounce(self, **kwargs):
            supplied = decisions[kwargs['event_id']]
            state = copy.deepcopy(engine.get_current_state().to_dict())
            requested_calls.append({'event_id': kwargs['event_id'], 'requested_context': kwargs['context'].value,
                'requested_target_service_box': kwargs['target_service_box'].value if kwargs['target_service_box'] else None,
                'supplied_fixture_context': supplied.decision_context.value,
                'supplied_fixture_decision': supplied.decision.value,
                'pre_event_public_state': state,
                'pre_event_resolver_state': engine.resolver.ball_play_state.value,
                'fixture_configured_expected_box': engine.get_current_state().get_expected_service_box().value,
                'identity_end_mapping_is_fixture_only': True})
            return supplied
    path, step7, end_guess = actual_step7()
    all_native_types = all(isinstance(ev.event_type, EventType) for ev in events)
    with EventAnalysisSequence(directory) as event_owner, ShotEvidenceSequence(directory) as shot_owner:
        if all_native_types:
            for ev in events:
                event_owner.events.append(ev)
        native = all_native_types
        namespace = dict(globals(), self=SimpleNamespace(scoring_engine=engine, line_call_engine=FixtureLineCalls()),
            disk_players=native, authoritative_enabled=authority, event_output=event_owner,
            detected_events=(event_owner.events if native else events) if authority else [],
            is_h_valid=True, ball_points=[], homography_matrix=None, frames=[])
        exec(step7, namespace)
        Linker, Classifier = linking.actual_classes()
        linked = Linker(Classifier()).link_shots(
            namespace['detected_events'], [], [], [], dead_event_ids=namespace['dead_event_ids'],
            **({'event_output': event_owner if authority else None, 'result_output': shot_owner} if native else {}))
        if not native:
            for value in linked:
                shot_owner.append(value)
        # Exact current Phase6 step10 + exports + return, same final real engine state.
        classes = capacity.consumers()
        step_path, statements, exports, returned, sanitize = capacity.phase6_parts()
        namespace.update(classes, self=SimpleNamespace(scoring_engine=engine), shot_evidences=shot_owner,
            shot_output=shot_owner, disk_players=True, authoritative_enabled=authority, fps=25.,
            p1_dist=None, p2_dist=None, p1_speed=None, p2_speed=None, p1_positions=[], p2_positions=[],
            speed_summary={'available': False}, p1_motion={'distance_estimator': 'SYNTHETIC_ONLY'},
            output_dir=str(directory), fps_proc=0., t_total=0.)
        exec(compile(ast.Module(body=[sanitize]+statements, type_ignores=[]), str(step_path), 'exec'), namespace)
        namespace['_sanitize'] = namespace['sanitize_json_value']
        exec(compile(ast.Module(body=exports, type_ignores=[]), str(step_path), 'exec'), namespace)
        shots = [value.to_dict() for value in shot_owner]
        rallies = capacity.normalize(namespace['rallies'])
        points = capacity.normalize(namespace['point_analytics'])
        summary = copy.deepcopy(namespace['match_analytics']['overview'])
        serve_stats = {str(player): classes['ServeAnalyzer'].compute_serve_statistics(
            shot_owner, namespace['line_call_evidences'], server_id=player) for player in (1, 2)}
        # Evaluate the exact current hard-coded end assignments against an
        # explicitly confirmed swapped-end fixture; no production end is inferred.
        swapped = MatchState(server_id=2, receiver_id=1, initial_server_id=1,
                             player_1_court_end=ServerCourtEnd.FAR_COURT)
        guess_namespace = dict(globals(), curr_st=swapped)
        exec(compile(ast.Module(body=end_guess, type_ignores=[]), str(path), 'exec'), guess_namespace)
        end_case = {'server_id': 2, 'fixture_confirmed_player_1_end': 'FAR_COURT',
            'fixture_confirmed_server_end': 'NEAR_COURT',
            'actual_hard_coded_server_end': guess_namespace['server_end'].value,
            'actual_hard_coded_expected_box': guess_namespace['expected_box'].value,
            'fixture_expected_box': swapped.get_expected_service_box().value,
            'is_production_identity_or_end_evidence': False}
        rows = namespace['scoring_outcomes']
        clock_probe = None
        if authority and fixture['name'].startswith('two_completed'):
            expected_members = {5, 7}
            isolated = [value for value in shot_owner if value.match_event_id in expected_members]
            start = next(ev for ev in events if ev.event_id == 5)
            terminal = next(ev for ev in events if ev.event_id == 8)
            elapsed = (int(terminal.evidence['source_pts']) - int(start.evidence['source_pts'])) * Fraction('1/1000')
            clock_probe = {'explicit_live_rally_contact_event_ids': [5, 7],
                'explicit_terminal_event_id': 8,
                'start_source_pts': start.evidence['source_pts'], 'end_source_pts': terminal.evidence['source_pts'],
                'source_time_base': '1/1000', 'eligible_same_camera_interval_fixture': True,
                'expected_exact_source_elapsed_fraction': str(elapsed), 'expected_source_elapsed_s': float(elapsed),
                'actual_current_fps_duration_s': classes['RallyAnalyzer'].analyze_rallies(isolated, fps=25.)[0].duration_s,
                'actual_timestamp_fallback_duration_s': classes['RallyAnalyzer'].analyze_rallies(isolated)[0].duration_s,
                'clock_accuracy_or_bounce_time_accuracy_qualified': False}
        finished = [row['event_id'] for row in engine.observed_transitions
                    if row['outcome']['is_point_ending'] and row['outcome']['winner_id'] is not None]
        result = {'name': fixture['name'], 'authority_enabled': authority,
            'step7_executed_actual_native_event_sequence': native,
            'step10_executed_actual_native_shot_sequence': True,
            'rule_signals_missing_from_native_event_type': sorted({ev.event_type.value for ev in events
                if not isinstance(ev.event_type, EventType)}),
            'synthetic_fixture_events': [{'event_id': ev.event_id, 'type': ev.event_type.value,
                'frame_index': ev.frame_index, 'timestamp_s': ev.timestamp_s, 'evidence': ev.evidence} for ev in events],
            'expected_from_explicit_fixture_rule_evidence': fixture['expected'] if authority else {
                'completed_points': None, 'point_measurements_available': False, 'points': []},
            'observed_real_scoring_completed_point_event_ids': finished,
            'observed_scoring_transition_trace': engine.observed_transitions,
            'actual_step7_export_rows': rows, 'actual_line_call_requests': requested_calls,
            'actual_dead_event_ids': sorted(namespace['dead_event_ids']), 'actual_linked_shots': shots,
            'actual_step10_rallies': rallies, 'actual_step10_points': points,
            'actual_match_overview': summary, 'actual_per_server_serve_statistics': serve_stats,
            'actual_final_state': engine.get_current_state().to_dict(),
            'isolated_known_rally_source_clock_probe': clock_probe,
            'confirmed_swapped_end_source_block_probe': end_case,
            'actual_loaded_export_hashes': {name: sha(directory/name) for name in
                ('rallies.json', 'point_analytics.json', 'match_analytics.json')},
            'full_history_and_probe_lists_are_small_fixture_instrumentation_not_bounded_capacity_evidence': True}
        assert shot_owner._readers == set()
    assert not list(directory.glob('shot_evidence_*')) and not list(directory.glob('event_analysis_*'))
    if authority:
        assert len(finished) == fixture['expected']['completed_points']
    else:
        assert not shots and not points and summary['total_points_played'] is None
    return result


def main():
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    before = {str(path.relative_to(ROOT)): sha(path) for path in PATHS}
    fixtures = list(cases())
    results = [run_case(fixture) for fixture in fixtures]
    results.append(run_case(fixtures[0], authority=False))
    first = results[0]
    let = results[1]
    checks = {
        'two_completed_points_must_not_collapse_to_one': first['actual_match_overview']['total_points_played'] == 2,
        'first_fault_contact_must_not_be_live_rally_member': first['actual_step10_rallies'][0]['shot_ids'] == [3, 4],
        'second_serve_attempt_must_be_counted': first['actual_per_server_serve_statistics']['1']['second_serves_attempted'] == 1,
        'known_rally_duration_must_use_eligible_source_pts_and_terminal_event':
            first['isolated_known_rally_source_clock_probe']['actual_current_fps_duration_s'] ==
            first['isolated_known_rally_source_clock_probe']['expected_source_elapsed_s'],
        'double_fault_must_be_counted': first['actual_per_server_serve_statistics']['2']['double_faults'] == 1,
        'service_bounce_must_request_service_context': all(row['requested_context'] == 'SERVE'
            for row in first['actual_line_call_requests'] if row['supplied_fixture_context'] == 'SERVE'),
        'swapped_end_must_not_use_fixed_identity_end': first['confirmed_swapped_end_source_block_probe']['actual_hard_coded_expected_box'] ==
            first['confirmed_swapped_end_source_block_probe']['fixture_expected_box'],
        'let_replay_must_not_double_official_first_attempts': let['actual_per_server_serve_statistics']['1']['first_serves_attempted'] == 1,
        'partial_unknown_point_must_not_report_one_played_point': results[3]['actual_match_overview']['total_points_played'] is None,
        'review_gap_must_not_report_one_completed_point': results[2]['actual_match_overview']['total_points_played'] == 0,
        'authority_off_remains_unavailable': results[4]['actual_match_overview']['total_points_played'] is None,
    }
    after = {str(path.relative_to(ROOT)): sha(path) for path in PATHS}
    assert before == after
    assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
    receipt = {'status': 'REPRODUCED_POINT_ATTEMPT_AND_SERVICE_CONTEXT_SEMANTICS_RED',
        'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter()-clock, 'cases': results, 'required_checks': checks,
        'failing_check_names': [name for name, passed in checks.items() if not passed],
        'source_bindings_before': before, 'source_bindings_after': after, 'sources_changed': [],
        'torch_imported': False, 'ultralytics_imported': False, 'model_used': False,
        'media_used': False, 'training_performed': False, 'source_clock_or_geometry_accuracy_exercised': False,
        'fixture_rule_evidence_only': True, 'qualification_evidence': False, 'full_match_capacity_proven': False,
        'python_executable': sys.executable, 'script_sha256': sha(__file__)}
    with (BASE/'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': str(BASE/'receipt.json'), 'required_checks': checks,
        'completed_real_points': first['observed_real_scoring_completed_point_event_ids'],
        'current_rally_count': len(first['actual_step10_rallies']),
        'current_point_count': len(first['actual_step10_points']),
        'sources_changed': [], 'torch_imported': False, 'ultralytics_imported': False}))
    assert all(checks.values()), 'Current native analytics loses explicit point/attempt/service/timing semantics'


if __name__ == '__main__':
    main()


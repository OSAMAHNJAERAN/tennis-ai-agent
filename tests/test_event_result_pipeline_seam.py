"""Event-result consumer contracts without model imports or video decoding.

The verifier cases execute the actual feature/candidate/verification flow on
synthetic coordinates. The Phase6 cases compile its exact event-related source
blocks, substituting detection and scoring/timeline boundaries only. Real
engine/point-ledger integration is covered by test_point_timeline.py and its
storage integration tests. This is not video or accuracy qualification.
"""
import ast
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.events import event_detector as detector_module
from src.scoring.point_outcome_resolver import PointOutcomeType
from src.scoring.point_timeline import ReviewedRuleSignal, merge_rule_inputs
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.landing_zone_sequence import LandingZoneSequence
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence
from src.visualization.landing_zones import landing_record


def _snapshot(analysis):
    return json.dumps({
        'candidates': [asdict(value) for value in analysis.candidates],
        'events': [asdict(value) for value in analysis.events],
        'traces': list(analysis.verification_traces),
        'settings': analysis.settings,
        'coordinate_system': analysis.coordinate_system,
    }, sort_keys=True, default=lambda value: value.value)


@pytest.mark.parametrize('scene', ['player_contact', 'court_bounce'])
@pytest.mark.parametrize('post_verify_gate', ['disabled', 'soft_activity', 'trusted_provenance'])
def test_actual_nonempty_verifier_results_and_gate_evidence_match_legacy(
        tmp_path, monkeypatch, scene, post_verify_gate):
    """A nonempty event assertion prevents vacuous disk/legacy equality."""
    trajectory = [TemporalBallPoint(
        index, index / 25., 300. + 3. * index,
        180. + 8. * abs(index - 24) if scene == 'player_contact' else 610. - 8. * abs(index - 30),
        state=BallState.DETECTED, confidence=.85,
    ) for index in range(60)]
    boxes = ([BBox(350., 130., 440., 350., .9, 1)] * len(trajectory)
             if scene == 'player_contact' else [None] * len(trajectory))
    absent = [None] * len(trajectory)
    settings = {}
    if post_verify_gate == 'soft_activity':
        settings.update(enable_activity_state_gating=True, activity_gating_penalise_possible_end=True)
        monkeypatch.setattr(detector_module, 'VisualActivityStateClassifier', lambda **kwargs: SimpleNamespace(
            classify_trajectory=lambda values, **kwargs: [detector_module.VisualActivityState.POSSIBLE_POINT_END] * len(values)))
    if post_verify_gate == 'trusted_provenance':
        settings['enable_provenance_gating'] = True
        monkeypatch.setattr(detector_module, 'ProvenanceAuditor', lambda **kwargs: SimpleNamespace(
            audit=lambda *args: SimpleNamespace(provenance_trust_score=1., low_trust_reason=None,
                                                to_dict=lambda: {'trusted_synthetic_fixture': True})))
    expected = detector_module.TennisEventDetector(**settings).analyze(
        trajectory, boxes, absent, fps=25., frame_size=(1280, 720))
    assert len(expected.events) > 0
    expected_type = detector_module.EventType.BOUNCE if scene == 'court_bounce' else detector_module.EventType.PLAYER_1_HIT
    assert any(value.event_type == expected_type for value in expected.events)
    assert any(value['verification_pass'] is False for value in expected.verification_traces)
    with TemporalBallPointSequence(tmp_path) as points, EventAnalysisSequence(tmp_path) as output:
        points.extend(trajectory)
        actual = detector_module.TennisEventDetector(**settings).analyze(
            points, boxes, absent, fps=25., frame_size=(1280, 720),
            scratch_directory=tmp_path, result_output=output)
        assert _snapshot(actual) == _snapshot(expected)
        passed = [value for value in actual.verification_traces if value['final_event_emitted']]
        assert len(passed) == len(actual.events)
        if post_verify_gate == 'soft_activity':
            assert all(value['activity_state'] == detector_module.VisualActivityState.POSSIBLE_POINT_END.value
                       for value in passed)
            ungated = detector_module.TennisEventDetector().analyze(
                trajectory, boxes, absent, fps=25., frame_size=(1280, 720))
            assert [value.confidence for value in actual.events] == pytest.approx(
                [value.confidence * .7 for value in ungated.events])
        if post_verify_gate == 'trusted_provenance':
            assert all(value['provenance_audit'] == {'trusted_synthetic_fixture': True} for value in passed)
    assert not list(tmp_path.glob('event_analysis_*'))
    assert not list(tmp_path.glob('event_features_*'))


def _phase6_event_source_blocks():
    """Extract exact consumers rather than a second implementation of them."""
    path = Path(__file__).resolve().parents[1] / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')

    def assignment(name):
        return next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))

    start, stop = assignment('event_output'), assignment('authoritative_events') + 1
    event_flow = run.body[start:stop]
    landing_start = assignment('landing_settings')
    landing_stop = next(index + 1 for index in range(landing_start, len(run.body))
                        if isinstance(run.body[index], ast.If)
                        and ast.unparse(run.body[index].test).startswith("landing_settings.get('enabled',"))
    landing_flow = run.body[landing_start:landing_stop]
    destinations = {'match_events.json', 'event_candidates.json', 'event_verification_trace.json'}
    exports = [node for node in run.body if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
               and node.value.func.id == 'write_json_array_artifact'
               and isinstance(node.value.args[0], ast.Call)
               and any(isinstance(argument, ast.Constant) and argument.value in destinations
                       for argument in node.value.args[0].args)]
    assert len(exports) == 3
    helper_names = {'sanitize_json_value', 'event_export_records', 'event_candidate_export_records'}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    assert len(helpers) == len(helper_names)
    return path, helpers, event_flow, landing_flow, exports


@pytest.mark.parametrize('authority', [False, True])
def test_exact_phase6_consumers_keep_diagnostics_separate_and_persist_landing_writeback(tmp_path, authority):
    source, helpers, event_flow, landing_flow, exports = _phase6_event_source_blocks()
    called = []
    retained = {}
    reviewed = {'review_status': 'confirmed', 'active_play': True, 'position_type': 'ground_contact',
                'ground_contact_source': 'SYNTHETIC_SEAM_FIXTURE', 'position_uncertainty_m': .1}
    source_identity = tmp_path / 'synthetic_event_source_identity.bin'
    source_identity.write_bytes(b'Synthetic source identity bytes for the event seam; not video or media.')
    expected_fingerprint = hashlib.sha256(source_identity.read_bytes()).hexdigest()
    candidate_frames = [object() for _ in range(10)]
    context_calls = []

    class AnalyzerBoundary:
        def analyze(self, **options):
            output = options['result_output']
            assert options['scratch_directory'] == str(tmp_path)
            assert isinstance(output, EventAnalysisSequence)
            retained['output'] = output
            # Deliberately unordered frames exercise the indexed scoring pass.
            for ident, frame in enumerate((9, 2, 4, 6), 1):
                output.candidates.append(detector_module.EventCandidate(
                    frame, frame / 25., .8, (3., 8.), 'DETECTED'))
                output.events.append(detector_module.TennisEvent(
                    ident, detector_module.EventType.BOUNCE, frame, frame / 25., None,
                    (3., 8.), (3., 8.), .9, 'DETECTED',
                    dict(reviewed) if ident != 2 else {}))
                output.verification_traces.append({'candidate_id': ident, 'synthetic_fixture': True})
            return detector_module.EventDetectionAnalysis(
                output.candidates, output.events, output.verification_traces, {}, {})

    class ScoringBoundary:
        def native_line_calls(self):
            # This seam deliberately supplies no line calls; real journal and
            # indexed banner behavior have separate actual-engine coverage.
            return ()

        @contextmanager
        def journal_context(self, directory):
            assert Path(directory) == tmp_path
            retained['scoring_context_open'] = True
            try:
                yield self
            finally:
                retained['scoring_context_open'] = False

        def get_current_state(self):
            return SimpleNamespace(to_dict=lambda: {'synthetic_scoring_boundary': True})

        def process_match_event(self, **options):
            called.append(options['event_id'])
            state = SimpleNamespace(get_score_summary_str=lambda: 'SYNTHETIC', serve_attempt=1,
                ball_state=SimpleNamespace(value='SYNTHETIC'), point_state=SimpleNamespace(value='SYNTHETIC'))
            outcome = SimpleNamespace(
                outcome_type=PointOutcomeType.DEAD_BALL_IGNORED if options['event_id'] == 3 else PointOutcomeType.RALLY_CONTINUES,
                winner_id=None, reason='SYNTHETIC_SEAM_FIXTURE')
            return state, outcome

    class TimelineBoundary:
        """Do not replace this event-consumer test with another rules test."""
        def __init__(self, owner, initial_state, *, source_frames):
            assert isinstance(owner, PointAnalysisSequence)
            assert owner.analysis_id == 'synthetic-event-seam' and owner.revision == 7
            assert owner.source_fingerprint == expected_fingerprint
            assert initial_state == {'synthetic_scoring_boundary': True}
            assert source_frames is candidate_frames
            self.owner = owner
            retained['point_owner'] = owner

    def context_boundary(event, source_frames, provider, *, analysis_id, source_fingerprint):
        assert source_frames is candidate_frames and provider is None
        assert analysis_id == 'synthetic-event-seam' and source_fingerprint == expected_fingerprint
        context_calls.append(event.event_id)
        return SimpleNamespace(event_id=event.event_id, synthetic_context=True)

    def timeline_event_boundary(event, context, engine, timeline, line_caller, **options):
        # Preserve the original explicit scoring stub and its dead-event rule.
        # Point legality/identity/reducer behavior belongs to the real timeline
        # integration tests; this checks the exact native dispatch to that seam.
        assert context.synthetic_context and context.event_id == event.event_id
        assert timeline.owner is retained['point_owner'] and timeline.owner._open() is not None
        assert engine is pipeline.scoring_engine and line_caller is pipeline.line_call_engine
        assert retained['scoring_context_open'] is True
        assert options == {'ball_points': [], 'homography_matrix': None,
                           'frames': [], 'geometry_available': False}
        state, outcome = engine.process_match_event(
            event_id=event.event_id, event_type=event.event_type.value,
            timestamp_seconds=event.timestamp_s, player_id=event.player_id,
            line_call=None, confidence=event.confidence)
        timeline.owner.put('transitions', f'synthetic-event-{event.event_id}',
            {'event_id': event.event_id, 'frame': event.frame_index,
             'outcome_type': outcome.outcome_type.value},
            event_id=event.event_id, frame=event.frame_index)
        return state, outcome, None

    pipeline = SimpleNamespace(event_detector=AnalyzerBoundary(), scoring_engine=ScoringBoundary(),
        line_call_engine=SimpleNamespace(), config={
        'event_detection': {'authoritative_enabled': authority},
        'match_configuration': {'match_id': 'synthetic-event-seam'}, 'analysis_revision': 7,
        'visualization': {'landing_zones': {'enabled': True, 'partition': 'tactical_3x3'}}})
    calibrations = [SimpleNamespace(is_valid=index != 6, image_to_court=np.eye(3)) for index in range(10)]
    namespace = dict(np=np, Any=object, List=list, Dict=dict, LineCallEvidence=object,
        os=os, time=time, json=json, hashlib=hashlib, EventAnalysisSequence=EventAnalysisSequence,
        PointAnalysisSequence=PointAnalysisSequence, PointTimeline=TimelineBoundary,
        ReviewedRuleSignal=ReviewedRuleSignal, merge_rule_inputs=merge_rule_inputs,
        candidate_event_context=context_boundary, process_timeline_event=timeline_event_boundary,
        EventType=detector_module.EventType, PointOutcomeType=PointOutcomeType,
        landing_record=landing_record, LandingZoneSequence=LandingZoneSequence,
        write_json_array_artifact=write_json_array_artifact,
        self=pipeline, output_dir=str(tmp_path), disk_players=True,
        input_video_path=str(source_identity), candidate_frames=candidate_frames,
        ball_points=[], p1_boxes=[], p2_boxes=[], camera_enabled=True, homography_matrix=None,
        fps=25., w=1280, h=720, t0=time.time(), is_h_valid=False, frames=[], frame_calibrations=calibrations)
    with ExitStack() as stack:
        namespace['export_stack'] = stack
        exec(compile(ast.Module(body=helpers + event_flow + landing_flow + exports,
                                type_ignores=[]), str(source), 'exec'), namespace)
        output = retained['output']
        assert output.path.exists() and output._open() is not None
        point_owner = retained['point_owner']
        assert point_owner.path.exists() and point_owner._open() is not None
        assert namespace['event_candidates'] is output.candidates
        assert namespace['event_verification_trace'] is output.verification_traces
        match = json.loads((tmp_path / 'match_events.json').read_text())
        candidates = json.loads((tmp_path / 'event_candidates.json').read_text())
        traces = json.loads((tmp_path / 'event_verification_trace.json').read_text())
        landings = json.loads((tmp_path / 'landing_zones.json').read_text())
        assert len(candidates['candidates']) == len(traces['traces']) == 4
        if authority:
            assert called == [2, 3, 4, 1]
            assert context_calls == called
            assert namespace['authoritative_events'] is output.events
            assert match['events_count'] == 4
            assert [value['event_id'] for value in match['events']] == [1, 2, 3, 4]
            assert {value['event_id'] for value in landings['records']} == {1}
            assert len(landings['records']) == 2  # Service + tactical presentations of the same event.
            evidence = match['events'][0]['evidence']
            assert evidence['analysis_id'] == 'synthetic-event-seam'
            assert evidence['analysis_revision'] == 7 and evidence['calibration_id'] == 'frame_9'
            assert evidence['contact_position_px'] == [3., 8.]
            assert output.events[0].evidence == evidence
            assert 'calibration_id' not in output.events[1].evidence  # Unreviewed.
            assert 'calibration_id' not in output.events[2].evidence  # Scoring dead ball retained, not highlighted.
            assert 'calibration_id' not in output.events[3].evidence  # Invalid contact calibration.
        else:
            assert called == [] and namespace['authoritative_events'] == []
            assert context_calls == []
            assert match['events_count'] == 0 and match['events'] == []
            assert landings['records'] == [] and landings['calibrations'] == {}
            assert len(output.events) == 4  # Diagnostic events must not gain authority through storage.
    assert not output.path.exists()
    assert not point_owner.path.exists() and point_owner.readers == set()
    assert not list(tmp_path.glob('event_analysis_*'))
    assert not list(tmp_path.glob('point_analysis_*'))

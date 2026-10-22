"""Fresh native shots and analytics after a private score correction.

Every measurement and reviewed binding below is synthetic. Actual rule, line
calling, shot linking, classification, and native analytics execute; this is
software evidence rather than a vision or physical accuracy qualification.
"""
import builtins
import copy
import gc
import hashlib
import json
import time
import tracemalloc
from types import SimpleNamespace

import pytest

from tests.test_corrected_rule_rebuild import (
    binding_amendment, build_sealed_base, canonical_sha, unchanged,
)


OBSERVATIONS = []


@pytest.fixture(autouse=True)
def no_model_imports(monkeypatch):
    from src.api.result_storage import ResultStorage
    parser_owners = []
    initializer = ResultStorage.__init__
    def observed_init(self, *args, **kwargs):
        initializer(self, *args, **kwargs)
        parser_owners.append(self)
    monkeypatch.setattr(ResultStorage, '__init__', observed_init)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
            raise AssertionError('Private shot reconstruction imported inference: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    yield
    assert all(owner._connection is None for owner in parser_owners)


def assert_only_parser_diagnostics(scratch):
    # Generic rule-source ResultStorage deliberately retains its closed
    # diagnostic database on exceptional context exit. Typed fresh event,
    # shot, rule journal and point owners must still remove their exact files.
    for path in scratch.iterdir():
        assert path.is_file()
        assert path.name.startswith('.result_storage_') and path.suffix == '.sqlite3'
        with path.open('rb') as stream:
            assert stream.read(16) == b'SQLite format 3\x00'


def advance_equivalent_generation(fixture):
    """Actual CAS publication with valid unchanged revision-zero source assets."""
    from src.api.analysis_generation import publish_analysis_generation
    from src.api.result_storage import ResultStorage
    files = {path.name: path for path in fixture.snapshot.files_directory.iterdir()
             if path.name not in {'result.json', 'summary.json'}}
    with ResultStorage(fixture.directory) as storage:
        return publish_analysis_generation(storage, {
            'summary': {'id': 'fixture', 'frames': fixture.snapshot.frames}, 'analysisRevision': 0,
        }, fixture.directory, files=files, analysis_revision=0,
            expected_generation=fixture.snapshot.generation)


def build_shot_base(tmp_path, *, frame_count=12, pose_enabled=True, configured_players=False,
                    receiver_hit=True, repeated_hits=False, config_tail='',
                    playable_source=False, metadata_size=None):
    """Extend the real sealed rule fixture with original native linked shots.

Original source events deliberately retain stale court coordinates. Their
accepted bounce call is fresh; original linked snapshots are captured only
after using that call. A new correction must rebuild this same distinction.
"""
    import numpy as np
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.events.event_detector import TennisEvent, EventType
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.match_state_codec import decode_match_state
    from src.scoring.point_timeline import (PointTimeline, ReviewedParticipantBinding,
        candidate_event_context, process_timeline_event, native_match_analytics)
    from src.scoring.rule_replay import write_rule_replay
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.shot_analysis.shot_classifier import TennisShotClassifier
    from src.shot_analysis.shot_linker import TennisShotLinker
    from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
    from src.utils.bbox_utils import BBox
    from src.utils.event_analysis_sequence import EventAnalysisSequence
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.scoring_journal import ScoringJournal
    from src.utils.shot_evidence_sequence import ShotEvidenceSequence

    fixture = build_sealed_base(tmp_path, frame_count=frame_count)
    directory = fixture.directory
    replacement_source = None
    if playable_source:
        import cv2
        replacement_source = directory / 'synthetic_playable_source.mp4'
        writer = cv2.VideoWriter(str(replacement_source), cv2.VideoWriter_fourcc(*'mp4v'),
                                 30., (1280, 960))
        assert writer.isOpened()
        try:
            for index in range(frame_count):
                writer.write(np.full((960, 1280, 3), index*16 % 240, dtype=np.uint8))
        finally:
            writer.release()
        fixture.source_sha = hashlib.sha256(replacement_source.read_bytes()).hexdigest()
        fixture.request['source_sha256'] = fixture.source_sha
    source_frames = [{0: row['interval'], 1: row['identity']}
                     for row in fixture.document['source_frames']]
    initial = decode_match_state(fixture.document['initial_state'])
    detections = json.loads(fixture.snapshot.resolve_file('detections.json').read_text(encoding='utf-8'))
    if playable_source:
        width, height = metadata_size if metadata_size is not None else (1280, 960)
        detections['metadata'].update(width=width, height=height)
    positions = [row['ball']['position_px'] for row in detections['frames']]
    for row in detections['frames']:
        row['player_1'] = {'bbox': [100.1234567890123, 300., 200., 700.],
                           'confidence': .9876543210123, 'source_track_id': 11}
        row['player_2'] = {'bbox': [100., 100., 200., 400.],
                           'confidence': .9765432101234, 'source_track_id': 22}
    detection_path = directory / 'detections_with_players.json'
    detection_path.write_text(json.dumps(detections, allow_nan=False), encoding='utf-8')
    config_path = directory / 'shot_run_config.yaml'
    config = ('model_system:\n  family: yolo26\n  vision_model_count: 3\n'
              'player_detection:\n  model: explicit-synthetic-selected-player-yolo26s-pose.pt\n'
              'line_calling:\n  contact_refinement_window: 3\n'
              'pose_estimation:\n  enabled: ' + ('true' if pose_enabled else 'false') + '\n'
              '  source: player_model\n'
              '  min_pose_confidence: 0.35\n'
              '  model_path: explicit-synthetic-selected-player-yolo26s-pose.pt\n'
              'shot_classification:\n  ambiguity_threshold: 0.15\n')
    if configured_players:
        config += ('  players_handedness:\n    2: RIGHT_HANDED\n'
                   '  court_orientation_sign:\n    2: 1.0\n')
    config_path.write_text(config + config_tail, encoding='utf-8')
    points = [TemporalBallPoint(index, index/30., *positions[index], confidence=.99,
                               state=BallState.DETECTED) for index in range(frame_count)]
    events = [TennisEvent(1, EventType.SERVE_CONTACT, 0, 0., 1,
                         tuple(positions[0]), (88., 88.), .99, 'EXPLICIT_SYNTHETIC_OBSERVATION',
                         {'fixture': 'synthetic-contact'}),
              TennisEvent(2, EventType.BOUNCE, 4, 4/30., None,
                         tuple(positions[4]), (99., 99.), .99, 'EXPLICIT_SYNTHETIC_OBSERVATION',
                         {'fixture': 'synthetic-bounce'})]
    if receiver_hit:
        for index, frame in enumerate(range(8, frame_count, 8) if repeated_hits else [8]):
            role = 2 if index % 2 == 0 else 1
            events.append(TennisEvent(3+index,
                EventType.PLAYER_2_HIT if role == 2 else EventType.PLAYER_1_HIT,
                frame, frame/30., role, tuple(positions[frame]), (77., 77.), .99,
                'EXPLICIT_SYNTHETIC_OBSERVATION', {'fixture': 'synthetic-receiver-hit'}))
    replay_path = directory / 'native_shot_rule_replay.json'
    p1 = [BBox(100.1234567890123, 300., 200., 700., .9876543210123, track_id=11)
          for _ in source_frames]
    p2 = [BBox(100., 100., 200., 400., .9765432101234, track_id=22)
          for _ in source_frames]
    with ScoringJournal(fixture.scratch) as journal:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        with PointAnalysisSequence(fixture.scratch, 'fixture', revision=0,
                    source_fingerprint=fixture.source_sha) as owner, \
             EventAnalysisSequence(fixture.scratch) as event_owner, \
             ShotEvidenceSequence(fixture.scratch) as shots:
            timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
            evaluator = TennisLineCallEngine()
            for event in events:
                interval = source_frames[event.frame_index][0]
                role = 2 if event.player_id == 2 else 1
                binding = ReviewedParticipantBinding(event.frame_index, interval['segment_id'], role, role,
                    'B' if role == 2 else 'A', 'track-B' if role == 2 else 'track-A',
                    'FAR_COURT' if role == 2 else 'NEAR_COURT', 0, 'synthetic-calibration', 0,
                    'EXPLICIT_SYNTHETIC_END_REVIEW', analysis_id='fixture',
                    source_fingerprint=fixture.source_sha, source_frame_index=interval['source_frame_index'],
                    source_pts=interval['source_pts'], source_time_base=interval['source_time_base'],
                    identity_epoch=0, participant_mapping_ref='EXPLICIT_SYNTHETIC_PARTICIPANT_MAP',
                    geometry_qualification_ref='EXPLICIT_SYNTHETIC_GEOMETRY_ONLY')
                context = candidate_event_context(event, source_frames, lambda *_: binding,
                    analysis_id='fixture', source_fingerprint=fixture.source_sha)
                _, _, call = process_timeline_event(event, context, engine, timeline, evaluator,
                    ball_points=points, homography_matrix=np.diag([.01, .01, 1.]), frames=(),
                    geometry_available=True)
                event_owner.events.append(event if call is None else
                    TennisEvent(event.event_id, event.event_type, event.frame_index, event.timestamp_s,
                        event.player_id, event.ball_position_px, tuple(call.bounce_position_m),
                        event.confidence, event.trajectory_state, event.evidence))
            TennisShotLinker(TennisShotClassifier()).link_shots(event_owner.events, points, p1, p2,
                dead_event_ids=owner.dead_event_ids(), result_output=shots, event_output=event_owner)
            timeline.attach_shots(shots)
            shots.seal()
            timeline.finish()
            analytics = native_match_analytics(owner, shots, engine.state, authoritative=True)
            original = {'shots': [shot.to_dict() for shot in shots],
                'overview': analytics['overview'], 'players': analytics['players'],
                'serve_analytics': analytics['serve_analytics'],
                'landings': list(analytics['ball_landing_heatmap']),
                'members': list(owner.member_rows())}
            write_rule_replay(replay_path, engine, owner, source_frames)
    files = {path.name: path for path in fixture.snapshot.files_directory.iterdir()
             if path.name not in {'result.json', 'summary.json'}}
    files.update({'native_rule_replay.json': replay_path, 'detections.json': detection_path,
                  'run_config.yaml': config_path})
    if replacement_source is not None:
        files['source.mp4'] = replacement_source
    with ResultStorage(directory) as storage:
        token = publish_analysis_generation(storage, {
            'summary': {'id': 'fixture', 'frames': len(source_frames)}, 'analysisRevision': 0,
        }, directory, files=files, analysis_revision=0, expected_generation=fixture.snapshot.generation)
    snapshot = resolve_generation(directory)
    document = json.loads(replay_path.read_text(encoding='utf-8'))
    fixture.request.update(base_generation=token,
        base_rule_replay_sha256=snapshot.file_sha256('native_rule_replay.json'))
    fixture.request['initial_state_amendment']['expected_initial_state_sha256'] = canonical_sha(document['initial_state'])
    fixture.snapshot, fixture.document, fixture.original = snapshot, document, original
    fixture.marker = (directory / '.result/current.store').read_bytes()
    fixture.manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in snapshot.files_directory.iterdir()}
    return fixture


def test_corrected_double_fault_rebuilds_native_shots_memberships_and_analytics(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.shot_analysis.shot_types import ShotType, ShotOutcome, ShotClassificationSource
    fixture = build_shot_base(tmp_path)
    assert fixture.original['overview']['live_shots_recorded'] == 2
    assert fixture.original['serve_analytics']['by_player']['1']['second_serves_in'] == 1
    assert len(fixture.original['landings']) == 1
    assert {row['play_scope'] for row in fixture.original['members']} == {'LEGAL_RALLY'}
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=fixture.request) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
            assert rebuilt.owner.sealed
            assert len(analysis.shots) == 2
            serve, receiver = analysis.shots
            assert serve.shot_type is ShotType.SERVE
            assert serve.outcome is ShotOutcome.FAULT
            assert serve.source_court_position_m is None
            assert serve.landing_court_position_m == pytest.approx((3., 8.))
            assert receiver.is_dead_ball
            assert receiver.shot_type is ShotType.UNKNOWN
            assert receiver.classification_source is ShotClassificationSource.ABSTENTION_UNKNOWN
            assert receiver.source_court_position_m is None
            assert receiver.landing_court_position_m is None
            assert rebuilt.owner.event(3)['outcome_type'] == 'DEAD_BALL_IGNORED'
            assert {row['play_scope'] for row in rebuilt.owner.member_rows()} == {'DOUBLE_FAULT', 'DEAD_AFTER_PLAY'}
            analytics = analysis.match_analytics
            assert analytics['overview']['live_shots_recorded'] == 0
            assert analytics['overview']['total_shots_recorded'] == 2
            assert analytics['overview']['total_points_played'] == 1
            assert len(analytics['ball_landing_heatmap']) == 0
            assert analytics['players']['player_1']['shots']['total'] == 0
            assert analytics['players']['player_2']['shots']['total'] == 0
            serves = analytics['serve_analytics']['by_player']['1']
            assert serves['second_serves_in'] == 0 and serves['double_faults'] == 1
            assert serves['second_serves_attempted'] == 1 and serves['contact_executions'] == 1
            assert analysis.audit['published'] is False
            assert analysis.audit['independent_accuracy_validated'] is False
            OBSERVATIONS.append({'original_live_shots': 2, 'corrected_live_shots': 0,
                'original_second_serves_in': 1, 'corrected_double_faults': 1,
                'corrected_receiver_suppressed': True, 'source_scope': 'SYNTHETIC_SOFTWARE_ONLY'})
            shots, event_owner = analysis.shots, analysis.events
        with pytest.raises(RuntimeError):
            shots[0]
        with pytest.raises(RuntimeError):
            event_owner[0]
    unchanged(fixture)


def test_disabled_pose_keeps_configured_native_geometry_classifier(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.shot_analysis.shot_types import ShotType, ShotClassificationSource
    fixture = build_shot_base(tmp_path, pose_enabled=False, configured_players=True)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
            receiver = analysis.shots[1]
            assert not receiver.is_dead_ball
            assert receiver.shot_type is ShotType.FOREHAND
            assert receiver.classification_source is ShotClassificationSource.GEOMETRY_BASELINE
            assert analysis.match_analytics['overview']['live_shots_recorded'] == 2
    unchanged(fixture)


def test_missing_required_pose_provider_refuses_without_silent_geometry_fallback(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path, configured_players=True)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with pytest.raises(RuntimeError, match='[Pp]ose|provider'):
            with relink_corrected_shots(rebuilt, fixture.scratch):
                pytest.fail('Configured reachable pose branch was silently omitted')
        assert not rebuilt.owner.sealed
        assert len(rebuilt.owner.member_rows()) == 0
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)


def test_configured_pose_is_not_requested_for_dead_receiver_or_semantic_serve(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path, configured_players=True)
    def forbidden(request):
        raise AssertionError('Dead or serve branch requested pose features')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=fixture.request) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch, pose_provider=forbidden) as analysis:
            assert analysis.shots[1].is_dead_ball
    unchanged(fixture)


def test_downstream_failure_closes_fresh_shots_and_preserves_sealed_base(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path)
    with pytest.raises(RuntimeError, match='downstream renderer'):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                            request=fixture.request) as rebuilt:
            with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
                shots, events, owner = analysis.shots, analysis.events, rebuilt.owner
                raise RuntimeError('Synthetic downstream renderer failure')
    with pytest.raises(RuntimeError):
        shots[0]
    with pytest.raises(RuntimeError):
        events[0]
    with pytest.raises(RuntimeError):
        owner.event(1)
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)


def test_live_pose_callback_receives_bound_sealed_inputs_and_actual_crop_defaults(tmp_path):
    from dataclasses import FrozenInstanceError
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.shot_analysis.shot_types import ShotType, ShotClassificationSource, PlayerHandedness
    fixture = build_shot_base(tmp_path, configured_players=True,
        config_tail='pose_crop_window_unused_by_original_phase6: 91\n')
    requests = []
    def measured_synthetic_features(request):
        requests.append(request)
        assert request.analysis_id == 'fixture' and request.analysis_revision == 1
        assert request.base_generation == fixture.snapshot.generation
        assert request.source_sha256 == fixture.source_sha
        assert request.run_config_sha256 == fixture.snapshot.file_sha256('run_config.yaml')
        assert request.selected_player_model_path == 'explicit-synthetic-selected-player-yolo26s-pose.pt'
        assert request.hit_frame == 8 and request.player_id == 2
        assert request.handedness is PlayerHandedness.RIGHT_HANDED
        assert request.court_orientation_sign == 1.
        assert request.window_half_size == 3 and request.crop_margin == .65
        assert len(request.frames) == len(request.player_boxes) == 12
        assert request.player_boxes[8].track_id == 22
        assert request.player_boxes[8].confidence == .9765432101234
        assert request.player_boxes[0] is None
        with pytest.raises(ValueError, match='window'):
            request.frames[0]
        with pytest.raises(FrozenInstanceError):
            request.hit_frame = 0
        # This is a declared synthetic feature provider; no real source frame
        # decode or checkpoint execution is asserted by returning these values.
        return {'valid': True, 'normalized_wrist_displacement': .5,
                'mean_pose_confidence': .9, 'sample_count': 7}
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch,
                                   pose_provider=measured_synthetic_features) as analysis:
            assert len(requests) == 1
            assert analysis.shots[1].shot_type is ShotType.FOREHAND
            assert analysis.shots[1].classification_source is ShotClassificationSource.YOLO11_POSE_TEMPORAL
            assert analysis.audit['pose_provider_calls'] == 1
            assert analysis.audit['source_video_opened_for_crops'] is False
            assert analysis.audit['selected_checkpoint_execution_verified'] is False
    unchanged(fixture)
    with pytest.raises(RuntimeError):
        len(requests[0].frames)


@pytest.mark.parametrize('transient', ['unavailable', 'segment', 'anchor', 'identity'])
def test_pose_crop_window_does_not_bridge_a_transient_intermediate_discontinuity(transient):
    from src.scoring.corrected_shot_rebuild import _WindowBoxes
    source = [{0: {'analysis_allowed': True, 'segment_id': 'camera-0',
        'anchor_frame_index': 0, 'measurement_transition_allowed': frame > 0,
        'source_frame_index': frame, 'source_time_base': '1/30', 'source_pts': str(frame)},
        1: {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0, 'observations': [
            {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}}
        for frame in range(12)]
    if transient == 'unavailable': source[7][0]['analysis_allowed'] = False
    elif transient == 'segment': source[7][0]['segment_id'] = 'camera-transient'
    elif transient == 'anchor': source[7][0]['anchor_frame_index'] = 7
    elif transient == 'identity': source[7][1]['epoch'] = 1
    boxes = [SimpleNamespace(track_id=22) for _ in source]
    bounded = _WindowBoxes(boxes, source, 8, 3)
    assert bounded[8] is boxes[8]
    assert bounded[9] is boxes[9]
    assert bounded[7] is None
    assert bounded[6] is None
    assert bounded[5] is None


@pytest.mark.parametrize('track_change', ['changed_neighbour', 'intermediate_swap', 'unknown_hit'])
def test_pose_crop_window_does_not_mix_selected_local_tracks(track_change):
    from src.scoring.corrected_shot_rebuild import _WindowBoxes
    source = [{0: {'analysis_allowed': True, 'segment_id': 'camera-0',
        'anchor_frame_index': 0, 'measurement_transition_allowed': frame > 0,
        'source_frame_index': frame, 'source_time_base': '1/30', 'source_pts': str(frame)},
        1: {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0, 'observations': [
            {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}}
        for frame in range(12)]
    boxes = [SimpleNamespace(track_id=22) for _ in source]
    if track_change == 'changed_neighbour': boxes[9].track_id = 23
    elif track_change == 'intermediate_swap': boxes[7].track_id = 23
    elif track_change == 'unknown_hit': boxes[8].track_id = None
    bounded = _WindowBoxes(boxes, source, 8, 3)
    assert bounded[8] is boxes[8]
    if track_change == 'changed_neighbour':
        assert bounded[9] is None and bounded[10] is None
        assert bounded[7] is boxes[7]
    elif track_change == 'intermediate_swap':
        assert bounded[7] is None and bounded[6] is None and bounded[5] is None
        assert bounded[9] is boxes[9]
    else:
        assert all(bounded[index] is None for index in (5, 6, 7, 9, 10, 11))


@pytest.mark.parametrize('response', [
    [], {}, {'valid': 1}, {'valid': True, 'normalized_wrist_displacement': False,
                           'mean_pose_confidence': .9},
    {'valid': True, 'normalized_wrist_displacement': .5, 'mean_pose_confidence': True},
    {'valid': True, 'normalized_wrist_displacement': float('nan'), 'mean_pose_confidence': .9},
    {'valid': True, 'normalized_wrist_displacement': .5, 'mean_pose_confidence': 1.1},
    {'valid': False, 'reason': 'oversized' * 3000},
])
def test_invalid_pose_callback_response_is_rejected_before_attaching_shots(tmp_path, response):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path, configured_players=True)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with pytest.raises((ValueError, TypeError)):
            with relink_corrected_shots(rebuilt, fixture.scratch,
                                       pose_provider=lambda _: copy.deepcopy(response)):
                pytest.fail('Invalid crop features yielded corrected analytics')
        assert not rebuilt.owner.sealed
        assert len(rebuilt.owner.member_rows()) == 0
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)


def test_mutable_root_configuration_and_detections_do_not_change_the_sealed_rebuild(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.shot_analysis.shot_types import ShotType
    fixture = build_shot_base(tmp_path, pose_enabled=False, configured_players=True)
    (fixture.directory / 'run_config.yaml').write_text('INVALID ROOT CONFIG', encoding='utf-8')
    (fixture.directory / 'detections.json').write_text('{}', encoding='utf-8')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
            assert analysis.shots[1].shot_type is ShotType.FOREHAND
            assert analysis.audit['run_config_sha256'] == fixture.snapshot.file_sha256('run_config.yaml')
    unchanged(fixture)


def test_corrected_shot_pipeline_keeps_owned_buffers_bounded_as_frames_and_shots_grow(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    samples = []
    for count in (257, 4097):
        parent = tmp_path / ('frames-' + str(count))
        parent.mkdir()
        fixture = build_shot_base(parent, frame_count=count, repeated_hits=True)
        expected_shots = 1 + len(range(8, count, 8))
        assert fixture.original['overview']['live_shots_recorded'] == expected_shots
        gc.collect()
        started = time.perf_counter()
        tracemalloc.start()
        try:
            with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                                request=fixture.request) as rebuilt:
                with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
                    assert len(analysis.events) == expected_shots + 1
                    assert len(analysis.shots) == expected_shots
                    assert len(rebuilt.owner.member_rows()) == expected_shots
                    assert analysis.match_analytics['overview']['live_shots_recorded'] == 0
                    assert len(analysis.match_analytics['ball_landing_heatmap']) == 0
                    retained, peak = tracemalloc.get_traced_memory()
            closed_retained, closed_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        samples.append({'frame_count': count, 'shot_count': expected_shots,
            'accepted_input_count': expected_shots+1, 'retained_bytes_inside': retained,
            'peak_bytes_inside': peak, 'retained_bytes_after_cleanup': closed_retained,
            'peak_bytes_including_cleanup': closed_peak,
            'elapsed_seconds': time.perf_counter()-started,
            'scope': 'FULL_PRIVATE_CORRECTED_RULE_SHOT_AND_NATIVE_ANALYTICS_CHAIN; FIXTURE_BUILD_EXCLUDED'})
        unchanged(fixture)
        assert not list(fixture.scratch.glob('*'))
    assert max(sample['peak_bytes_including_cleanup'] for sample in samples) < 24*1024*1024
    assert samples[1]['peak_bytes_including_cleanup'] < samples[0]['peak_bytes_including_cleanup']+4*1024*1024
    assert max(sample['retained_bytes_inside'] for sample in samples) < 8*1024*1024
    OBSERVATIONS.append({'corrected_shot_capacity': samples,
                         'native_sqlite_and_numpy_allocations_outside_tracemalloc': True})


@pytest.mark.parametrize('advance_at', ['entry', 'crop_provider', 'consumer_exit'])
def test_current_generation_advance_rejects_obsolete_fresh_shot_work(tmp_path, advance_at):
    from src.api.analysis_generation import resolve_generation
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path, configured_players=advance_at == 'crop_provider')
    request = binding_amendment(fixture) if advance_at == 'crop_provider' else fixture.request
    advanced = []
    def advancing_provider(pose_request):
        advanced.append(advance_equivalent_generation(fixture))
        return {'valid': True, 'normalized_wrist_displacement': .5, 'mean_pose_confidence': .9}
    with pytest.raises(ValueError, match='current generation'):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
            if advance_at == 'entry':
                advanced.append(advance_equivalent_generation(fixture))
            with relink_corrected_shots(rebuilt, fixture.scratch,
                                       pose_provider=advancing_provider) as analysis:
                assert advance_at == 'consumer_exit'
                assert len(analysis.shots) == 2
                advanced.append(advance_equivalent_generation(fixture))
    assert len(advanced) == 1
    assert advanced[0] != fixture.snapshot.generation
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.snapshot.files_directory.iterdir()} == fixture.manifest
    assert_only_parser_diagnostics(fixture.scratch)


def test_live_pose_callback_decodes_actual_sealed_frames_with_bounded_cache_and_cleanup(tmp_path):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    fixture = build_shot_base(tmp_path, configured_players=True, playable_source=True)
    (fixture.directory / 'source.mp4').write_bytes(b'CONFLICTING MUTABLE SOURCE ALIAS')
    decoders, requests = [], []
    def read_actual_synthetic_frames(request):
        requests.append(request)
        assert request.source_sha256 == fixture.snapshot.file_sha256('source.mp4')
        for index in range(5, 12):
            frame = request.frames[index]
            assert frame.shape == (960, 1280, 3)
            assert float(frame.mean()) == pytest.approx(index*16, abs=6)
        decoder = request.frames.frames.sequence
        decoders.append(decoder)
        assert decoder._cap.isOpened()
        assert len(decoder._cache) <= 8
        assert decoder.metadata.fps == pytest.approx(30.)
        return {'valid': True, 'normalized_wrist_displacement': .5, 'mean_pose_confidence': .9}
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch,
                                   pose_provider=read_actual_synthetic_frames) as analysis:
            assert analysis.audit['source_video_opened_for_crops'] is True
            assert analysis.audit['selected_checkpoint_execution_verified'] is False
            assert analysis.audit['pose_provider_calls'] == 1
    assert len(decoders) == len(requests) == 1
    assert not decoders[0]._cap.isOpened() and not decoders[0]._cache
    with pytest.raises(RuntimeError):
        requests[0].frames[8]
    unchanged(fixture)
    assert not list(fixture.scratch.glob('*'))
    OBSERVATIONS.append({'actual_synthetic_media': {
        'codec': 'mp4v', 'frame_count': 12, 'fps': 30., 'width': 1280, 'height': 960,
        'crop_global_frame_indices_read': list(range(5, 12)), 'maximum_cache_frames': 8,
        'decoder_closed': True, 'selected_model_execution': False}})


def test_actual_crop_decode_rejects_detection_pixel_dimensions_from_another_source(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.utils.video_frame_sequence import VideoFrameSequence
    fixture = build_shot_base(tmp_path, configured_players=True, playable_source=True,
                             metadata_size=(1920, 1080))
    decoders, actual_decoders = [], []
    initializer = VideoFrameSequence.__init__
    def observed_decoder(self, *args, **kwargs):
        initializer(self, *args, **kwargs)
        actual_decoders.append(self)
    monkeypatch.setattr(VideoFrameSequence, '__init__', observed_decoder)
    def mismatched_source_crop(request):
        try:
            request.frames[8]
        finally:
            # On failed first admission the owned decoder is still registered
            # with ExitStack, even if the lazy .sequence was never accepted.
            decoders.append(request.frames)
        pytest.fail('Mismatched source dimensions reached the pose feature provider')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=binding_amendment(fixture)) as rebuilt:
        with pytest.raises(ValueError, match='dimension|size|resolution|clock'):
            with relink_corrected_shots(rebuilt, fixture.scratch,
                                       pose_provider=mismatched_source_crop):
                pytest.fail('Incompatible crop source yielded analytics')
        assert not rebuilt.owner.sealed
        assert len(rebuilt.owner.member_rows()) == 0
    assert len(decoders) == 1
    assert len(actual_decoders) == 1
    assert not actual_decoders[0]._cap.isOpened() and not actual_decoders[0]._cache
    with pytest.raises(RuntimeError):
        len(decoders[0])
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)

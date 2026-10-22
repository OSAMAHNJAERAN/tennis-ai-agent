def build_sealed_base(tmp_path, *, frame_count=12, reviewed_let=False):
    import numpy as np
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.match_state import MatchState, BallPlayState, PointState
    from src.scoring.match_state_codec import encode_match_state
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.scoring.point_timeline import PointTimeline, ReviewedParticipantBinding, ReviewedRuleSignal, candidate_event_context, process_timeline_event
    from src.scoring.rule_replay import write_rule_replay
    from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.scoring_journal import ScoringJournal
    from tests.test_correction_sources import write_observation_assets
    directory = tmp_path / 'analysis'
    directory.mkdir()
    scratch = tmp_path / 'private-rebuild'
    scratch.mkdir()
    source = directory / 'source.mp4'
    source.write_bytes(b'EXPLICIT SYNTHETIC SOURCE; NO VIDEO DECODE OR MATCH ACCURACY CLAIM')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_frames = [{0: {'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'frame_index': index, 'timestamp_seconds': index / 30.0, 'source_pts': str(10 ** 18 + index * 100), 'source_time_base': '1/3000', 'source_frame_index': index, 'segment_id': 'synthetic-camera-0', 'analysis_allowed': True, 'measurement_transition_allowed': index > 0, 'anchor_frame_index': 0, 'review_evidence_id': 'EXPLICIT_SYNTHETIC_NORMAL_SPEED'}, 1: {'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'epoch': 0, 'revision': 0, 'observations': [{'track_key': 'track-A', 'player_id': 'A', 'state': 'CONFIRMED'}, {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}} for index in range(frame_count)]
    positions = [[300.0, 800.0 - 10.0 * abs(index - 4)] if index <= 8 else [300.0, 760.0] for index in range(frame_count)]
    homography = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
    files = write_observation_assets(directory, source_frames, homography=homography, positions=positions, fps=30.0)
    files['source.mp4'] = source
    config = directory / 'run_config.yaml'
    config.write_text('line_calling:\n  contact_refinement_window: 3\n', encoding='utf-8')
    files[config.name] = config
    replay_path = directory / 'native_rule_replay.json'
    initial = MatchState(match_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', serve_attempt=2, ball_state=BallPlayState.DEAD, point_state=PointState.SECOND_SERVE_WAITING, score_source='EXPLICIT_SYNTHETIC_REVIEWED_INITIAL_SCORE')
    points = [TemporalBallPoint(index, index / 30.0, *positions[index], confidence=0.99, state=BallState.DETECTED) for index in range(frame_count)]
    events = [SimpleNamespace(event_id=1, frame_index=0, timestamp_s=0.0, event_type=SimpleNamespace(value='SERVE_CONTACT'), player_id=1, confidence=0.99, ball_position_px=positions[0], court_position_m=None, trajectory_state='EXPLICIT_SYNTHETIC_OBSERVATION', evidence='SYNTHETIC_CONTACT'), SimpleNamespace(event_id=2, frame_index=4, timestamp_s=4 / 30.0, event_type=SimpleNamespace(value='BOUNCE'), player_id=None, confidence=0.99, ball_position_px=positions[4], court_position_m=None, trajectory_state='EXPLICIT_SYNTHETIC_OBSERVATION', evidence='SYNTHETIC_BOUNCE')]
    if reviewed_let:
        restart = copy.deepcopy(events[0])
        restart.event_id, restart.frame_index, restart.timestamp_s = (4, 3, 3 / 30.0)
        restart.ball_position_px = positions[3]
        signal = SimpleNamespace(event_id=3, frame_index=2, timestamp_s=2 / 30.0)
        events[1:1] = [signal, restart]
    with ScoringJournal(directory) as journal:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        with PointAnalysisSequence(directory, 'match_7a26c0de7a26c0de7a26c0de7a26c0de', revision=0, source_fingerprint=source_sha) as owner:
            timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
            evaluator = TennisLineCallEngine()
            for event in events:
                interval = source_frames[event.frame_index][0]
                binding = ReviewedParticipantBinding(event.frame_index, interval['segment_id'], 1, 1, 'A', 'track-A', 'NEAR_COURT', 0, 'synthetic-calibration', 0, 'EXPLICIT_SYNTHETIC_END_REVIEW', analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', source_fingerprint=source_sha, source_frame_index=interval['source_frame_index'], source_pts=interval['source_pts'], source_time_base=interval['source_time_base'], identity_epoch=0, participant_mapping_ref='EXPLICIT_SYNTHETIC_PARTICIPANT_MAP', geometry_qualification_ref='EXPLICIT_SYNTHETIC_GEOMETRY_ONLY')
                context = candidate_event_context(event, source_frames, lambda *_: binding, analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', source_fingerprint=source_sha)
                assert context.binding_current
                if reviewed_let and event.event_id == 3:
                    event = ReviewedRuleSignal(3, 2, 2 / 30.0, 'SERVICE_LET', 'EXPLICIT_SYNTHETIC_REVIEWED_LET', context)
                process_timeline_event(event, context, engine, timeline, evaluator, ball_points=points, homography_matrix=np.asarray(homography), frames=(), geometry_available=True)
            timeline.finish()
            assert engine.state.points_p2 == 0 and engine.state.score_confirmed
            write_rule_replay(replay_path, engine, owner, source_frames)
    files[replay_path.name] = replay_path
    with ResultStorage(directory) as storage:
        token = publish_analysis_generation(storage, {'summary': {'id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'frames': len(source_frames)}, 'analysisRevision': 0}, directory, files=files, analysis_revision=0)
    snapshot = resolve_generation(directory)
    assert snapshot.generation == token
    document = json.loads(replay_path.read_text(encoding='utf-8'))
    corrected_initial = encode_match_state(initial)
    corrected_initial['fields']['points_p2'] = 1
    corrected_initial['fields']['service_side']['value'] = 'AD'
    request = {'schema_version': 1, 'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'source_sha256': source_sha, 'base_generation': token, 'base_analysis_revision': 0, 'base_rule_replay_sha256': snapshot.file_sha256(replay_path.name), 'correction_id': 'SYNTHETIC-REVIEW-1', 'reviewer_id': 'SYNTHETIC-REVIEWER', 'evidence_ref': 'SYNTHETIC-SCOREBOARD-REFERENCE', 'reason': 'Explicit software fixture: initial receiver score was 15, not zero.', 'initial_state_amendment': {'expected_initial_state_sha256': canonical_sha(document['initial_state']), 'replacement_initial_state': corrected_initial}, 'input_amendments': []}
    return SimpleNamespace(directory=directory, scratch=scratch, source_sha=source_sha, request=request, snapshot=snapshot, document=document, marker=(directory / '.result/current.store').read_bytes(), manifest={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.files_directory.iterdir()}, events=events)

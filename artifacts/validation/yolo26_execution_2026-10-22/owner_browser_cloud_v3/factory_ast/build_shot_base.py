def build_shot_base(tmp_path, *, frame_count=12, pose_enabled=True, configured_players=False, receiver_hit=True, repeated_hits=False, config_tail='', playable_source=False, metadata_size=None):
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
    from src.scoring.point_timeline import PointTimeline, ReviewedParticipantBinding, candidate_event_context, process_timeline_event, native_match_analytics
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
        writer = cv2.VideoWriter(str(replacement_source), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 960))
        assert writer.isOpened()
        try:
            for index in range(frame_count):
                writer.write(np.full((960, 1280, 3), index * 16 % 240, dtype=np.uint8))
        finally:
            writer.release()
        _browser_source(replacement_source)
        fixture.source_sha = hashlib.sha256(replacement_source.read_bytes()).hexdigest()
        fixture.request['source_sha256'] = fixture.source_sha
    source_frames = [{0: row['interval'], 1: row['identity']} for row in fixture.document['source_frames']]
    initial = decode_match_state(fixture.document['initial_state'])
    detections = json.loads(fixture.snapshot.resolve_file('detections.json').read_text(encoding='utf-8'))
    if playable_source:
        width, height = metadata_size if metadata_size is not None else (1280, 960)
        detections['metadata'].update(width=width, height=height)
    positions = [row['ball']['position_px'] for row in detections['frames']]
    for row in detections['frames']:
        row['player_1'] = {'bbox': [100.1234567890123, 300.0, 200.0, 700.0], 'confidence': 0.9876543210123, 'source_track_id': 11}
        row['player_2'] = {'bbox': [100.0, 100.0, 200.0, 400.0], 'confidence': 0.9765432101234, 'source_track_id': 22}
    detection_path = directory / 'detections_with_players.json'
    detection_path.write_text(json.dumps(detections, allow_nan=False), encoding='utf-8')
    config_path = directory / 'shot_run_config.yaml'
    config = 'model_system:\n  family: yolo26\n  vision_model_count: 3\nplayer_detection:\n  model: explicit-synthetic-selected-player-yolo26s-pose.pt\nline_calling:\n  contact_refinement_window: 3\npose_estimation:\n  enabled: ' + ('true' if pose_enabled else 'false') + '\n  source: player_model\n  min_pose_confidence: 0.35\n  model_path: explicit-synthetic-selected-player-yolo26s-pose.pt\nshot_classification:\n  ambiguity_threshold: 0.15\n'
    if configured_players:
        config += '  players_handedness:\n    2: RIGHT_HANDED\n  court_orientation_sign:\n    2: 1.0\n'
    config_path.write_text(config + config_tail, encoding='utf-8')
    points = [TemporalBallPoint(index, index / 30.0, *positions[index], confidence=0.99, state=BallState.DETECTED) for index in range(frame_count)]
    events = [TennisEvent(1, EventType.SERVE_CONTACT, 0, 0.0, 1, tuple(positions[0]), (88.0, 88.0), 0.99, 'EXPLICIT_SYNTHETIC_OBSERVATION', {'match_7a26c0de7a26c0de7a26c0de7a26c0de': 'synthetic-contact'}), TennisEvent(2, EventType.BOUNCE, 4, 4 / 30.0, None, tuple(positions[4]), (99.0, 99.0), 0.99, 'EXPLICIT_SYNTHETIC_OBSERVATION', {'match_7a26c0de7a26c0de7a26c0de7a26c0de': 'synthetic-bounce'})]
    if receiver_hit:
        for index, frame in enumerate(range(8, frame_count, 8) if repeated_hits else [8]):
            role = 2 if index % 2 == 0 else 1
            events.append(TennisEvent(3 + index, EventType.PLAYER_2_HIT if role == 2 else EventType.PLAYER_1_HIT, frame, frame / 30.0, role, tuple(positions[frame]), (77.0, 77.0), 0.99, 'EXPLICIT_SYNTHETIC_OBSERVATION', {'match_7a26c0de7a26c0de7a26c0de7a26c0de': 'synthetic-receiver-hit'}))
    replay_path = directory / 'native_shot_rule_replay.json'
    p1 = [BBox(100.1234567890123, 300.0, 200.0, 700.0, 0.9876543210123, track_id=11) for _ in source_frames]
    p2 = [BBox(100.0, 100.0, 200.0, 400.0, 0.9765432101234, track_id=22) for _ in source_frames]
    with ScoringJournal(fixture.scratch) as journal:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        with PointAnalysisSequence(fixture.scratch, 'match_7a26c0de7a26c0de7a26c0de7a26c0de', revision=0, source_fingerprint=fixture.source_sha) as owner, EventAnalysisSequence(fixture.scratch) as event_owner, ShotEvidenceSequence(fixture.scratch) as shots:
            timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
            evaluator = TennisLineCallEngine()
            for event in events:
                interval = source_frames[event.frame_index][0]
                role = 2 if event.player_id == 2 else 1
                binding = ReviewedParticipantBinding(event.frame_index, interval['segment_id'], role, role, 'B' if role == 2 else 'A', 'track-B' if role == 2 else 'track-A', 'FAR_COURT' if role == 2 else 'NEAR_COURT', 0, 'synthetic-calibration', 0, 'EXPLICIT_SYNTHETIC_END_REVIEW', analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', source_fingerprint=fixture.source_sha, source_frame_index=interval['source_frame_index'], source_pts=interval['source_pts'], source_time_base=interval['source_time_base'], identity_epoch=0, participant_mapping_ref='EXPLICIT_SYNTHETIC_PARTICIPANT_MAP', geometry_qualification_ref='EXPLICIT_SYNTHETIC_GEOMETRY_ONLY')
                context = candidate_event_context(event, source_frames, lambda *_: binding, analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', source_fingerprint=fixture.source_sha)
                _, _, call = process_timeline_event(event, context, engine, timeline, evaluator, ball_points=points, homography_matrix=np.diag([0.01, 0.01, 1.0]), frames=(), geometry_available=True)
                event_owner.events.append(event if call is None else TennisEvent(event.event_id, event.event_type, event.frame_index, event.timestamp_s, event.player_id, event.ball_position_px, tuple(call.bounce_position_m), event.confidence, event.trajectory_state, event.evidence))
            TennisShotLinker(TennisShotClassifier()).link_shots(event_owner.events, points, p1, p2, dead_event_ids=owner.dead_event_ids(), result_output=shots, event_output=event_owner)
            timeline.attach_shots(shots)
            shots.seal()
            timeline.finish()
            analytics = native_match_analytics(owner, shots, engine.state, authoritative=True)
            original = {'shots': [shot.to_dict() for shot in shots], 'overview': analytics['overview'], 'players': analytics['players'], 'serve_analytics': analytics['serve_analytics'], 'landings': list(analytics['ball_landing_heatmap']), 'members': list(owner.member_rows())}
            write_rule_replay(replay_path, engine, owner, source_frames)
    files = {path.name: path for path in fixture.snapshot.files_directory.iterdir() if path.name not in {'result.json', 'summary.json'}}
    files.update({'native_rule_replay.json': replay_path, 'detections.json': detection_path, 'run_config.yaml': config_path})
    if replacement_source is not None:
        files['source.mp4'] = replacement_source
    with ResultStorage(directory) as storage:
        token = publish_analysis_generation(storage, {'summary': {'id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'frames': len(source_frames)}, 'analysisRevision': 0}, directory, files=files, analysis_revision=0, expected_generation=fixture.snapshot.generation)
    snapshot = resolve_generation(directory)
    document = json.loads(replay_path.read_text(encoding='utf-8'))
    fixture.request.update(base_generation=token, base_rule_replay_sha256=snapshot.file_sha256('native_rule_replay.json'))
    fixture.request['initial_state_amendment']['expected_initial_state_sha256'] = canonical_sha(document['initial_state'])
    fixture.snapshot, fixture.document, fixture.original = (snapshot, document, original)
    fixture.marker = (directory / '.result/current.store').read_bytes()
    fixture.manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.files_directory.iterdir()}
    return fixture

def build_complete_base(tmp_path, *, frame_count=12, omit=(), provenance_present=True, provenance_change=None, with_optional=False):
    """Publish actual native-format complete observations plus original bytes.

    Original rule/shot/timeline assets are written from the real verified
    unchanged reproducer. Null metric/pose/racket evidence remains unavailable.
    The original annotation slot is an explicitly synthetic unannotated video;
    every correction must replace it with actual freshly rendered overlays.
    """
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.analysis_resume import REQUIRED_PIPELINE_FILES
    from src.api.result_storage import ResultStorage
    from src.api.results import build_result, IDENTITY_SCOPE, PARTICIPANT_SEMANTICS
    from src.court.court_geometry import TennisCourtGeometry
    from src.scoring.corrected_exports import _physical_events
    from src.scoring.point_timeline import native_match_analytics
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    from src.scoring.rule_replay_source import _source_event
    from src.utils.analytics_json_artifact import write_membership_analytics, write_match_analytics
    from src.utils.json_array_artifact import write_json_array_artifact
    from src.utils.point_timeline_artifact import publish_point_timeline_manifest
    fixture = build_shot_base(tmp_path, frame_count=frame_count, repeated_hits=frame_count > 12, playable_source=True)
    directory = fixture.directory / 'complete-original-assets'
    directory.mkdir()
    files = {path.name: path for path in fixture.snapshot.files_directory.iterdir() if path.name not in {'result.json', 'summary.json'}}
    for name, path in tuple(files.items()):
        destination = directory / name
        shutil.copyfile(path, destination)
        files[name] = destination
    detections = read(files['detections.json'])
    count, fps = (frame_count, 30.0)
    samples = [{'frame_index': index, 'timestamp_seconds': index / fps, 'ground_position_m': None, 'speed_kmh': None, 'position_fit_rmse_m': None} for index in range(count)]
    trajectories = [{'frame_index': index, 'timestamp_seconds': index / fps, 'x_px': frame['ball']['position_px'][0], 'y_px': frame['ball']['position_px'][1], 'court_x_m': None, 'court_y_m': None, 'speed_kmh': None, 'confidence': frame['ball']['confidence'], 'state': frame['ball']['state'], 'source': 'EXPLICIT_SYNTHETIC_IMAGE_OBSERVATION'} for index, frame in enumerate(detections['frames'])]
    court = read(files['court_geometry.json'])
    canonical = TennisCourtGeometry.get_canonical_keypoints().tolist()
    court.update(reprojection_error_px=0.25, canonical_keypoints_m=canonical, court_keypoints_px=[[100 * x, 100 * y] for x, y in canonical])
    files['court_geometry.json'] = object_file(directory, 'court_geometry.json', court)
    files['trajectories.json'] = object_file(directory, 'trajectories.json', {'schema_version': '1.0', 'ball_trajectory': trajectories})
    files['player_motion.json'] = object_file(directory, 'player_motion.json', {'schema_version': '2.0', 'metric_measurements_available': False, 'ground_position_source': 'UNAVAILABLE_UNQUALIFIED_SYNTHETIC_FIXTURE', 'pose_frames': [], **{key: {'samples': samples, 'observed_distance_m': None, 'mean_speed_kmh': None, 'max_speed_kmh': None} for key in ('player_1', 'player_2')}})
    files['racket_tracking.json'] = object_file(directory, 'racket_tracking.json', {'schema_version': '1.0', 'enabled': False, 'frames': []})
    people = []
    for index, frame in enumerate(detections['frames']):
        people.append({'frame_index': index, 'timestamp_seconds': index / fps, 'people': [{'bbox': frame['player_' + str(role)]['bbox'], 'confidence': frame['player_' + str(role)]['confidence'], 'sourceTrackId': frame['player_' + str(role)]['source_track_id'], 'courtRole': role, 'participantStatus': 'COURT_ROLE_CANDIDATE'} for role in (1, 2)]})
    files['image_people.json'] = object_file(directory, 'image_people.json', {'schema_version': '1.0', 'coordinateSystem': 'normalized_video_pixels', 'identityScope': IDENTITY_SCOPE, 'participantSemantics': PARTICIPANT_SEMANTICS, 'accuracyValidated': False, 'frames': people})
    files['media_metadata.json'] = object_file(directory, 'media_metadata.json', {'schema_version': '1.0', 'original': {'width': 1280, 'height': 960, 'fps': fps, 'frames': count, 'durationSeconds': count / fps, 'sourceSha256': fixture.source_sha}, 'normalization': {'status': 'EXPLICIT_SYNTHETIC_ALREADY_CFR_FIXTURE'}, 'timeMapping': {'status': 'FRAME_MAPPING_UNAVAILABLE', 'validMeasurementTransitionIntervals': []}})
    inference = directory / 'inference_config.yaml'
    shutil.copyfile(files['run_config.yaml'], inference)
    files['inference_config.yaml'] = inference
    for name, key in (('event_candidates.json', 'event_candidates'), ('event_verification_trace.json', 'event_verification_trace'), ('player_metrics.json', 'player_metrics'), ('ball_metrics.json', 'ball_metrics'), ('temporal_spacing_audit.json', 'records'), ('temporal_detour_audit.json', 'records'), ('camera_intervals.json', 'intervals'), ('match_identity.json', 'frames'), ('stationary_candidate_audit.json', 'records'), ('pixel_motion_audit.json', 'records'), ('patch_persistence_audit.json', 'records')):
        files[name] = object_file(directory, name, {'schema_version': '1.0', 'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'scope': 'EXPLICIT_SYNTHETIC_EMPTY_AUDIT_NOT_QUALIFIED', 'independent_accuracy_validated': False, key: []})
    files['camera_intervals.json'] = object_file(directory, 'camera_intervals.json', {'schema_version': '1.0', 'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'frames': [row['interval'] for row in fixture.document['source_frames']]})
    files['match_identity.json'] = object_file(directory, 'match_identity.json', {'schema_version': '1.0', 'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'frames': [row['identity'] for row in fixture.document['source_frames']]})
    replay_scratch = fixture.scratch / 'original-native-export'
    replay_scratch.mkdir()
    with reproduce_rule_timeline(files['native_rule_replay.json'], replay_scratch, artifact_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'), source_sha256=fixture.source_sha, analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', analysis_revision=0) as replay:
        owner, engine = (replay.owner, replay.engine)
        analytics = native_match_analytics(owner, replay.shots, engine.state, authoritative=True)
        for name, kind in (('point_analytics.json', 'points'), ('rallies.json', 'rallies')):
            files[name] = directory / name
            write_membership_analytics(files[name], kind, owner.rows(kind), lambda value: value, schema_version='2.0')
        files['match_analytics.json'] = directory / 'match_analytics.json'
        write_match_analytics(files['match_analytics.json'], analytics, lambda value: value)
        physical = []
        for row in owner.rows('replay_inputs'):
            event, _ = _source_event(row['source_input'], replay.source_frames, 'match_7a26c0de7a26c0de7a26c0de7a26c0de', fixture.source_sha)
            if row['source_input']['reviewed_evidence_ref'] is None:
                physical.append(event)
        for name, key, rows, metadata in (('serve_executions.json', 'serve_executions', owner.rows('attempts'), {'schema_version': '2.0', 'analysis_revision': 0}), ('shot_play_scope.json', 'shots', owner.member_rows(), {'schema_version': '2.0', 'analysis_revision': 0}), ('scoring_events.json', 'scoring_events', owner.rows('transitions'), {'schema_version': '2.0'}), ('shot_events.json', 'shot_events', (shot.to_dict() for shot in replay.shots), {'schema_version': '1.0'}), ('line_calls.json', 'line_calls', (call.to_dict() for call in engine.native_line_calls()), {'schema_version': '1.0'}), ('match_events.json', 'events', _physical_events((event for event in physical)), {'schema_version': '1.0', 'analysis_id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'analysis_revision': 0, 'events_count': len(physical)})):
            files[name] = directory / name
            write_json_array_artifact(files[name], metadata, key, rows, allow_nan=False)
        files['score_history.json'] = directory / 'score_history.json'
        engine.save_history_json(files['score_history.json'])
        files['match_state.json'] = object_file(directory, 'match_state.json', {'schema_version': '1.0', **engine.state.to_dict(), 'authoritative_events_enabled': True, 'semantic_role': 'INFERRED_SCORING_STATE'})
    publish_point_timeline_manifest(directory, analysis_id='match_7a26c0de7a26c0de7a26c0de7a26c0de', analysis_revision=0, source_sha256=fixture.source_sha, scoring_revision=0, authoritative_events_enabled=True)
    files['point_timeline_manifest.json'] = directory / 'point_timeline_manifest.json'
    files['metrics.json'] = object_file(directory, 'metrics.json', {'schema_version': '1.0', 'total_frames': count, 'processing_fps': 3.14159, 'authoritative_events_enabled': True, 'production_qualified': False, 'ball_backend': 'EXPLICIT_SYNTHETIC_FIXTURE', 'tracking_strategy': 'EXPLICIT_SYNTHETIC_RETAINED_OBSERVATIONS'})
    for name in ('annotated.mp4', 'original.mp4', 'annotated_web.mp4'):
        destination = directory / name
        shutil.copyfile(files['source.mp4'], destination)
        files[name] = destination
    if with_optional:
        from src.api.media_timing import frame_mapping
        stats = directory / 'normalization_frame_stats.txt'
        stats.write_text(''.join((f'{index} {index} {10 ** 18 + index * 100} 1/3000 {index} 1/30\n' for index in range(count))), encoding='utf-8')
        mapping = directory / 'source_frame_mapping.jsonl'
        frame_mapping(stats, mapping, fps, source_frame_count=count)
        files[stats.name], files[mapping.name] = (stats, mapping)
        files['pose_tracking.json'] = object_file(directory, 'pose_tracking.json', {'schema_version': '1.0', 'pose_frames': [], 'scope': 'EXPLICIT_SYNTHETIC_NO_POSE_OBSERVATIONS'})
        source_credit = {'id': 'explicit-synthetic-source', 'title': 'EXPLICIT SYNTHETIC FIXTURE; NO REAL MEDIA RIGHTS ASSERTION', 'author': 'Synthetic <author> & fixture', 'sourcePage': 'https://commons.wikimedia.org/wiki/File:Synthetic_fixture_only.webm', 'license': 'CC BY 4.0', 'licenseUrl': 'https://creativecommons.org/licenses/by/4.0/', 'derivativeLicense': 'CC BY 4.0', 'modifications': 'EXPLICIT SYNTHETIC original test transform.', 'sourceSha256': fixture.source_sha, 'inputSha256': fixture.source_sha}
        files['media_source.json'] = object_file(directory, 'media_source.json', source_credit)
        files['annotated_web_credit.json'] = object_file(directory, 'annotated_web_credit.json', {'schemaVersion': 1, 'analysisId': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'inputSha256': fixture.source_sha, 'sourceSha256': fixture.source_sha, 'annotatedSha256': '0' * 64, 'modifications': 'STALE_OLD_ANNOTATION_MUST_NOT_BE_COPIED'})
    assert set(REQUIRED_PIPELINE_FILES) <= set(files)
    job = {'id': 'match_7a26c0de7a26c0de7a26c0de7a26c0de', 'filename': 'SYNTHETIC SOFTWARE TEST - correction workflow.mp4', 'extension': '.mp4', 'owner_id': 'synthetic-owner', 'started_at': '2026-10-18T03:00:00+00:00', 'completed_at': '2026-10-18T03:00:01+00:00'}
    with ResultStorage(fixture.directory) as storage:
        original_result = build_result(directory, job, _storage=storage)
        provenance = copy.deepcopy(original_result['provenance'])
        provenance.update(bundleVersion='EXPLICIT_SYNTHETIC_CAPTURED_INFERENCE', bundleSha256='d' * 64, configSha256=hashlib.sha256(inference.read_bytes()).hexdigest(), checkpointHashes={role: {'filename': 'explicit-synthetic-' + role + '.pt', 'sha256': digit * 64} for role, digit in (('players', 'a'), ('ball', 'b'), ('court', 'c'))})
        provenance.update(provenance_change or {})
        if provenance_present:
            original_result['provenance'] = provenance
        else:
            original_result.pop('provenance', None)
        token = publish_analysis_generation(storage, original_result, fixture.directory, files={name: path for name, path in files.items() if name not in set(omit)}, analysis_revision=0, expected_generation=fixture.snapshot.generation)
    fixture.snapshot = resolve_generation(fixture.directory)
    fixture.request.update(base_generation=token, base_rule_replay_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'))
    fixture.marker = (fixture.directory / '.result/current.store').read_bytes()
    fixture.manifest = digests(fixture.snapshot.files_directory)
    fixture.job, fixture.inference_provenance = (job, provenance if provenance_present else None)
    return fixture

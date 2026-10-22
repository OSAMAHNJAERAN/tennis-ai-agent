import { RawInitialStateSchema, ReviewSourceSchema, type CorrectionStatus } from '../data/correctionApi';
import { NativeForecastSchema, type NativeForecastRow } from '../data/nativeForecast';
export const analysisId = 'match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
export const generation = 'b'.repeat(64); export const sourceHash = 'c'.repeat(64);
export function reviewFixture() {
  const initial = RawInitialStateSchema.parse({ schema_version: 1, type: 'MatchState', fields: {
    match_id: analysisId, format: { enum: 'MatchFormat', value: 'BEST_OF_3' }, set_format: { enum: 'SetFormat', value: 'TIE_BREAK_SET' },
    player_1_id: 1, player_2_id: 2, server_id: 1, receiver_id: 2, initial_server_id: 1, current_set: 1, current_game: 1,
    sets_won_p1: 0, sets_won_p2: 0, games_p1: 0, games_p2: 0, points_p1: 0, points_p2: 0, serve_attempt: 1,
    service_side: { enum: 'ServiceSide', value: 'DEUCE' }, tie_break_active: false, tie_break_points_p1: 0, tie_break_points_p2: 0,
    point_state: { enum: 'PointState', value: 'WAITING_FOR_SERVE' }, ball_state: { enum: 'BallPlayState', value: 'NOT_STARTED' },
    completed_sets: [], match_complete: false, winner_id: null, last_point_winner: null, last_point_reason: '', state_version: 0,
    tie_break_points_to_win: 7, deciding_set_tie_break_points_to_win: null, deciding_set_format: null,
    player_1_court_end: { enum: 'ServerCourtEnd', value: 'NEAR_COURT' }, initial_state_confirmed: true, score_confirmed: true,
    score_source: 'configured_initial_state', match_status: 'IN_PROGRESS',
  } });
  return ReviewSourceSchema.parse({ schema_version: 1, analysis_id: analysisId, source_sha256: sourceHash, base_generation: generation,
    base_analysis_revision: 0, base_rule_replay_sha256: 'd'.repeat(64), initial_state: initial, initial_state_sha256: 'e'.repeat(64),
    inputs: [{ input_fingerprint: 'f'.repeat(64), source_input: { event_id: 4, frame_index: 10, timestamp_s: 1, player_id: 1,
      confidence: .8, ball_position_px: [12, 22], court_position_m: null, trajectory_state: 'DETECTED', evidence: { original: true },
      event_type: 'SERVE_CONTACT', reviewed_evidence_ref: null, context: { frame_index: 10, source_pts: '9007199254740993',
        source_time_base: '1/10', source_frame_index: 10, segment_id: 'seg-1', analysis_allowed: true, review_evidence_ref: 'review-1',
        binding: null, binding_audit_matches: false, anchor_frame_index: null } } }],
    offset: 0, limit: 20, total_inputs: 1, next_offset: null, is_current: true, measurement_edits_supported: false, independent_accuracy_validated: false });
}
export function correctionStatus(status: CorrectionStatus['status'] = 'QUEUED'): CorrectionStatus {
  return { id: 'correction_' + 'a'.repeat(32), analysis_id: analysisId, status, stage: 'Waiting for worker', step: 0,
    attempt: 0, created_at: '2026-10-18T00:00:00Z', started_at: null, completed_at: null, error: null, cancel_requested: false,
    base_generation: generation, base_analysis_revision: 0, analysis_revision: 1, generation: status === 'COMPLETED' ? 'a'.repeat(64) : null };
}
export function nativeFixture() {
  const forecast = { schema_version: '1.0', match_id: analysisId, revision: 1, score_state_version: 0, timestamp_seconds: 0,
    source: 'statistical_tennis_rules', calibration_status: 'not_calibrated', status: 'prior_only', sample_count: 0, reason: 'Explicit neutral prior.', winner_id: null,
    probabilities: { game: { player_1: .5, player_2: .5 }, set: { player_1: .5, player_2: .5 }, match: { player_1: .5, player_2: .5 } },
    serve_estimates: [1, 2].map(player_id => ({ player_id, probability: .5, points_won: 0, points_played: 0, alpha: 2, beta: 2 })),
    prior: { alpha: 2, beta: 2, source: 'explicit_neutral_engineering_prior' }, assumptions: 'Independent stationary points.' };
  const first = { schema_version: '1.0', analysis_id: analysisId, analysis_revision: 1, source_sha256: sourceHash,
    base_generation: generation, model_signature: 'd'.repeat(64), phase: 'INITIAL', event_id: null, event_uid: null, point_uid: null,
    frame_index: 0, timestamp_seconds: 0, source_pts: '9007199254740993', source_time_base: '1/10', source_frame_index: 0,
    uses_points_through_event_id: null, raw_state_sha256: 'e'.repeat(64), forecast };
  const before = { ...first, phase: 'BEFORE_EVENT', event_id: 4, event_uid: 'event-4', frame_index: 10, source_frame_index: 10,
    timestamp_seconds: 1, source_pts: '9007199254741003', forecast: { ...forecast, timestamp_seconds: 1 } };
  const after = { ...before, phase: 'AFTER_EVENT', forecast: { ...before.forecast, status: 'unavailable', reason: 'Point requires review.', probabilities: null } };
  return NativeForecastSchema.parse({ schemaVersion: '1.0', status: 'UNQUALIFIED', analysisId, analysisRevision: 1,
    sourceSha256: sourceHash, baseGeneration: generation, modelSignature: 'd'.repeat(64), calibrationStatus: 'not_calibrated',
    independentAccuracyValidated: false, humanReviewDependent: true, modelSpec: {}, historyPolicy: 'INITIAL_AND_BEFORE_AFTER_EVERY_ACCEPTED_EVENT',
    current: after.forecast, history: [first, before, after] });
}
export function cloneRows(rows: NativeForecastRow[]) { return structuredClone(rows); }

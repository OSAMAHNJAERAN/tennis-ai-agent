import { z } from 'zod';
import { apiJson, ApiError } from './analysisApi';

const count = z.number().int().nonnegative().safe();
const positive = z.number().int().positive().safe();
const hash = z.string().regex(/^[a-f0-9]{64}$/);
const tagged = <T extends string, V extends [string, ...string[]]>(name: T, values: V) => z.object({ enum: z.literal(name), value: z.enum(values) }).strict();
const setFormat = tagged('SetFormat', ['TIE_BREAK_SET', 'ADVANTAGE_SET']);
export const RawInitialStateSchema = z.object({ schema_version: z.literal(1), type: z.literal('MatchState'), fields: z.object({
  match_id: z.string().min(1), format: tagged('MatchFormat', ['BEST_OF_3', 'BEST_OF_5']), set_format: setFormat,
  player_1_id: positive, player_2_id: positive, server_id: positive, receiver_id: positive, initial_server_id: positive,
  current_set: positive, current_game: positive, sets_won_p1: count, sets_won_p2: count,
  games_p1: count, games_p2: count, points_p1: count, points_p2: count,
  serve_attempt: z.union([z.literal(1), z.literal(2)]), service_side: tagged('ServiceSide', ['DEUCE', 'AD']),
  tie_break_active: z.boolean(), tie_break_points_p1: count, tie_break_points_p2: count,
  point_state: tagged('PointState', ['WAITING_FOR_SERVE', 'FIRST_SERVE_IN_PROGRESS', 'SECOND_SERVE_WAITING', 'SECOND_SERVE_IN_PROGRESS', 'RALLY_LIVE', 'POINT_COMPLETE', 'POINT_REVIEW_PENDING']),
  ball_state: tagged('BallPlayState', ['NOT_STARTED', 'SERVE_STARTED', 'LIVE', 'DEAD', 'LET_REPLAY', 'POINT_COMPLETE']),
  completed_sets: z.array(z.object({ type: z.literal('CompletedSetScore'), fields: z.object({ set_number: positive,
    games_p1: count, games_p2: count, tie_break_p1: count.nullable(), tie_break_p2: count.nullable(), winner_id: positive }).strict() }).strict()),
  match_complete: z.boolean(), winner_id: positive.nullable(), last_point_winner: positive.nullable(), last_point_reason: z.string(),
  state_version: count, tie_break_points_to_win: positive, deciding_set_tie_break_points_to_win: positive.nullable(),
  deciding_set_format: setFormat.nullable(), player_1_court_end: tagged('ServerCourtEnd', ['NEAR_COURT', 'FAR_COURT']),
  initial_state_confirmed: z.boolean(), score_confirmed: z.boolean(), score_source: z.string(), match_status: z.string(),
}).strict() }).strict();
const exactPts = z.union([z.string().regex(/^-?(0|[1-9]\d*)$/), z.number().int().safe()]);
const binding = z.object({ participant_id: positive, court_end: z.enum(['NEAR_COURT', 'FAR_COURT']) }).passthrough();
export const SourceInputSchema = z.object({
  event_id: count, frame_index: count, timestamp_s: z.number().finite().nonnegative(), player_id: positive.nullable(),
  confidence: z.number().finite().min(0).max(1).nullable(), ball_position_px: z.tuple([z.number().finite(), z.number().finite()]).nullable(),
  court_position_m: z.tuple([z.number().finite(), z.number().finite()]).nullable(), trajectory_state: z.string().nullable(),
  evidence: z.unknown(), event_type: z.string(), reviewed_evidence_ref: z.string().nullable(),
  context: z.object({ frame_index: count, source_pts: exactPts.nullable(), source_time_base: z.string().nullable(),
    source_frame_index: count.nullable(), segment_id: z.string().nullable(), analysis_allowed: z.boolean(),
    review_evidence_ref: z.string().nullable(), binding: binding.nullable(), binding_audit_matches: z.boolean(), anchor_frame_index: count.nullable() }).passthrough(),
}).passthrough();
export const ReviewSourceSchema = z.object({
  schema_version: z.literal(1), analysis_id: z.string(), source_sha256: hash, base_generation: hash,
  base_analysis_revision: count, base_rule_replay_sha256: hash, initial_state: RawInitialStateSchema,
  initial_state_sha256: hash, inputs: z.array(z.object({ source_input: SourceInputSchema, input_fingerprint: hash })).max(50),
  offset: count, limit: positive.max(50), total_inputs: count, next_offset: count.nullable(), is_current: z.boolean(),
  measurement_edits_supported: z.literal(false), independent_accuracy_validated: z.literal(false),
});
export const CorrectionStatusSchema = z.object({
  id: z.string().regex(/^correction_[a-f0-9]{32}$/), analysis_id: z.string(),
  status: z.enum(['QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'CONFLICT', 'CANCELLED']), stage: z.string(), step: count,
  attempt: count, created_at: z.string(), started_at: z.string().nullable(), completed_at: z.string().nullable(), error: z.string().nullable(),
  cancel_requested: z.boolean(), base_generation: hash, base_analysis_revision: count, analysis_revision: count.nullable(), generation: hash.nullable(),
}).superRefine((row, ctx) => {
  if ((row.analysis_revision !== null && row.analysis_revision !== row.base_analysis_revision + 1) ||
      (row.status === 'COMPLETED' && (row.generation === null || row.analysis_revision === null)) ||
      (row.status !== 'COMPLETED' && row.generation !== null)) ctx.addIssue({ code: 'custom', message: 'Correction completion and generation disagree.' });
});
export type ReviewSource = z.infer<typeof ReviewSourceSchema>;
export type SourceInput = z.infer<typeof SourceInputSchema>;
export type RawInitialState = z.infer<typeof RawInitialStateSchema>;
export type CorrectionStatus = z.infer<typeof CorrectionStatusSchema>;
export interface CorrectionCommand {
  schema_version: 1; correction_id: string; base_generation: string; base_analysis_revision: number;
  source_sha256: string; base_rule_replay_sha256: string; reason: string; evidence_ref: string;
  initial_state_amendment: null | { expected_initial_state_sha256: string; replacement_initial_state: RawInitialState };
  input_amendments: { event_id: number; expected_input_fingerprint: string; replacement_source_input: SourceInput }[];
}
const nativeText = (maximum: number) => z.string().min(1).max(maximum).refine(value => !!value.trim() && !Array.from(value).some(char => char.charCodeAt(0) < 32));
export const CorrectionCommandSchema = z.object({ schema_version: z.literal(1), correction_id: nativeText(128),
  base_generation: hash, base_analysis_revision: count, source_sha256: hash, base_rule_replay_sha256: hash,
  reason: nativeText(4000), evidence_ref: nativeText(2000),
  initial_state_amendment: z.object({ expected_initial_state_sha256: hash, replacement_initial_state: RawInitialStateSchema }).nullable(),
  input_amendments: z.array(z.object({ event_id: count, expected_input_fingerprint: hash, replacement_source_input: SourceInputSchema })).max(128),
}).strict();
const path = (id: string) => `/api/analyses/${encodeURIComponent(id)}`;
function decode<S extends z.ZodType>(schema: S, value: unknown, message: string): z.output<S> {
  const parsed = schema.safeParse(value); if (!parsed.success) throw new Error(message); return parsed.data;
}
function checkStatus(value: unknown, id: string) { const parsed = decode(CorrectionStatusSchema, value, 'The server returned an invalid correction status. Refresh the saved history before retrying.'); if (parsed.analysis_id !== id) throw new Error('Correction belongs to a different analysis.'); return parsed; }
async function post(url: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  const value: unknown = await response.json().catch(() => null);
  if (!response.ok) { const detail = value && typeof value === 'object' && 'detail' in value ? value.detail : null;
    throw new ApiError(typeof detail === 'string' ? detail : 'Correction request failed. Retry the same request after reconnecting.', response.status); }
  return value;
}
export const correctionApi = {
  async session(signal?: AbortSignal) { return decode(z.object({ authenticated: z.boolean(), required: z.boolean() }), await apiJson('/api/auth/session', signal), 'Account access could not be checked. Reconnect before reviewing this analysis.'); },
  async source(id: string, generation: string, offset = 0, signal?: AbortSignal) {
    const source = decode(ReviewSourceSchema, await apiJson(`${path(id)}/review-source?generation=${generation}&offset=${offset}&limit=20`, signal), 'The saved review evidence is incomplete or unsupported. Your replay remains available.');
    if (source.analysis_id !== id || source.base_generation !== generation || source.initial_state.fields.match_id !== id) throw new Error('Review evidence belongs to a different analysis generation.');
    return source;
  },
  async list(id: string, signal?: AbortSignal) { return decode(z.array(CorrectionStatusSchema), await apiJson(`${path(id)}/corrections`, signal), 'Saved correction history could not be verified. Reconnect before submitting.').map(row => checkStatus(row, id)); },
  async status(id: string, command: string, signal?: AbortSignal) { return checkStatus(await apiJson(`${path(id)}/corrections/${encodeURIComponent(command)}`, signal), id); },
  async submit(id: string, command: CorrectionCommand, signal?: AbortSignal) { return checkStatus(await post(`${path(id)}/corrections`, command, signal), id); },
  async cancel(id: string, command: string, signal?: AbortSignal) { return checkStatus(await post(`${path(id)}/corrections/${encodeURIComponent(command)}/cancel`, {}, signal), id); },
};

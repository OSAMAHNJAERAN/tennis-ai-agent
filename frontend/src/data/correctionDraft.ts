import type { CorrectionCommand, RawInitialState, ReviewSource, SourceInput } from './correctionApi';

export const PHYSICAL_TYPES = ['SERVE_CONTACT', 'PLAYER_1_HIT', 'PLAYER_2_HIT', 'BOUNCE', 'UNKNOWN_EVENT'] as const;
export const REVIEWED_TYPES = ['SERVICE_LET', 'CONSEQUENTIAL_GAP', 'POINT_OUTCOME_UNKNOWN'] as const;
export function initialEditable(raw: RawInitialState): boolean {
  const f = raw.fields;
  return !f.match_complete && f.winner_id === null && f.match_status === 'IN_PROGRESS' &&
    ['WAITING_FOR_SERVE', 'SECOND_SERVE_WAITING', 'POINT_REVIEW_PENDING'].includes(f.point_state.value) &&
    (!f.tie_break_active ? f.points_p1 <= 4 && f.points_p2 <= 4 : f.games_p1 === 6 && f.games_p2 === 6);
}
export interface ScoreDraft { p1: number; p2: number; server: number; serve: 1 | 2 }
export function initialScoreDraft(raw: RawInitialState): ScoreDraft {
  const f = raw.fields; return { p1: f.tie_break_active ? f.tie_break_points_p1 : f.points_p1,
    p2: f.tie_break_active ? f.tie_break_points_p2 : f.points_p2, server: f.tie_break_active ? f.initial_server_id : f.server_id, serve: f.serve_attempt };
}
export function amendInitial(source: ReviewSource, draft: ScoreDraft): RawInitialState {
  if (!initialEditable(source.initial_state)) throw new Error('This initial playing context is read-only.');
  const raw = structuredClone(source.initial_state); const f = raw.fields;
  if (![draft.p1, draft.p2].every(n => Number.isSafeInteger(n) && n >= 0 && n <= 1000)) throw new Error('Enter whole point counts from 0 to 1000.');
  if (![f.player_1_id, f.player_2_id].includes(draft.server)) throw new Error('Choose a match participant as server.');
  if (f.tie_break_active) {
    const target = f.deciding_set_tie_break_points_to_win && f.current_set === (f.format.value === 'BEST_OF_3' ? 3 : 5) ? f.deciding_set_tie_break_points_to_win : f.tie_break_points_to_win;
    if (Math.max(draft.p1, draft.p2) >= target && Math.abs(draft.p1 - draft.p2) >= 2) throw new Error('A completed tie-break needs a full set transition and is read-only here.');
    f.tie_break_points_p1 = draft.p1; f.tie_break_points_p2 = draft.p2; f.initial_server_id = draft.server;
    f.server_id = Math.floor((draft.p1 + draft.p2 + 1) / 2) % 2 === 0 ? draft.server : (draft.server === f.player_1_id ? f.player_2_id : f.player_1_id);
  } else {
    if (Math.max(draft.p1, draft.p2) > 4 || (Math.max(draft.p1, draft.p2) === 4 && Math.min(draft.p1, draft.p2) !== 3)) throw new Error('Advantage requires 40 for the other player; completed games require a full transition.');
    f.points_p1 = draft.p1; f.points_p2 = draft.p2; f.server_id = draft.server;
  }
  f.receiver_id = f.server_id === f.player_1_id ? f.player_2_id : f.player_1_id;
  f.service_side.value = (draft.p1 + draft.p2) % 2 === 0 ? 'DEUCE' : 'AD';
  f.serve_attempt = draft.serve; f.point_state.value = draft.serve === 1 ? 'WAITING_FOR_SERVE' : 'SECOND_SERVE_WAITING';
  f.ball_state.value = draft.serve === 1 ? 'NOT_STARTED' : 'DEAD';
  f.initial_state_confirmed = true; f.score_confirmed = true; f.score_source = 'human_reviewed_initial_state';
  return raw;
}
export function amendInput(original: SourceInput, type: string, participant: number | null, end: string | null): SourceInput {
  const allowed: readonly string[] = original.reviewed_evidence_ref === null ? PHYSICAL_TYPES : REVIEWED_TYPES;
  if (!allowed.includes(type)) throw new Error('This event interpretation is unsupported.');
  const row = structuredClone(original); row.event_type = type;
  // Detector roles remain original measurements. Only an existing reviewed
  // binding can name a participant; no UI-created identity proof is allowed.
  if (row.context.binding && participant !== null && end !== null) {
    if (![1, 2].includes(participant) || !['NEAR_COURT', 'FAR_COURT'].includes(end)) throw new Error('Choose a supported reviewed participant and court end.');
    row.context.binding.participant_id = participant; row.context.binding.court_end = end as 'NEAR_COURT' | 'FAR_COURT';
  }
  return row;
}
export function commandBase(source: ReviewSource, correctionId: string, reason: string, evidence: string): CorrectionCommand {
  if (!source.is_current) throw new Error('A newer analysis is available. Refresh before submitting a correction.');
  const text = (value: string, label: string, maximum: number) => {
    const normalized = value.replace(/[\t\r\n]+/g, ' ').trim();
    if (!normalized) throw new Error(`Enter the ${label}.`);
    if (normalized.length > maximum) throw new Error(`The ${label} must be ${maximum} characters or fewer.`);
    if (Array.from(normalized).some(char => char.charCodeAt(0) < 32)) throw new Error(`The ${label} contains unsupported control characters.`);
    return normalized;
  };
  return { schema_version: 1, correction_id: correctionId, base_generation: source.base_generation,
    base_analysis_revision: source.base_analysis_revision, source_sha256: source.source_sha256,
    base_rule_replay_sha256: source.base_rule_replay_sha256, reason: text(reason, 'reason for correction', 4000), evidence_ref: text(evidence, 'video evidence reference', 2000),
    initial_state_amendment: null, input_amendments: [] };
}

import { z } from 'zod';
import { apiJson } from './analysisApi';

export interface Participant { name: string | null; profile_id: string | null }
export interface MatchMetadata {
  participants: Participant[]; tournament: string | null; match_date: string | null;
  match_type: 'singles'; role_assignment: 'unconfirmed' | 'same' | 'swapped'; result: string | null;
}
export interface PointRecord { winner: number; server: number; recorded_at: string; score: string[]; games: number[]; sets: number[] }
export interface ManualScore {
  state: { games_p1: number; games_p2: number; sets_won_p1: number; sets_won_p2: number; server_id: number; match_complete: boolean; winner_id: number | null; completed_sets: { games_p1: number; games_p2: number }[] };
  display_points: string[]; history: PointRecord[];
  statistics: { points_won: number; service_points_won: number; service_points: number; return_points_won: number; return_points: number }[];
}
export interface WinPrediction {
  status: 'unavailable' | 'prior_only' | 'uncalibrated' | 'confirmed_result'; reason: string;
  calibration_status?: 'not_calibrated'; sample_count?: number; revision?: number; winner_id?: number | null;
  probabilities?: Record<'game' | 'set' | 'match', { player_1: number; player_2: number }> | null;
  history?: (Omit<WinPrediction, 'history'> & { timestamp_seconds?: number | null; time_base?: 'source_seconds' | 'point_sequence'; point_id?: string })[];
}
export interface ProductMatch {
  id: string; source: 'upload' | 'manual'; analysis_id: string | null; revision: number;
  created_at: string; updated_at: string; metadata: MatchMetadata; metadata_source: string;
  status: 'DRAFT' | 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'LIVE' | 'PAUSED';
  filename: string | null; stage: string | null; error: string | null; notes: string | null;
  score: ManualScore | null;
  analysis_summary: { durationSeconds: number; frames: number } | null;
  predictions: WinPrediction;
}
export interface PlayerProfile { id: string; name: string; created_at: string }
export interface PlayerDetail extends PlayerProfile {
  matches: ProductMatch[];
  movement: { match_id: string; date: string | null; duration_seconds: number; confirmed_role: number; metrics: { total_distance_m: number; avg_speed_kmh: number; max_speed_kmh?: number; coverage_pct: number } }[];
}
export const emptyMetadata = (): MatchMetadata => ({ participants: [{ name: null, profile_id: null }, { name: null, profile_id: null }], tournament: null, match_date: null, match_type: 'singles', role_assignment: 'unconfirmed', result: null });
export const participantName = (match: ProductMatch, index: number) => match.metadata.participants[index]?.name || `Player ${index + 1}`;
export const matchTitle = (match: ProductMatch) => match.metadata.participants.some(p => p.name) ? `${participantName(match, 0)} vs ${participantName(match, 1)}` : match.filename || 'Untitled live match';
export const statusLabel = (status: string) => ({ UPLOADED: 'Queued', PROCESSING: 'Processing', COMPLETED: 'Completed', FAILED: 'Failed', LIVE: 'Live', PAUSED: 'Paused', DRAFT: 'Draft' })[status] || status;

export async function productJson<T>(path: string): Promise<T> { return await apiJson(path) as T; }
export async function mutate<T>(path: string, body: unknown, method = 'POST'): Promise<T> {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the match details and try again.');
  return data as T;
}
export async function listMatches() {
  const data = await apiJson('/api/matches');
  z.array(z.object({ id: z.string(), source: z.enum(['upload', 'manual']), metadata: z.object({ participants: z.array(z.object({ name: z.string().nullable(), profile_id: z.string().nullable() })).length(2) }) })).parse(data);
  return data as ProductMatch[];
}

export interface InsightResult {
  status: 'available' | 'unavailable'; matchId: string; source: 'telemetry' | 'manual_score' | 'coach' | 'prediction_provider';
  generatedAt: string | null; text: string | null; evidence: { metric: string; value: number; startFrame?: number; endFrame?: number }[];
}

export const predictionService = {
  get: async (matchId: string): Promise<InsightResult> => ({ status: 'unavailable', matchId, source: 'prediction_provider', generatedAt: null, text: null, evidence: [] }),
};

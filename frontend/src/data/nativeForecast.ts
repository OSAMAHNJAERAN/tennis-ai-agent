import { z } from 'zod';

const hash = z.string().regex(/^[a-f0-9]{64}$/);
const count = z.number().int().nonnegative().safe();
const probability = z.number().finite().min(0).max(1);
const pair = z.object({ player_1: probability, player_2: probability }).refine(p => Math.abs(p.player_1 + p.player_2 - 1) < 1e-8);
export const ForecastValueSchema = z.object({
  schema_version: z.literal('1.0'), match_id: z.string(), revision: count,
  score_state_version: count, timestamp_seconds: z.number().finite().nonnegative(),
  source: z.literal('statistical_tennis_rules'), calibration_status: z.literal('not_calibrated'),
  status: z.enum(['prior_only', 'uncalibrated', 'unavailable', 'confirmed_result']),
  sample_count: count, reason: z.string(), winner_id: z.number().int().positive().nullable(),
  probabilities: z.object({ game: pair, set: pair, match: pair }).nullable(),
  serve_estimates: z.array(z.object({ player_id: z.number().int().positive(), probability,
    points_won: count, points_played: count, alpha: z.number().positive(), beta: z.number().positive() })).length(2),
  prior: z.object({ alpha: z.number().positive(), beta: z.number().positive(), source: z.string() }),
  assumptions: z.string(), withheld_reason_code: z.string().optional(),
}).superRefine((value, ctx) => {
  const predictive = value.status === 'prior_only' || value.status === 'uncalibrated';
  if (predictive !== (value.probabilities !== null) || (value.status === 'confirmed_result') !== (value.winner_id !== null)
      || (value.status === 'prior_only' && value.sample_count !== 0)) ctx.addIssue({ code: 'custom', message: 'Forecast status and payload differ.' });
});
export const NativeForecastRowSchema = z.object({
  schema_version: z.literal('1.0'), analysis_id: z.string(), analysis_revision: count,
  source_sha256: hash, base_generation: hash, model_signature: hash,
  phase: z.enum(['INITIAL', 'BEFORE_EVENT', 'AFTER_EVENT']), event_id: count.nullable(),
  event_uid: z.string().nullable(), point_uid: z.string().nullable(), frame_index: count,
  timestamp_seconds: z.number().finite().nonnegative(),
  source_pts: z.union([z.string().regex(/^-?(0|[1-9]\d*)$/), z.number().int().safe()]),
  source_time_base: z.string().regex(/^[1-9]\d*(\/[1-9]\d*)?$/), source_frame_index: count,
  uses_points_through_event_id: count.nullable(), raw_state_sha256: hash, forecast: ForecastValueSchema,
});
export const NativeForecastSchema = z.object({
  schemaVersion: z.literal('1.0'), status: z.literal('UNQUALIFIED'), analysisId: z.string(),
  analysisRevision: count, sourceSha256: hash, baseGeneration: hash, modelSignature: hash,
  calibrationStatus: z.literal('not_calibrated'), independentAccuracyValidated: z.literal(false),
  humanReviewDependent: z.literal(true), modelSpec: z.record(z.string(), z.unknown()),
  historyPolicy: z.literal('INITIAL_AND_BEFORE_AFTER_EVERY_ACCEPTED_EVENT'),
  current: ForecastValueSchema, history: z.array(NativeForecastRowSchema).min(1),
});
export type NativeForecast = z.infer<typeof NativeForecastSchema>;
export type NativeForecastRow = z.infer<typeof NativeForecastRowSchema>;

export function forecastBindingValid(value: NativeForecast, id: string, revision: number | undefined, frames: number, source?: string): boolean {
  if (value.analysisId !== id || value.analysisRevision !== revision || !source || value.sourceSha256 !== source) return false;
  const seen = new Set<number>(); let frame = -1; let timestamp = -1;
  for (let i = 0; i < value.history.length; i++) {
    const row = value.history[i]!; const previous = value.history[i - 1];
    if (row.analysis_id !== id || row.analysis_revision !== revision || row.source_sha256 !== source ||
        row.base_generation !== value.baseGeneration || row.model_signature !== value.modelSignature ||
        row.frame_index >= frames || row.frame_index < frame || row.timestamp_seconds < timestamp ||
        row.forecast.match_id !== id || row.forecast.revision !== revision || row.forecast.timestamp_seconds !== row.timestamp_seconds) return false;
    if (i === 0) { if (row.phase !== 'INITIAL' || row.event_id !== null || row.uses_points_through_event_id !== null) return false; }
    else if (i % 2 === 1) {
      if (row.phase !== 'BEFORE_EVENT' || row.event_id === null || seen.has(row.event_id)) return false;
    } else {
      if (row.phase !== 'AFTER_EVENT' || row.event_id === null || row.event_id !== previous?.event_id ||
          row.event_uid !== previous.event_uid || row.frame_index !== previous.frame_index ||
          row.timestamp_seconds !== previous.timestamp_seconds || row.source_pts !== previous.source_pts ||
          row.source_time_base !== previous.source_time_base || row.source_frame_index !== previous.source_frame_index) return false;
      seen.add(row.event_id);
    }
    if (row.uses_points_through_event_id !== null && !seen.has(row.uses_points_through_event_id)) return false;
    frame = row.frame_index; timestamp = row.timestamp_seconds;
  }
  return value.history.length % 2 === 1 && JSON.stringify(value.current) === JSON.stringify(value.history.at(-1)!.forecast);
}

// A history bucket containing an unavailable prefix remains a gap. Each column seeks
// its retained source frame, never a synthetic bucket timestamp.
export function forecastBins(history: NativeForecastRow[], maximum = 600): NativeForecastRow[] {
  const limit = Math.max(1, Math.min(600, Math.floor(maximum)));
  if (history.length <= limit) return history;
  return Array.from({ length: limit }, (_, index) => {
    const start = Math.floor(index * history.length / limit); const end = Math.floor((index + 1) * history.length / limit);
    let selected = history[end - 1]!;
    for (let i = start; i < end; i++) if (history[i]!.forecast.status === 'unavailable') { selected = history[i]!; break; }
    return selected;
  });
}

export function forecastAtFrame(history: NativeForecastRow[], frame: number): NativeForecastRow | null {
  let low = 0; let high = history.length;
  while (low < high) { const middle = (low + high) >>> 1; if (history[middle]!.frame_index <= frame) low = middle + 1; else high = middle; }
  return history[low - 1] ?? null;
}

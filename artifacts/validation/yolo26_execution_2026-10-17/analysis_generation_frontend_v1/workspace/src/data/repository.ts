import rawSnapshot from "./generated/analysis-snapshot.json";
import { AnalysisRunSchema, AnalysisRunSummarySchema, SnapshotSchema, type AnalysisRun, type AnalysisRunSummary } from "./contracts";
import { apiJson } from './analysisApi';

export interface AnalysisRepository {
  listRuns(): Promise<AnalysisRunSummary[]>;
  getRun(id: string): Promise<AnalysisRun>;
  getVideoUrl(id: string): string | null;
}

export class RunNotFoundError extends Error {
  constructor(id: string) {
    super(`Analysis run ${id} was not found in the offline snapshot.`);
    this.name = "RunNotFoundError";
  }
}

export const snapshot = SnapshotSchema.parse(rawSnapshot);

export class StaticAnalysisRepository implements AnalysisRepository {
  async listRuns() {
    return snapshot.runs;
  }

  async getRun(id: string) {
    const run = snapshot.details[id];
    if (!run) throw new RunNotFoundError(id);
    return run;
  }

  getVideoUrl(id: string) {
    return snapshot.details[id]?.videoUrl ?? null;
  }
}

// Decode sections independently: one malformed optional artifact cannot hide
// a usable video or the other validated evidence.
export function parseLiveRun(raw: unknown): AnalysisRun {
  if (!raw || typeof raw !== 'object') throw new Error('The saved analysis is invalid.');
  const data = { ...raw } as Record<string, unknown>;
  const warnings = Array.isArray(data.warnings) ? data.warnings.filter((v): v is string => typeof v === 'string') : [];
  for (const key of ['detections', 'trajectories', 'court', 'playerMetrics', 'ballMetrics', 'pipelineMetrics', 'matchState', 'poseTracking', 'racketTracking', 'imagePeople', 'playerTracking', 'courtSegments', 'cameraTracking', 'mediaSource', 'pointAnalysis'] as const) {
    const parsed = AnalysisRunSchema.shape[key].safeParse(data[key]);
    if (!parsed.success) { data[key] = null; warnings.push(key === 'pointAnalysis' ? 'Point and serve evidence is unavailable for this analysis.' : `Analysis unavailable for this section: ${key}.`); }
  }
  for (const key of ['events', 'lineCalls', 'landingZones'] as const) {
    const parsed = AnalysisRunSchema.shape[key].safeParse(data[key]);
    if (!parsed.success) { data[key] = []; warnings.push(`Analysis unavailable for this section: ${key}.`); }
  }
  const result = AnalysisRunSchema.safeParse({ ...data, warnings });
  if (!result.success) throw new Error('The saved analysis has invalid metadata. Upload the video again to retry.');
  const points = result.data.pointAnalysis;
  if (points && (points.analysisId !== result.data.summary.id ||
      (result.data.analysisRevision != null && points.analysisRevision !== result.data.analysisRevision))) {
    result.data.pointAnalysis = null;
    result.data.warnings?.push('Point and serve evidence is unavailable for this analysis.');
  }
  const people = result.data.imagePeople;
  if (people) {
    const metadata = result.data.detections?.metadata;
    const width = metadata?.width; const height = metadata?.height;
    const aligned = people.frames.length === result.data.summary.frames && people.frames.every((frame, index) =>
      frame.frame_index === index && Math.abs(frame.timestamp_seconds - index / result.data.summary.fps) <= .01 &&
      (frame.people ?? []).every(person => typeof width === 'number' && typeof height === 'number' &&
        person.bbox[2] <= width && person.bbox[3] <= height));
    if (!aligned) {
      result.data.imagePeople = null;
      result.data.warnings?.push('Analysis unavailable for this section: imagePeople.');
    }
  }
  return result.data;
}

export class LiveAnalysisRepository extends StaticAnalysisRepository {
  async listRuns() {
    // Preserve demos while the server is offline; a requested live result never
    // silently falls back to a different analysis.
    const summaries = await apiJson('/api/runs').then(raw => AnalysisRunSummarySchema.array().parse(raw)).catch(() => []);
    return [...summaries, ...snapshot.runs];
  }

  async getRun(id: string) {
    if (id in snapshot.details) return super.getRun(id);
    if (!/^match_[a-f0-9]{32}$/.test(id)) throw new RunNotFoundError(id);
    const result = parseLiveRun(await apiJson(`/api/analyses/${encodeURIComponent(id)}/result`));
    if (result.summary.id !== id) throw new Error('The server returned a different analysis. Reload this session to retry.');
    return result;
  }

  getVideoUrl(id: string) {
    return /^match_[a-f0-9]{32}$/.test(id) ? `/api/analyses/${id}/media/source` : super.getVideoUrl(id);
  }
}

export const analysisRepository = new LiveAnalysisRepository();

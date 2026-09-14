import rawSnapshot from "./generated/analysis-snapshot.json";
import { SnapshotSchema, type AnalysisRun, type AnalysisRunSummary } from "./contracts";

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

export const analysisRepository = new StaticAnalysisRepository();


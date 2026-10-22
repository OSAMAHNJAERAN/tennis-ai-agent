import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseLiveRun, LiveAnalysisRepository, snapshot } from './repository';
import { analysisId, generation, nativeFixture } from '../test/correctionFixtures';

afterEach(() => vi.unstubAllGlobals());
describe('native forecast result admission', () => {
  it('withholds a foreign causal history while preserving existing video evidence', () => {
    const base = snapshot.details[snapshot.defaultRunId]!; const native = nativeFixture();
    const run = parseLiveRun({ ...base, summary: { ...base.summary, id: analysisId }, analysisRevision: 1, nativeForecast: native });
    // An absent independent source binding cannot be inferred from the forecast itself.
    expect(run.nativeForecast).toBeNull(); expect(run.videoUrl).toBe(base.videoUrl); expect(run.warnings?.some(value => value.includes('causal history'))).toBe(true);
    const malformed = parseLiveRun({ ...base, nativeForecast: { ...native, independentAccuracyValidated: true } });
    expect(malformed.nativeForecast).toBeNull(); expect(malformed.videoUrl).toBe(base.videoUrl);
  });
  it('requests only the exact saved generation and rejects substitution', async () => {
    const base = snapshot.details[snapshot.defaultRunId]!;
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ...base, summary: { ...base.summary, id: analysisId }, analysisGeneration: generation }) });
    vi.stubGlobal('fetch', fetch); const repo = new LiveAnalysisRepository();
    expect((await repo.getRun(analysisId, generation)).analysisGeneration).toBe(generation);
    expect(fetch).toHaveBeenCalledWith(`/api/analyses/${analysisId}/result?generation=${generation}`, { signal: undefined });
    await expect(repo.getRun(analysisId, 'a'.repeat(64))).rejects.toThrow('different saved generation');
    await expect(repo.getRun(analysisId, '../current')).rejects.toThrow('Invalid saved');
  });
});

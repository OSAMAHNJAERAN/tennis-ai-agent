import { afterEach, describe, expect, it, vi } from 'vitest';
import { correctionApi, CorrectionStatusSchema } from './correctionApi';
import { commandBase } from './correctionDraft';
import { analysisId, correctionStatus, generation, reviewFixture } from '../test/correctionFixtures';

afterEach(() => vi.unstubAllGlobals());
describe('owner correction HTTP contracts', () => {
  it('pins a bounded review page to the requested saved generation', async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => reviewFixture() }); vi.stubGlobal('fetch', fetch);
    expect((await correctionApi.source(analysisId, generation)).initial_state.fields.match_id).toBe(analysisId);
    expect(fetch).toHaveBeenCalledWith(`/api/analyses/${analysisId}/review-source?generation=${generation}&offset=0&limit=20`, { signal: undefined });
    const foreign = reviewFixture(); foreign.base_generation = 'f'.repeat(64); fetch.mockResolvedValue({ ok: true, json: async () => foreign });
    await expect(correctionApi.source(analysisId, generation)).rejects.toThrow('different analysis generation');
  });
  it('submits the native body without client supplied reviewer identities and propagates owner denial', async () => {
    const fetch = vi.fn().mockResolvedValueOnce({ ok: true, json: async () => correctionStatus() }).mockResolvedValueOnce({ ok: false, status: 403, json: async () => ({ detail: 'Owner access required.' }) });
    vi.stubGlobal('fetch', fetch); const command = commandBase(reviewFixture(), 'stable-id', 'Reviewed', 'video 00:01');
    await correctionApi.submit(analysisId, command); const request = fetch.mock.calls[0]![1];
    expect(JSON.parse(request.body)).toEqual(command); expect(request.method).toBe('POST');
    await expect(correctionApi.cancel(analysisId, correctionStatus().id)).rejects.toThrow('Owner access required.');
  });
  it('rejects another analysis status and inconsistent completed generation metadata', async () => {
    const foreign = correctionStatus(); foreign.analysis_id = 'another'; vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => foreign }));
    await expect(correctionApi.status(analysisId, foreign.id)).rejects.toThrow('different analysis');
    expect(CorrectionStatusSchema.safeParse({ ...correctionStatus('COMPLETED'), generation: null }).success).toBe(false);
    expect(CorrectionStatusSchema.safeParse({ ...correctionStatus('COMPLETED'), analysis_revision: 9 }).success).toBe(false);
  });
});

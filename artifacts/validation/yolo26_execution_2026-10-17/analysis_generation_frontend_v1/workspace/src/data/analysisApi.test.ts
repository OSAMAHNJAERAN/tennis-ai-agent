import { afterEach, describe, expect, it, vi } from 'vitest';
import { listJobs } from './analysisApi';

const job = { id: 'match_a', filename: 'tennis.mp4', status: 'UPLOADED', stage: 'Queued', step: 0,
  created_at: '2026-10-09T00:00:00Z', started_at: null, completed_at: null, error: null,
  size_bytes: 1024, cancel_requested: false, retry_of: null };

afterEach(() => vi.unstubAllGlobals());

describe('persisted job API contract', () => {
  it('accepts boolean cancellation flags in actual job history', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [job] }));
    expect((await listJobs())[0]?.cancel_requested).toBe(false);
  });
  it('keeps malformed storage details out of the user error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [{ ...job, cancel_requested: 0 }] }));
    await expect(listJobs()).rejects.toThrow('The server returned invalid analysis history. Reconnect before retrying.');
  });
});

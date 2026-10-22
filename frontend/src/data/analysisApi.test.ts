import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { listJobs } from './analysisApi';

const job = { id: 'match_a', filename: 'tennis.mp4', status: 'UPLOADED', stage: 'Queued', step: 0,
  created_at: '2026-10-09T00:00:00Z', started_at: null, completed_at: null, error: null,
  size_bytes: 1024, cancel_requested: false, retry_of: null };

beforeEach(() => {
  const entries = new Map<string, string>();
  vi.stubGlobal('localStorage', { getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => { entries.set(key, value); } });
});
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

describe('retry acknowledgement contract', () => {
  const session = { required: true, authenticated: true, user: { id: 'alice' } };
  const child = { ...job, id: 'match_child', retry_of: 'match_failed' };
  const reply = (value: unknown, status = 202) => ({ ok: status < 400, status, json: async () => value });

  async function retryApi() {
    vi.resetModules();
    return import('./analysisApi');
  }

  it('keeps its Idempotency-Key after a lost response and acknowledges only the requested source', async () => {
    const { retryAnalysis } = await retryApi();
    let posts = 0;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      void init;
      if (path === '/api/auth/session') return reply(session);
      if (++posts === 1) throw new TypeError('Connection lost');
      return reply(child);
    });
    vi.stubGlobal('fetch', fetcher);
    await expect(retryAnalysis('match_failed', 'alice')).rejects.toThrow('Connection lost');
    await expect(retryAnalysis('match_failed', 'alice')).resolves.toMatchObject(child);
    const requests = fetcher.mock.calls.filter(([path]) => path !== '/api/auth/session');
    expect(requests).toHaveLength(2);
    const init = fetcher.mock.calls as unknown as [string, RequestInit][];
    const tokens = init.filter(([path]) => path !== '/api/auth/session').map(([, options]) => (options.headers as Record<string, string>)['Idempotency-Key']);
    expect(tokens[0]).toMatch(/^[A-Za-z0-9_-]{16,128}$/); expect(tokens[1]).toBe(tokens[0]);
    await retryAnalysis('match_failed', 'alice');
    const latest = (fetcher.mock.calls as unknown as [string, RequestInit][]).filter(([path]) => path !== '/api/auth/session').at(-1)!;
    expect((latest[1].headers as Record<string, string>)['Idempotency-Key']).not.toBe(tokens[0]);
  });

  it.each([{ ...child, retry_of: 'match_other' }, { ...child, id: 'match_failed' }, { ...child, id: '../wrong' },
    { ...child, id: 'match_child\n' }])(
    'retains the saved intent for an invalid child acknowledgement $id/$retry_of', async invalid => {
      const { retryAnalysis } = await retryApi();
      let result = invalid;
      const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
        void init; return reply(path === '/api/auth/session' ? session : result);
      });
      vi.stubGlobal('fetch', fetcher);
      await expect(retryAnalysis('match_failed', 'alice')).rejects.toThrow('invalid retry acknowledgement');
      const token = (fetcher.mock.calls.find(([path]) => path !== '/api/auth/session')![1]!.headers as Record<string, string>)['Idempotency-Key'];
      result = child;
      await retryAnalysis('match_failed', 'alice');
      const last = fetcher.mock.calls.filter(([path]) => path !== '/api/auth/session').at(-1)!;
      expect((last[1]!.headers as Record<string, string>)['Idempotency-Key']).toBe(token);
    },
  );

  it('refuses a previous account response and keeps its key for later reconciliation', async () => {
    const { retryAnalysis } = await retryApi();
    let owner = 'alice';
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      void init;
      if (path === '/api/auth/session') return reply({ ...session, user: { id: owner } });
      owner = 'bob'; return reply(child);
    });
    vi.stubGlobal('fetch', fetcher);
    await expect(retryAnalysis('match_failed', 'alice')).rejects.toThrow('account changed');
    const { retryIntents } = await import('./retryIntent');
    const token = (fetcher.mock.calls.find(([path]) => path !== '/api/auth/session')![1]!.headers as Record<string, string>)['Idempotency-Key'];
    expect(retryIntents.key('alice', 'match_failed')).toBe(token);
  });
});

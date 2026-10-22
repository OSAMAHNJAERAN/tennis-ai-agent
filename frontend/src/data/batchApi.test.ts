import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const metadata = { participants: [{ name: 'Alice', profile_id: null }, { name: null, profile_id: null }],
  tournament: 'Original cup', match_date: null, match_type: 'singles' as const, role_assignment: 'unconfirmed' as const, result: null };
const file = () => new File(['1234567890123456'], 'first.mp4');
const session = { id: 'upload_one', filename: 'first.mp4', size_bytes: 16, offset: 0,
  status: 'UPLOADING', analysis_id: null, clip_index: 0 };
const batch = { id: 'batch_one', mode: 'independent', uploads: [session], grouping_status: 'unverified' };
const options = { fileKeys: ['first-content'] };
const reply = (data: unknown, status = 201) => ({ ok: status < 400, status, json: async () => data });
const local = { required: false, authenticated: false, user: null };

beforeEach(() => {
  vi.resetModules();
  const values = new Map<string, string>();
  vi.stubGlobal('localStorage', { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); } });
});
afterEach(() => vi.unstubAllGlobals());

describe('atomic batch creation', () => {
  it('reuses the same full metadata request and key after lost response and reload', async () => {
    let posts = 0;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      void init; if (path === '/api/auth/session') return reply(local);
      if (++posts === 1) throw new TypeError('Connection lost');
      return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const before = await import('./batchApi');
    await expect(before.createUploadBatch([file()], 'independent', metadata, 'local', options)).rejects.toThrow('Connection lost');
    vi.resetModules();
    const after = await import('./batchApi');
    await expect(after.createUploadBatch([file()], 'independent', metadata, 'local', options)).resolves.toMatchObject(batch);
    const sent = fetcher.mock.calls.filter(([path]) => path === '/api/batches');
    expect(sent).toHaveLength(2);
    expect(sent[1]![1]!.body).toBe(sent[0]![1]!.body);
    expect((sent[1]![1]!.headers as Record<string, string>)['Idempotency-Key']).toBe((sent[0]![1]!.headers as Record<string, string>)['Idempotency-Key']);
    expect(JSON.parse(sent[0]![1]!.body as string)).toEqual({ mode: 'independent', metadata,
      files: [{ filename: 'first.mp4', size_bytes: 16, clip_index: 0 }] });
    expect(after.rememberedUpload('local', 'first-content')).toBe(session.id);
    expect(after.rememberedUpload('bob', 'first-content')).toBeNull();
    expect(fetcher.mock.calls.some(([path]) => path === '/api/matches')).toBe(false);
  });

  it('blocks edited metadata and exposes the original request for explicit restoration', async () => {
    let lost = true;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      void init; if (path === '/api/auth/session') return reply(local);
      if (lost) throw new TypeError('Uncertain'); return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).rejects.toThrow('Uncertain');
    await expect(api.createUploadBatch([file()], 'independent', { ...metadata, tournament: 'Edited' }, 'local', options)).rejects.toMatchObject({
      name: 'CreationIntentConflict', request: { payload: { metadata } },
    });
    expect(fetcher.mock.calls.filter(([path]) => path === '/api/batches')).toHaveLength(1);
    lost = false;
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).resolves.toMatchObject(batch);
  });

  it.each([
    { ...batch, uploads: [] }, { ...batch, mode: 'same_match' },
    { ...batch, uploads: [{ ...session, filename: 'other.mp4' }] },
    { ...batch, uploads: [{ ...session, id: 'upload_one\n' }] },
    { ...batch, uploads: [{ ...session, offset: 17 }] },
    { ...batch, uploads: [{ ...session, clip_index: 9 }] },
    { ...batch, uploads: [{ ...session, status: 'COMPLETED', analysis_id: null }] },
  ])('retains its key when acknowledgement does not match the submitted files', async invalid => {
    let result: unknown = invalid;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => { void init; return reply(path === '/api/auth/session' ? local : result); });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).rejects.toThrow('acknowledgement');
    expect(api.rememberedUpload('local', 'first-content')).toBeNull();
    result = batch;
    await api.createUploadBatch([file()], 'independent', metadata, 'local', options);
    const sent = fetcher.mock.calls.filter(([path]) => path === '/api/batches');
    expect(sent[1]![1]!.headers).toEqual(sent[0]![1]!.headers);
  });

  it('rejects distinct files acknowledged with the same saved session identity', async () => {
    const duplicate = { ...batch, uploads: [session, { ...session, filename: 'second.mp4', clip_index: 1 }] };
    vi.stubGlobal('fetch', vi.fn(async (path: string) => reply(path === '/api/auth/session' ? local : duplicate)));
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file(), new File(['1234567890123456'], 'second.mp4')], 'independent', metadata, 'local',
      { fileKeys: ['first-content', 'second-content'] })).rejects.toThrow('duplicate upload sessions');
    expect(api.rememberedUpload('local', 'first-content')).toBeNull();
  });

  it('does not acknowledge or remember a response after the current account changes', async () => {
    let current = true;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      void init; if (path === '/api/auth/session') return reply(local);
      current = false; return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', { ...options, isCurrent: () => current })).rejects.toThrow('account changed');
    expect(api.rememberedUpload('local', 'first-content')).toBeNull();
  });

  it('sends required idempotency key for single-session creation too', async () => {
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => { void init; return reply(path === '/api/auth/session' ? local : session); });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadSession(file(), 'local', 'match_draft', options)).resolves.toMatchObject(session);
    const sent = fetcher.mock.calls.find(([path]) => path === '/api/uploads')!;
    expect(JSON.parse(sent[1]!.body as string)).toEqual({ filename: 'first.mp4', size_bytes: 16, match_id: 'match_draft' });
    const key = (sent[1]!.headers as Record<string, string>)['Idempotency-Key']!;
    expect(key.length).toBeGreaterThanOrEqual(16); expect(key).not.toMatch(/[^A-Za-z0-9_-]/);
  });
});

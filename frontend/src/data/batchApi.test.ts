import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const metadata = { participants: [{ name: 'Alice', profile_id: null }, { name: null, profile_id: null }],
  tournament: 'Original cup', match_date: null, match_type: 'singles' as const, role_assignment: 'unconfirmed' as const, result: null };
const file = () => new File(['1234567890123456'], 'first.mp4');
const session = { id: 'upload_one', filename: 'first.mp4', size_bytes: 16, offset: 0,
  status: 'UPLOADING', analysis_id: null, clip_index: 0 };
const batch = { id: 'batch_one', mode: 'independent', uploads: [session], grouping_status: 'unverified' };
const options = { fileKeys: ['first-content'] };
const reply = (data: unknown, status = 201, headers: Record<string, string> = {}) =>
  ({ ok: status < 400, status, headers: new Headers(headers), json: async () => data });
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
    expect((sent[0]![1]!.headers as Record<string, string>)['Upload-Creation-Owner']).toBe('local');
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
    expect((sent[1]!.headers as Record<string, string>)['Upload-Creation-Owner']).toBe('local');
  });

  it('permits corrected metadata after the server durably rejects that exact creation key', async () => {
    let posts = 0;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === '/api/auth/session') return reply(local);
      const key = (init!.headers as Record<string, string>)['Idempotency-Key']!;
      if (++posts === 1) return reply({ detail: 'Storage limit reached' }, 413, { 'Upload-Creation-Rejected': key });
      return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).rejects.toThrow('Storage limit reached');
    expect(api.rememberedUpload('local', 'first-content')).toBeNull();
    vi.resetModules();
    const after = await import('./batchApi');
    await expect(after.createUploadBatch([file()], 'independent', { ...metadata, tournament: 'Corrected cup' }, 'local', options)).resolves.toMatchObject(batch);
    const sent = fetcher.mock.calls.filter(([path]) => path === '/api/batches');
    expect((sent[1]![1]!.headers as Record<string, string>)['Idempotency-Key']).not.toBe((sent[0]![1]!.headers as Record<string, string>)['Idempotency-Key']);
  });

  it.each([
    { status: 413, marker: 'absent' },
    { status: 422, marker: 'different_creation_key' },
    { status: 409, marker: 'duplicate' },
    { status: 500, marker: 'matching' },
  ])('preserves uncertain intent without an exact trusted 4xx rejection: $status/$marker', async ({ status, marker }) => {
    let posts = 0;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === '/api/auth/session') return reply(local);
      const key = (init!.headers as Record<string, string>)['Idempotency-Key']!;
      const headers: Record<string, string> = marker === 'absent' ? {} : { 'Upload-Creation-Rejected': marker === 'matching' ? key
        : marker === 'duplicate' ? `${key}, ${key}` : marker };
      return ++posts === 1 ? reply({ detail: 'Unconfirmed' }, status, headers) : reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).rejects.toThrow('Unconfirmed');
    await expect(api.createUploadBatch([file()], 'independent', { ...metadata, tournament: 'Edited' }, 'local', options)).rejects.toMatchObject({ name: 'CreationIntentConflict' });
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).resolves.toMatchObject(batch);
    const sent = fetcher.mock.calls.filter(([path]) => path === '/api/batches');
    expect((sent[1]![1]!.headers as Record<string, string>)['Idempotency-Key']).toBe((sent[0]![1]!.headers as Record<string, string>)['Idempotency-Key']);
  });

  it('retains the rejected key when account identity changes before rejection confirmation', async () => {
    let owner = 'alice'; let posts = 0;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === '/api/auth/session') return reply({ required: true, authenticated: true, user: { id: owner } });
      const key = (init!.headers as Record<string, string>)['Idempotency-Key']!;
      if (++posts === 1) { owner = 'bob'; return reply({ detail: 'Rejected' }, 422, { 'Upload-Creation-Rejected': key }); }
      return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'alice', options)).rejects.toThrow('account changed');
    expect(api.rememberedUpload('alice', 'first-content')).toBeNull();
    owner = 'alice';
    await expect(api.createUploadBatch([file()], 'independent', { ...metadata, tournament: 'Edited' }, 'alice', options)).rejects.toMatchObject({ name: 'CreationIntentConflict' });
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'alice', options)).resolves.toMatchObject(batch);
    const sent = fetcher.mock.calls.filter(([path]) => path === '/api/batches');
    expect((sent[1]![1]!.headers as Record<string, string>)['Idempotency-Key']).toBe((sent[0]![1]!.headers as Record<string, string>)['Idempotency-Key']);
    expect((sent[0]![1]!.headers as Record<string, string>)['Upload-Creation-Owner']).toBe('alice');
  });

  it('does not clear rejected intent when the caller has observed an account change or pause', async () => {
    let current = true; let rejected = true;
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === '/api/auth/session') return reply(local);
      const key = (init!.headers as Record<string, string>)['Idempotency-Key']!;
      if (rejected) { current = false; return reply({ detail: 'Rejected' }, 422, { 'Upload-Creation-Rejected': key }); }
      return reply(batch);
    });
    vi.stubGlobal('fetch', fetcher);
    const api = await import('./batchApi');
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', { ...options, isCurrent: () => current })).rejects.toThrow('account changed');
    current = true; rejected = false;
    await expect(api.createUploadBatch([file()], 'independent', { ...metadata, tournament: 'Edited' }, 'local', options)).rejects.toMatchObject({ name: 'CreationIntentConflict' });
    await expect(api.createUploadBatch([file()], 'independent', metadata, 'local', options)).resolves.toMatchObject(batch);
  });
});

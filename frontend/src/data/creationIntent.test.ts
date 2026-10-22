import { describe, expect, it, vi } from 'vitest';
import { CreationIntentConflict, createCreationIntentStore, type CreationRequest } from './creationIntent';

function request(): CreationRequest {
  return { kind: 'batch', payload: { mode: 'independent', files: [{ filename: 'first.mp4', size_bytes: 16, clip_index: 0 }],
    metadata: { participants: [{ name: 'Alice', profile_id: null }, { name: null, profile_id: null }], tournament: 'Original cup',
      match_date: null, match_type: 'singles', role_assignment: 'unconfirmed', result: null } }, fileKeys: ['first-file-content'] };
}
function browserStorage() {
  const values = new Map<string, string>();
  return { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
}
const first = 'creation_intent_00000001';
const second = 'creation_intent_00000002';

describe('uncertain upload creation', () => {
  it('persists the full original payload and reuses its key after reload and lost response', async () => {
    const storage = browserStorage(); const input = request();
    const before = createCreationIntentStore(() => storage, () => first);
    await expect(before.run('alice', input, async () => { throw new Error('Lost response'); })).rejects.toThrow('Lost response');
    const after = createCreationIntentStore(() => storage, () => second);
    const send = vi.fn(async (saved: { key: string; request: CreationRequest }) => saved);
    const result = await after.run('alice', input, send);
    expect(result).toEqual({ key: first, request: input });
  });

  it('blocks edited metadata, file identity or mode until the original request is restored', async () => {
    const input = request(); const store = createCreationIntentStore(() => browserStorage(), () => first);
    await expect(store.run('alice', input, async () => { throw new Error('Unconfirmed'); })).rejects.toThrow('Unconfirmed');
    const edited = structuredClone(input);
    if (edited.kind === 'batch') edited.payload.metadata.tournament = 'Edited cup';
    const send = vi.fn();
    expect(() => store.run('alice', edited, send)).toThrow(CreationIntentConflict);
    expect(() => store.run('alice', { ...input, fileKeys: ['different-content'] }, send)).toThrow(CreationIntentConflict);
    if (input.kind === 'batch') expect(() => store.run('alice', { ...input, payload: { ...input.payload, mode: 'same_match' } }, send)).toThrow(CreationIntentConflict);
    expect(send).not.toHaveBeenCalled();
    const recovered = await store.run('alice', request(), async saved => saved);
    expect(recovered.request).toEqual(request()); expect(recovered.key).toBe(first);
  });

  it('coalesces rapid repeated submission but isolates different owners', async () => {
    const keys = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second);
    const store = createCreationIntentStore(() => null, keys);
    let finish: (id: string) => void = () => undefined;
    const send = vi.fn(() => new Promise<string>(resolve => { finish = resolve; }));
    const one = store.run('alice', request(), send); const two = store.run('alice', request(), send);
    expect(one).toBe(two); await Promise.resolve(); expect(send).toHaveBeenCalledOnce();
    expect(await store.run('bob', request(), async saved => saved.key)).toBe(second);
    finish('batch_saved'); await expect(one).resolves.toBe('batch_saved');
  });

  it('permits a fresh intent only after the matching validated acknowledgement', async () => {
    const keys = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second);
    const storage = browserStorage(); const store = createCreationIntentStore(() => storage, keys);
    expect(await store.run('alice', request(), async saved => saved.key)).toBe(first);
    store.acknowledge('alice', second);
    expect(await store.run('alice', request(), async saved => saved.key)).toBe(first);
    store.acknowledge('alice', first);
    expect(await createCreationIntentStore(() => storage, () => second).run('alice', request(), async saved => saved.key)).toBe(second);
  });

  it('keeps the same request in memory if browser storage is disabled', async () => {
    const store = createCreationIntentStore(() => { throw new DOMException('Blocked', 'SecurityError'); }, () => first);
    await expect(store.run('alice', request(), async () => { throw new Error('Lost'); })).rejects.toThrow('Lost');
    expect(await store.run('alice', request(), async saved => saved.key)).toBe(first);
  });

  it('fails closed for corrupted persisted uncertain creation rather than silently replacing it', () => {
    const storage = { getItem: () => JSON.stringify({ key: first + '\n', request: request() }), setItem: vi.fn() };
    const store = createCreationIntentStore(() => storage, () => second);
    expect(() => store.run('alice', request(), vi.fn())).toThrow('saved upload request cannot be read');
    expect(storage.setItem).not.toHaveBeenCalled();
  });
});

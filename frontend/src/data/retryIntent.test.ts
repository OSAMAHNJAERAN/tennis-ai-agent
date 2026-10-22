import { describe, expect, it, vi } from 'vitest';
import { createRetryIntentStore } from './retryIntent';

function browserStorage() {
  const values = new Map<string, string>();
  return { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); } };
}
const first = 'retry_intent_00000001';
const second = 'retry_intent_00000002';

describe('saved analysis retry intent', () => {
  it('restores the same uncertain intent after a browser reload', async () => {
    const storage = browserStorage();
    const before = createRetryIntentStore(() => storage, () => first);
    await expect(before.run('alice', 'match_failed', async () => { throw new Error('Response lost'); })).rejects.toThrow('Response lost');
    const after = createRetryIntentStore(() => storage, () => second);
    expect(after.key('alice', 'match_failed')).toBe(first);
  });

  it('records acknowledged child identity and starts fresh explicit intent after acknowledgement', () => {
    const storage = browserStorage();
    const before = createRetryIntentStore(() => storage, () => first);
    before.acknowledge('alice', 'match_failed', before.key('alice', 'match_failed'), 'match_child');
    const after = createRetryIntentStore(() => storage, () => second);
    expect(after.key('alice', 'match_failed')).toBe(second);
    expect(createRetryIntentStore(() => storage, () => first).key('alice', 'match_child')).toBe(first);
  });

  it('keeps owners and failed child sources in separate namespaces', () => {
    const keys = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second).mockReturnValueOnce('retry_intent_00000003');
    const store = createRetryIntentStore(() => browserStorage(), keys);
    expect(store.key('alice', 'match_failed')).toBe(first);
    expect(store.key('bob', 'match_failed')).toBe(second);
    expect(store.key('alice', 'match_child')).toBe('retry_intent_00000003');
    expect(store.key('alice', 'match_failed')).toBe(first);
  });

  it('reuses the same in-flight promise before any request finishes', async () => {
    const store = createRetryIntentStore(() => null, () => first);
    let finish: (value: string) => void = () => undefined;
    const send = vi.fn(() => new Promise<string>(resolve => { finish = resolve; }));
    const one = store.run('alice', 'match_failed', send);
    const two = store.run('alice', 'match_failed', send);
    expect(one).toBe(two);
    await Promise.resolve();
    expect(send).toHaveBeenCalledExactlyOnceWith(first);
    finish('match_child');
    await expect(one).resolves.toBe('match_child');
  });

  it('falls back to stable memory when browser storage access is disabled', () => {
    const keys = vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second);
    const store = createRetryIntentStore(() => { throw new DOMException('Blocked', 'SecurityError'); }, keys);
    expect(store.key('alice', 'match_failed')).toBe(first);
    expect(store.key('alice', 'match_failed')).toBe(first);
    store.acknowledge('alice', 'match_failed', first, 'match_child');
    expect(store.key('alice', 'match_failed')).toBe(second);
  });

  it('ignores stale acknowledgements and malformed persisted tokens', () => {
    const storage = { getItem: () => '{"key":"bad key","childId":null}', setItem: vi.fn() };
    const store = createRetryIntentStore(() => storage, () => first);
    expect(store.key('alice', 'match_failed')).toBe(first);
    store.acknowledge('alice', 'match_failed', second, 'match_wrong');
    expect(store.key('alice', 'match_failed')).toBe(first);
  });

  it('regenerates persisted tokens with a trailing newline instead of repeating a server-invalid key', () => {
    const storage = { getItem: () => JSON.stringify({ key: first + '\n', childId: null }), setItem: vi.fn() };
    const store = createRetryIntentStore(() => storage, () => second);
    expect(store.key('alice', 'match_failed')).toBe(second);
    expect(storage.setItem).toHaveBeenCalledWith(expect.any(String), JSON.stringify({ key: second, childId: null }));
    expect(() => createRetryIntentStore(() => null, () => first + '\n').key('alice', 'match_failed')).toThrow('Unable to create');
  });
});

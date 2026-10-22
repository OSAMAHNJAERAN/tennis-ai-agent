/** Persist uncertain retry intent; acknowledgement alone permits a fresh key. */
interface RetryRecord { key: string; childId: string | null }
type RetryStorage = Pick<Storage, 'getItem' | 'setItem'>;
const validKey = (value: string) => value.length >= 16 && value.length <= 128 && !/[^A-Za-z0-9_-]/.test(value);

export function createRetryIntentStore(
  storage: () => RetryStorage | null = () => localStorage,
  newKey: () => string = () => crypto.randomUUID(),
) {
  const memory = new Map<string, RetryRecord>();
  const pending = new Map<string, Promise<unknown>>();
  const address = (owner: string, source: string) => `astra.analysis.retry.v1:${encodeURIComponent(owner)}:${encodeURIComponent(source)}`;
  function save(id: string, record: RetryRecord) {
    memory.set(id, record);
    try { storage()?.setItem(id, JSON.stringify(record)); } catch { /* The same intent remains stable in memory. */ }
  }
  function key(owner: string, source: string) {
    const id = address(owner, source);
    let record = memory.get(id);
    if (!record) {
      try {
        const value: unknown = JSON.parse(storage()?.getItem(id) ?? 'null');
        if (value && typeof value === 'object' && 'key' in value && typeof value.key === 'string' && validKey(value.key)
          && 'childId' in value && (value.childId === null || typeof value.childId === 'string')) {
          record = { key: value.key, childId: value.childId };
        }
      } catch { /* Browser storage may be blocked or contain an older value. */ }
    }
    if (record && record.childId === null) { memory.set(id, record); return record.key; }
    const token = newKey();
    if (!validKey(token)) throw new Error('Unable to create a saved retry request. Reconnect before retrying.');
    save(id, { key: token, childId: null });
    return token;
  }
  function acknowledge(owner: string, source: string, token: string, childId: string) {
    const id = address(owner, source);
    const record = memory.get(id);
    if (!record || record.key !== token || record.childId !== null) return;
    // Keep an acknowledgement marker rather than depending on removeItem:
    // disabled storage still has a reliable in-memory acknowledgement.
    save(id, { key: token, childId });
  }
  function run<T>(owner: string, source: string, send: (token: string) => Promise<T>): Promise<T> {
    const id = address(owner, source);
    const active = pending.get(id);
    if (active) return active as Promise<T>;
    const token = key(owner, source);
    const request = Promise.resolve().then(() => send(token)).finally(() => {
      if (pending.get(id) === request) pending.delete(id);
    });
    pending.set(id, request);
    return request;
  }
  return { key, acknowledge, run };
}

export const retryIntents = createRetryIntentStore();

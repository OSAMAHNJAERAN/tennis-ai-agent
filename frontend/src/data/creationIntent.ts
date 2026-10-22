import { z } from 'zod';

const metadata = z.object({ participants: z.array(z.object({ name: z.string().nullable(), profile_id: z.string().nullable() })).length(2),
  tournament: z.string().nullable(), match_date: z.string().nullable(), match_type: z.literal('singles'),
  role_assignment: z.enum(['unconfirmed', 'same', 'swapped']), result: z.string().nullable() });
const file = z.object({ filename: z.string().min(1), size_bytes: z.number().int().positive(),
  match_id: z.string().optional(), clip_index: z.number().int().nonnegative().optional() });
const requestSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('batch'), payload: z.object({ mode: z.enum(['independent', 'same_match']),
    files: z.array(file).min(1).max(16), metadata }), fileKeys: z.array(z.string().min(1)).min(1).max(16) }),
  z.object({ kind: z.literal('upload'), payload: file, fileKeys: z.array(z.string().min(1)).length(1) }),
]);
export type CreationRequest = z.infer<typeof requestSchema>;
interface SavedCreation { key: string; request: CreationRequest }
type IntentStorage = Pick<Storage, 'getItem' | 'setItem'>;
const validKey = (value: string) => value.length >= 16 && value.length <= 128 && !/[^A-Za-z0-9_-]/.test(value);
const savedSchema = z.object({ key: z.string().refine(validKey), request: requestSchema });
const canonical = (value: unknown): string => JSON.stringify(value, (_key, item: unknown) =>
  item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);

export class CreationIntentConflict extends Error {
  request: CreationRequest;
  constructor(request: CreationRequest) {
    const names = request.kind === 'batch' ? request.payload.files.map(item => item.filename) : [request.payload.filename];
    super(`A previous upload request is still unconfirmed: ${names.join(', ')}. Restore its original match details and files, then retry the saved request.`);
    this.name = 'CreationIntentConflict'; this.request = request;
  }
}

/** One uncertain creation per account prevents edited forms from duplicating it. */
export function createCreationIntentStore(storage: () => IntentStorage | null = () => localStorage,
  newKey: () => string = () => crypto.randomUUID()) {
  const memory = new Map<string, SavedCreation | null>();
  const pending = new Map<string, Promise<unknown>>();
  const address = (owner: string) => `astra.upload.creation.v1:${encodeURIComponent(owner)}`;
  function read(owner: string): SavedCreation | null {
    if (memory.has(owner)) return memory.get(owner) ?? null;
    let raw: string | null;
    try { raw = storage()?.getItem(address(owner)) ?? null; } catch { return null; }
    if (!raw || raw === 'null') return null;
    try {
      const saved = savedSchema.parse(JSON.parse(raw));
      memory.set(owner, saved); return saved;
    } catch {
      // A corrupt uncertain request cannot safely authorize another creation.
      throw new Error('The saved upload request cannot be read. Check your saved batches before starting another upload.');
    }
  }
  function save(owner: string, value: SavedCreation | null) {
    memory.set(owner, value);
    try { storage()?.setItem(address(owner), JSON.stringify(value)); } catch { /* Stable same-runtime fallback. */ }
  }
  function run<T>(owner: string, input: CreationRequest, send: (saved: SavedCreation) => Promise<T>): Promise<T> {
    const request = requestSchema.parse(input);
    if (request.kind === 'batch' && request.fileKeys.length !== request.payload.files.length)
      throw new Error('Upload file identities do not match the saved request.');
    let saved = read(owner);
    if (saved && canonical(saved.request) !== canonical(request)) throw new CreationIntentConflict(saved.request);
    const active = pending.get(owner);
    if (active) return active as Promise<T>;
    if (!saved) {
      const key = newKey();
      if (!validKey(key)) throw new Error('Unable to save an upload request. Reconnect before uploading.');
      saved = { key, request }; save(owner, saved);
    }
    const admitted = saved;
    const promise = Promise.resolve().then(() => send(admitted)).finally(() => {
      if (pending.get(owner) === promise) pending.delete(owner);
    });
    pending.set(owner, promise); return promise;
  }
  function acknowledge(owner: string, key: string) {
    if (read(owner)?.key === key) save(owner, null);
  }
  return { run, acknowledge };
}

export const creationIntents = createCreationIntentStore();

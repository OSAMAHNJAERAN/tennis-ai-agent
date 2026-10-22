import { z } from 'zod';
import { ApiError, apiJson, JobSchema, retryOwnerScope, type AnalysisJob } from './analysisApi';
import { mutate, type MatchMetadata } from './productApi';
import { creationIntents, type CreationRequest } from './creationIntent';

export const UploadSessionSchema = z.object({ id: z.string(), filename: z.string(), size_bytes: z.number().int().nonnegative(),
  offset: z.number().int().nonnegative(), status: z.string(), analysis_id: z.string().nullable() }).passthrough();
export type UploadSession = z.infer<typeof UploadSessionSchema>;
export interface UploadBatch { id: string; mode: 'independent' | 'same_match'; uploads: UploadSession[]; grouping_status?: string; reason?: string }

/** Metadata selects a resume candidate; sampled bytes bind it to the reselected file. */
export async function fileResumeKey(file: File): Promise<string> {
  const first = await file.slice(0, 65536).arrayBuffer();
  const last = await file.slice(Math.max(0, file.size - 65536)).arrayBuffer();
  const bytes = new Uint8Array(first.byteLength + last.byteLength);
  bytes.set(new Uint8Array(first)); bytes.set(new Uint8Array(last), first.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  return JSON.stringify([file.name, file.size, file.lastModified, hash]);
}

const resumeMemory = new Map<string, string>();
const resumeAddress = (owner: string, key: string) => `astra.upload.sessions.v2:${encodeURIComponent(owner)}:${encodeURIComponent(key)}`;
export function rememberUpload(owner: string, key: string, sessionId: string) {
  const address = resumeAddress(owner, key); resumeMemory.set(address, sessionId);
  try { localStorage.setItem(address, sessionId); } catch { /* Stable memory fallback when browser storage is disabled. */ }
}
export function rememberedUpload(owner: string, key: string): string | null {
  const address = resumeAddress(owner, key);
  if (resumeMemory.has(address)) return resumeMemory.get(address) ?? null;
  try { return localStorage.getItem(address); } catch { return null; }
}

interface CreationOptions { signal?: AbortSignal; isCurrent?: () => boolean; fileKeys?: string[] }
const safeId = (value: string, prefix: string) => value.startsWith(prefix) && value.length > prefix.length && !/[^A-Za-z0-9_-]/.test(value);

async function ownerCurrent(owner: string, options: CreationOptions) {
  if (options.signal?.aborted || options.isCurrent?.() === false
    || retryOwnerScope(await apiJson('/api/auth/session', options.signal)) !== owner
    || options.signal?.aborted || options.isCurrent?.() === false)
    throw new Error('Your account changed. Reconnect before resuming your saved upload.');
}

async function sendCreation<T>(owner: string, request: CreationRequest, options: CreationOptions,
  validate: (data: unknown) => T, sessions: (value: T) => UploadSession[]): Promise<T> {
  return creationIntents.run(owner, request, async saved => {
    await ownerCurrent(owner, options);
    const response = await fetch(saved.request.kind === 'batch' ? '/api/batches' : '/api/uploads', {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': saved.key },
      body: JSON.stringify(saved.request.payload), signal: options.signal,
    });
    const data: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : null;
      throw new ApiError(typeof detail === 'string' ? detail : 'Upload creation could not be confirmed. Retry the saved request.', response.status);
    }
    const result = validate(data);
    await ownerCurrent(owner, options);
    sessions(result).forEach((session, index) => rememberUpload(owner, saved.request.fileKeys[index]!, session.id));
    creationIntents.acknowledge(owner, saved.key);
    return result;
  });
}

function checkedSession(value: unknown, expected: { filename: string; size_bytes: number; clip_index?: number }): UploadSession {
  const parsed = UploadSessionSchema.safeParse(value);
  if (!parsed.success || !safeId(parsed.data.id, 'upload_') || parsed.data.filename !== expected.filename
    || parsed.data.size_bytes !== expected.size_bytes || parsed.data.offset > parsed.data.size_bytes
    || !['UPLOADING', 'COMPLETED', 'CANCELLED', 'FAILED'].includes(parsed.data.status)
    || parsed.data.status === 'COMPLETED' && parsed.data.analysis_id === null
    || expected.clip_index !== undefined && parsed.data.clip_index !== undefined && parsed.data.clip_index !== expected.clip_index
    || parsed.data.analysis_id !== null && !safeId(parsed.data.analysis_id, 'match_'))
    throw new Error('The server returned an invalid upload acknowledgement. Retry the saved request.');
  return parsed.data;
}

export async function createUploadBatch(files: File[], mode: UploadBatch['mode'], metadata: MatchMetadata, owner: string,
  options: CreationOptions = {}): Promise<UploadBatch> {
  const request: CreationRequest = { kind: 'batch', payload: { mode, metadata, files: files.map((file, index) => ({
    filename: file.name, size_bytes: file.size, clip_index: index,
  })) }, fileKeys: options.fileKeys ?? await Promise.all(files.map(fileResumeKey)) };
  return sendCreation(owner, request, options, data => {
    const envelope = z.object({ id: z.string(), mode: z.enum(['independent', 'same_match']), uploads: z.array(z.unknown()),
      grouping_status: z.string().optional(), reason: z.string().optional() }).safeParse(data);
    if (!envelope.success || !safeId(envelope.data.id, 'batch_') || envelope.data.mode !== mode || envelope.data.uploads.length !== files.length)
      throw new Error('The server returned an invalid batch acknowledgement. Retry the saved request.');
    const uploads = envelope.data.uploads.map((item, index) => checkedSession(item, request.payload.files[index]!));
    if (new Set(uploads.map(item => item.id)).size !== uploads.length)
      throw new Error('The server returned duplicate upload sessions. Retry the saved request.');
    return { ...envelope.data, uploads };
  }, batch => batch.uploads);
}

export async function createUploadSession(file: File, owner: string, matchId?: string, options: CreationOptions = {}): Promise<UploadSession> {
  const request: CreationRequest = { kind: 'upload', payload: { filename: file.name, size_bytes: file.size,
    ...(matchId ? { match_id: matchId } : {}) }, fileKeys: options.fileKeys ?? [await fileResumeKey(file)] };
  return sendCreation(owner, request, options, data => checkedSession(data, request.payload), session => [session]);
}

function sendChunk(file: Blob, id: string, offset: number, total: number, onProgress: (value: number) => void, signal: AbortSignal): Promise<UploadSession> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const clean = () => signal.removeEventListener('abort', abort);
    xhr.open('PATCH', `/api/uploads/${encodeURIComponent(id)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.setRequestHeader('Upload-Offset', String(offset)); xhr.responseType = 'json';
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(Math.min(99, Math.floor((offset + event.loaded) / total * 100))); };
    xhr.onload = () => { clean();
      if (xhr.status < 200 || xhr.status >= 300) { reject(new Error(typeof xhr.response?.detail === 'string' ? xhr.response.detail : 'Upload interrupted. Resume to check the saved offset.')); return; }
      const result = UploadSessionSchema.safeParse(xhr.response);
      if (result.success) resolve(result.data); else reject(new Error('Invalid upload status. Reconnect before resuming.'));
    };
    xhr.onerror = () => { clean(); reject(new Error('Connection lost. Your received chunks are saved; resume this file.')); };
    xhr.onabort = () => { clean(); reject(new DOMException('Upload paused', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { clean(); reject(new DOMException('Upload paused', 'AbortError')); return; }
    xhr.send(file);
  });
}

export async function resumeVideoUpload(file: File, id: string, onProgress: (value: number) => void, signal: AbortSignal): Promise<AnalysisJob> {
  let session = UploadSessionSchema.parse(await apiJson(`/api/uploads/${encodeURIComponent(id)}`, signal));
  if (session.filename !== file.name || session.size_bytes !== file.size || session.offset > file.size)
    throw new Error('This upload belongs to a different file. Choose the original file to resume.');
  if (session.analysis_id) return JobSchema.parse(await apiJson(`/api/analyses/${session.analysis_id}`, signal));
  onProgress(Math.floor(session.offset / file.size * 100));
  while (session.offset < file.size) {
    const offset = session.offset;
    session = await sendChunk(file.slice(offset, Math.min(offset + 8 * 1024 * 1024, file.size)), id, offset, file.size, onProgress, signal);
    if (session.offset <= offset || session.offset > file.size) throw new Error('Upload offset did not advance safely. Reconnect before resuming.');
  }
  if (signal.aborted) throw new DOMException('Upload paused', 'AbortError');
  onProgress(100);
  return JobSchema.parse(await mutate(`/api/uploads/${encodeURIComponent(id)}/complete`, {}));
}

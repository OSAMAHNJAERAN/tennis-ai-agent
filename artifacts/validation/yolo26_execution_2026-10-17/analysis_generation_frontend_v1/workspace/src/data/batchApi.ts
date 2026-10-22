import { z } from 'zod';
import { apiJson, JobSchema, type AnalysisJob } from './analysisApi';
import { mutate } from './productApi';

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

const resumeStorage = 'astra.upload.sessions.v1';
export function rememberUpload(key: string, sessionId: string) {
  try { const entries = JSON.parse(localStorage.getItem(resumeStorage) ?? '{}'); entries[key] = sessionId;
    localStorage.setItem(resumeStorage, JSON.stringify(entries)); } catch { /* Server state remains durable when browser storage is disabled. */ }
}
export function rememberedUpload(key: string): string | null {
  try { const value = JSON.parse(localStorage.getItem(resumeStorage) ?? '{}')[key]; return typeof value === 'string' ? value : null; } catch { return null; }
}

export async function createUploadBatch(files: File[], mode: UploadBatch['mode'], matchIds: string[]) {
  const data = await mutate<UploadBatch>('/api/batches', { mode, files: files.map((file, index) => ({
    filename: file.name, size_bytes: file.size, match_id: matchIds[index], clip_index: index,
  })) });
  return { ...data, uploads: data.uploads.map(session => UploadSessionSchema.parse(session)) };
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

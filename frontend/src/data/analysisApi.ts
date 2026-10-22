import { z } from 'zod';
import { retryIntents } from './retryIntent';

export const JobSchema = z.object({
  id: z.string(), filename: z.string(),
  status: z.enum(['UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED']),
  stage: z.string(), step: z.number().int().min(0).max(11),
  created_at: z.string(), started_at: z.string().nullable(), completed_at: z.string().nullable(),
  error: z.string().nullable(), size_bytes: z.number().nonnegative(),
  cancel_requested: z.boolean().optional(), retry_of: z.string().nullable().optional(),
});
export type AnalysisJob = z.infer<typeof JobSchema>;
export const HealthSchema = z.object({ max_upload_bytes: z.number().positive(), max_duration_seconds: z.number().positive(),
  auth_required: z.boolean().optional(), resumable_uploads: z.boolean().optional(), batch_max_files: z.number().int().positive().optional() });

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.name = 'ApiError'; this.status = status; }
}

export async function apiJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : null;
    throw new ApiError(typeof detail === 'string' ? detail : 'The analysis server is unavailable. Check the connection and retry.', response.status);
  }
  return data;
}

export async function listJobs(signal?: AbortSignal) {
  const result = z.array(JobSchema).safeParse(await apiJson('/api/analyses', signal));
  if (!result.success) throw new Error('The server returned invalid analysis history. Reconnect before retrying.');
  return result.data;
}

const RetrySessionSchema = z.object({ required: z.boolean(), authenticated: z.boolean(),
  user: z.object({ id: z.string().min(1) }).nullable() });

export function retryOwnerScope(value: unknown): string | null {
  const session = RetrySessionSchema.safeParse(value);
  if (!session.success) return null;
  if (!session.data.required) return 'local';
  return session.data.authenticated ? session.data.user?.id ?? null : null;
}

/** Reuse the saved key after any uncertain response, including refresh. */
export function retryAnalysis(sourceId: string, ownerId: string, isCurrent: () => boolean = () => true): Promise<AnalysisJob> {
  return retryIntents.run(ownerId, sourceId, async token => {
    const requireOwner = async () => {
      if (!isCurrent() || retryOwnerScope(await apiJson('/api/auth/session')) !== ownerId || !isCurrent())
        throw new Error('Your account changed. Refresh your analysis history before retrying.');
    };
    await requireOwner();
    const response = await fetch(`/api/analyses/${encodeURIComponent(sourceId)}/retry`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': token }, body: '{}',
    });
    const data: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : null;
      throw new ApiError(typeof detail === 'string' ? detail : 'Retry could not be confirmed. Reconnect and retry the saved request.', response.status);
    }
    const result = JobSchema.safeParse(data);
    if (!result.success || result.data.id === sourceId || !result.data.id.startsWith('match_')
      || result.data.id.length <= 6 || /[^A-Za-z0-9_-]/.test(result.data.id)
      || result.data.retry_of !== sourceId)
      throw new Error('The server returned an invalid retry acknowledgement. Reconnect and retry the saved request.');
    await requireOwner();
    retryIntents.acknowledge(ownerId, sourceId, token, result.data.id);
    return result.data;
  });
}

export function uploadVideo(file: File, onProgress: (percent: number) => void, signal: AbortSignal, matchId?: string): Promise<AnalysisJob> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const clean = () => signal.removeEventListener('abort', abort);
    xhr.open('POST', `/api/analyses?filename=${encodeURIComponent(file.name)}${matchId ? `&match_id=${encodeURIComponent(matchId)}` : ""}`);
    xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
    xhr.responseType = 'json';
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.onload = () => {
      clean();
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail: unknown = xhr.response?.detail;
        reject(new Error(typeof detail === 'string' ? detail : 'Upload failed. Check the analysis server and retry.'));
        return;
      }
      const result = JobSchema.safeParse(xhr.response);
      if (result.success) resolve(result.data);
      else reject(new Error('The server returned an invalid job. Refresh the analysis history before retrying.'));
    };
    xhr.onerror = () => { clean(); reject(new Error('Connection lost. Check analysis history before uploading again.')); };
    xhr.onabort = () => { clean(); reject(new DOMException('Upload canceled', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) { clean(); reject(new DOMException('Upload canceled', 'AbortError')); return; }
    xhr.send(file);
  });
}

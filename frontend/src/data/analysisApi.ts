import { z } from 'zod';

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

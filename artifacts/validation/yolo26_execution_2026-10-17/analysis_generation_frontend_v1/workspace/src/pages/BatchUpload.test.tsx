import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import Upload from './Upload';
import { apiJson, listJobs, type AnalysisJob } from '../data/analysisApi';
import { mutate } from '../data/productApi';
import { createUploadBatch, resumeVideoUpload } from '../data/batchApi';

vi.mock('../data/analysisApi', async original => ({ ...await original<typeof import('../data/analysisApi')>(), apiJson: vi.fn(), listJobs: vi.fn() }));
vi.mock('../data/productApi', async original => ({ ...await original<typeof import('../data/productApi')>(), mutate: vi.fn() }));
vi.mock('../data/batchApi', async original => ({ ...await original<typeof import('../data/batchApi')>(), createUploadBatch: vi.fn(), resumeVideoUpload: vi.fn(),
  fileResumeKey: async (file: File) => file.name, rememberedUpload: () => null, rememberUpload: vi.fn() }));
const job = (name: string, id: string): AnalysisJob => ({ id, filename: name, status: 'UPLOADED', stage: 'Waiting', step: 0,
  created_at: '2026-10-08T00:00:00Z', started_at: null, completed_at: null, error: null, size_bytes: 16 });
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(apiJson).mockImplementation(async path => {
    if (path === '/api/health') return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true };
    if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
    if (path === '/api/players') return [];
    if (path === '/api/auth/session') return { required: false, authenticated: false, user: null };
    return { id: 'batch_one', mode: 'independent', uploads: [] };
  });
  vi.mocked(listJobs).mockResolvedValue([]);
  vi.mocked(mutate).mockResolvedValue({ id: 'session_draft' });
  vi.mocked(createUploadBatch).mockResolvedValue({ id: 'batch_one', mode: 'independent', uploads: [
    { id: 'upload_one', filename: 'first.mp4', size_bytes: 16, offset: 0, status: 'UPLOADING', analysis_id: null },
    { id: 'upload_two', filename: 'second.mp4', size_bytes: 16, offset: 0, status: 'UPLOADING', analysis_id: null },
  ] });
});

describe('independent file progress', () => {
  it('keeps the second upload usable when the first fails', async () => {
    vi.mocked(resumeVideoUpload).mockRejectedValueOnce(new Error('Connection lost. Resume this file.'))
      .mockResolvedValueOnce(job('second.mp4', 'match_second'));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), [new File(['1234567890123456'], 'first.mp4'), new File(['1234567890123456'], 'second.mp4')]);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    const files = within(screen.getByLabelText('Selected videos'));
    expect(await files.findByRole('alert')).toHaveTextContent('Connection lost. Resume this file.');
    expect(await files.findByText('Received · Saved analysis created')).toBeInTheDocument();
    expect(resumeVideoUpload).toHaveBeenCalledTimes(2);
    expect(createUploadBatch).toHaveBeenCalledOnce();
    expect(await screen.findByRole('region', { name: 'Saved batch' })).toBeInTheDocument();
  });
  it('requires explicit same-match order and keeps grouping unverified', async () => {
    vi.mocked(resumeVideoUpload).mockResolvedValue(job('second.mp4', 'match_second'));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), [new File(['1234567890123456'], 'first.mp4'), new File(['1234567890123456'], 'second.mp4')]);
    await user.click(screen.getByLabelText('Clips from the same match, in the order shown'));
    expect(screen.getByText(/Identity, overlap and timing require review/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    await waitFor(() => expect(createUploadBatch).toHaveBeenCalledWith(expect.any(Array), 'same_match', ['session_draft', 'session_draft']));
  });
});

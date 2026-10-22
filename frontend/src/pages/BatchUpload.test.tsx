import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import Upload from './Upload';
import { apiJson, listJobs, type AnalysisJob } from '../data/analysisApi';
import { mutate } from '../data/productApi';
import { createUploadBatch, rememberUpload, resumeVideoUpload } from '../data/batchApi';
import { CreationIntentConflict, type CreationRequest } from '../data/creationIntent';
import { emptyMetadata } from '../data/productApi';

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
    await waitFor(() => expect(createUploadBatch).toHaveBeenCalledWith(expect.any(Array), 'same_match', emptyMetadata(), 'local',
      expect.objectContaining({ signal: expect.any(AbortSignal), isCurrent: expect.any(Function), fileKeys: ['first.mp4', 'second.mp4'] })));
    expect(mutate).not.toHaveBeenCalled();
  });

  it('restores original metadata after an uncertain creation instead of posting edited details as a new batch', async () => {
    const original = { ...emptyMetadata(), tournament: 'Original cup' };
    const saved: CreationRequest = { kind: 'batch', payload: { mode: 'independent', metadata: original, files: [
      { filename: 'first.mp4', size_bytes: 16, clip_index: 0 }, { filename: 'second.mp4', size_bytes: 16, clip_index: 1 },
    ] }, fileKeys: ['first.mp4', 'second.mp4'] };
    vi.mocked(createUploadBatch).mockRejectedValueOnce(new CreationIntentConflict(saved));
    vi.mocked(resumeVideoUpload).mockImplementation(async file => job(file.name, 'match_'+file.name.replace('.mp4', '')));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), [new File(['1234567890123456'], 'first.mp4'), new File(['1234567890123456'], 'second.mp4')]);
    await user.type(screen.getByLabelText('Tournament'), 'Edited cup');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('previous upload request is still unconfirmed');
    await user.click(screen.getByRole('button', { name: 'Restore saved match details' }));
    expect(screen.getByLabelText('Tournament')).toHaveValue('Original cup');
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    await waitFor(() => expect(createUploadBatch).toHaveBeenCalledTimes(2));
    expect(vi.mocked(createUploadBatch).mock.calls[1]?.[2]).toEqual(original);
    expect(mutate).not.toHaveBeenCalled();
  });

  it('prevents rapid repeated Analyze clicks while atomic creation is pending', async () => {
    let finish: (value: Awaited<ReturnType<typeof createUploadBatch>>) => void = () => undefined;
    vi.mocked(createUploadBatch).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    vi.mocked(resumeVideoUpload).mockResolvedValue(job('first.mp4', 'match_first'));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    await userEvent.setup().upload(screen.getByLabelText('Choose tennis video'), new File(['1234567890123456'], 'first.mp4'));
    const analyze = screen.getByRole('button', { name: 'Analyze match' });
    await waitFor(() => expect(analyze).toBeEnabled());
    fireEvent.click(analyze); fireEvent.click(analyze);
    await waitFor(() => expect(createUploadBatch).toHaveBeenCalledOnce());
    await act(async () => finish({ id: 'batch_one', mode: 'independent', uploads: [
      { id: 'upload_one', filename: 'first.mp4', size_bytes: 16, offset: 0, status: 'UPLOADING', analysis_id: null },
    ] }));
  });

  it('aborts pending creation and discards its late response after sign-out', async () => {
    let signedIn = true;
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/players') return [];
      if (path === '/api/auth/session') return { required: true, authenticated: signedIn, user: signedIn ? { id: 'alice', username: 'alice' } : null };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true, auth_required: true };
    });
    vi.mocked(mutate).mockImplementation(async () => { signedIn = false; return {}; });
    let finish: (value: Awaited<ReturnType<typeof createUploadBatch>>) => void = () => undefined;
    vi.mocked(createUploadBatch).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), new File(['1234567890123456'], 'first.mp4'));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    await waitFor(() => expect(createUploadBatch).toHaveBeenCalledOnce());
    const signal = vi.mocked(createUploadBatch).mock.calls[0]?.[4]?.signal;
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    expect(signal?.aborted).toBe(true);
    await act(async () => finish({ id: 'batch_one', mode: 'independent', uploads: [
      { id: 'upload_one', filename: 'first.mp4', size_bytes: 16, offset: 0, status: 'UPLOADING', analysis_id: null },
    ] }));
    expect(resumeVideoUpload).not.toHaveBeenCalled(); expect(rememberUpload).not.toHaveBeenCalled();
    expect(screen.queryByRole('region', { name: 'Saved batch' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Selected videos')).not.toBeInTheDocument();
  });

  it('aborts file transfer and ignores late progress and accepted jobs after sign-out', async () => {
    let signedIn = true;
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/players') return [];
      if (path.startsWith('/api/batches/')) return { id: 'batch_one', mode: 'independent', uploads: [] };
      if (path === '/api/auth/session') return { required: true, authenticated: signedIn, user: signedIn ? { id: 'alice', username: 'alice' } : null };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true, auth_required: true };
    });
    vi.mocked(mutate).mockImplementation(async () => { signedIn = false; return {}; });
    let finish: (value: AnalysisJob) => void = () => undefined;
    let progress: (value: number) => void = () => undefined;
    let transferSignal: AbortSignal | undefined;
    vi.mocked(resumeVideoUpload).mockImplementation((_file, _id, onProgress, signal) => {
      progress = onProgress; transferSignal = signal;
      return new Promise(resolve => { finish = resolve; });
    });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), [new File(['1234567890123456'], 'first.mp4'), new File(['1234567890123456'], 'second.mp4')]);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    await waitFor(() => expect(resumeVideoUpload).toHaveBeenCalledOnce());
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    expect(transferSignal?.aborted).toBe(true);
    await act(async () => { progress(90); finish(job('first.mp4', 'match_alice_late')); });
    expect(resumeVideoUpload).toHaveBeenCalledOnce();
    expect(screen.queryByRole('heading', { name: 'first.mp4' })).not.toBeInTheDocument();
    expect(screen.queryByRole('progressbar', { name: 'Video upload progress' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Saved batch' })).not.toBeInTheDocument();
  });
});

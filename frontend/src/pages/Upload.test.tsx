import { act, render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Upload from './Upload';
import { ApiError, apiJson, listJobs, retryAnalysis, uploadVideo, type AnalysisJob } from '../data/analysisApi';
import { mutate } from '../data/productApi';
import type { PublicDemo } from '../data/publicDemos';

vi.mock('../data/productApi', async importOriginal => ({
  ...await importOriginal<typeof import('../data/productApi')>(), mutate: vi.fn(),
}));

vi.mock('../data/analysisApi', async importOriginal => ({
  ...await importOriginal<typeof import('../data/analysisApi')>(),
  apiJson: vi.fn(), listJobs: vi.fn(), retryAnalysis: vi.fn(), uploadVideo: vi.fn(),
}));
const job: AnalysisJob = { id: 'match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', filename: 'tennis.mp4', status: 'UPLOADED',
  stage: 'Waiting for the analysis worker', step: 0, created_at: '2026-09-19T00:00:00Z',
  started_at: null, completed_at: null, error: null, size_bytes: 20 };
const exampleJob: AnalysisJob = { ...job, id: 'match_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', filename: 'public-example.mp4', status: 'COMPLETED', stage: 'Analysis complete' };
const demo: PublicDemo = {
  id: 'licensed-example', title: 'Licensed public example', analysisId: exampleJob.id,
  previewUrl: `/api/analyses/${exampleJob.id}/media/annotated`, qualificationEvidence: false,
  attribution: { author: 'Camera author', sourcePage: 'https://commons.wikimedia.org/wiki/File:Tennis.webm', license: 'CC BY-SA 4.0', licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/', modifications: 'Annotated.' },
};

beforeEach(() => {
  vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [] } : path === '/api/players' ? [] : { max_upload_bytes: 1024, max_duration_seconds: 7200 });
  vi.mocked(mutate).mockResolvedValue({ id: 'session_draft' });
  vi.mocked(listJobs).mockResolvedValue([]);
});
afterEach(() => vi.clearAllMocks());

describe('video upload workflow', () => {
  it('discards delayed initial history when the owner changes between the before and after checks', async () => {
    const alice = { ...job, filename: 'alice-private.mp4', status: 'FAILED' as const };
    const bob = { ...job, id: 'match_bob', filename: 'bob-private.mp4', status: 'FAILED' as const };
    let owner = 'alice';
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/auth/session') return { required: true, authenticated: true, user: { id: owner, username: owner } };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true, auth_required: true };
    });
    let finish: (value: AnalysisJob[]) => void = () => undefined;
    vi.mocked(listJobs).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; })).mockResolvedValue([bob]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    await waitFor(() => expect(listJobs).toHaveBeenCalledOnce());
    owner = 'bob';
    await act(async () => finish([alice]));
    const row = within((await screen.findByRole('heading', { name: bob.filename })).closest('article')!);
    expect(screen.queryByRole('heading', { name: alice.filename })).not.toBeInTheDocument();
    vi.mocked(retryAnalysis).mockResolvedValueOnce({ ...job, id: 'match_bob_child', retry_of: bob.id });
    await userEvent.setup().click(row.getByRole('button', { name: 'Retry analysis' }));
    expect(retryAnalysis).toHaveBeenCalledExactlyOnceWith(bob.id, 'bob', expect.any(Function));
  });

  it('invalidates a pending cancel acknowledgement after a refresh observes 401', async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(apiJson).mockImplementation(async path => {
        if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
        if (path === '/api/auth/session') return { required: true, authenticated: true, user: { id: 'alice', username: 'alice' } };
        return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true, auth_required: true };
      });
      vi.mocked(listJobs).mockResolvedValue([job]);
      let finish: (value: AnalysisJob) => void = () => undefined;
      vi.mocked(mutate).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
      render(<MemoryRouter><Upload /></MemoryRouter>);
      await act(async () => undefined);
      fireEvent.click(screen.getByRole('button', { name: 'Cancel analysis' }));
      vi.mocked(listJobs).mockRejectedValue(new ApiError('Sign in to access your matches.', 401));
      await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
      expect(screen.getByText('Sign in to upload videos and view your private analysis history.')).toBeInTheDocument();
      await act(async () => finish({ ...job, status: 'FAILED', cancel_requested: true }));
      expect(screen.queryByRole('heading', { name: job.filename })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Retry analysis' })).not.toBeInTheDocument();
    } finally { vi.useRealTimers(); }
  });

  it('clears native file bytes when a history refresh observes an expired owner session', async () => {
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/players') return [];
      if (path === '/api/auth/session') return { required: true, authenticated: true, user: { id: 'alice', username: 'alice' } };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true, auth_required: true };
    });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const chooser = screen.getByLabelText('Choose tennis video') as HTMLInputElement;
    await userEvent.setup().upload(chooser,
      new File(['private-video'], 'alice-private.mp4', { type: 'video/mp4' }));
    expect(chooser.files).toHaveLength(1);
    expect(chooser.value).toContain('alice-private.mp4');
    vi.mocked(listJobs).mockRejectedValue(new ApiError('Sign in to access your matches.', 401));
    await waitFor(() => expect(screen.getByText('Sign in to upload videos and view your private analysis history.')).toBeInTheDocument(), { timeout: 4000 });
    expect(chooser).toHaveValue('');
    expect(chooser.files).toHaveLength(0);
    expect(screen.queryByLabelText('Selected videos')).not.toBeInTheDocument();
  });

  it('withholds account history if an authenticated server cannot identify its owner', async () => {
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/auth/session') return null;
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, auth_required: true };
    });
    vi.mocked(listJobs).mockResolvedValue([job]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    expect(await screen.findByRole('alert')).toHaveTextContent('Your account could not be confirmed');
    expect(screen.queryByRole('heading', { name: job.filename })).not.toBeInTheDocument();
  });

  it('uses the retry seam once and disables rapid repeat clicks until acknowledged', async () => {
    const failed = { ...job, status: 'FAILED' as const, error: 'Interrupted' };
    const child = { ...job, id: 'match_child', filename: 'retried.mp4', retry_of: failed.id };
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/auth/session') return { required: false, authenticated: false, user: null };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true };
    });
    vi.mocked(listJobs).mockResolvedValue([failed]);
    let finish: (value: AnalysisJob) => void = () => undefined;
    vi.mocked(retryAnalysis).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const button = await screen.findByRole('button', { name: 'Retry analysis' });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button); fireEvent.click(button);
    expect(retryAnalysis).toHaveBeenCalledExactlyOnceWith(failed.id, 'local', expect.any(Function));
    expect(screen.getByRole('button', { name: 'Retrying analysis…' })).toBeDisabled();
    vi.mocked(listJobs).mockResolvedValue([child, failed]);
    await act(async () => finish(child));
    expect(await screen.findByRole('heading', { name: child.filename })).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it('keeps retry errors actionable and treats a failed child as its own retry source', async () => {
    const failed = { ...job, status: 'FAILED' as const, error: 'Interrupted' };
    const child = { ...failed, id: 'match_child', filename: 'failed-child.mp4', retry_of: failed.id };
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/auth/session') return { required: false, authenticated: false, user: null };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true };
    });
    vi.mocked(listJobs).mockResolvedValue([failed]);
    vi.mocked(retryAnalysis).mockRejectedValueOnce(new Error('Connection lost. Retry the saved request.')).mockImplementationOnce(async () => {
      vi.mocked(listJobs).mockResolvedValue([child, failed]); return child;
    }).mockResolvedValueOnce({ ...job, id: 'match_grandchild', retry_of: child.id });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    const retry = await screen.findByRole('button', { name: 'Retry analysis' });
    await waitFor(() => expect(retry).toBeEnabled());
    await user.click(retry);
    expect(await screen.findByRole('alert')).toHaveTextContent('Retry the saved request');
    expect(retry).toBeEnabled();
    await user.click(retry);
    const childRow = within((await screen.findByRole('heading', { name: child.filename })).closest('article')!);
    await user.click(childRow.getByRole('button', { name: 'Retry analysis' }));
    expect(vi.mocked(retryAnalysis).mock.calls.map(([source]) => source)).toEqual([failed.id, failed.id, child.id]);
  });

  it('suppresses an earlier account retry response after sign-out', async () => {
    const failed = { ...job, status: 'FAILED' as const, error: 'Interrupted' };
    let signedIn = true;
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') return { schemaVersion: 1, demos: [] };
      if (path === '/api/auth/session') return { required: true, authenticated: signedIn, user: signedIn ? { id: 'alice', username: 'alice' } : null };
      return { max_upload_bytes: 1024, max_duration_seconds: 7200, resumable_uploads: true };
    });
    vi.mocked(listJobs).mockResolvedValue([failed]);
    let finish: (value: AnalysisJob) => void = () => undefined;
    vi.mocked(retryAnalysis).mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    vi.mocked(mutate).mockImplementation(async () => { signedIn = false; vi.mocked(listJobs).mockResolvedValue([]); return {}; });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    const retry = await screen.findByRole('button', { name: 'Retry analysis' });
    await waitFor(() => expect(retry).toBeEnabled());
    await user.click(retry);
    const isCurrent = vi.mocked(retryAnalysis).mock.calls[0]![2]!;
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    expect(isCurrent()).toBe(false);
    await act(async () => finish({ ...job, id: 'match_old_account_child', filename: 'private-old-account.mp4', retry_of: failed.id }));
    expect(screen.queryByRole('heading', { name: 'private-old-account.mp4' })).not.toBeInTheDocument();
  });

  it('treats authentication-required as account access while retaining a healthy service', async () => {
    vi.mocked(listJobs).mockRejectedValue(new ApiError('Sign in to access your matches.', 401));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), new File(['video'], 'tennis.mp4', { type: 'video/mp4' }));
    expect(await screen.findByText('Sign in to upload videos and view your private analysis history.')).toBeInTheDocument();
    expect(screen.queryByText('Analysis server unavailable')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze match' })).toBeDisabled();
  });
  it('accepts a selected video and opens persistent processing status', async () => {
    vi.mocked(uploadVideo).mockImplementation(async () => {
      vi.mocked(listJobs).mockResolvedValue([job]);
      return job;
    });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), new File(['video'], 'tennis.mp4', { type: 'video/mp4' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    expect(await screen.findByRole('heading', { name: 'Analyzing your tennis match…' })).toBeInTheDocument();
    expect(uploadVideo).toHaveBeenCalledOnce();
  });

  it('rejects an unsupported dropped file without uploading', async () => {
    render(<MemoryRouter><Upload /></MemoryRouter>);
    fireEvent.drop(screen.getByRole('heading', { name: 'Drop your tennis video here' }).parentElement!,
      { dataTransfer: { files: [new File(['x'], 'bad.exe')] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Choose an MP4, MOV or AVI');
    expect(uploadVideo).not.toHaveBeenCalled();
  });

  it('shows actual upload progress while keeping cancellation available', async () => {
    let finish: (value: AnalysisJob) => void = () => undefined;
    vi.mocked(uploadVideo).mockImplementation((_file, onProgress) => {
      onProgress(40);
      return new Promise(resolve => { finish = resolve; });
    });
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText('Choose tennis video'), new File(['video'], 'tennis.mp4', { type: 'video/mp4' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze match' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Analyze match' }));
    expect(screen.getByRole('progressbar', { name: 'Video upload progress' })).toHaveAttribute('value', '40');
    expect(screen.getByRole('button', { name: 'Cancel upload' })).toBeEnabled();
    await act(async () => finish(job));
  });

  it('restores completion and a deep result link from job history', async () => {
    vi.mocked(listJobs).mockResolvedValue([{ ...job, status: 'COMPLETED', stage: 'Analysis complete' }]);
    render(<MemoryRouter initialEntries={[`/upload?job=${job.id}`]}><Upload /></MemoryRouter>);
    expect(await screen.findByRole('heading', { name: 'Your analysis is ready' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open analysis' })).toHaveAttribute('href', `/analysis/${job.id}`);
  });

  it('shows a recoverable connection error rather than losing a job', async () => {
    vi.mocked(listJobs).mockRejectedValue(new Error('Server unavailable'));
    render(<MemoryRouter><Upload /></MemoryRouter>);
    expect(await screen.findByRole('alert')).toHaveTextContent('Your saved jobs will reappear');
    expect(screen.getByRole('button', { name: 'Reconnect' })).toBeEnabled();
  });

  it('excludes public examples from history without removing the active job', async () => {
    vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [demo] } : { max_upload_bytes: 1024, max_duration_seconds: 7200 });
    vi.mocked(listJobs).mockResolvedValue([exampleJob, job]);
    render(<MemoryRouter initialEntries={[`/upload?job=${exampleJob.id}`]}><Upload /></MemoryRouter>);
    const history = within(screen.getByRole('region', { name: 'Analysis history' }));
    expect(await history.findByRole('heading', { name: job.filename })).toBeInTheDocument();
    expect(history.getByText('1 sessions')).toBeInTheDocument();
    expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Current analysis' })).toHaveTextContent(exampleJob.filename);
    expect(screen.getByRole('link', { name: 'Open analysis' })).toHaveAttribute('href', `/analysis/${exampleJob.id}`);
    expect(apiJson).toHaveBeenCalledWith('/api/demos');
  });

  it('shows the private archive empty state when only public examples exist', async () => {
    vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [demo] } : { max_upload_bytes: 1024, max_duration_seconds: 7200 });
    vi.mocked(listJobs).mockResolvedValue([exampleJob]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const history = within(screen.getByRole('region', { name: 'Analysis history' }));
    expect(await history.findByText(/Your uploaded sessions will appear here/)).toBeInTheDocument();
    expect(history.getByText('0 sessions')).toBeInTheDocument();
    expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
  });

  it.each(['loading', 'unavailable'] as const)('keeps saved completion and queued details available while public example filtering is %s', async state => {
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') {
        if (state === 'loading') return new Promise(() => undefined);
        throw new Error('Catalog unavailable');
      }
      return { max_upload_bytes: 1024, max_duration_seconds: 7200 };
    });
    const completedJob: AnalysisJob = { ...job, id: 'match_cccccccccccccccccccccccccccccccc', filename: 'saved-completed.mp4', status: 'COMPLETED', stage: 'Analysis complete' };
    vi.mocked(listJobs).mockResolvedValue([exampleJob, completedJob, job]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const history = within(screen.getByRole('region', { name: 'Analysis history' }));
    expect(await history.findByRole('link', { name: `Open analysis for ${completedJob.filename}` })).toHaveAttribute('href', `/analysis/${completedJob.id}`);
    expect(history.getByRole('heading', { name: job.filename })).toBeInTheDocument();
    expect(history.getByRole('heading', { name: exampleJob.filename })).toBeInTheDocument();
    expect(history.getByRole('heading', { name: 'Saved analyses' })).toBeInTheDocument();
    expect(history.getByText('3 saved analyses')).toBeInTheDocument();
    expect(history.getByRole('status')).toHaveTextContent('Public example filtering is unavailable');
    expect(history.getByRole('status')).toHaveTextContent('may include public examples');
    const queuedRecord = within(history.getByRole('heading', { name: job.filename }).closest('article')!);
    await userEvent.setup().click(queuedRecord.getByRole('button', { name: 'Details' }));
    expect(screen.getByRole('region', { name: 'Current analysis' })).toHaveTextContent(job.filename);
    expect(history.getByRole('link', { name: `Open analysis for ${completedJob.filename}` })).toBeInTheDocument();
  });

  it('keeps saved analyses accessible and restores public filtering when the catalog recovers', async () => {
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') throw new Error('Catalog unavailable');
      return { max_upload_bytes: 1024, max_duration_seconds: 7200 };
    });
    vi.mocked(listJobs).mockResolvedValue([exampleJob, job]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const history = within(screen.getByRole('region', { name: 'Analysis history' }));
    expect(await history.findByRole('heading', { name: job.filename })).toBeInTheDocument();
    expect(history.getByRole('status')).toHaveTextContent('Public example filtering is unavailable');
    expect(history.getByText('2 saved analyses')).toBeInTheDocument();
    expect(history.getByRole('link', { name: `Open analysis for ${exampleJob.filename}` })).toHaveAttribute('href', `/analysis/${exampleJob.id}`);
    expect(history.getByRole('link', { name: 'My analyses' })).toHaveAttribute('href', '/match-center');
    vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [demo] } : { max_upload_bytes: 1024, max_duration_seconds: 7200 });
    await userEvent.setup().click(history.getByRole('button', { name: 'Retry history' }));
    await waitFor(() => expect(history.getByText('1 sessions')).toBeInTheDocument());
    expect(history.getByRole('heading', { name: job.filename })).toBeInTheDocument();
    expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
    expect(history.queryByRole('status')).not.toBeInTheDocument();
  });

  it.each(['unavailable', 'malformed'] as const)('retains verified public filtering when a later catalog is %s', async state => {
    vi.useFakeTimers();
    try {
      let catalogRequests = 0;
      vi.mocked(apiJson).mockImplementation(async path => {
        if (path === '/api/demos') {
          if (++catalogRequests === 1) return { schemaVersion: 1, demos: [demo] };
          if (state === 'unavailable') throw new Error('Catalog unavailable');
          return { schemaVersion: 1, demos: [{ id: 'incomplete-public-record' }] };
        }
        return { max_upload_bytes: 1024, max_duration_seconds: 7200 };
      });
      const completedJob: AnalysisJob = { ...job, id: 'match_cccccccccccccccccccccccccccccccc', filename: 'saved-completed.mp4', status: 'COMPLETED', stage: 'Analysis complete' };
      vi.mocked(listJobs).mockResolvedValue([exampleJob, completedJob, job]);
      render(<MemoryRouter><Upload /></MemoryRouter>);
      await act(async () => undefined);
      const history = within(screen.getByRole('region', { name: 'Analysis history' }));
      expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
      expect(history.getByText('2 sessions')).toBeInTheDocument();
      await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
      expect(history.getByRole('status')).toHaveTextContent('Public example filtering is unavailable');
      expect(history.getByRole('link', { name: `Open analysis for ${completedJob.filename}` })).toHaveAttribute('href', `/analysis/${completedJob.id}`);
      expect(history.getByRole('button', { name: 'Details' })).toBeEnabled();
      expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
      expect(history.getByText('2 saved analyses')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

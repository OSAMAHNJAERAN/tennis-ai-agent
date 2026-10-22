import { act, render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Upload from './Upload';
import { apiJson, listJobs, uploadVideo, type AnalysisJob } from '../data/analysisApi';
import { mutate } from '../data/productApi';
import type { PublicDemo } from '../data/publicDemos';

vi.mock('../data/productApi', async importOriginal => ({
  ...await importOriginal<typeof import('../data/productApi')>(), mutate: vi.fn(),
}));

vi.mock('../data/analysisApi', async importOriginal => ({
  ...await importOriginal<typeof import('../data/analysisApi')>(),
  apiJson: vi.fn(), listJobs: vi.fn(), uploadVideo: vi.fn(),
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

  it('offers recovery without classifying unverified public examples as private sessions', async () => {
    vi.mocked(apiJson).mockImplementation(async path => {
      if (path === '/api/demos') throw new Error('Catalog unavailable');
      return { max_upload_bytes: 1024, max_duration_seconds: 7200 };
    });
    vi.mocked(listJobs).mockResolvedValue([exampleJob, job]);
    render(<MemoryRouter><Upload /></MemoryRouter>);
    const history = within(screen.getByRole('region', { name: 'Analysis history' }));
    expect(await history.findByRole('alert')).toHaveTextContent('Unable to verify your session archive');
    expect(history.getByText('— sessions')).toBeInTheDocument();
    expect(history.queryByRole('heading', { name: exampleJob.filename })).not.toBeInTheDocument();
    expect(history.getByRole('link', { name: 'My analyses' })).toHaveAttribute('href', '/match-center');
    vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [demo] } : { max_upload_bytes: 1024, max_duration_seconds: 7200 });
    await userEvent.setup().click(history.getByRole('button', { name: 'Retry history' }));
    expect(await history.findByRole('heading', { name: job.filename })).toBeInTheDocument();
    expect(history.getByText('1 sessions')).toBeInTheDocument();
  });
});

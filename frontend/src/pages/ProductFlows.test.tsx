import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ProductHome, { MatchCenter } from './ProductHome';
import NewLiveMatch, { ManualLiveDesk } from './ManualLive';
import { LegacyAnalysis } from './ProductWorkspace';
import { apiJson } from '../data/analysisApi';
import { emptyMetadata, mutate, type ProductMatch } from '../data/productApi';
import { AnalysisProvider, useAnalysis } from '../app/AnalysisContext';
import { analysisRepository, snapshot } from '../data/repository';

vi.mock('../data/analysisApi', async original => ({ ...await original<typeof import('../data/analysisApi')>(), apiJson: vi.fn() }));
vi.mock('../data/productApi', async original => ({ ...await original<typeof import('../data/productApi')>(), mutate: vi.fn() }));

const live: ProductMatch = {
  id: 'session_live', source: 'manual', analysis_id: null, revision: 0,
  created_at: '2026-09-19T00:00:00Z', updated_at: '2026-09-19T00:00:00Z',
  metadata: { ...emptyMetadata(), participants: [{ name: 'Avery', profile_id: null }, { name: 'Jordan', profile_id: null }], tournament: 'Club final' },
  metadata_source: 'user', status: 'LIVE', filename: null, stage: null, error: null, notes: '', analysis_summary: null,
  score: { state: { games_p1: 0, games_p2: 0, sets_won_p1: 0, sets_won_p2: 0, server_id: 1, match_complete: false, winner_id: null, completed_sets: [] }, display_points: ['0', '0'], history: [], statistics: [0, 1].map(() => ({ points_won: 0, service_points: 0, service_points_won: 0, return_points: 0, return_points_won: 0 })) },
  predictions: { status: 'unavailable', reason: 'No prediction service is connected.' },
};
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [] } : path === '/api/matches' ? [live] : path === '/api/players' ? [] : path.includes('by-analysis') ? { ...live, id: 'exact-match' } : { status: 'ok', worker_available: true });
});
afterEach(() => vi.restoreAllMocks());

describe('product journeys', () => {
  it.each([ProductHome, MatchCenter])('keeps a failed library request distinct from an empty library in %s', async Page => {
    vi.mocked(apiJson).mockRejectedValue(new Error('Service unavailable'));
    render(<MemoryRouter><Page /></MemoryRouter>);
    expect(await screen.findByRole('alert')).toHaveTextContent('Service unavailable');
    expect(screen.queryByText(/No matches in progress|Completed matches will appear|No matches match this view/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled();
  });

  it('focuses home on uploads and keeps manual records out of the video library', async () => {
    const analysis = vi.spyOn(analysisRepository, 'getRun');
    render(<MemoryRouter><ProductHome /></MemoryRouter>);
    expect(await screen.findByRole('link', { name: /Upload video/ })).toHaveAttribute('href', '/upload');
    expect(screen.queryByRole('link', { name: /Live match analysis/ })).not.toBeInTheDocument();
    expect(screen.queryByText('Avery vs Jordan')).not.toBeInTheDocument();
    expect(analysis).not.toHaveBeenCalled();
    expect(screen.queryByText(/98.7%|Hawk-Eye|74%/)).not.toBeInTheDocument();
  });

  it('filters match records by names and status', async () => {
    vi.mocked(apiJson).mockResolvedValue([{ ...live, source: 'upload', status: 'PROCESSING', filename: 'club.mp4' }]);
    render(<MemoryRouter><MatchCenter /></MemoryRouter>);
    await screen.findByText('Avery vs Jordan');
    fireEvent.change(screen.getByLabelText('Find an analysis'), { target: { value: 'missing' } });
    expect(screen.getByText(/No analyses match this view/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Find an analysis'), { target: { value: 'club' } });
    expect(screen.getByText('Avery vs Jordan')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Status'), { target: { value: 'completed' } });
    expect(screen.queryByText('Avery vs Jordan')).not.toBeInTheDocument();
  });

  it('requires an initial server and sends explicit manual match details', async () => {
    vi.mocked(mutate).mockResolvedValue(live);
    render(<MemoryRouter><NewLiveMatch /></MemoryRouter>);
    const user = userEvent.setup();
    expect(screen.getByRole('button', { name: 'Start live match' })).toBeDisabled();
    await user.type(screen.getByLabelText('Player 1'), 'Avery');
    await user.selectOptions(screen.getByLabelText('Initial server'), '2');
    await user.click(screen.getByRole('button', { name: 'Start live match' }));
    expect(mutate).toHaveBeenCalledWith('/api/matches', expect.objectContaining({ source: 'manual', initial_server: 2, best_of: 3, metadata: expect.objectContaining({ role_assignment: 'unconfirmed' }) }));
  });

  it('retries an uncertain point with exactly the same idempotency key', async () => {
    const user = userEvent.setup();
    vi.mocked(mutate).mockRejectedValueOnce(new Error('Connection lost')).mockResolvedValueOnce(live);
    render(<ManualLiveDesk match={live} onUpdate={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: 'Point · Avery' }));
    await screen.findByText('Connection lost');
    expect(screen.getByRole('button', { name: 'Point · Jordan' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Retry same update' }));
    const calls = vi.mocked(mutate).mock.calls.slice(-2);
    expect(calls[0]?.[1]).toEqual(calls[1]?.[1]);
  });

  it('resolves an explicit legacy ID ahead of a conflicting query and preserves event', async () => {
    render(<MemoryRouter initialEntries={['/analysis/match_exact?run=match_wrong&event=7']}><Routes><Route path="/analysis/:analysisId" element={<LegacyAnalysis />} /><Route path="/matches/:matchId" element={<div>Resolved exact match</div>} /></Routes></MemoryRouter>);
    expect(await screen.findByText('Resolved exact match')).toBeInTheDocument();
    expect(apiJson).toHaveBeenCalledWith('/api/matches/by-analysis/match_exact');
    expect(apiJson).not.toHaveBeenCalledWith('/api/matches/by-analysis/match_wrong');
  });

  it('keeps explicit run selection independent of query parameters', async () => {
    const id = snapshot.defaultRunId;
    const getRun = vi.spyOn(analysisRepository, 'getRun').mockResolvedValue(snapshot.details[id]!);
    vi.spyOn(analysisRepository, 'listRuns').mockResolvedValue(snapshot.runs);
    function ReadSelection() { const { selectedRunId } = useAnalysis(); return <span>{selectedRunId}</span>; }
    render(<MemoryRouter initialEntries={['/?run=wrong']}><AnalysisProvider runId={id}><ReadSelection /></AnalysisProvider></MemoryRouter>);
    await waitFor(() => expect(getRun).toHaveBeenCalledWith(id));
    expect(screen.getByText(id)).toBeInTheDocument();
  });
});

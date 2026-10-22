import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnalysisCorrection } from './AnalysisCorrection';
import { correctionApi } from '../data/correctionApi';
import { ApiError } from '../data/analysisApi';
import { snapshot } from '../data/repository';
import { analysisId, correctionStatus, generation, reviewFixture } from '../test/correctionFixtures';

vi.mock('../data/correctionApi', async original => ({ ...await original<typeof import('../data/correctionApi')>(), correctionApi: {
  session: vi.fn(), source: vi.fn(), list: vi.fn(), submit: vi.fn(), status: vi.fn(), cancel: vi.fn(),
} }));
const run = { ...snapshot.details[snapshot.defaultRunId]!, analysisGeneration: generation, analysisRevision: 0,
  summary: { ...snapshot.details[snapshot.defaultRunId]!.summary, id: analysisId } };
function show() { const onCompleted = vi.fn(); const onSelectEvent = vi.fn(); const onSeekFrame = vi.fn(); const rendered = render(<AnalysisCorrection run={run} onCompleted={onCompleted} onSelectEvent={onSelectEvent} onSeekFrame={onSeekFrame} />); return { ...rendered, onCompleted, onSelectEvent, onSeekFrame }; }
async function editAndSubmit() {
  await screen.findByRole('heading', { name: 'Review and correct analysis' });
  fireEvent.change(screen.getByLabelText('Event 4 interpretation'), { target: { value: 'UNKNOWN_EVENT' } });
  fireEvent.change(screen.getByLabelText('Reason for correction'), { target: { value: 'Contact cannot be confirmed.' } });
  fireEvent.change(screen.getByLabelText('Video evidence reference'), { target: { value: 'Original video 00:01' } });
  fireEvent.click(screen.getByLabelText(/I reviewed the source video/));
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit correction and rebuild' })); });
}
beforeEach(() => {
  vi.resetAllMocks(); sessionStorage.clear();
  vi.mocked(correctionApi.session).mockResolvedValue({ authenticated: true, required: true });
  vi.mocked(correctionApi.source).mockResolvedValue(reviewFixture()); vi.mocked(correctionApi.list).mockResolvedValue([]);
  vi.mocked(correctionApi.submit).mockResolvedValue(correctionStatus()); vi.mocked(correctionApi.status).mockResolvedValue(correctionStatus());
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('owner analysis correction workflow', () => {
  it('keeps guests and nonowners read-only without correction controls', async () => {
    vi.mocked(correctionApi.session).mockResolvedValue({ authenticated: false, required: true }); show();
    expect(await screen.findByText(/This analysis is read-only/)).toBeInTheDocument(); expect(correctionApi.source).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /Submit correction/ })).not.toBeInTheDocument(); cleanup();
    vi.mocked(correctionApi.session).mockResolvedValue({ authenticated: true, required: true });
    vi.mocked(correctionApi.source).mockRejectedValue(new ApiError('Forbidden', 403)); show();
    expect(await screen.findByText(/This analysis is read-only/)).toBeInTheDocument(); expect(correctionApi.submit).not.toHaveBeenCalled();
  });
  it('requires explicit authorization, submits exact cloned evidence and keeps the replay usable', async () => {
    const { onSelectEvent, onSeekFrame } = show(); await screen.findByLabelText('Event 4 interpretation');
    expect(screen.getByRole('button', { name: 'Submit correction and rebuild' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Event 4 · frame 10' })); expect(onSelectEvent).toHaveBeenCalledWith(4); expect(onSeekFrame).toHaveBeenCalledWith(10);
    await editAndSubmit(); await waitFor(() => expect(correctionApi.submit).toHaveBeenCalledOnce());
    const [id, body] = vi.mocked(correctionApi.submit).mock.calls[0]!; expect(id).toBe(analysisId);
    expect(body).not.toHaveProperty('reviewer_id'); expect(body).not.toHaveProperty('analysis_id');
    expect(body.base_generation).toBe(generation); expect(body.input_amendments[0]!.replacement_source_input).toEqual({ ...reviewFixture().inputs[0]!.source_input, event_type: 'UNKNOWN_EVENT' });
    expect(body.initial_state_amendment).toBeNull(); expect(await screen.findByText('QUEUED')).toBeInTheDocument();
  });
  it('retries an ambiguous network response using the same retained command after remount', async () => {
    vi.mocked(correctionApi.submit).mockRejectedValueOnce(new Error('Connection lost.')).mockResolvedValueOnce(correctionStatus());
    const first = show(); await editAndSubmit(); expect(await screen.findByRole('alert')).toHaveTextContent('Connection lost.');
    const body = vi.mocked(correctionApi.submit).mock.calls[0]![1]; first.unmount(); show();
    const retry = await screen.findByRole('button', { name: 'Retry saved submission' }); await act(async () => { fireEvent.click(retry); });
    await waitFor(() => expect(correctionApi.submit).toHaveBeenCalledTimes(2)); expect(vi.mocked(correctionApi.submit).mock.calls[1]![1]).toEqual(body);
    expect(await screen.findByText('QUEUED')).toBeInTheDocument();
  });
  it('restores persisted active status, polls completion and reloads the exact generation once', async () => {
    vi.mocked(correctionApi.list).mockResolvedValue([correctionStatus('PROCESSING')]);
    vi.mocked(correctionApi.status).mockResolvedValue(correctionStatus('COMPLETED'));
    vi.useFakeTimers(); let view!: ReturnType<typeof show>; await act(async () => { view = show(); }); expect(screen.getByText('PROCESSING')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
    expect(view.onCompleted).toHaveBeenCalledExactlyOnceWith('a'.repeat(64)); expect(view.onSelectEvent).toHaveBeenCalledWith(null);
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); }); expect(correctionApi.status).toHaveBeenCalledTimes(1);
  });
  it('preserves a historical generation until the reviewer explicitly opens a completed successor', async () => {
    const source = reviewFixture(); source.is_current = false; vi.mocked(correctionApi.source).mockResolvedValue(source);
    vi.mocked(correctionApi.list).mockResolvedValue([correctionStatus('COMPLETED')]); const view = show();
    await screen.findByText(/This generation has been superseded/); expect(view.onCompleted).not.toHaveBeenCalled(); expect(correctionApi.status).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText('Saved correction history (1)')); fireEvent.click(screen.getByRole('button', { name: 'Open saved result' }));
    expect(view.onCompleted).toHaveBeenCalledExactlyOnceWith('a'.repeat(64));
  });
  it.each([false, true])('does not follow an active correction for another base generation (current source: %s)', async isCurrent => {
    const source = reviewFixture(); source.is_current = isCurrent; vi.mocked(correctionApi.source).mockResolvedValue(source);
    const unrelated = { ...correctionStatus('PROCESSING'), base_generation: 'c'.repeat(64) };
    vi.mocked(correctionApi.list).mockResolvedValue([unrelated]);
    vi.mocked(correctionApi.status).mockResolvedValue(correctionStatus('COMPLETED'));
    vi.useFakeTimers(); let view!: ReturnType<typeof show>; await act(async () => { view = show(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(correctionApi.status).not.toHaveBeenCalled(); expect(view.onCompleted).not.toHaveBeenCalled();
    expect(screen.getByText('Saved correction history (1)')).toBeInTheDocument();
  });
  it('cancels the owned command and stops polling after unmount', async () => {
    vi.mocked(correctionApi.list).mockResolvedValue([correctionStatus('PROCESSING')]);
    vi.mocked(correctionApi.cancel).mockResolvedValue(correctionStatus('CANCELLED'));
    const view = show(); fireEvent.click(await screen.findByRole('button', { name: 'Cancel correction' }));
    await screen.findByText('CANCELLED'); expect(correctionApi.cancel).toHaveBeenCalledWith(analysisId, correctionStatus().id, expect.any(AbortSignal));
    vi.useFakeTimers(); view.unmount(); await act(async () => { await vi.advanceTimersByTimeAsync(6000); }); expect(correctionApi.status).not.toHaveBeenCalled();
  });
  it('stops owned-command polling and removes controls when owner access is revoked', async () => {
    vi.mocked(correctionApi.list).mockResolvedValue([correctionStatus('PROCESSING')]);
    vi.mocked(correctionApi.status).mockRejectedValue(new ApiError('Owner access required.', 403));
    vi.useFakeTimers(); await act(async () => { show(); }); await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
    expect(screen.getByText(/This analysis is read-only/)).toBeInTheDocument(); expect(screen.queryByRole('button', { name: /Submit correction/ })).not.toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); }); expect(correctionApi.status).toHaveBeenCalledTimes(1);
  });
  it('supports explicit local-owner review and withholds stale-generation submission', async () => {
    vi.mocked(correctionApi.session).mockResolvedValue({ authenticated: false, required: false });
    const source = reviewFixture(); source.is_current = false; vi.mocked(correctionApi.source).mockResolvedValue(source); show();
    expect(await screen.findByText(/as the local owner/)).toBeInTheDocument(); expect(screen.getByRole('button', { name: 'Submit correction and rebuild' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Load current analysis' })).toBeEnabled();
  });
  it('keeps curated examples read-only even if the local owner is authenticated', () => {
    render(<AnalysisCorrection run={run} readOnly onCompleted={vi.fn()} onSelectEvent={vi.fn()} />);
    expect(screen.getByText(/This curated example is read-only/)).toBeInTheDocument();
    expect(correctionApi.session).not.toHaveBeenCalled(); expect(correctionApi.source).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /Submit correction/ })).not.toBeInTheDocument();
  });
});

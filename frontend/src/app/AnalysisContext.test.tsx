import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AnalysisProvider, useAnalysis } from './AnalysisContext';
import { analysisRepository, snapshot } from '../data/repository';

function Consumer() { const context = useAnalysis(); const location = useLocation(); return <>
  <span>{context.run?.analysisGeneration ?? 'waiting'}</span><span data-testid="location">{location.search}</span>
  <button onClick={() => context.reloadGeneration('a'.repeat(64))}>Corrected</button><button onClick={context.reloadCurrent}>Current</button>
</>; }
afterEach(() => vi.restoreAllMocks());
describe('analysis generation refresh', () => {
  it('requests the exact completed generation and clears the stale event selection', async () => {
    const id = 'match_' + 'a'.repeat(32); const base = snapshot.details[snapshot.defaultRunId]!;
    vi.spyOn(analysisRepository, 'listRuns').mockResolvedValue([]);
    const load = vi.spyOn(analysisRepository, 'getRun').mockImplementation(async (_id, generation) => ({ ...base, analysisGeneration: generation ?? 'b'.repeat(64) }));
    render(<MemoryRouter initialEntries={[`/analysis/${id}?event=99`]}><AnalysisProvider runId={id}><Consumer /></AnalysisProvider></MemoryRouter>);
    await screen.findByText('b'.repeat(64)); fireEvent.click(screen.getByText('Corrected'));
    await screen.findByText('a'.repeat(64)); expect(load).toHaveBeenLastCalledWith(id, 'a'.repeat(64));
    expect(screen.getByTestId('location').textContent).toBe('?generation=' + 'a'.repeat(64));
    fireEvent.click(screen.getByText('Current')); await waitFor(() => expect(load).toHaveBeenLastCalledWith(id));
    expect(screen.getByTestId('location').textContent).toBe('');
  });
});

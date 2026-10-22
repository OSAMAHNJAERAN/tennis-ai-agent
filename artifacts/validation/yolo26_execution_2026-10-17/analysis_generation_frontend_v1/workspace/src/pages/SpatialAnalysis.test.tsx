import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useAnalysis } from '../app/AnalysisContext';
import type { AnalysisRun } from '../data/contracts';
import { snapshot } from '../data/repository';
import MatchAnalysis from './SpatialAnalysis';

vi.mock('../app/AnalysisContext', () => ({ useAnalysis: vi.fn() }));
vi.mock('../components/SynchronizedTelemetryView', () => ({
  SynchronizedTelemetryView: () => <div aria-label="Synchronized replay" />,
}));

const id = 'match_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const generation = '1a'.repeat(32);
const downloads = [
  ['readableReport', 'readable-report', 'Download report'],
  ['annotatedVideo', 'annotated', 'Download annotated video'],
  ['reportJson', 'report', 'Analysis JSON'],
  ['csv', 'csv', 'Movement CSV'],
  ['originalVideo', 'original', 'Original video'],
] as const;

function run(): AnalysisRun {
  const base = snapshot.details[snapshot.defaultRunId]!;
  return { ...base, summary: { ...base.summary, id },
    annotatedVideoUrl: `/api/analyses/${id}/media/annotated`, exports: undefined };
}

function show(value: AnalysisRun) {
  vi.mocked(useAnalysis).mockReturnValue({ run: value, runs: [value.summary],
    selectedRunId: id, selectedEventId: null, loading: false, error: null,
    selectRun: vi.fn(), selectEvent: vi.fn(), retry: vi.fn() });
  render(<MatchAnalysis />);
}

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe('analysis download generation binding', () => {
  it('uses all five supplied export URLs with the same immutable generation', () => {
    const exports = Object.fromEntries(downloads.map(([key, path]) => [key,
      `/api/analyses/${id}/download/${path}?generation=${generation}`]));
    show({ ...run(), analysisGeneration: generation, exports });
    for (const [key, , label] of downloads) {
      expect(screen.getByRole('link', { name: label })).toHaveAttribute('href', exports[key]);
    }
    expect(screen.getAllByRole('link')).toHaveLength(5);
  });

  it('keeps the historical download routes for a legacy or static result', () => {
    show(run());
    for (const [, path, label] of downloads) {
      expect(screen.getByRole('link', { name: label })).toHaveAttribute('href', `/api/analyses/${id}/download/${path}`);
    }
  });

  it('does not create current-generation routes when pinned exports are missing or unavailable', () => {
    const readableReport = `/api/analyses/${id}/download/readable-report?generation=${generation}`;
    show({ ...run(), analysisGeneration: generation, exports: { readableReport, csv: null } });
    expect(screen.getByRole('link', { name: 'Download report' })).toHaveAttribute('href', readableReport);
    expect(screen.getAllByRole('link')).toHaveLength(1);
    expect(screen.queryByRole('link', { name: 'Movement CSV' })).not.toBeInTheDocument();
  });

  it('honors an unavailable legacy export while retaining its other supplied URLs', () => {
    const reportJson = `/api/analyses/${id}/download/report?v=18d-ab`;
    show({ ...run(), exports: { reportJson, originalVideo: null } });
    expect(screen.getByRole('link', { name: 'Analysis JSON' })).toHaveAttribute('href', reportJson);
    expect(screen.queryByRole('link', { name: 'Original video' })).not.toBeInTheDocument();
  });
});

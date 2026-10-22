import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ProductExamples from './ProductExamples';
import { useAnalysis } from '../app/AnalysisContext';
import { apiJson } from '../data/analysisApi';
import { analysisRepository, parseLiveRun, snapshot } from '../data/repository';
import { AnalysisRunSchema } from '../data/contracts';

vi.mock('../data/analysisApi', async original => ({ ...await original<typeof import('../data/analysisApi')>(), apiJson: vi.fn() }));
vi.mock('./SpatialAnalysis', () => ({ default: function Replay() { const { run } = useAnalysis(); return <p>Replay analysis {run?.summary.id}</p>; } }));
vi.mock('./LiveRunAnalytics', () => ({ default: () => <p>Movement evidence</p> }));
vi.mock('./Reports', () => ({ default: () => <p>Report evidence</p> }));

const analysisId = `match_${'b'.repeat(32)}`;
const attribution = { author: 'Licensed camera operator', sourcePage: 'https://commons.wikimedia.org/wiki/File:Tennis.webm', license: 'CC BY-SA 4.0', licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/', derivativeLicense: 'CC BY-SA 4.0', modifications: ['Excerpted', 'Normalized and annotated'] };
const excerpt = { startSeconds: 6.5, endSeconds: 16.5, startFrame: 390, endFrameExclusive: 990, sourceTimeBase: '1/60000', firstSourcePts: '390000', lastSourcePts: '989000' };
const demo = { id: 'elevated-court', title: 'Tennis from above', analysisId, previewUrl: `/api/analyses/${analysisId}/media/annotated`, attribution, excerpt, qualificationEvidence: false };
const baseline = snapshot.details[snapshot.defaultRunId]!;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiJson).mockResolvedValue({ schemaVersion: 1, demos: [demo] });
  vi.spyOn(analysisRepository, 'getRun').mockResolvedValue({ ...baseline, summary: { ...baseline.summary, id: analysisId } });
  vi.spyOn(analysisRepository, 'listRuns').mockResolvedValue([]);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('public example analysis selection', () => {
  it('selects the curated analysis even with a conflicting run query', async () => {
    render(<MemoryRouter initialEntries={['/examples/elevated-court?run=wrong']}><Routes><Route path="/examples/:exampleId" element={<ProductExamples />} /></Routes></MemoryRouter>);
    expect(await screen.findByText(`Replay analysis ${analysisId}`)).toBeInTheDocument();
    expect(analysisRepository.getRun).toHaveBeenCalledWith(analysisId);
    expect(analysisRepository.getRun).not.toHaveBeenCalledWith('wrong');
    expect(screen.getByText(/Video by Licensed camera operator/)).toBeInTheDocument();
    expect(screen.getByText(/Source excerpt: 6.50–16.50 seconds/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Original source' })).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('does not substitute a historical phase or a default analysis for an unknown example', async () => {
    render(<MemoryRouter initialEntries={[`/examples/${snapshot.defaultRunId}`]}><Routes><Route path="/examples/:exampleId" element={<ProductExamples />} /></Routes></MemoryRouter>);
    expect(await screen.findByRole('heading', { name: 'Example unavailable' })).toBeInTheDocument();
    await waitFor(() => expect(analysisRepository.getRun).not.toHaveBeenCalled());
  });

  it('preserves source attribution, hashes and excerpt metadata in the export contract', () => {
    const mediaSource = { ...attribution, id: demo.id, title: demo.title, sourceSha256: 'a'.repeat(64), inputSha256: 'b'.repeat(64), excerpt };
    const parsed = AnalysisRunSchema.parse({ ...baseline, mediaSource });
    expect(JSON.parse(JSON.stringify(parsed)).mediaSource).toEqual(mediaSource);
  });

  it('keeps valid analysis evidence usable when optional source attribution is malformed', () => {
    const parsed = parseLiveRun({ ...baseline, mediaSource: { author: 'Incomplete credit' } });
    expect(parsed.mediaSource).toBeNull();
    expect(parsed.videoUrl).toBe(baseline.videoUrl);
    expect(parsed.warnings).toContain('Analysis unavailable for this section: mediaSource.');
  });
});

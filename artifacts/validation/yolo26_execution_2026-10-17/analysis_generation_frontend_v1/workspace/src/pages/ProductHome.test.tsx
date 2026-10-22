import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ProductHome, { LocalAnalysisPreviews, PublicDemoGallery } from './ProductHome';
import { emptyMetadata, type ProductMatch } from '../data/productApi';
import { analysisRepository, snapshot } from '../data/repository';
import type { AnalysisRun } from '../data/contracts';
import { apiJson } from '../data/analysisApi';
import { MediaSourceCredit } from '../components/MediaSourceCredit';
import { PublicDemosSchema, type PublicDemo } from '../data/publicDemos';

vi.mock('../data/analysisApi', async original => ({ ...await original<typeof import('../data/analysisApi')>(), apiJson: vi.fn() }));

const observers: Array<{ callback: IntersectionObserverCallback; target: Element | null }> = [];
const paused = new WeakMap<HTMLMediaElement, boolean>();
let motionListener: (() => void) | undefined;
let reducedMotion = false;

function match(index: number): ProductMatch {
  return {
    id: `session_${index}`, analysis_id: `match_${String(index).padStart(32, '0')}`, source: 'upload', status: 'COMPLETED',
    revision: 0, created_at: '2026-09-30T00:00:00Z', updated_at: '2026-09-30T00:00:00Z',
    metadata: emptyMetadata(), metadata_source: 'user', filename: `court-${index}.mp4`, stage: null, error: null,
    notes: '', score: null, analysis_summary: { durationSeconds: 8, frames: 240 },
    predictions: { status: 'unavailable', reason: 'Unavailable' },
  };
}

function run(id: string): AnalysisRun {
  const baseline = snapshot.details[snapshot.defaultRunId]!;
  return { ...baseline, summary: { ...baseline.summary, id }, annotatedVideoUrl: `/api/analyses/${id}/media/annotated` };
}

function demo(index: number): PublicDemo {
  const analysisId = match(index).analysis_id!;
  return {
    id: `licensed-${index}`, title: `Licensed tennis ${index}`, analysisId,
    previewUrl: `/api/analyses/${analysisId}/media/annotated`, qualificationEvidence: false,
    attribution: { author: `Camera author ${index}`, sourcePage: `https://commons.wikimedia.org/wiki/File:Tennis_${index}.webm`, license: 'CC BY-SA 4.0', licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/', derivativeLicense: 'CC BY-SA 4.0', modifications: 'Excerpted, normalized, and annotated.' },
  };
}

async function visibility(index: number, visible: boolean) {
  const observer = observers[index]!;
  await act(async () => observer.callback([{ target: observer.target, isIntersecting: visible, intersectionRatio: visible ? 1 : 0 } as IntersectionObserverEntry], {} as IntersectionObserver));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [] } : []);
  observers.length = 0;
  reducedMotion = false;
  motionListener = undefined;
  vi.stubGlobal('IntersectionObserver', class {
    state: { callback: IntersectionObserverCallback; target: Element | null };
    constructor(callback: IntersectionObserverCallback) { this.state = { callback, target: null }; observers.push(this.state); }
    observe(target: Element) { this.state.target = target; }
    disconnect() {}
  });
  vi.spyOn(window, 'matchMedia').mockImplementation(query => ({
    get matches() { return reducedMotion; }, media: query, onchange: null,
    addEventListener: (_type: string, listener: EventListenerOrEventListenerObject) => { motionListener = listener as () => void; },
    removeEventListener: () => undefined, addListener: () => undefined, removeListener: () => undefined,
    dispatchEvent: () => false,
  }));
  vi.spyOn(HTMLMediaElement.prototype, 'paused', 'get').mockImplementation(function (this: HTMLMediaElement) { return paused.get(this) ?? true; });
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
    paused.set(this, false); this.dispatchEvent(new Event('play')); return Promise.resolve();
  });
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
    paused.set(this, true); this.dispatchEvent(new Event('pause'));
  });
  vi.spyOn(analysisRepository, 'getRun').mockImplementation(async id => run(id));
});

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('local homepage previews', () => {
  it('loads secondary analyses only on viewport entry and pauses off-screen media', async () => {
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1), match(2)]} /></MemoryRouter>);
    const first = await screen.findByLabelText('Annotated preview: court-1.mp4');
    expect(analysisRepository.getRun).toHaveBeenCalledTimes(1);
    expect(first).toHaveAttribute('preload', 'metadata');
    expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();
    await visibility(0, true);
    await waitFor(() => expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1));
    await visibility(0, false);
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    await visibility(1, true);
    expect(await screen.findByLabelText('Annotated preview: court-2.mp4')).toHaveAttribute('preload', 'none');
    expect(analysisRepository.getRun).toHaveBeenCalledWith(match(2).analysis_id);
  });

  it('stops automatic playback when reduced motion changes without creating a manual override', async () => {
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    await waitFor(() => expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1));
    await act(async () => { reducedMotion = true; motionListener?.(); });
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })).toBeVisible();
    await visibility(0, false);
    await visibility(0, true);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })); });
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
  });

  it('provides a visible play fallback if the browser rejects autoplay', async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new DOMException('Blocked', 'NotAllowedError'));
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    expect(await screen.findByText('Play video')).toBeVisible();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })); });
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Play preview: court-1.mp4' })).not.toBeInTheDocument());
  });

  it('keeps a newer playing preview when an earlier manual request rejects', async () => {
    reducedMotion = true;
    let rejectEarlier: (reason: unknown) => void = () => undefined;
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(() =>
      new Promise<void>((_resolve, reject) => { rejectEarlier = reject; }));
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' }));
    // The manual intent effect may also retry while the earlier request is pending.
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Play preview: court-1.mp4' })).not.toBeInTheDocument());
    await act(async () => { rejectEarlier(new DOMException('Earlier request', 'NotAllowedError')); });
    expect(screen.queryByRole('button', { name: 'Play preview: court-1.mp4' })).not.toBeInTheDocument();
    expect(screen.queryByText('Play video')).not.toBeInTheDocument();
  });

  it('keeps a paused preview after its canceled manual request rejects', async () => {
    reducedMotion = true;
    const rejections: Array<(reason: unknown) => void> = [];
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementation(() =>
      new Promise<void>((_resolve, reject) => { rejections.push(reject); }));
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' }));
    await visibility(0, false);
    await act(async () => { for (const reject of rejections) reject(new DOMException('Canceled', 'AbortError')); });
    expect(screen.getByText('Play preview')).toBeVisible();
    expect(screen.queryByText('Play video')).not.toBeInTheDocument();
  });

  it('shows a fallback when the current manual start rejects after its play event', async () => {
    reducedMotion = true;
    let rejectStart: (reason: unknown) => void = () => undefined;
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(function (this: HTMLMediaElement) {
      // A browser leaves paused state and queues play before a buffering start resolves.
      paused.set(this, false);
      this.dispatchEvent(new Event('play'));
      return new Promise<void>((_resolve, reject) => { rejectStart = reject; });
    });
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Play preview: court-1.mp4' })).not.toBeInTheDocument());
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1);
    await act(async () => { rejectStart(new Error('Playback start failed')); });
    expect(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })).toBeVisible();
    expect(screen.getByText('Play video')).toBeVisible();
  });

  it('waits for explicit playback when reduced motion is already enabled', async () => {
    reducedMotion = true;
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })); });
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1);
  });

  it('cancels an outstanding play request when a buffering preview leaves the viewport', async () => {
    let finishPlayback: (() => void) | undefined;
    vi.mocked(HTMLMediaElement.prototype.play).mockImplementationOnce(() => new Promise<void>(resolve => { finishPlayback = resolve; }));
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    await screen.findByLabelText('Annotated preview: court-1.mp4');
    await visibility(0, true);
    await visibility(0, false);
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    await act(async () => finishPlayback?.());
  });

  it('pauses a hidden tab and preserves a native-control pause on viewport return', async () => {
    let hidden = false;
    vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    const video = await screen.findByLabelText('Annotated preview: court-1.mp4') as HTMLVideoElement;
    await visibility(0, true);
    await act(async () => { hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalledTimes(1);
    await act(async () => { hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
    paused.set(video, true);
    fireEvent.pause(video);
    await visibility(0, false);
    await visibility(0, true);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('button', { name: 'Play preview: court-1.mp4' })).toBeVisible();
  });

  it('does not play a result returned for a different analysis ID', async () => {
    vi.mocked(analysisRepository.getRun).mockResolvedValue(run('match_wrong'));
    render(<MemoryRouter><LocalAnalysisPreviews matches={[match(1)]} /></MemoryRouter>);
    expect(await screen.findByText('Annotated preview unavailable for this analysis.')).toBeInTheDocument();
    expect(screen.queryByLabelText('Annotated preview: court-1.mp4')).not.toBeInTheDocument();
  });
});

describe('licensed public example previews', () => {
  it('replaces the catalog loading preview with the fetched result generation', async () => {
    const item = { ...demo(1), previewUrl: `${demo(1).previewUrl}?v=18d-ab` };
    const generation = '1a'.repeat(32);
    const fetched = { ...run(item.analysisId), analysisGeneration: generation,
      annotatedVideoUrl: `/api/analyses/${item.analysisId}/media/annotated?generation=${generation}` };
    let resolveRun: (value: AnalysisRun) => void = () => undefined;
    vi.mocked(analysisRepository.getRun).mockReturnValue(new Promise<AnalysisRun>(resolve => { resolveRun = resolve; }));
    render(<MemoryRouter><PublicDemoGallery demos={[item]} /></MemoryRouter>);
    const loadingVideo = screen.getByLabelText(`Annotated preview: ${item.title}`);
    expect(loadingVideo).toHaveAttribute('src', item.previewUrl);
    await act(async () => resolveRun(fetched));
    const fetchedVideo = screen.getByLabelText(`Annotated preview: ${item.title}`);
    expect(fetchedVideo).toHaveAttribute('src', fetched.annotatedVideoUrl);
    expect(fetchedVideo).not.toBe(loadingVideo);
    expect(screen.getByRole('link', { name: `Open analysis: ${item.title}` })).toHaveAttribute('href', `/examples/${item.id}`);
  });

  it('can display a fetched generation after the catalog loading media fails', async () => {
    const item = { ...demo(1), previewUrl: `${demo(1).previewUrl}?generation=${'a'.repeat(64)}` };
    const generation = 'b'.repeat(64);
    const fetched = { ...run(item.analysisId), analysisGeneration: generation,
      annotatedVideoUrl: `/api/analyses/${item.analysisId}/media/annotated?generation=${generation}` };
    let resolveRun: (value: AnalysisRun) => void = () => undefined;
    vi.mocked(analysisRepository.getRun).mockReturnValue(new Promise<AnalysisRun>(resolve => { resolveRun = resolve; }));
    render(<MemoryRouter><PublicDemoGallery demos={[item]} /></MemoryRouter>);
    fireEvent.error(screen.getByLabelText(`Annotated preview: ${item.title}`));
    expect(screen.queryByLabelText(`Annotated preview: ${item.title}`)).not.toBeInTheDocument();
    await act(async () => resolveRun(fetched));
    expect(screen.getByLabelText(`Annotated preview: ${item.title}`)).toHaveAttribute('src', fetched.annotatedVideoUrl);
  });

  it('removes the catalog fallback after an invalid fetched analysis', async () => {
    const item = demo(1);
    vi.mocked(analysisRepository.getRun).mockResolvedValue(run(demo(2).analysisId));
    render(<MemoryRouter><PublicDemoGallery demos={[item]} /></MemoryRouter>);
    expect(await screen.findByText('Annotated preview unavailable for this analysis.')).toBeInTheDocument();
    expect(screen.queryByLabelText(`Annotated preview: ${item.title}`)).not.toBeInTheDocument();
  });

  it('links every credited example to its own curated identity and matching media', async () => {
    const demos = [demo(1), demo(2), demo(3)];
    render(<MemoryRouter><PublicDemoGallery demos={demos} /></MemoryRouter>);
    await visibility(1, true);
    await visibility(2, true);
    for (const item of demos) {
      expect(await screen.findByLabelText(`Annotated preview: ${item.title}`)).toHaveAttribute('src', item.previewUrl);
      expect(screen.getByRole('link', { name: `Open analysis: ${item.title}` })).toHaveAttribute('href', `/examples/${item.id}`);
      expect(screen.getByText(new RegExp(`Video by ${item.attribution.author}`))).toBeInTheDocument();
      expect(analysisRepository.getRun).toHaveBeenCalledWith(item.analysisId);
    }
    expect(screen.getAllByRole('link', { name: 'Original source' })).toHaveLength(3);
    expect(screen.queryByText('Your local analysis')).not.toBeInTheDocument();
  });

  it('rejects a preview bound to another analysis and repeated catalog identities', () => {
    expect(PublicDemosSchema.safeParse({ schemaVersion: 1, demos: [{ ...demo(1), previewUrl: demo(2).previewUrl }] }).success).toBe(false);
    expect(PublicDemosSchema.safeParse({ schemaVersion: 1, demos: [demo(1), { ...demo(1), id: 'different-title' }] }).success).toBe(false);
  });

  it('accepts media revisions only on the matching analysis route', () => {
    const item = demo(1);
    const parse = (previewUrl: string) => PublicDemosSchema.safeParse({ schemaVersion: 1, demos: [{ ...item, previewUrl }] }).success;
    expect(parse(`${item.previewUrl}?v=18d-ab`)).toBe(true);
    expect(parse(`${demo(2).previewUrl}?v=18d-ab`)).toBe(false);
    expect(parse(`${item.previewUrl}?v=18d-ab&redirect=https://outside.example`)).toBe(false);
    expect(parse(`https://outside.example${item.previewUrl}?v=18d-ab`)).toBe(false);
  });

  it('accepts exactly one immutable generation only on the matching analysis route', () => {
    const item = demo(1);
    const generation = '1a'.repeat(32);
    const parse = (previewUrl: string) => PublicDemosSchema.safeParse({ schemaVersion: 1, demos: [{ ...item, previewUrl }] }).success;
    expect(parse(`${item.previewUrl}?generation=${generation}`)).toBe(true);
    for (const url of [
      `${demo(2).previewUrl}?generation=${generation}`,
      `${item.previewUrl}?generation=${generation.toUpperCase()}`,
      `${item.previewUrl}?generation=${generation.slice(1)}`,
      `${item.previewUrl}?generation=${generation}0`,
      `${item.previewUrl}?generation=`,
      `${item.previewUrl}?generation=${generation}&generation=${generation}`,
      `${item.previewUrl}?generation=${generation}&v=18d-ab`,
      `${item.previewUrl}?v=18d-ab&generation=${generation}`,
      `${item.previewUrl}?generation=${generation}&redirect=https://outside.example`,
      `${item.previewUrl}?generation=${generation}#fragment`,
      `https://outside.example${item.previewUrl}?generation=${generation}`,
    ]) expect(parse(url)).toBe(false);
  });

  it('renders unsafe credit links as text while keeping license attribution', () => {
    const source = { ...demo(1).attribution, sourcePage: 'javascript:alert(1)', licenseUrl: 'data:text/html,test' };
    render(<MediaSourceCredit source={source} />);
    expect(screen.getByText('Original source')).toBeInTheDocument();
    expect(screen.getByText('CC BY-SA 4.0')).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('keeps public examples out of the private recent-analysis gallery', async () => {
    vi.mocked(apiJson).mockImplementation(async path => path === '/api/demos' ? { schemaVersion: 1, demos: [demo(1)] } : [match(1), match(2)]);
    render(<MemoryRouter><ProductHome /></MemoryRouter>);
    const own = await screen.findByRole('region', { name: 'Your recent analyses' });
    expect(within(own).getByText('court-2.mp4')).toBeInTheDocument();
    expect(within(own).queryByText('court-1.mp4')).not.toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Example analyses' })).toHaveTextContent('Licensed tennis 1');
  });

  it('keeps a catalog failure distinct from an unavailable example collection', async () => {
    vi.mocked(apiJson).mockImplementation(async path => { if (path === '/api/demos') throw new Error('Catalog unavailable'); return [match(1)]; });
    render(<MemoryRouter><ProductHome /></MemoryRouter>);
    expect(await screen.findByRole('button', { name: 'Retry examples' })).toBeEnabled();
    expect(screen.queryByText('Public demo videos are not available yet.')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Upload video' })).toHaveAttribute('href', '/upload');
    expect(screen.getByRole('heading', { name: 'Recent saved analyses' })).toBeInTheDocument();
    expect(screen.getByText('court-1.mp4')).toBeInTheDocument();
  });
});

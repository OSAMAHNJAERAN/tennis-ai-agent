import { Link, useSearchParams } from 'react-router-dom';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, ArrowUpRight, CloudArrowUp, PlayCircle } from '@phosphor-icons/react';
import { useProductResource } from '../hooks/useProductResource';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { usePublicDemos } from '../hooks/usePublicDemos';
import { matchTitle, statusLabel, type ProductMatch } from '../data/productApi';
import { analysisRepository } from '../data/repository';
import type { AnalysisRun } from '../data/contracts';
import { Card, PageHeader } from '../components/ui';
import { MediaSourceCredit } from '../components/MediaSourceCredit';
import type { MediaAttribution, PublicDemo } from '../data/publicDemos';
import { CourtEmblem, SocietyAction, SocietyMarquee, SocietyWorkflow, useSocietyMotion } from '../components/Society';

export function ProductError({ error, retry }: { error: string; retry: () => void }) {
  return error ? <div className="product-alert" role="alert"><strong>Connection needs attention</strong><p>{error}</p><button type="button" className="button" onClick={retry}>Retry</button></div> : null;
}

export function MatchList({ matches }: { matches: ProductMatch[] }) {
  return <div className="product-match-list">{matches.map(match => <Link className="product-match-row" to={`/matches/${match.id}`} key={match.id}><div><strong>{matchTitle(match)}</strong><small>{match.filename || 'Uploaded video'}{match.metadata.match_date ? ` · ${match.metadata.match_date}` : ''}{match.metadata.tournament ? ` · ${match.metadata.tournament}` : ''}</small>{match.error || match.stage ? <small>{match.error || match.stage}</small> : null}</div><span className={`product-status status-${match.status.toLowerCase()}`}>{statusLabel(match.status)}</span><ArrowRight size={18} aria-hidden="true" /></Link>)}</div>;
}

export function ServiceStatus() {
  const health = useProductResource<{ status: string; worker_available: boolean }>('/api/health', 10000);
  return <Card className="product-panel"><h2>Analysis service</h2><p>Connection: {health.error ? 'Unavailable' : health.data ? 'Connected' : 'Checking…'}</p><p>Processing: {health.error || !health.data ? 'Unknown' : health.data.worker_available ? 'Worker available' : 'Worker not running'}</p><Link className="astra-text-button" to="/settings">Service details <ArrowRight size={16} aria-hidden="true" /></Link></Card>;
}

interface Preview {
  analysisId: string;
  title: string;
  href: string;
  label: string;
  previewUrl?: string;
  attribution?: MediaAttribution;
}

function AnalysisPreview({ preview, priority }: { preview: Preview; priority: boolean }) {
  const articleRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const playbackIntent = useRef(false);
  const playRequest = useRef(0);
  const [viewport, setViewport] = useState({ visible: false, entered: false });
  const [pageVisible, setPageVisible] = useState(() => !document.hidden);
  const [playback, setPlayback] = useState<'paused' | 'playing' | 'blocked'>('paused');
  const [automaticPlayback, setAutomaticPlayback] = useState(true);
  const [manualPlayback, setManualPlayback] = useState(false);
  const [state, setState] = useState<{ id: string; run: AnalysisRun | null; error: string }>({ id: '', run: null, error: '' });
  const [failedMediaUrl, setFailedMediaUrl] = useState<string | null>(null);
  const reducedMotion = useReducedMotion();
  const id = preview.analysisId;
  const loadPreview = priority || viewport.entered;
  const run = state.id === id ? state.run : null;
  const mediaUrl = run?.annotatedVideoUrl ?? (loadPreview && state.id !== id ? preview.previewUrl ?? null : null);
  const mediaError = Boolean(mediaUrl && failedMediaUrl === mediaUrl);
  const shouldPlay = viewport.visible && pageVisible && (manualPlayback || (automaticPlayback && !reducedMotion));

  useEffect(() => {
    const article = articleRef.current;
    if (!article) return;
    const update = (visible: boolean) => setViewport(previous => ({ visible, entered: previous.entered || visible }));
    if (typeof IntersectionObserver !== 'undefined') {
      const observer = new IntersectionObserver(entries => {
        const entry = entries[0];
        if (entry) update(entry.isIntersecting && entry.intersectionRatio >= 0.15);
      }, { threshold: [0, 0.15] });
      observer.observe(article);
      return () => observer.disconnect();
    }
    const checkVisibility = () => {
      const bounds = article.getBoundingClientRect();
      update(bounds.bottom > 0 && bounds.top < window.innerHeight && bounds.right > 0 && bounds.left < window.innerWidth);
    };
    const frame = requestAnimationFrame(checkVisibility);
    window.addEventListener('scroll', checkVisibility, { passive: true });
    window.addEventListener('resize', checkVisibility);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('scroll', checkVisibility);
      window.removeEventListener('resize', checkVisibility);
    };
  }, []);

  useEffect(() => {
    const onVisibilityChange = () => setPageVisible(!document.hidden);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  useEffect(() => {
    if (!loadPreview) return;
    let active = true;
    void analysisRepository.getRun(id).then(run => {
      if (!active) return;
      const hasEvidence = Boolean(run.detections?.frames.length || run.trajectories?.ballTrajectory.length);
      setState({ id, run: run.summary.id === id && run.annotatedVideoUrl && hasEvidence ? run : null, error: '' });
    }).catch(() => { if (active) setState({ id, run: null, error: 'Preview unavailable. Open the analysis to inspect its saved results.' }); });
    return () => { active = false; };
  }, [id, loadPreview]);

  const startPlayback = useCallback((video: HTMLVideoElement) => {
    const request = ++playRequest.current;
    void video.play().catch((error: unknown) => {
      if (request !== playRequest.current || (error instanceof DOMException && error.name === 'AbortError')) return;
      setPlayback('blocked');
    });
  }, []);

  useEffect(() => () => { ++playRequest.current; }, [mediaUrl, mediaError]);

  useEffect(() => {
    const video = videoRef.current;
    const previousIntent = playbackIntent.current;
    playbackIntent.current = shouldPlay;
    if (!video || mediaError) return;
    if (shouldPlay && video.paused) {
      startPlayback(video);
    } else if (!shouldPlay && (!video.paused || previousIntent)) {
      // Pause also cancels a pending play() while the browser buffers media.
      ++playRequest.current;
      video.pause();
    }
  }, [shouldPlay, mediaUrl, mediaError, startPlayback]);

  function playPreview() {
    const video = videoRef.current;
    if (!video) return;
    setManualPlayback(true);
    playbackIntent.current = true;
    startPlayback(video);
  }

  function onPlay() {
    setPlayback('playing');
    // A native-controls click is a manual request only if the visibility effect
    // did not already request playback. Autoplay must keep respecting reduced motion.
    if (!playbackIntent.current) setManualPlayback(true);
  }

  function onPause() {
    ++playRequest.current;
    setPlayback('paused');
    if (playbackIntent.current) {
      setAutomaticPlayback(false);
      setManualPlayback(false);
    }
  }

  return <article ref={articleRef} className="product-video-preview">
    <div className="product-preview-stage">
      {mediaUrl && !mediaError ? <><video ref={videoRef} key={mediaUrl} src={mediaUrl} controls muted loop playsInline preload={priority ? 'metadata' : 'none'} aria-label={`Annotated preview: ${preview.title}`} onPlay={onPlay} onPause={onPause} onError={() => setFailedMediaUrl(mediaUrl)} />{playback !== 'playing' ? <button className="product-preview-play" type="button" onClick={playPreview} aria-label={`Play preview: ${preview.title}`}><PlayCircle size={32} weight="fill" aria-hidden="true" /><span>{playback === 'blocked' ? 'Play video' : 'Play preview'}</span></button> : null}</> : <div className="product-preview-unavailable"><PlayCircle size={30} aria-hidden="true" /><p>{!loadPreview ? 'Scroll here to load this analyzed video.' : state.id !== id ? 'Loading analyzed video…' : mediaError ? 'The preview video could not load. Open the analysis to retry playback.' : state.error || 'Annotated preview unavailable for this analysis.'}</p></div>}
    </div>
    <div className="product-preview-caption"><div><span className="product-hint">{preview.label}</span><h3>{preview.title}</h3>{run ? <><p>{run.summary.durationSeconds.toFixed(1)} seconds · Annotated video</p>{!run.summary.capabilities.court ? <p>Image tracking only · Court positions and speeds unavailable.</p> : null}</> : null}</div><Link to={preview.href} aria-label={`Open analysis: ${preview.title}`}><ArrowRight size={22} aria-hidden="true" /></Link></div>
    {preview.attribution ? <MediaSourceCredit source={preview.attribution} /> : null}
  </article>;
}

export function LocalAnalysisPreviews({ matches, label = 'Your local analysis' }: { matches: ProductMatch[]; label?: string }) {
  const completed = matches.filter(match => match.source === 'upload' && match.status === 'COMPLETED' && match.analysis_id).slice(0, 3);
  return completed.length ? <div className={`product-video-gallery gallery-count-${completed.length}`}>{completed.map((match, index) => <AnalysisPreview key={match.analysis_id} preview={{ analysisId: match.analysis_id!, title: matchTitle(match), href: `/matches/${match.id}`, label }} priority={index === 0} />)}</div> : null;
}

export function PublicDemoGallery({ demos }: { demos: PublicDemo[] }) {
  return <div className={`product-video-gallery gallery-count-${Math.min(demos.length, 3)}`}>{demos.slice(0, 3).map((demo, index) => <AnalysisPreview key={demo.analysisId} priority={index === 0} preview={{ analysisId: demo.analysisId, title: demo.title, href: `/examples/${encodeURIComponent(demo.id)}`, label: demo.excerpt ? 'Example analysis · Source excerpt' : 'Example analysis', previewUrl: demo.previewUrl, attribution: demo.attribution }} />)}</div>;
}

function PreviewEmptyState() {
  return <div className="product-video-empty"><div className="product-empty-court" aria-hidden="true"><span /><span /><span /></div><div className="product-video-empty-copy"><PlayCircle size={36} aria-hidden="true" /><h2>Your footage belongs here.</h2><p>Upload a tennis video to see the annotated action and open its full analysis.</p><Link className="button button-primary" to="/upload"><CloudArrowUp size={18} aria-hidden="true" />Upload video</Link><small>Public demo videos are not available yet.</small></div></div>;
}

export default function ProductHome() {
  const resource = useProductResource<ProductMatch[]>('/api/matches', 5000);
  const catalog = usePublicDemos();
  const publicAnalysisIds = new Set(catalog.demos.map(demo => demo.analysisId));
  const matches = (resource.data ?? []).filter(match => match.source === 'upload' && !publicAnalysisIds.has(match.analysis_id ?? ''));
  const active = matches.filter(match => ['PROCESSING', 'UPLOADED'].includes(match.status));
  const completed = matches.filter(match => match.status === 'COMPLETED' && match.analysis_id);
  const [graphicsPaused, setGraphicsPaused] = useState(false);
  const motionScope = useSocietyMotion();
  return <div ref={motionScope} className={`page product-page product-home society-home${graphicsPaused ? ' motion-is-paused' : ''}`}>
    <header className="society-entry-intro">
      <div><span className="society-label">[ AI TENNIS INTELLIGENCE ]</span><h1><span className="society-reveal-line"><span>See the game.</span></span><span className="society-reveal-line"><span>See it <em>differently.</em></span></span></h1></div>
      <div className="society-entry-copy"><p>Your footage. A new perspective.<br />Follow the ball, read the movement, and discover the evidence behind your game.</p><Link className="society-cta" to="/upload"><span>Analyze your match</span><ArrowUpRight size={22} aria-hidden="true" /></Link></div>
    </header>
    <section id="match-intelligence" className="society-media-section society-entry-media" aria-label="Example analyses"><div className="product-preview-intro society-section-heading"><h2>Watch the analysis <em>in action.</em></h2><Link to="/examples">Explore examples <ArrowRight size={18} aria-hidden="true" /></Link></div>{catalog.loading ? <div className="product-preview-loading" role="status">Loading example videos…</div> : catalog.error ? <div className="product-demo-unavailable" role="status"><p>Example videos could not be loaded. You can still upload and open your own analyses.</p><button className="button" type="button" onClick={catalog.refresh}>Retry examples</button><Link className="button" to="/upload">Upload video</Link></div> : catalog.demos.length ? <><PublicDemoGallery demos={catalog.demos} /><p className="product-demo-note">Real processed clips shared under their source licenses. These examples demonstrate the workflow; their measurements have not been independently validated.</p></> : <PreviewEmptyState />}</section>
    <SocietyMarquee paused={graphicsPaused} onToggle={() => setGraphicsPaused(paused => !paused)} />
    <ProductError error={resource.error} retry={resource.refresh} />
    {!resource.data && !resource.error ? <p role="status">Loading your analyses…</p> : completed.length ? <section className="society-media-section society-local-section" aria-label="Your recent analyses"><div className="product-preview-intro society-section-heading"><div><span className="society-label">[ YOUR LOCAL LIBRARY ]</span><h2>{catalog.loading || catalog.error ? 'Recent saved analyses' : 'Your recent analyzed videos'}</h2></div><span>Saved on this device · Local library</span></div><LocalAnalysisPreviews matches={matches} label={catalog.loading || catalog.error ? 'Saved analysis' : 'Your local analysis'} /></section> : null}
    <section className="society-manifesto"><span className="society-label">[ INTELLIGENCE WITH EVIDENCE ]</span><p>{'Every movement tells a story. Every estimate needs evidence. Look beyond the score.'.split(' ').map((word, index) => <span className="society-manifesto-word" key={index}>{index === 2 ? <span className="society-inline-court" aria-hidden="true"><CourtEmblem compact /></span> : null}{word} </span>)}</p><span className="society-manifesto-mark" aria-hidden="true">↗</span><div className="society-hero-art society-entry-art"><div className="society-art-caption"><span>COURT / COMPUTER VISION</span><span>+</span></div><CourtEmblem /><div className="society-art-footer"><span>A different angle on every rally.</span><span>[ ASTRA ]</span></div></div></section>
    <SocietyWorkflow />
    <div className="product-home-library"><section><div className="product-section-title"><h2>{active.length ? 'Processing your videos' : 'Your analysis library'}</h2><Link to="/match-center">My analyses <ArrowRight size={15} aria-hidden="true" /></Link></div>{active.length ? <MatchList matches={active} /> : resource.data && !resource.error ? <p className="product-empty">{matches.length ? 'Open My analyses to revisit saved results and downloads.' : 'Your uploaded videos and completed results will appear here.'}</p> : null}</section><p className="product-home-note">Measurements depend on the camera view and visible evidence. Missing tracks and uncertain estimates stay visible in the analysis.</p></div>
    <SocietyAction />
  </div>;
}

export function MatchCenter() {
  const resource = useProductResource<ProductMatch[]>('/api/matches', 5000);
  const catalog = usePublicDemos();
  const publicAnalysisIds = new Set(catalog.demos.map(demo => demo.analysisId));
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const filter = params.get('status') ?? 'all';
  const matches = (resource.data ?? []).filter(match => match.source === 'upload' && !publicAnalysisIds.has(match.analysis_id ?? '') && (filter === 'all' || (filter === 'processing' ? ['UPLOADED', 'PROCESSING'].includes(match.status) : match.status.toLowerCase() === filter)) && `${matchTitle(match)} ${match.filename ?? ''} ${match.metadata.tournament ?? ''}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="page product-page"><PageHeader eyebrow="YOUR VIDEO LIBRARY" title="My analyses" description="Follow processing, revisit a replay, and download your results." action={<Link className="button button-primary" to="/upload"><CloudArrowUp size={18} aria-hidden="true" />Upload video</Link>} /><div className="product-filters"><label>Find an analysis<input type="search" placeholder="Player, tournament or filename" value={query} onChange={event => setQuery(event.target.value)} /></label><label>Status<select value={filter} onChange={event => setParams(event.target.value === 'all' ? {} : { status: event.target.value })}>{['all', 'processing', 'completed', 'failed'].map(value => <option key={value} value={value}>{value[0]!.toUpperCase() + value.slice(1)}</option>)}</select></label></div><ProductError error={resource.error} retry={resource.refresh} />{!resource.data && !resource.error ? <p role="status">Loading analyses…</p> : matches.length ? <MatchList matches={matches} /> : resource.data && !resource.error ? <p className="product-empty">No analyses match this view. Adjust your search or <Link to="/upload">upload a video</Link>.</p> : null}</div>;
}

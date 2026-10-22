import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, CheckCircle, CloudArrowUp, FilmStrip, WarningCircle } from '@phosphor-icons/react';
import { apiJson, HealthSchema, listJobs, uploadVideo, type AnalysisJob } from '../data/analysisApi';
import '../styles/upload.css';
import { MatchMetadataForm } from '../components/MatchMetadataForm';
import { emptyMetadata, mutate, type ProductMatch } from '../data/productApi';
import { JobSchema } from '../data/analysisApi';
import { usePublicDemos } from '../hooks/usePublicDemos';

export default function Upload() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [metadata, setMetadata] = useState(emptyMetadata);
  const draftId = useRef<string | null>(null);
  const autoOpen = useRef(false);
  const activeId = params.get('job');
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const catalog = usePublicDemos();
  const publicAnalysisIds = new Set(catalog.demos.map(demo => demo.analysisId));
  const historyJobs = jobs.filter(job => !publicAnalysisIds.has(job.id));
  const historyReady = !catalog.loading && !catalog.error;
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [limit, setLimit] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const uploadController = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement | null>(null);
  const activeJob = jobs.find(job => job.id === activeId);
  const uploading = progress !== null;

  useEffect(() => () => uploadController.current?.abort(), []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const [health, nextJobs] = await Promise.all([
          apiJson('/api/health', controller.signal).then(value => HealthSchema.parse(value)),
          listJobs(controller.signal),
        ]);
        if (controller.signal.aborted) return;
        if (activeId && !nextJobs.some(job => job.id === activeId)) {
          nextJobs.unshift(JobSchema.parse(await apiJson(`/api/analyses/${encodeURIComponent(activeId)}`, controller.signal)));
        }
        setLimit(health.max_upload_bytes); setJobs(nextJobs); setConnectionError(null);
      } catch (caught) {
        if (!controller.signal.aborted) setConnectionError(caught instanceof Error ? caught.message : 'Unable to reach the analysis server.');
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 2500);
      }
    };
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refreshKey, activeId]);
  useEffect(() => {
    if (autoOpen.current && activeJob?.status === 'COMPLETED') {
      autoOpen.current = false;
      navigate(`/analysis/${activeJob.id}`);
    }
  }, [activeJob, navigate]);

  function choose(next: File | undefined) {
    if (!next || uploading) return;
    if (!/\.(mp4|mov|avi)$/i.test(next.name)) { setFile(null); setError('Choose an MP4, MOV or AVI video.'); return; }
    if (!next.size || (limit !== null && next.size > limit)) { setFile(null); setError('The video is empty or exceeds the upload limit. Choose a smaller video.'); return; }
    setFile(next); setError(null); draftId.current = null;
  }

  async function submit() {
    if (!file || uploading) return;
    const controller = new AbortController();
    uploadController.current = controller;
    setError(null); setProgress(0);
    try {
      if (!draftId.current) {
        const match = await mutate<ProductMatch>('/api/matches', { source: 'upload', metadata });
        draftId.current = match.id;
      } else {
        const current = await apiJson(`/api/matches/${draftId.current}`) as ProductMatch;
        if (current.analysis_id) { navigate(`/matches/${current.id}`); return; }
        await mutate(`/api/matches/${draftId.current}/metadata`, { metadata, expected_revision: current.revision }, 'PATCH');
      }
      if (controller.signal.aborted) return;
      const job = await uploadVideo(file, setProgress, controller.signal, draftId.current);
      autoOpen.current = true;
      setJobs(current => [job, ...current.filter(item => item.id !== job.id)]);
      setParams({ job: job.id }); setFile(null); setRefreshKey(key => key + 1);
      if (input.current) input.current.value = '';
    } catch (caught) {
      if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : 'Upload failed. Please try again.');
    } finally {
      setProgress(null); uploadController.current = null;
    }
  }

  return <div className="page upload-page">
    <header className="upload-heading">
      <div><span className="astra-eyebrow">YOUR NEXT SESSION</span><h1>Bring your match<br /><em>into focus.</em></h1></div>
      <p>Every match has a story.<br />Upload yours to explore player movement, ball tracking and a new perspective on your court.</p>
    </header>
    <div className="upload-layout">
      <section className="upload-panel" aria-label="Upload tennis video">
        <div className="upload-panel-caption"><span>VIDEO ANALYSIS</span><span>START WITH YOUR FOOTAGE <ArrowRight size={14} aria-hidden="true" /></span></div>
        <div className={`upload-dropzone ${dragging ? 'is-dragging' : ''}`} onDragOver={event => { event.preventDefault(); if (!uploading) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files[0]); }}>
          <svg className="upload-court-lines" viewBox="0 0 720 420" preserveAspectRatio="none" aria-hidden="true"><path d="M36 28H684V392H36Z M116 28V392 M604 28V392 M36 210H684 M116 118H604 M116 302H604 M360 118V302" /></svg>
          <CloudArrowUp size={42} weight="light" aria-hidden="true" />
          <h2>{file ? file.name : 'Drop your tennis video here'}</h2>
          <p>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to upload` : `MP4, MOV or AVI${limit ? ` · Up to ${(limit / 1024 / 1024).toFixed(0)} MB` : ''}`}</p>
          <label className={`upload-file-label ${uploading ? 'is-disabled' : ''}`}>Choose video<ArrowRight size={40} weight="light" aria-hidden="true" /><input ref={input} type="file" accept=".mp4,.mov,.avi" disabled={uploading} onChange={event => choose(event.target.files?.[0])} aria-label="Choose tennis video" /></label>
          <span className="upload-dropzone-note" aria-hidden="true">A NEW VIEW OF YOUR GAME</span>
        </div>
        {file ? <MatchMetadataForm value={metadata} onChange={setMetadata} disabled={uploading} /> : null}
        {error ? <p className="upload-error" role="alert"><WarningCircle size={20} />{error}</p> : null}
        {uploading ? <div className="upload-progress" role="status"><span>{progress === 100 ? 'Upload received. Creating your analysis…' : `Uploading video · ${progress}%`}</span><progress max={100} value={progress ?? 0} aria-label="Video upload progress" /><button type="button" className="astra-text-button" onClick={() => uploadController.current?.abort()}>Cancel upload</button></div> : <button className="upload-submit" type="button" disabled={!file || limit === null || connectionError !== null} onClick={() => void submit()}>Analyze match<ArrowRight size={18} /></button>}
        <p className="upload-hint">For the clearest results, use a mostly static camera with the court and both players visible. Processing continues when you leave this page after upload.</p>
      </section>
      <aside className="upload-guide"><span className="astra-eyebrow">FROM FOOTAGE TO FEEDBACK</span><h2>Your game.<br /><em>A fresh perspective.</em></h2><ol><li><b>01</b><div><h3>Upload your session</h3><p>Each upload creates a separate saved analysis.</p></div></li><li><b>02</b><div><h3>Follow the analysis</h3><p>See each processing stage as the tennis pipeline tracks your match.</p></div></li><li><b>03</b><div><h3>Review every movement</h3><p>Open synchronized video, court replay, movement maps and statistics.</p></div></li></ol><div className="upload-guide-footer"><span>LESS GUESSWORK.<br />MORE PERSPECTIVE.</span><ArrowRight size={42} weight="light" aria-hidden="true" /></div></aside>
    </div>
    {connectionError ? <div className="upload-connection" role="alert"><WarningCircle size={22} /><div><strong>Analysis server unavailable</strong><p>{connectionError} Your saved jobs will reappear when the connection returns.</p><button type="button" className="astra-text-button" onClick={() => setRefreshKey(key => key + 1)}>Reconnect</button></div></div> : null}
    {activeJob ? <section className={`upload-active status-${activeJob.status.toLowerCase()}`} aria-label="Current analysis" aria-live="polite"><div>{activeJob.status === 'COMPLETED' ? <CheckCircle size={28} /> : <FilmStrip size={28} />}<span><h2>{activeJob.status === 'COMPLETED' ? 'Your analysis is ready' : activeJob.status === 'FAILED' ? 'Analysis could not finish' : 'Analyzing your tennis match…'}</h2><p>{activeJob.filename}</p></span></div><p>{activeJob.error ?? activeJob.stage}{activeJob.status === 'PROCESSING' && activeJob.step > 0 ? ` · Stage ${activeJob.step} of 11` : ''}</p>{activeJob.status === 'COMPLETED' ? <Link className="upload-result-link" to={`/analysis/${activeJob.id}`}>Open analysis<ArrowRight size={18} /></Link> : null}</section> : null}
    <section className="upload-history" aria-label="Analysis history">
      <div className="upload-history-heading"><div><span className="astra-eyebrow">THE SESSION ARCHIVE</span><h2>{historyReady ? 'Your analyses' : 'Saved analyses'}</h2></div><span>{historyJobs.length} {historyReady ? 'sessions' : 'saved analyses'}</span></div>
      {!historyReady ? <div className="upload-hint" role="status"><p>Public example filtering is unavailable. Showing saved analyses, which may include public examples.</p>{catalog.error ? <><button type="button" className="astra-text-button" onClick={catalog.refresh}>Retry history</button><Link className="astra-text-button" to="/match-center">My analyses</Link></> : null}</div> : null}
      {!historyJobs.length && !connectionError ? <p className="upload-hint upload-history-empty">{historyReady ? 'Your uploaded sessions will appear here. Historical examples are available in the separate Examples section.' : 'Saved analyses will appear here.'}</p> : null}
      {historyJobs.map(job => <article key={job.id}><FilmStrip size={24} weight="light" aria-hidden="true" /><div><h3>{job.filename}</h3><p>{new Date(job.created_at).toLocaleString()} · {job.error ?? job.stage}</p></div><span className={`upload-job-state status-${job.status.toLowerCase()}`}>{job.status === 'UPLOADED' ? 'Queued' : job.status.toLowerCase()}</span>{job.status === 'COMPLETED' ? <Link to={`/analysis/${job.id}`} aria-label={`Open analysis for ${job.filename}`}>View<ArrowRight size={16} /></Link> : <button type="button" className="astra-text-button" onClick={() => setParams({ job: job.id })}>Details</button>}</article>)}
    </section>
  </div>;
}

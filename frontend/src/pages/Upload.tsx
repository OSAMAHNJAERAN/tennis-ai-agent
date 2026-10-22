import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, CheckCircle, CloudArrowUp, FilmStrip, WarningCircle } from '@phosphor-icons/react';
import { ApiError, apiJson, HealthSchema, listJobs, retryAnalysis, retryOwnerScope, uploadVideo, type AnalysisJob } from '../data/analysisApi';
import '../styles/upload.css';
import { MatchMetadataForm } from '../components/MatchMetadataForm';
import { emptyMetadata, mutate, type ProductMatch } from '../data/productApi';
import { JobSchema } from '../data/analysisApi';
import { usePublicDemos } from '../hooks/usePublicDemos';
import { AccountAccess } from '../components/AccountAccess';
import { createUploadBatch, fileResumeKey, rememberedUpload, rememberUpload, resumeVideoUpload, type UploadBatch } from '../data/batchApi';
import { CreationIntentConflict, type CreationRequest } from '../data/creationIntent';

export default function Upload() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [metadata, setMetadata] = useState(emptyMetadata);
  const draftId = useRef<string | null>(null);
  const autoOpen = useRef(false);
  const activeId = params.get('job');
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const catalog = usePublicDemos();
  const historyReady = !catalog.loading && !catalog.error;
  const [lastVerifiedDemos, setLastVerifiedDemos] = useState(catalog.demos);
  if (historyReady && lastVerifiedDemos !== catalog.demos) setLastVerifiedDemos(catalog.demos);
  const publicAnalysisIds = new Set((historyReady ? catalog.demos : lastVerifiedDemos).map(demo => demo.analysisId));
  const historyJobs = jobs.filter(job => !publicAnalysisIds.has(job.id));
  const [files, setFiles] = useState<File[]>([]);
  const file = files[0] ?? null;
  const [fileStates, setFileStates] = useState<Record<string, { progress: number; error?: string; job?: AnalysisJob }>>({});
  const [mode, setMode] = useState<UploadBatch['mode']>('independent');
  const [resumable, setResumable] = useState(false);
  const [activeBatch, setActiveBatch] = useState<UploadBatch | null>(null);
  const [savedCreation, setSavedCreation] = useState<CreationRequest | null>(null);
  const [resumingSaved, setResumingSaved] = useState(false);
  const sessions = useRef<Record<string, string>>({});
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [accessRequired, setAccessRequired] = useState(false);
  const [limit, setLimit] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [retryOwner, setRetryOwner] = useState<string | null>(null);
  const [pendingJobs, setPendingJobs] = useState<Record<string, boolean>>({});
  const accountGeneration = useRef(0);
  const retryOwnerRef = useRef<string | null>(null);
  const pendingActions = useRef(new Set<string>());
  const uploadController = useRef<AbortController | null>(null);
  const activeFileUpload = useRef<{ key: string; controller: AbortController } | null>(null);
  const input = useRef<HTMLInputElement | null>(null);
  const activeJob = jobs.find(job => job.id === activeId);
  const uploading = progress !== null;

  useEffect(() => () => { accountGeneration.current += 1; uploadController.current?.abort(); }, []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const clearOwner = (owner: string | null) => {
      accountGeneration.current += 1; retryOwnerRef.current = owner; setRetryOwner(owner);
      setJobs([]); setActiveBatch(null); setPendingJobs({}); pendingActions.current = new Set();
      uploadController.current?.abort(); activeFileUpload.current?.controller.abort();
      uploadController.current = null; activeFileUpload.current = null; sessions.current = {};
      draftId.current = null; autoOpen.current = false; setProgress(null); setFileStates({}); setFiles([]);
      setMetadata(emptyMetadata()); setSavedCreation(null); setResumingSaved(false);
      if (activeId || params.get('batch')) setParams({});
    };
    const refresh = async () => {
      const generation = accountGeneration.current;
      try {
        const [health, sessionBefore] = await Promise.all([
          apiJson('/api/health', controller.signal).then(value => {
            const health = HealthSchema.parse(value);
            if (!controller.signal.aborted) { setLimit(health.max_upload_bytes); setResumable(health.resumable_uploads === true); }
            return health;
          }),
          apiJson('/api/auth/session', controller.signal).catch(() => null),
        ]);
        if (controller.signal.aborted || generation !== accountGeneration.current) return;
        const ownerBefore = retryOwnerScope(sessionBefore);
        const nextJobs = await listJobs(controller.signal);
        if (activeId && !nextJobs.some(job => job.id === activeId)) {
          nextJobs.unshift(JobSchema.parse(await apiJson(`/api/analyses/${encodeURIComponent(activeId)}`, controller.signal)));
        }
        const batchId = params.get('batch');
        const batch = batchId ? await apiJson(`/api/batches/${encodeURIComponent(batchId)}`, controller.signal) as UploadBatch : null;
        // Bracket private reads, including the initial load: a parallel session
        // request could otherwise label Alice's delayed history with Bob's owner.
        const owner = retryOwnerScope(await apiJson('/api/auth/session', controller.signal).catch(() => null));
        if (controller.signal.aborted || generation !== accountGeneration.current) return;
        if (ownerBefore !== owner || retryOwnerRef.current !== null && retryOwnerRef.current !== owner) {
          clearOwner(owner); setRefreshKey(key => key + 1); return;
        }
        if (health.auth_required === true && owner === null) {
          clearOwner(null);
          setConnectionError('Your account could not be confirmed. Reconnect before viewing private analyses.'); return;
        }
        retryOwnerRef.current = owner; setRetryOwner(owner);
        setLimit(health.max_upload_bytes); setJobs(nextJobs); setConnectionError(null);
        setAccessRequired(false);
        setResumable(health.resumable_uploads === true);
        if (batch) setActiveBatch(batch);
      } catch (caught) {
        if (!controller.signal.aborted && generation === accountGeneration.current) {
          if (caught instanceof ApiError && caught.status === 401) {
            clearOwner(null); setAccessRequired(true); setConnectionError(null);
          }
          else setConnectionError(caught instanceof Error ? caught.message : 'Unable to reach the analysis server.');
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(refresh, 2500);
      }
    };
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refreshKey, activeId, params, setParams]);
  useEffect(() => {
    if (autoOpen.current && activeJob?.status === 'COMPLETED') {
      autoOpen.current = false;
      navigate(`/analysis/${activeJob.id}`);
    }
  }, [activeJob, navigate]);

  async function updateJob(job: AnalysisJob) {
    const owner = retryOwnerRef.current;
    if (pendingActions.current.has(job.id) || job.status === 'FAILED' && owner === null) return;
    const generation = accountGeneration.current;
    const actions = pendingActions.current;
    const current = () => generation === accountGeneration.current && owner === retryOwnerRef.current;
    actions.add(job.id); setPendingJobs(previous => ({ ...previous, [job.id]: true })); setError(null);
    try {
      const updated = job.status === 'FAILED'
        ? await retryAnalysis(job.id, owner!, current)
        : JobSchema.parse(await mutate(`/api/analyses/${encodeURIComponent(job.id)}/cancel`, {}));
      if (!current()) return;
      setJobs(previous => [updated, ...previous.filter(item => item.id !== updated.id)]); setRefreshKey(key => key + 1);
    } catch (caught) {
      if (current()) setError(caught instanceof Error ? caught.message : 'Unable to update the job. Reconnect and retry.');
    } finally {
      actions.delete(job.id);
      if (current()) setPendingJobs(previous => ({ ...previous, [job.id]: false }));
    }
  }

  const fileKey = (next: File) => `${next.name}:${next.size}:${next.lastModified}`;
  function choose(selected: FileList | File[] | null) {
    if (!selected || uploading) return;
    const next = Array.from(selected);
    if (!next.length) return;
    if (next.length > 16) { setError('Choose up to 16 videos per batch.'); return; }
    if (next.some(item => !/\.(mp4|mov|avi)$/i.test(item.name))) { setFiles([]); setError('Choose an MP4, MOV or AVI video.'); return; }
    if (next.some(item => !item.size || (limit !== null && item.size > limit))) { setFiles([]); setError('A video is empty or exceeds the upload limit. Choose a smaller video.'); return; }
    if (new Set(next.map(fileKey)).size !== next.length) { setError('The same file is selected twice. Remove the duplicate before uploading.'); return; }
    setFiles(next); setFileStates({}); setError(null); draftId.current = null; setSavedCreation(null); setResumingSaved(false);
  }

  async function submit() {
    if (!file || uploading || uploadController.current) return;
    const controller = new AbortController();
    uploadController.current = controller;
    const generation = accountGeneration.current;
    const owner = retryOwnerRef.current;
    const current = () => generation === accountGeneration.current && owner === retryOwnerRef.current && !controller.signal.aborted;
    setError(null); setSavedCreation(null); setProgress(0);
    try {
      if (resumable) {
        if (owner === null) throw new Error('Reconnect to confirm your account before creating an upload.');
        const candidates = await Promise.all(files.map(async next => {
          const key = await fileResumeKey(next);
          if (!current()) throw new DOMException('Upload paused', 'AbortError');
          const known = sessions.current[fileKey(next)] ?? rememberedUpload(owner, key);
          if (known) {
            try { const state = await apiJson(`/api/uploads/${encodeURIComponent(known)}`, controller.signal) as { status: string };
              if (current() && state.status !== 'CANCELLED') { sessions.current[fileKey(next)] = known; return { file: next, key, known }; }
            } catch { /* A different signed-in owner cannot reuse this session. */ }
          }
          return { file: next, key, known: null };
        }));
        if (!current()) return;
        setResumingSaved(candidates.some(item => item.known !== null));
        const missing = candidates.filter(item => !item.known);
        if (missing.length) {
          const batch = await createUploadBatch(missing.map(item => item.file), mode, metadata, owner,
            { signal: controller.signal, isCurrent: current, fileKeys: missing.map(item => item.key) });
          if (!current()) return;
          missing.forEach((item, index) => { const session = batch.uploads[index];
            if (!session) throw new Error('The batch is missing an upload session. Reconnect before resuming.');
            sessions.current[fileKey(item.file)] = session.id; rememberUpload(owner, item.key, session.id);
          });
          setActiveBatch(batch); setParams({ batch: batch.id });
        }
        for (const next of files) {
          if (!current()) break;
          const key = fileKey(next);
          const fileController = new AbortController();
          const pauseFile = () => fileController.abort();
          controller.signal.addEventListener('abort', pauseFile, { once: true });
          activeFileUpload.current = { key, controller: fileController };
          setFileStates(current => ({ ...current, [key]: { progress: current[key]?.progress ?? 0 } }));
          try {
            const sessionId = sessions.current[key];
            if (!sessionId) throw new Error('No saved upload session for this file. Reconnect and retry.');
            const accepted = await resumeVideoUpload(next, sessionId, value => {
              if (current()) { setProgress(value); setFileStates(previous => ({ ...previous, [key]: { progress: value } })); }
            }, fileController.signal);
            if (!current()) return;
            setFileStates(current => ({ ...current, [key]: { progress: 100, job: accepted } }));
            setJobs(current => [accepted, ...current.filter(item => item.id !== accepted.id)]);
            if (files.length === 1) { autoOpen.current = true; setParams(current => { current.set('job', accepted.id); return current; }); }
          } catch (caught) {
            if (current()) setFileStates(previous => ({ ...previous, [key]: { progress: previous[key]?.progress ?? 0, error: caught instanceof Error ? caught.message : 'Upload failed. Resume this file.' } }));
          } finally {
            controller.signal.removeEventListener('abort', pauseFile);
            if (activeFileUpload.current?.controller === fileController) activeFileUpload.current = null;
          }
        }
        if (current()) setRefreshKey(key => key + 1);
        return;
      }
      if (files.length > 1) throw new Error('Reconnect to a server with batch uploading before sending multiple videos.');
      if (!draftId.current) {
        const match = await mutate<ProductMatch>('/api/matches', { source: 'upload', metadata });
        if (!current()) return;
        draftId.current = match.id;
      } else {
        const draft = await apiJson(`/api/matches/${draftId.current}`) as ProductMatch;
        if (!current()) return;
        if (draft.analysis_id) { navigate(`/matches/${draft.id}`); return; }
        await mutate(`/api/matches/${draftId.current}/metadata`, { metadata, expected_revision: draft.revision }, 'PATCH');
      }
      if (!current()) return;
      const job = await uploadVideo(file, value => { if (current()) setProgress(value); }, controller.signal, draftId.current);
      if (!current()) return;
      autoOpen.current = true;
      setJobs(current => [job, ...current.filter(item => item.id !== job.id)]);
      setParams({ job: job.id }); setFiles([]); setRefreshKey(key => key + 1);
      if (input.current) input.current.value = '';
    } catch (caught) {
      if (current()) {
        if (caught instanceof CreationIntentConflict) setSavedCreation(caught.request);
        setError(caught instanceof Error ? caught.message : 'Upload failed. Please try again.');
      }
    } finally {
      if (uploadController.current === controller) { setProgress(null); uploadController.current = null; }
    }
  }

  return <div className="page upload-page">
    <header className="upload-heading">
      <div><span className="astra-eyebrow">YOUR NEXT SESSION</span><h1>Bring your match<br /><em>into focus.</em></h1></div>
      <p>Every match has a story.<br />Upload yours to explore player movement, ball tracking and a new perspective on your court.</p>
    </header>
    <AccountAccess onChange={() => {
      accountGeneration.current += 1; retryOwnerRef.current = null; setRetryOwner(null);
      pendingActions.current = new Set(); setPendingJobs({}); setJobs([]); setError(null);
      uploadController.current?.abort(); activeFileUpload.current?.controller.abort();
      uploadController.current = null; activeFileUpload.current = null; sessions.current = {};
      draftId.current = null; autoOpen.current = false; setProgress(null); setFileStates({}); setFiles([]);
      setMetadata(emptyMetadata()); setSavedCreation(null); setResumingSaved(false);
      setActiveBatch(null); setParams({}); setRefreshKey(key => key + 1);
    }} />
    <div className="upload-layout">
      <section className="upload-panel" aria-label="Upload tennis video">
        <div className="upload-panel-caption"><span>VIDEO ANALYSIS</span><span>START WITH YOUR FOOTAGE <ArrowRight size={14} aria-hidden="true" /></span></div>
        <div className={`upload-dropzone ${dragging ? 'is-dragging' : ''}`} onDragOver={event => { event.preventDefault(); if (!uploading) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files); }}>
          <svg className="upload-court-lines" viewBox="0 0 720 420" preserveAspectRatio="none" aria-hidden="true"><path d="M36 28H684V392H36Z M116 28V392 M604 28V392 M36 210H684 M116 118H604 M116 302H604 M360 118V302" /></svg>
          <CloudArrowUp size={42} weight="light" aria-hidden="true" />
          <h2>{files.length > 1 ? `${files.length} tennis videos selected` : file ? file.name : 'Drop your tennis video here'}</h2>
          <p>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Ready to upload` : `MP4, MOV or AVI${limit ? ` · Up to ${(limit / 1024 / 1024).toFixed(0)} MB` : ''}`}</p>
          <label className={`upload-file-label ${uploading ? 'is-disabled' : ''}`}>Choose video<ArrowRight size={40} weight="light" aria-hidden="true" /><input ref={input} type="file" multiple accept=".mp4,.mov,.avi" disabled={uploading} onChange={event => choose(event.target.files)} aria-label="Choose tennis video" /></label>
          <span className="upload-dropzone-note" aria-hidden="true">A NEW VIEW OF YOUR GAME</span>
        </div>
        {file ? <><div className="upload-files" aria-label="Selected videos">{files.map((item, index) => {
          const state = fileStates[fileKey(item)];
          return <article key={fileKey(item)}><div><strong>{index + 1}. {item.name}</strong><span>{(item.size / 1024 ** 2).toFixed(1)} MB</span></div>
            <p>{state?.job ? 'Received · Saved analysis created' : state?.error ?? (state ? `${state.progress}% received · Choose Analyze match to resume` : 'Ready to upload')}</p>
            {state?.error ? <p role="alert">{state.error}</p> : null}
            {state && !state.job ? <progress value={state.progress} max={100} aria-label={`Upload progress for ${item.name}`} /> : null}
            {state?.job ? <Link to={`/upload?job=${state.job.id}`}>Follow processing</Link> : null}
            {resumable && !state?.job ? <button type="button" className="astra-text-button" onClick={async () => {
              const generation = accountGeneration.current;
              const key = fileKey(item); const sessionId = sessions.current[key];
              if (activeFileUpload.current?.key === key) activeFileUpload.current.controller.abort();
              try { if (sessionId) await mutate(`/api/uploads/${encodeURIComponent(sessionId)}`, {}, 'DELETE');
                if (generation !== accountGeneration.current) return;
                delete sessions.current[key]; setFiles(current => current.filter(next => fileKey(next) !== key));
              } catch (caught) { if (generation === accountGeneration.current) setError(caught instanceof Error ? caught.message : 'Unable to cancel this file. Reconnect and retry.'); }
            }}>Cancel {item.name}</button> : null}
          </article>;
        })}</div><fieldset className="upload-mode" disabled={uploading}><legend>How should these videos be organized?</legend>
          <label><input type="radio" name="batch-mode" checked={mode === 'independent'} onChange={() => setMode('independent')} />Independent analyses</label>
          <label><input type="radio" name="batch-mode" checked={mode === 'same_match'} onChange={() => setMode('same_match')} />Clips from the same match, in the order shown</label>
          {mode === 'same_match' ? <p>Confirm the order shown above before uploading. Identity, overlap and timing require review before a combined score or report is available.</p> : null}
        </fieldset><MatchMetadataForm value={metadata} onChange={setMetadata} disabled={uploading} />
        {resumingSaved ? <p className="upload-hint">Saved uploads keep their original match details. Edits here apply only to newly created uploads.</p> : null}</> : null}
        {error ? <p className="upload-error" role="alert"><WarningCircle size={20} />{error}</p> : null}
        {savedCreation?.kind === 'batch' ? <button type="button" className="astra-text-button" disabled={uploading} onClick={() => {
          setMetadata(savedCreation.payload.metadata); setMode(savedCreation.payload.mode); setSavedCreation(null);
          setError('Saved match details restored. Reselect the original files if needed, then choose Analyze match to recover the saved request.');
        }}>Restore saved match details</button> : null}
        {uploading ? <div className="upload-progress" role="status"><span>{progress === 100 ? 'Upload received. Creating your analysis…' : `Uploading video · ${progress}%`}</span><progress max={100} value={progress ?? 0} aria-label="Video upload progress" /><button type="button" className="astra-text-button" onClick={() => uploadController.current?.abort()}>Cancel upload</button></div> : <button className="upload-submit" type="button" disabled={!file || limit === null || connectionError !== null || accessRequired} onClick={() => void submit()}>Analyze match<ArrowRight size={18} /></button>}
        {accessRequired ? <p className="upload-hint" role="status">Sign in to upload videos and view your private analysis history.</p> : null}
        <p className="upload-hint">For the clearest results, use a mostly static camera with the court and both players visible. Processing continues when you leave this page after upload.</p>
      </section>
      <aside className="upload-guide"><span className="astra-eyebrow">FROM FOOTAGE TO FEEDBACK</span><h2>Your game.<br /><em>A fresh perspective.</em></h2><ol><li><b>01</b><div><h3>Upload your session</h3><p>Each upload creates a separate saved analysis.</p></div></li><li><b>02</b><div><h3>Follow the analysis</h3><p>See each processing stage as the tennis pipeline tracks your match.</p></div></li><li><b>03</b><div><h3>Review every movement</h3><p>Open synchronized video, court replay, movement maps and statistics.</p></div></li></ol><div className="upload-guide-footer"><span>LESS GUESSWORK.<br />MORE PERSPECTIVE.</span><ArrowRight size={42} weight="light" aria-hidden="true" /></div></aside>
    </div>
    {activeBatch ? <section className="upload-batch" aria-label="Saved batch"><h2>Batch saved · {activeBatch.uploads.length} videos</h2>
      <p>{activeBatch.mode === 'same_match' ? activeBatch.reason ?? 'Clip grouping awaits verification. Each video has its own analysis.' : 'Each video processes independently. A failed file can be retried separately.'}</p>
      <p>Received chunks persist after refresh. Reselect the original files to resume an interrupted transfer.</p></section> : null}
    {connectionError ? <div className="upload-connection" role="alert"><WarningCircle size={22} /><div><strong>Analysis server unavailable</strong><p>{connectionError} Your saved jobs will reappear when the connection returns.</p><button type="button" className="astra-text-button" onClick={() => setRefreshKey(key => key + 1)}>Reconnect</button></div></div> : null}
    {activeJob ? <section className={`upload-active status-${activeJob.status.toLowerCase()}`} aria-label="Current analysis" aria-live="polite"><div>{activeJob.status === 'COMPLETED' ? <CheckCircle size={28} /> : <FilmStrip size={28} />}<span><h2>{activeJob.status === 'COMPLETED' ? 'Your analysis is ready' : activeJob.status === 'FAILED' ? 'Analysis could not finish' : 'Analyzing your tennis match…'}</h2><p>{activeJob.filename}</p></span></div><p>{activeJob.error ?? activeJob.stage}{activeJob.status === 'PROCESSING' && activeJob.step > 0 ? ` · Stage ${activeJob.step} of 11` : ''}</p>{activeJob.status === 'COMPLETED' ? <Link className="upload-result-link" to={`/analysis/${activeJob.id}`}>Open analysis<ArrowRight size={18} /></Link> : null}</section> : null}
    <section className="upload-history" aria-label="Analysis history">
      <div className="upload-history-heading"><div><span className="astra-eyebrow">THE SESSION ARCHIVE</span><h2>{historyReady ? 'Your analyses' : 'Saved analyses'}</h2></div><span>{historyJobs.length} {historyReady ? 'sessions' : 'saved analyses'}</span></div>
      {!historyReady ? <div className="upload-hint" role="status"><p>Public example filtering is unavailable. Showing saved analyses, which may include public examples.</p>{catalog.error ? <><button type="button" className="astra-text-button" onClick={catalog.refresh}>Retry history</button><Link className="astra-text-button" to="/match-center">My analyses</Link></> : null}</div> : null}
      {!historyJobs.length && !connectionError && !accessRequired ? <p className="upload-hint upload-history-empty">{historyReady ? 'Your uploaded sessions will appear here. Historical examples are available in the separate Examples section.' : 'Saved analyses will appear here.'}</p> : null}
      {historyJobs.map(job => <article key={job.id}><FilmStrip size={24} weight="light" aria-hidden="true" /><div><h3>{job.filename}</h3><p>{new Date(job.created_at).toLocaleString()} · {job.error ?? job.stage}</p></div><span className={`upload-job-state status-${job.status.toLowerCase()}`}>{job.cancel_requested ? (job.status === 'FAILED' ? 'Cancelled' : 'Cancellation requested') : job.status === 'UPLOADED' ? 'Queued' : job.status.toLowerCase()}</span>{job.status === 'COMPLETED' ? <Link to={`/analysis/${job.id}`} aria-label={`Open analysis for ${job.filename}`}>View<ArrowRight size={16} /></Link> : <><button type="button" className="astra-text-button" onClick={() => setParams({ job: job.id })}>Details</button>{resumable ? <button type="button" className="astra-text-button" disabled={pendingJobs[job.id] || job.status === 'FAILED' && retryOwner === null} onClick={() => void updateJob(job)}>{pendingJobs[job.id] ? (job.status === 'FAILED' ? 'Retrying analysis…' : 'Cancelling analysis…') : job.status === 'FAILED' ? 'Retry analysis' : 'Cancel analysis'}</button> : null}</>}</article>)}
    </section>
  </div>;
}

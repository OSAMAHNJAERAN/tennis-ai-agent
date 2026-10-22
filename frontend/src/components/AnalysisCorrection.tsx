import { useEffect, useRef, useState } from 'react';
import type { AnalysisRun } from '../data/contracts';
import { ApiError } from '../data/analysisApi';
import { correctionApi, CorrectionCommandSchema, type CorrectionCommand, type CorrectionStatus, type ReviewSource, type SourceInput } from '../data/correctionApi';
import { amendInitial, amendInput, commandBase, initialEditable, initialScoreDraft, PHYSICAL_TYPES, REVIEWED_TYPES, type ScoreDraft } from '../data/correctionDraft';
import '../styles/correction.css';

const activeStatus = (row: CorrectionStatus | null) => !!row && ['QUEUED', 'PROCESSING'].includes(row.status);
const message = (error: unknown) => error instanceof Error ? error.message : 'The correction server is unavailable.';
const accessDenied = (error: unknown) => error instanceof ApiError && [401, 403, 404].includes(error.status);
const storageKey = (id: string, generation: string) => `astra.correction.pending.${id}.${generation}`;
function retainedCommand(id: string, generation: string): CorrectionCommand | null {
  try { const raw: unknown = JSON.parse(sessionStorage.getItem(storageKey(id, generation)) ?? 'null'); const parsed = CorrectionCommandSchema.safeParse(raw);
    return parsed.success && parsed.data.base_generation === generation ? parsed.data : null; } catch { return null; }
}

export function AnalysisCorrection(props: { run: AnalysisRun; onCompleted: (generation: string) => void; onSelectEvent: (id: number | null) => void; onRefresh?: () => void; onSeekFrame?: (frame: number) => void; readOnly?: boolean }) {
  if (props.readOnly) return <section className="analysis-correction correction-readonly"><h2>Review status</h2><p>This curated example is read-only. Open your own uploaded analysis to review and submit corrections.</p></section>;
  return <CorrectionWorkspace key={`${props.run.summary.id}:${props.run.analysisGeneration ?? ''}`} {...props} />;
}
function CorrectionWorkspace({ run, onCompleted, onSelectEvent, onRefresh, onSeekFrame }: { run: AnalysisRun; onCompleted: (generation: string) => void; onSelectEvent: (id: number | null) => void; onRefresh?: () => void; onSeekFrame?: (frame: number) => void }) {
  const id = run.summary.id; const generation = run.analysisGeneration;
  const [source, setSource] = useState<ReviewSource | null>(null);
  const [admission, setAdmission] = useState<'loading' | 'readonly' | 'owner' | 'error'>('loading');
  const [localOwner, setLocalOwner] = useState(false); const [reload, setReload] = useState(0);
  const [history, setHistory] = useState<CorrectionStatus[]>([]); const [current, setCurrent] = useState<CorrectionStatus | null>(null);
  const [pending, setPending] = useState(false); const [error, setError] = useState<string | null>(null);
  const [retryCommand, setRetryCommand] = useState<CorrectionCommand | null>(() => generation ? retainedCommand(id, generation) : null);
  const [score, setScore] = useState<ScoreDraft | null>(null); const [changeScore, setChangeScore] = useState(false);
  const [changes, setChanges] = useState<Record<number, SourceInput>>({}); const [reason, setReason] = useState(''); const [evidence, setEvidence] = useState('');
  const [authorized, setAuthorized] = useState(false); const errorRef = useRef<HTMLDivElement>(null);
  const mounted = useRef(true); const completed = useRef<string | null>(null);
  const operations = useRef<AbortController | null>(null);
  const active = activeStatus(current); const locked = pending || active || !!retryCommand;
  useEffect(() => { mounted.current = true; operations.current = new AbortController(); return () => { mounted.current = false; operations.current?.abort(); }; }, []);
  useEffect(() => { if (error) errorRef.current?.focus(); }, [error]);
  useEffect(() => {
    if (!generation || !/^match_[a-f0-9]{32}$/.test(id)) return;
    const controller = new AbortController();
    (async () => {
      const session = await correctionApi.session(controller.signal);
      if (session.required && !session.authenticated) { setAdmission('readonly'); return; }
      const [nextSource, rows] = await Promise.all([correctionApi.source(id, generation, 0, controller.signal), correctionApi.list(id, controller.signal)]);
      if (controller.signal.aborted) return;
      if (nextSource.base_analysis_revision !== run.analysisRevision || (run.pointAnalysis && nextSource.source_sha256 !== run.pointAnalysis.sourceSha256)) throw new Error('Review evidence differs from the displayed analysis revision or source.');
      setLocalOwner(!session.required); setSource(nextSource); setScore(initialScoreDraft(nextSource.initial_state)); setHistory(rows);
      setCurrent(nextSource.is_current ? rows.find(row => activeStatus(row) && row.base_generation === generation) ?? null : null); setAdmission('owner'); setError(null);
    })().catch(caught => { if (controller.signal.aborted) return;
      if (accessDenied(caught)) setAdmission('readonly');
      else { setAdmission('error'); setError(message(caught)); }
    });
    return () => controller.abort();
  }, [id, generation, reload, run.analysisRevision, run.pointAnalysis]);
  useEffect(() => {
    if (!current || !activeStatus(current)) return;
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try { const next = await correctionApi.status(id, current.id, controller.signal); if (controller.signal.aborted) return;
        setCurrent(next); setError(null); if (activeStatus(next)) timer = setTimeout(() => void poll(), 1500);
      } catch (caught) { if (!controller.signal.aborted) { if (accessDenied(caught)) { setAdmission('readonly'); setCurrent(null); }
        else { setError(message(caught)); timer = setTimeout(() => void poll(), 4000); } } }
    };
    timer = setTimeout(() => void poll(), 1000);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, current]);
  useEffect(() => {
    if (current?.status === 'COMPLETED' && current.generation && completed.current !== current.id) {
      completed.current = current.id; onSelectEvent(null); onCompleted(current.generation);
    }
  }, [current, onCompleted, onSelectEvent]);

  async function submit(command: CorrectionCommand) {
    setPending(true); setError(null); setRetryCommand(command);
    try { sessionStorage.setItem(storageKey(id, generation!), JSON.stringify(command)); } catch { /* In-memory retry remains available when browser storage is disabled. */ }
    try {
      const row = await correctionApi.submit(id, command, operations.current?.signal);
      if (!mounted.current) return;
      setCurrent(row); setRetryCommand(null); setHistory(rows => [row, ...rows.filter(value => value.id !== row.id)]);
      try { sessionStorage.removeItem(storageKey(id, generation!)); } catch { /* Optional browser persistence. */ }
    } catch (caught) { if (mounted.current) {
      setError(message(caught));
      if (accessDenied(caught)) { setAdmission('readonly'); setCurrent(null); }
      if (caught instanceof ApiError && [400, 401, 403, 404, 409, 413, 422].includes(caught.status)) {
        setRetryCommand(null); try { sessionStorage.removeItem(storageKey(id, generation!)); } catch { /* Optional storage. */ }
      }
    } }
    finally { if (mounted.current) setPending(false); }
  }
  function prepare(event: React.FormEvent) {
    event.preventDefault(); if (!source || !score) return;
    try {
      if (!authorized) throw new Error('Confirm your review and authorize rebuilding this analysis.');
      const command = commandBase(source, crypto.randomUUID(), reason, evidence);
      if (changeScore) { const raw = amendInitial(source, score);
        if (JSON.stringify(raw) !== JSON.stringify(source.initial_state)) command.initial_state_amendment = { expected_initial_state_sha256: source.initial_state_sha256, replacement_initial_state: raw }; }
      command.input_amendments = source.inputs.flatMap(row => { const replacement = changes[row.source_input.event_id]; return replacement && JSON.stringify(replacement) !== JSON.stringify(row.source_input)
        ? [{ event_id: row.source_input.event_id, expected_input_fingerprint: row.input_fingerprint, replacement_source_input: replacement }] : []; });
      if (!command.initial_state_amendment && !command.input_amendments.length) throw new Error('Choose an initial score or event interpretation to change.');
      void submit(CorrectionCommandSchema.parse(command));
    } catch (caught) { setError(message(caught)); }
  }
  async function page(offset: number) {
    if (!generation) return; setPending(true); setError(null);
    try { const next = await correctionApi.source(id, generation, offset, operations.current?.signal); if (mounted.current) { setSource(next); setChanges({}); } }
    catch (caught) { if (mounted.current) { if (accessDenied(caught)) { setAdmission('readonly'); setCurrent(null); } else setError(message(caught)); } } finally { if (mounted.current) setPending(false); }
  }
  if (!generation || !/^match_[a-f0-9]{32}$/.test(id) || admission === 'readonly') return <section className="analysis-correction correction-readonly"><h2>Review status</h2><p>This analysis is read-only. Corrections require access to the original owner’s saved source evidence.</p></section>;
  if (admission === 'loading') return <p role="status">Checking access to saved review evidence…</p>;
  return <section className="analysis-correction" aria-label="Review and correct analysis">
    <header><div><span className="astra-eyebrow">Human review</span><h2>Review and correct analysis</h2></div><span className="correction-badge">Revision {run.analysisRevision ?? 0}</span></header>
    <p>Rebuild score, point evidence, forecasts, reports and annotated video together. Your current replay remains available while processing. Reviewed outputs remain unqualified.</p>
    {error ? <div className="correction-error" role="alert" tabIndex={-1} ref={errorRef}>{error}</div> : null}
    {admission === 'error' ? <button type="button" className="button" onClick={() => { setAdmission('loading'); setReload(value => value + 1); }}>Reconnect review evidence</button> : null}
    {current ? <div className="correction-status" role="status"><strong>{current.status.replaceAll('_', ' ')}</strong><span>{current.stage} · attempt {current.attempt}</span><small>Request {current.id}</small>{current.error ? <p>{current.error}</p> : null}
      {active ? <button type="button" className="button" disabled={pending || current.cancel_requested} onClick={async () => { setPending(true); try { const next = await correctionApi.cancel(id, current.id, operations.current?.signal); if (mounted.current) setCurrent(next); } catch (caught) { if (mounted.current) setError(message(caught)); } finally { if (mounted.current) setPending(false); } }}>{current.cancel_requested ? 'Cancellation requested' : 'Cancel correction'}</button> : null}
      {current.status === 'COMPLETED' && current.generation ? <button className="button" type="button" onClick={() => { onSelectEvent(null); onCompleted(current.generation!); }}>Open corrected generation</button> : null}
    </div> : null}
    {retryCommand ? <div className="correction-status"><p>A submission is awaiting confirmation. Retrying sends the same saved command and cannot create another copy.</p><button type="button" className="button" disabled={pending} onClick={() => void submit(retryCommand)}>{pending ? 'Submitting…' : 'Retry saved submission'}</button><button type="button" className="astra-text-button" disabled={pending} onClick={() => { setRetryCommand(null); try { sessionStorage.removeItem(storageKey(id, generation)); } catch { /* Optional storage. */ } }}>Discard local draft</button></div> : null}
    {source && score ? <form onSubmit={prepare}>
      {!source.is_current ? <p role="status">This generation has been superseded. Reload the current analysis before starting a correction. <button type="button" className="button" onClick={onRefresh}>Load current analysis</button></p> : null}
      <fieldset disabled={locked || !source.is_current}><legend>Initial score context</legend>
        <p>Set {source.initial_state.fields.current_set} · games {source.initial_state.fields.games_p1}–{source.initial_state.fields.games_p2} · {source.initial_state.fields.tie_break_active ? 'Tie-break' : 'Standard game'}. Completed sets and playing phase are retained.</p>
        {initialEditable(source.initial_state) ? <><label className="correction-check"><input type="checkbox" checked={changeScore} onChange={event => setChangeScore(event.target.checked)} />Review initial score and serve context</label>
          {changeScore ? <div className="correction-grid">{(['p1', 'p2'] as const).map((key, index) => <label key={key}>Player {index + 1} initial points{source.initial_state.fields.tie_break_active ? <input type="number" min={0} max={1000} step={1} value={score[key]} onChange={event => setScore({ ...score, [key]: Number(event.target.value) })} /> : <select value={score[key]} onChange={event => setScore({ ...score, [key]: Number(event.target.value) })}>{['0', '15', '30', '40', 'Adv'].map((label, value) => <option key={value} value={value}>{label}</option>)}</select>}</label>)}
          <label>{source.initial_state.fields.tie_break_active ? 'First tie-break server' : 'Server'}<select value={score.server} onChange={event => setScore({ ...score, server: Number(event.target.value) })}>{[source.initial_state.fields.player_1_id, source.initial_state.fields.player_2_id].map((player, index) => <option key={player} value={player}>Player {index + 1}</option>)}</select></label>
          <label>Serve attempt<select value={score.serve} onChange={event => setScore({ ...score, serve: Number(event.target.value) as 1 | 2 })}><option value={1}>First serve</option><option value={2}>Second serve</option></select></label><small>Receiver and service side are derived from server and point parity.</small></div> : null}</> : <p>This initial playing phase or completed result requires a full state review and is read-only in this form.</p>}
      </fieldset>
      <fieldset disabled={locked || !source.is_current}><legend>Existing event interpretations</legend><p>Source times, coordinates, detector roles, confidence and identity provenance are retained. Select an event to compare the replay.</p>
        {source.inputs.map(row => { const original = row.source_input; const input = changes[original.event_id] ?? original; const proof = input.context.binding; const types = original.reviewed_evidence_ref === null ? PHYSICAL_TYPES : REVIEWED_TYPES;
          const update = (type: string, participant: number | null = proof?.participant_id ?? null, end: string | null = proof?.court_end ?? null) => setChanges(previous => ({ ...previous, [original.event_id]: amendInput(original, type, participant, end) }));
          return <div className="correction-event" key={original.event_id}><button type="button" className="astra-text-button" onClick={() => { onSelectEvent(original.event_id); onSeekFrame?.(original.frame_index); }}>Event {original.event_id} · frame {original.frame_index}</button><label>Interpretation<select aria-label={`Event ${original.event_id} interpretation`} value={input.event_type} onChange={event => update(event.target.value)}>{types.map(type => <option key={type} value={type}>{type.toLowerCase().replaceAll('_', ' ')}</option>)}</select></label>
            {proof ? <><label>Reviewed participant<select aria-label={`Event ${original.event_id} participant`} value={proof.participant_id} onChange={event => update(input.event_type, Number(event.target.value))}><option value={1}>Player 1</option><option value={2}>Player 2</option></select></label><label>Reviewed court end<select aria-label={`Event ${original.event_id} court end`} value={proof.court_end} onChange={event => update(input.event_type, proof.participant_id, event.target.value)}><option value="NEAR_COURT">Near court</option><option value="FAR_COURT">Far court</option></select></label></> : <small>No current reviewed participant binding; identity is read-only.</small>}
          </div>;
        })}
        {!source.inputs.length ? <p>No accepted source events in this saved generation.</p> : null}
        <div className="correction-pagination"><span>{source.offset + (source.inputs.length ? 1 : 0)}–{source.offset + source.inputs.length} of {source.total_inputs} events</span><button type="button" className="button" disabled={source.offset === 0 || Object.keys(changes).length > 0} onClick={() => void page(Math.max(0, source.offset - source.limit))}>Previous events</button><button type="button" className="button" disabled={source.next_offset === null || Object.keys(changes).length > 0} onClick={() => void page(source.next_offset!)}>Next events</button></div><small>Submit or reset this page’s edits before changing pages.</small>
      </fieldset>
      <fieldset disabled={locked || !source.is_current}><legend>Review evidence and authorization</legend><div className="correction-grid"><label>Reason for correction<textarea value={reason} maxLength={4000} required onChange={event => setReason(event.target.value)} /></label><label>Video evidence reference<input value={evidence} maxLength={2000} placeholder="For example: original video, 00:12–00:15" required onChange={event => setEvidence(event.target.value)} /></label></div>
        <label className="correction-check"><input type="checkbox" checked={authorized} onChange={event => setAuthorized(event.target.checked)} />I reviewed the source video and authorize this correction{localOwner ? ' as the local owner' : ' as the signed-in owner'}.</label>
        <div className="correction-actions"><button type="submit" className="button" disabled={!authorized}>Submit correction and rebuild</button><button type="button" className="astra-text-button" onClick={() => { setChanges({}); setChangeScore(false); setScore(initialScoreDraft(source.initial_state)); }}>Reset edits</button></div>
      </fieldset>
    </form> : null}
    {history.length ? <details><summary>Saved correction history ({history.length})</summary><ul>{history.map(row => <li key={row.id}>{row.status} · {row.created_at} · revision {row.base_analysis_revision}{row.status === 'COMPLETED' && row.generation ? <button type="button" className="astra-text-button" onClick={() => onCompleted(row.generation!)}>Open saved result</button> : null}</li>)}</ul></details> : null}
  </section>;
}

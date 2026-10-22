import { useState } from 'react';
import { Link, Navigate, useLocation, useParams, useSearchParams } from 'react-router-dom';
import { AnalysisProvider, useAnalysis } from '../app/AnalysisContext';
import { useProductResource } from '../hooks/useProductResource';
import { matchTitle, mutate, participantName, statusLabel, type ProductMatch } from '../data/productApi';
import { MatchMetadataForm } from '../components/MatchMetadataForm';
import { Card, DataError, SkeletonPage } from '../components/ui';
import { ProductError } from './ProductHome';
import { ManualLiveDesk, ManualStatistics } from './ManualLive';
import SpatialAnalysis from './SpatialAnalysis';
import SessionAnalytics from './LiveRunAnalytics';
import Reports from './Reports';
import Events from './Events';
import BallTracking from './BallTracking';
import type { FrameRange } from '../lib/tennisTelemetry';
import { MediaSourceCredit } from '../components/MediaSourceCredit';
import { WinProbability } from '../components/WinProbability';

function Notes({ match, onUpdate }: { match: ProductMatch; onUpdate: (m: ProductMatch) => void }) {
  const [notes, setNotes] = useState(match.notes ?? '');
  const [baseRevision, setBaseRevision] = useState(match.revision);
  const [error, setError] = useState(''); const [busy, setBusy] = useState(false); const [saved, setSaved] = useState(false);
  async function save() { setBusy(true); setError(''); try { const result = await mutate<ProductMatch>(`/api/matches/${match.id}/notes`, { notes, expected_revision: baseRevision }, 'PUT'); onUpdate(result); setBaseRevision(result.revision); setSaved(true); } catch(e) { setError(e instanceof Error ? e.message : 'Unable to save notes.'); } finally { setBusy(false); } }
  return <Card className="product-panel"><h2>Your match notes</h2><p className="product-hint">User notes are separate from measured observations and AI suggestions.</p><label className="sr-only" htmlFor="match-notes">Match notes</label><textarea id="match-notes" className="product-note" maxLength={10000} value={notes} onChange={e => { setNotes(e.target.value); setSaved(false); }} /><button className="button" disabled={busy} onClick={() => void save()}>{busy ? 'Saving…' : 'Save notes'}</button>{saved ? <span role="status"> Notes saved</span> : null}{error ? <p role="alert">{error}</p> : null}</Card>;
}
function Details({ match, onUpdate }: { match: ProductMatch; onUpdate: (m: ProductMatch) => void }) {
  const [metadata, setMetadata] = useState(match.metadata); const [baseRevision, setBaseRevision] = useState(match.revision); const [error, setError] = useState(''); const [busy, setBusy] = useState(false); const [saved, setSaved] = useState(false);
  async function save() { setBusy(true); setError(''); try { const result = await mutate<ProductMatch>(`/api/matches/${match.id}/metadata`, { metadata, expected_revision: baseRevision }, 'PATCH'); onUpdate(result); setMetadata(result.metadata); setBaseRevision(result.revision); setSaved(true); } catch(e) { setError(e instanceof Error ? e.message : 'Unable to save details.'); } finally { setBusy(false); } }
  return <Card className="product-panel"><h2>Match information</h2><MatchMetadataForm value={metadata} onChange={value => { setMetadata(value); setSaved(false); }} disabled={busy} />{match.source === 'upload' ? <><label>Recorded result <input maxLength={120} value={metadata.result ?? ''} placeholder="Unknown — enter only if known" onChange={e => setMetadata({ ...metadata, result: e.target.value || null })} /></label><h3>Confirm tracked identities</h3><p>The tracker labels near/far court roles. Confirm only if these roles belong to the same people throughout this entire clip. Leave unconfirmed if players change ends or the camera switches.</p><label>Whole-clip role assignment<select value={metadata.role_assignment} onChange={e => setMetadata({ ...metadata, role_assignment: e.target.value as typeof metadata.role_assignment })}><option value="unconfirmed">Unconfirmed — keep role-based statistics</option><option value="same">I confirm: P1 = {metadata.participants[0]?.name || 'Player 1'}, P2 = {metadata.participants[1]?.name || 'Player 2'} throughout</option><option value="swapped">I confirm: P1 = {metadata.participants[1]?.name || 'Player 2'}, P2 = {metadata.participants[0]?.name || 'Player 1'} throughout</option></select></label><p className="product-hint">Changing participants clears confirmation. Save participants first, then confirm roles. Named movement history also requires linked profiles.</p></> : null}<button className="button button-primary" disabled={busy} onClick={() => void save()}>{busy ? 'Saving…' : 'Save match details'}</button>{saved ? <span role="status"> Details saved</span> : null}{error ? <p role="alert">{error}</p> : null}</Card>;
}
function RecordedPanels({ section }: { section: string }) {
  const { run, error, loading, retry } = useAnalysis();
  const [range, setRange] = useState<FrameRange | undefined>(undefined);
  if (loading) return <SkeletonPage />;
  if (error || !run) return <DataError message={error?.message ?? 'Analysis unavailable.'} onRetry={retry} />;
  return <>{run.mediaSource ? <MediaSourceCredit source={run.mediaSource} expanded /> : null}<section className="product-workspace-section" hidden={section !== 'replay'}><SpatialAnalysis active={section === 'replay'} range={range} onRangeChange={setRange} /></section><section className="product-workspace-section" hidden={section !== 'statistics'}>{section === 'statistics' ? <><SessionAnalytics /><details><summary>Ball tracking evidence</summary><BallTracking /></details>{run.summary.capabilities.events ? <details><summary>Events and rally evidence</summary><Events /></details> : null}</> : null}</section>{section === 'details' ? <details><summary>Analysis exports and quality</summary><Reports /></details> : null}</>;
}
export default function ProductWorkspace() {
  const { matchId = '' } = useParams();
  const resource = useProductResource<ProductMatch>(`/api/matches/${encodeURIComponent(matchId)}`, 5000);
  const match = resource.data;
  const [params] = useSearchParams();
  const requested = params.get('section') ?? 'replay';
  const section = ['replay', 'statistics', 'insights', 'details'].includes(requested) ? requested : 'replay';
  if (!match) return <div className="page product-page"><Link to="/match-center">Back to Match Center</Link>{resource.error ? <ProductError error={resource.error} retry={resource.refresh} /> : <SkeletonPage />}</div>;
  const tabLink = (value: string) => { const next = new URLSearchParams(params); next.set('section', value); next.delete('run'); return `?${next}`; };
  return <div className="page product-page"><Link className="product-breadcrumb" to="/match-center">← Match Center</Link><header><span className="astra-eyebrow">{match.source === 'manual' ? 'MANUALLY SCORED' : 'RECORDED MATCH'}</span><h1>{matchTitle(match)}</h1><div className="product-match-summary"><span>{statusLabel(match.status)}</span><span>{match.metadata.tournament || 'Tournament unknown'}</span><span>{match.metadata.match_date || 'Match date unknown'}</span><span>{match.source === 'manual' && match.score?.state.winner_id ? `${participantName(match, match.score.state.winner_id - 1)} won` : match.metadata.result || 'Result unknown'}</span></div><p className="product-hint">{match.source === 'upload' ? 'Uploaded' : 'Created'} {new Date(match.created_at).toLocaleString()} · Match details: {match.metadata_source}</p></header><ProductError error={resource.error} retry={resource.refresh} />
    <nav className="product-workspace-tabs" aria-label="Match sections">{[['replay', match.source === 'manual' ? 'Saved score' : 'Replay'], ['statistics', 'Movement & tracking'], ['insights', 'Notes'], ['details', 'Details & exports']].map(([value, label]) => <Link key={value} to={tabLink(value!)} aria-current={section === value ? 'page' : undefined}>{label}</Link>)}</nav>
    {section === 'details' ? <Details key={match.id} match={match} onUpdate={resource.replace} /> : null}
    {match.source === 'manual' ? <><section hidden={section !== 'replay'}><ManualLiveDesk match={match} onUpdate={resource.replace} /></section>{section === 'statistics' ? <ManualStatistics match={match} /> : null}</> : null}
    {match.source === 'upload' && match.status === 'COMPLETED' && match.analysis_id ? <AnalysisProvider key={match.analysis_id} runId={match.analysis_id}><RecordedPanels section={section} /></AnalysisProvider> : match.source === 'upload' && section !== 'details' && section !== 'insights' ? <Card className="product-panel"><h2>{match.status === 'FAILED' ? 'Analysis could not finish' : 'Analyzing your match'}</h2><p>{match.error || match.stage || 'Waiting for upload'}</p>{match.status === 'FAILED' ? <Link className="button" to="/upload">Upload a new video</Link> : <p className="product-hint">You can leave this page; accepted processing continues.</p>}</Card> : null}
    {section === 'insights' ? <><WinProbability key={`forecast-${match.id}`} match={match} /><Notes key={match.id} match={match} onUpdate={resource.replace} /></> : null}
    {section === 'details' ? <>{match.source === 'manual' ? <button className="button" onClick={() => { const url = URL.createObjectURL(new Blob([JSON.stringify(match, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = `${match.id}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }}>Download match record</button> : null}</> : null}
  </div>;
}

export function LegacyAnalysis() {
  const { analysisId } = useParams(); const location = useLocation(); const [params] = useSearchParams();
  const id = analysisId ?? params.get('run');
  if (!id) return <Navigate to={['/agent', '/scouting'].includes(location.pathname) ? '/insights' : '/match-center'} replace />;
  if (!id.startsWith('match_')) return <Navigate to={`/examples/${encodeURIComponent(id)}?${params}`} replace />;
  return <ResolveLegacy id={id} section={location.pathname === '/analytics' || location.pathname === '/ball' || location.pathname === '/events' || location.pathname === '/line-calls' ? 'statistics' : location.pathname === '/agent' || location.pathname === '/scouting' ? 'insights' : location.pathname === '/reports' ? 'details' : 'replay'} />;
}
function ResolveLegacy({ id, section }: { id: string; section: string }) {
  const resource = useProductResource<ProductMatch>(`/api/matches/by-analysis/${encodeURIComponent(id)}`); const [params] = useSearchParams();
  if (resource.error) return <DataError message={resource.error} onRetry={resource.refresh} />;
  if (!resource.data) return <SkeletonPage />;
  const next = new URLSearchParams(params); next.delete('run'); next.set('section', section);
  return <Navigate to={`/matches/${resource.data.id}?${next}`} replace />;
}

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MatchMetadataForm } from '../components/MatchMetadataForm';
import { emptyMetadata, mutate, participantName, type MatchMetadata, type ProductMatch } from '../data/productApi';
import { Card, PageHeader } from '../components/ui';

export default function NewLiveMatch() {
  const [metadata, setMetadata] = useState<MatchMetadata>(emptyMetadata);
  const [bestOf, setBestOf] = useState(3);
  const [server, setServer] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('');
    try { const match = await mutate<ProductMatch>('/api/matches', { source: 'manual', metadata, best_of: bestOf, initial_server: Number(server) }); navigate(`/matches/${match.id}`); }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to create match.'); }
    finally { setBusy(false); }
  }
  return <div className="page product-page"><PageHeader eyebrow="MANUAL LIVE DESK" title="Follow the match, point by point." description="You record the point winner. Astra keeps the score and statistics from your log. No video stream or external feed is connected." /><Card className="product-panel"><form onSubmit={e => void submit(e)}><MatchMetadataForm value={metadata} onChange={setMetadata} disabled={busy} /><div className="product-inline"><label>Match format<select value={bestOf} disabled={busy} onChange={e => setBestOf(Number(e.target.value))}><option value={3}>Best of three sets</option><option value={5}>Best of five sets</option></select></label><label>Initial server<select required value={server} disabled={busy} onChange={e => setServer(e.target.value)}><option value="">Select the first server</option>{metadata.participants.map((p, i) => <option key={i} value={i + 1}>{p.name || `Player ${i + 1}`}</option>)}</select></label></div><p className="product-hint">Starts at 0–0. Advantage games; seven-point tiebreaks at 6–6 in every set. Log from the beginning for complete match statistics.</p>{error ? <p className="product-alert" role="alert">{error}</p> : null}<button className="button button-primary" disabled={busy || !server}>{busy ? 'Creating…' : 'Start live match'}</button></form></Card></div>;
}

export function ManualLiveDesk({ match, onUpdate }: { match: ProductMatch; onUpdate: (match: ProductMatch) => void }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Record<string, unknown> | null>(null);
  const score = match.score!;
  async function send(body: Record<string, unknown>) {
    setBusy(true); setError('');
    try { onUpdate(await mutate<ProductMatch>(`/api/matches/${match.id}/score-commands`, body)); setPending(null); }
    catch (e) { setError(e instanceof Error ? e.message : 'Score update failed.'); setPending(body); }
    finally { setBusy(false); }
  }
  function command(action: string, winner?: number, point_index?: number) {
    void send({ command_id: crypto.randomUUID(), expected_revision: match.revision, action, ...(winner ? { winner } : {}), ...(point_index !== undefined ? { point_index } : {}) });
  }
  const complete = score.state.match_complete;
  return <div className="product-columns"><Card className="product-panel"><h2>{complete ? 'Match complete' : match.status === 'PAUSED' ? 'Scoring paused' : 'Live scoreboard'}</h2><p className="product-hint">Manually scored · Last update {new Date(match.updated_at).toLocaleString()}</p><div className="product-score"><small>Player</small><small>Sets</small><small>Games</small><small>Points</small></div>{[0, 1].map(i => <div className="product-score" key={i}><strong>{participantName(match, i)}{score.state.server_id === i + 1 && !complete ? ' · serving' : ''}</strong><b>{i ? score.state.sets_won_p2 : score.state.sets_won_p1}</b><b>{i ? score.state.games_p2 : score.state.games_p1}</b><b>{score.display_points[i]}</b></div>)}{score.state.completed_sets.length ? <p>Completed sets: {score.state.completed_sets.map(s => `${s.games_p1}–${s.games_p2}`).join(', ')}</p> : null}{complete ? <p><strong>{participantName(match, (score.state.winner_id ?? 1) - 1)} wins.</strong></p> : null}
    <div className="product-point-actions">{[1, 2].map(winner => <button key={winner} className="button button-primary" disabled={busy || Boolean(pending) || complete || match.status === 'PAUSED'} onClick={() => command('point', winner)}>Point · {participantName(match, winner - 1)}</button>)}</div><div className="product-inline"><button className="button" disabled={busy || Boolean(pending) || !score.history.length} onClick={() => command('undo')}>Undo last point</button>{!complete ? <button className="button" disabled={busy || Boolean(pending)} onClick={() => command(match.status === 'PAUSED' ? 'resume' : 'pause')}>{match.status === 'PAUSED' ? 'Resume scoring' : 'Pause scoring'}</button> : null}</div>
    {error ? <div className="product-alert" role="alert"><p>{error}</p><p>Review the current score before continuing. An uncertain update can be retried safely.</p>{pending ? <button className="button" disabled={busy} onClick={() => void send(pending)}>Retry same update</button> : null}<button className="button" disabled={busy} onClick={async () => { try { const response = await fetch(`/api/matches/${match.id}`); if (!response.ok) throw new Error('Unable to refresh score.'); onUpdate(await response.json()); setPending(null); setError(''); } catch (e) { setError(String(e)); } }}>Refresh score</button></div> : null}
    </Card>
    <Card className="product-panel"><h2>Point history</h2><p className="product-hint">Correcting a winner recalculates the score and server order from the beginning.</p>{!score.history.length ? <p>No points recorded yet.</p> : <ol className="product-history">{score.history.map((point, index) => <li key={index}><div><strong>{index + 1}. {participantName(match, point.winner - 1)}</strong><small className="product-hint"> · {new Date(point.recorded_at).toLocaleTimeString()}</small><p>{point.games.join('–')} games · {point.score.join('–')}</p></div><button className="button button-quiet" disabled={busy || Boolean(pending)} aria-label={`Change winner of point ${index + 1}`} onClick={() => command('correct', 3 - point.winner, index)}>Change winner</button></li>)}</ol>}</Card></div>;
}

export function ManualStatistics({ match }: { match: ProductMatch }) {
 const score = match.score!;
 return <Card className="product-panel">    <h3>Recorded statistics</h3><div className="product-table-wrap"><table className="product-table"><thead><tr><th>Player</th><th>Points won</th><th>Service points</th><th>Return points</th></tr></thead><tbody>{score.statistics.map((s, i) => <tr key={i}><th>{participantName(match, i)}</th><td>{s.points_won}</td><td>{s.service_points_won}/{s.service_points}</td><td>{s.return_points_won}/{s.return_points}</td></tr>)}</tbody></table></div><p className="product-hint">Fractions show points won / recorded opportunities. They do not establish serve type, shot quality or rally length.</p></Card>;
}

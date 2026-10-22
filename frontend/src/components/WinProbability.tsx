import { useId, useState } from 'react';
import { participantName, type ProductMatch } from '../data/productApi';
import { Card } from './ui';

export function WinProbability({ match }: { match: ProductMatch }) {
  const id = useId();
  const [horizon, setHorizon] = useState<'game' | 'set' | 'match'>('match');
  const forecast = match.predictions;
  const values = forecast.probabilities?.[horizon];
  const valid = values && [values.player_1, values.player_2].every(p => Number.isFinite(p) && p >= 0 && p <= 1)
    && Math.abs(values.player_1 + values.player_2 - 1) < .000001;
  if (forecast.status === 'confirmed_result') return <Card className="product-panel"><h2>Confirmed result</h2>
    <p>{forecast.winner_id === 1 || forecast.winner_id === 2 ? `${participantName(match, forecast.winner_id - 1)} wins.` : 'Winner unavailable.'}</p><p>{forecast.reason}</p></Card>;
  const history = (forecast.history ?? []).filter(row => row.status !== 'unavailable' && row.probabilities?.[horizon]
    && Number.isFinite(row.probabilities[horizon].player_1));
  const step = Math.max(1, Math.ceil(history.length / 300));
  const sampled = history.filter((_, index) => index % step === 0 || index === history.length - 1);
  const x = (index: number) => 30 + index / Math.max(1, sampled.length - 1) * 530;
  const path = sampled.map((row, index) => `${index ? 'L' : 'M'}${x(index).toFixed(2)},${(125 - 100 * row.probabilities![horizon].player_1).toFixed(2)}`).join(' ');
  return <Card className="product-panel win-probability"><h2>Win probability</h2>
    <p className="product-hint">{forecast.status === 'prior_only' ? 'Neutral prior + current score' : 'Statistical estimate · Independent calibration pending'}</p>
    {forecast.status === 'unavailable' || !valid ? <p>{forecast.reason || 'Insufficient confirmed point data.'}</p> : <>
      <label>Probability of winning the <select value={horizon} onChange={event => setHorizon(event.target.value as typeof horizon)}><option value="game">Current game</option><option value="set">Current set</option><option value="match">Match</option></select></label>
      <div className="probability-players">{[values.player_1, values.player_2].map((probability, index) => <div key={index}><span>{participantName(match, index)}</span><strong>{(probability * 100).toFixed(1)}%</strong></div>)}</div>
      {sampled.length > 1 ? <figure><svg viewBox="0 0 590 155" role="img" aria-labelledby={id}><title id={id}>{participantName(match, 0)} probability after each recorded point</title>
        <path d="M30 25H560M30 75H560M30 125H560" stroke="currentColor" opacity=".15" fill="none" />
        <text x="0" y="30" fontSize="11">100%</text><text x="0" y="80" fontSize="11">50%</text><text x="0" y="130" fontSize="11">0%</text>
        <path d={path} stroke="var(--society-ink, #1e1e1e)" strokeWidth="2" fill="none" /></svg><figcaption>{participantName(match, 0)} · Ordered point history · {forecast.sample_count ?? 0} confirmed points</figcaption></figure> : null}
      <p>{forecast.reason}</p><p className="product-hint">Uses the score, server and earlier confirmed points. A higher probability does not guarantee a win. This estimate has not passed independent calibration.</p>
    </>}
  </Card>;
}

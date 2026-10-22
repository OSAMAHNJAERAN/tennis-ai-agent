import { useMemo } from 'react';
import type { NativeForecast as Forecast } from '../data/nativeForecast';
import { forecastAtFrame, forecastBins } from '../data/nativeForecast';
import '../styles/correction.css';

export function NativeForecast({ value, frame, onSeek }: { value: Forecast; frame: number; onSeek: (frame: number) => void }) {
  const columns = useMemo(() => forecastBins(value.history), [value.history]);
  const row = forecastAtFrame(value.history, frame); const current = row?.forecast;
  let position = 0; for (let index = 0; index < columns.length; index++) if (columns[index]!.frame_index <= frame) position = index;
  return <section className="native-forecast" aria-label="Native causal forecasts">
    <h2>Score-based forecast</h2><span className="correction-badge">Unqualified · not calibrated</span>
    <p>Uses reviewed points available at the replay time. Independent prediction accuracy has not been validated.</p>
    {!current ? <p role="status">No forecast at this frame.</p> : <>
      <strong>{current.status === 'confirmed_result' ? `Confirmed result · Player ${current.winner_id}` : current.status === 'unavailable' ? 'Forecast withheld' : current.status === 'prior_only' ? 'Neutral prior · no performance points' : `Posterior estimate · ${current.sample_count} completed points`}</strong>
      <p>{current.reason}</p>
      {current.probabilities ? <dl className="forecast-values">{(['game', 'set', 'match'] as const).map(horizon => <div key={horizon}><dt>{horizon}</dt><dd><span>P1 {(100 * current.probabilities![horizon].player_1).toFixed(1)}%</span><span>P2 {(100 * current.probabilities![horizon].player_2).toFixed(1)}%</span></dd></div>)}</dl> : null}
      <small>Frame {row!.frame_index} · {row!.phase.toLowerCase().replaceAll('_', ' ')} · revision {value.analysisRevision}</small>
    </>}
    <div className="forecast-history" role="group" aria-label="Match probability history; gaps are withheld forecasts">
      {columns.map((entry, index) => <button key={index} type="button" className={entry.forecast.probabilities ? 'forecast-column' : 'forecast-gap'}
        tabIndex={-1}
        aria-label={`Seek to frame ${entry.frame_index}: ${entry.forecast.status === 'unavailable' ? 'forecast withheld' : entry.forecast.status === 'confirmed_result' ? 'confirmed result' : `Player 1 match ${(100 * entry.forecast.probabilities!.match.player_1).toFixed(1)} percent`}`}
        title={`Frame ${entry.frame_index} · ${entry.forecast.status}`} onClick={() => onSeek(entry.frame_index)}>
        {entry.forecast.probabilities ? <span style={{ height: `${100 * entry.forecast.probabilities.match.player_1}%` }} /> : <span>·</span>}
      </button>)}
    </div><label className="forecast-seek">Forecast history position<input type="range" min={0} max={Math.max(0, columns.length - 1)} value={position} aria-valuetext={`Frame ${columns[position]!.frame_index}`} onChange={event => onSeek(columns[Number(event.target.value)]!.frame_index)} /></label><small>Choose a history column or use the slider to seek the shared replay. Gaps retain unavailable prefixes.</small>
  </section>;
}

import { useMemo, type CSSProperties, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts';
import { useAnalysis } from '../app/AnalysisContext';
import { SpatialHeatmapPanel } from '../components/spatial/SpatialHeatmapPanel';
import { PointTimelineEvidence } from '../components/PointTimelineEvidence';
import { DataError, SkeletonPage } from '../components/ui';
import { trackingDistribution } from '../lib/analytics';
import type { AnalysisRun, TrackingState } from '../data/contracts';
import '../styles/society-analytics.css';

const PLAYER_COLORS = ['#350aff', '#9b482b'] as const;
const TRACKING_COLORS: Record<TrackingState, string> = {
  DETECTED: '#bcf015',
  TRACKED: '#350aff',
  PREDICTED: '#8674cc',
  INTERPOLATED: '#a8a7a3',
  OCCLUDED: '#9b482b',
  MISSING: '#1e1e1e',
};

function elapsedTime(seconds: number, precise = false) {
  const rounded = precise ? Math.round(seconds * 100) / 100 : Math.floor(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded - minutes * 60;
  return `${minutes}:${precise ? remainder.toFixed(2).padStart(5, '0') : String(remainder).padStart(2, '0')}`;
}

function SpeedTooltip({ active, label, payload }: Pick<TooltipContentProps, 'active' | 'label' | 'payload'>) {
  if (!active || !payload.length) return null;
  return <div className="analytics-tooltip" role="status" aria-live="polite" aria-atomic="true">
    <p>{typeof label === 'number' ? elapsedTime(label, true) : label}<span>elapsed video</span></p>
    {[1, 2].map(id => {
      const value = payload.find(item => item.dataKey === `player${id}`)?.value;
      return <div className={`analytics-tooltip-player analytics-player-${id}`} key={id}>
        <span><i aria-hidden="true" />Player {id}</span>
        <strong>{typeof value === 'number' ? <>{value.toFixed(1)} <small>km/h</small></> : <small>No observation</small>}</strong>
      </div>;
    })}
  </div>;
}

export function LiveRunView({ children, unavailable = false }: { children: ReactNode; unavailable?: boolean }) {
  const { run } = useAnalysis();
  if (!run?.summary.id.startsWith('match_')) return children;
  if (unavailable) return <div className="page society-analytics analytics-unavailable"><h1>Analysis unavailable for this section</h1><p>This video analysis does not establish player names, tournament records, rankings or scouting predictions.</p><Link className="analytics-replay-link" to={`/analysis/${run.summary.id}`}>Return to your match analysis <span aria-hidden="true">↗</span></Link></div>;
  return <LiveRunAnalytics run={run} />;
}

function LiveRunAnalytics({ run }: { run: AnalysisRun }) {
  const location = useLocation();
  const replayParams = new URLSearchParams(location.search);
  replayParams.set('section', 'replay');
  const replayHref = location.pathname.startsWith('/matches/') || location.pathname.startsWith('/examples/') ? `${location.pathname}?${replayParams}` : `/analysis/${run.summary.id}`;
  const data = useMemo(() => {
    const stride = Math.max(1, Math.ceil(run.summary.frames / 600));
    const series = [];
    for (let index = 0; index < run.summary.frames; index += stride) {
      const values = [run.trajectories?.player1Speeds, run.trajectories?.player2Speeds].map(samples => {
        const interval = samples?.slice(index, index + stride);
        return interval?.length && interval.every(value => value !== null) ? interval.reduce((sum, value) => sum + value!, 0) / interval.length : null;
      });
      series.push({ seconds: Number((index / run.summary.fps).toFixed(2)), player1: values[0], player2: values[1] });
    }
    return series;
  }, [run]);
  const distribution = trackingDistribution(run.trajectories?.ballTrajectory ?? []);
  const trackedFrames = distribution.reduce((total, item) => total + item.count, 0);
  const hasSpeed = data.some(value => value.player1 != null || value.player2 != null);

  return <div className="page society-analytics">
    <header className="analytics-heading">
      <span className="analytics-kicker">Recorded match analytics</span>
      <h1>The movement<br /><em>record.</em></h1>
      <div className="analytics-heading-bottom">
        <p>Evidence from {(run.summary.durationSeconds / 60).toFixed(2)} minutes of video. Player IDs identify near/far court roles, separate from the match participants. Confirm whole-clip identities in Details before using named movement history.</p>
        <Link className="analytics-replay-link" to={replayHref}>Open synchronized replay <span aria-hidden="true">↗</span></Link>
      </div>
    </header>

    <section className="analytics-section analytics-comparison" aria-labelledby="analytics-players-title">
      <div className="analytics-section-heading"><h2 id="analytics-players-title">Two players.<br /><em>Every interval.</em></h2><p>Speed and distance estimates use observed, contiguous ground-position intervals.</p></div>
      <div className="analytics-player-rows">{[1, 2].map(id => {
        const player = id === 1 ? run.playerMetrics?.player_1 : run.playerMetrics?.player_2;
        const metrics = [
          { label: 'Distance covered', value: player?.total_distance_m, unit: 'm' },
          { label: 'Average speed', value: player?.avg_speed_kmh, unit: 'km/h' },
          { label: 'Maximum speed', value: player?.max_speed_kmh, unit: 'km/h' },
          { label: 'Mapped frame coverage', value: player?.coverage_pct, unit: '%' },
        ];
        return <article className={`analytics-player-row analytics-player-${id}`} key={id}>
          <div className="analytics-player-identity"><i aria-hidden="true">{id === 1 ? '●' : '◆'}</i><h3>Player {id}</h3><span>Court role ID</span></div>
          <dl className="analytics-player-metrics">{metrics.map(metric => <div key={metric.label}><dt>{metric.label}</dt><dd>{metric.value?.toFixed(1) ?? '—'}<small>{metric.unit}</small></dd></div>)}</dl>
          {!player ? <p className="analytics-player-unavailable">Analysis unavailable for this section.</p> : null}
        </article>;
      })}</div>
    </section>

    <section className="analytics-section analytics-speed" aria-labelledby="analytics-speed-title">
      <div className="analytics-section-heading"><div><span className="analytics-kicker">Observed movement</span><h2 id="analytics-speed-title">The pace of play.</h2></div><p>Speed estimates in km/h. Missing intervals remain gaps; longer sessions show interval averages.</p></div>
      <div className="analytics-chart-topline"><span>Speed <small>/ km/h</small></span><div className="analytics-chart-key" aria-label="Speed chart legend"><span className="analytics-player-1"><i aria-hidden="true" />Player 1</span><span className="analytics-player-2"><i aria-hidden="true" />Player 2</span></div></div>
      {hasSpeed ? <figure className="analytics-speed-figure">
        <figcaption className="sr-only">Observed player speed over elapsed video time. Player 1 is a solid blue line; Player 2 is a dashed orange line. Gaps indicate unavailable observations.</figcaption>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} accessibilityLayer margin={{ top: 18, right: 12, bottom: 10, left: 0 }}>
            <CartesianGrid vertical={false} stroke="#1e1e1e" strokeOpacity={0.12} />
            <XAxis dataKey="seconds" type="number" domain={['dataMin', 'dataMax']} tickFormatter={value => elapsedTime(Number(value))} tickLine={false} axisLine={{ stroke: '#1e1e1e', strokeOpacity: 0.2 }} minTickGap={48} tickMargin={14} tick={{ fill: '#66645f', fontSize: 11, fontFamily: 'Geist Mono Variable, monospace' }} />
            <YAxis tickLine={false} axisLine={false} width={44} tickMargin={12} tick={{ fill: '#66645f', fontSize: 11, fontFamily: 'Geist Mono Variable, monospace' }} />
            <Tooltip filterNull={false} content={props => <SpeedTooltip {...props} />} cursor={{ stroke: '#1e1e1e', strokeOpacity: 0.45, strokeDasharray: '2 4' }} isAnimationActive={false} />
            <Line dataKey="player1" name="Player 1" stroke={PLAYER_COLORS[0]} strokeWidth={2.25} dot={false} activeDot={{ r: 4, stroke: '#efeeec', strokeWidth: 2 }} connectNulls={false} isAnimationActive={false} />
            <Line dataKey="player2" name="Player 2" stroke={PLAYER_COLORS[1]} strokeWidth={2.25} strokeDasharray="7 5" dot={false} activeDot={{ r: 4, stroke: '#efeeec', strokeWidth: 2 }} connectNulls={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="analytics-chart-footnote"><span>Elapsed video <small>/ min:sec</small></span><span>Whole-clip observations</span></div>
      </figure> : <p className="analytics-empty" role="status">Movement speed analysis is unavailable for this section.</p>}
    </section>

    <section className="analytics-section analytics-tracking" aria-labelledby="analytics-tracking-title">
      <div className="analytics-section-heading"><div><span className="analytics-kicker">Ball tracking confidence and availability</span><h2 id="analytics-tracking-title">The visible evidence.</h2></div><p>Tracking coverage describes available observations, not detection accuracy. Per-frame confidence is retained in the analysis export and replay evidence.</p></div>
      {trackedFrames ? <>
        <div className="analytics-tracking-total"><strong>{trackedFrames.toLocaleString()}</strong><span>ball trajectory frames<br />represented in this analysis</span></div>
        <div className="analytics-tracking-bar" aria-hidden="true">{distribution.filter(value => value.count > 0).map(value => <span key={value.state} style={{ width: `${value.count / trackedFrames * 100}%`, backgroundColor: TRACKING_COLORS[value.state] }} />)}</div>
        <dl className="analytics-tracking-states">{distribution.map(value => <div key={value.state} style={{ '--tracking-color': TRACKING_COLORS[value.state] } as CSSProperties}><dt><i aria-hidden="true" />{value.state.toLowerCase()}</dt><dd>{value.count.toLocaleString()}<small>frames</small></dd></div>)}</dl>
      </> : <p className="analytics-empty" role="status">Ball tracking analysis is unavailable for this section.</p>}
      <p className="analytics-availability-note">Qualified ball speed and bounce placement are unavailable. Any point and shot records below remain provisional evidence for review.</p>
    </section>

    <PointTimelineEvidence analysis={run.pointAnalysis} />

    <section className="analytics-section analytics-positioning" aria-labelledby="analytics-position-title">
      <div className="analytics-section-heading"><div><span className="analytics-kicker">Player positioning and heatmaps</span><h2 id="analytics-position-title">Where the play lives.</h2></div><p>Explore mapped positions across the full analysis. The same density scale applies to both players.</p></div>
      <SpatialHeatmapPanel run={run} range={[0, Math.max(0, run.summary.frames - 1)]} reducedMotion />
    </section>
    {run.warnings?.length ? <details className="analytics-notices"><summary>Measurement availability <span aria-hidden="true">+</span></summary><ul>{run.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></details> : null}
  </div>;
}

export default function SessionAnalytics() {
  const { run, loading, error, retry } = useAnalysis();
  if (loading) return <SkeletonPage />;
  if (!run || error) return <DataError message={error?.message ?? 'Analysis unavailable.'} onRetry={retry} />;
  return <LiveRunAnalytics run={run} />;
}

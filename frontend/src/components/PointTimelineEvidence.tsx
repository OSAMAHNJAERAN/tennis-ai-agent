import { useState } from 'react';
import type { PointAnalysis } from '../data/contracts';

const POINTS_PER_PAGE = 20;
const EXECUTIONS_PER_POINT = 10;

function seconds(value: number | null) {
  return value == null ? 'Unavailable' : `${value.toFixed(2)} s`;
}

function participant(value: 1 | 2 | null) {
  return value == null ? 'Not established' : `Participant ${value === 1 ? 'A' : 'B'}`;
}

function pointStatus(point: PointAnalysis['points'][number]) {
  if (point.isReviewPending || point.status === 'REVIEW_REQUIRED') return 'Human review needed';
  if (point.status === 'PARTIAL_UNANCHORED') return 'Partial point · start not established';
  if (point.status === 'COMPLETED') return 'Recorded completed point';
  if (point.status === 'OPEN') return 'Point incomplete';
  return 'Point status unresolved';
}

function serveStatus(status: string) {
  switch (status) {
    case 'SERVICE_LET': return 'Service let';
    case 'FIRST_SERVE_FAULT': return 'First fault';
    case 'DOUBLE_FAULT': return 'Double fault';
    case 'SERVE_IN_PLAY': return 'Serve in play';
    case 'REVIEW_REQUIRED': return 'Human review needed';
    default: return 'Outcome unresolved';
  }
}

function PointRecords({ analysis }: { analysis: PointAnalysis }) {
  const [pageIndex, setPageIndex] = useState(0);
  const lastPage = Math.max(0, Math.ceil(analysis.points.length / POINTS_PER_PAGE) - 1);
  const page = Math.min(pageIndex, lastPage);
  const first = page * POINTS_PER_PAGE;
  const points = analysis.points.slice(first, first + POINTS_PER_PAGE);
  // Keep rendered rows bounded for long matches, including their executions.
  const visibleIds = new Set(points.map(point => point.id));
  const attempts = new Map<string, { rows: PointAnalysis['serveExecutions']; total: number }>();
  for (const attempt of analysis.serveExecutions) {
    if (!visibleIds.has(attempt.pointId)) continue;
    const preview = attempts.get(attempt.pointId) ?? { rows: [], total: 0 };
    preview.total += 1;
    if (preview.rows.length < EXECUTIONS_PER_POINT) preview.rows.push(attempt);
    attempts.set(attempt.pointId, preview);
  }

  if (!points.length) return <p className="analytics-availability-note">No point records are available. This does not establish that no points were played.</p>;

  return <details className="analytics-notices">
    <summary>Point-by-point evidence <span aria-hidden="true">+</span></summary>
    <p>Outcomes remain provisional. Point-specific replay is unavailable until its video timing is verified.</p>
    <ol start={first + 1} aria-label="Point evidence">
      {points.map((point, index) => <li key={point.id}>
        <details>
          <summary>Point {first + index + 1} · {pointStatus(point)}</summary>
          <dl>
            <div><dt>Recorded server</dt><dd>{participant(point.serverId)}</dd></div>
            <div><dt>Recorded winner</dt><dd>{participant(point.winnerId)}</dd></div>
            <div><dt>Legal rally strokes including serve</dt><dd>{point.totalStrokesIncludingServe}</dd></div>
            <div><dt>Legal rally hits excluding serve</dt><dd>{point.rallyHitsExcludingServe}</dd></div>
            <div><dt>Rally playing time</dt><dd>{seconds(point.durationSeconds)}</dd></div>
          </dl>
          {point.endingReason ? <p>Recorded ending reason: {point.endingReason}</p> : <p>The reason the point ended is not established.</p>}
          {attempts.get(point.id)?.total ? <><ul aria-label={`Serve executions for point ${first + index + 1}`}>
            {attempts.get(point.id)!.rows.map(attempt => <li key={attempt.id}>
              {attempt.officialAttempt === 1 ? 'First' : 'Second'} serve{attempt.replayIndex ? ` · replay ${attempt.replayIndex}` : ''}: {serveStatus(attempt.status)}
            </li>)}
          </ul>{attempts.get(point.id)!.total > EXECUTIONS_PER_POINT ? <p>Showing the first {EXECUTIONS_PER_POINT} of {attempts.get(point.id)!.total} serve executions. Additional executions are retained in the analysis data.</p> : null}</> : <p>Serve execution evidence is unavailable.</p>}
        </details>
      </li>)}
    </ol>
    {lastPage > 0 ? <nav aria-label="Point evidence pages">
      <button className="button" type="button" disabled={page === 0} onClick={() => setPageIndex(page - 1)}>Previous points</button>
      <span aria-live="polite"> {first + 1}–{first + points.length} of {analysis.points.length} points </span>
      <button className="button" type="button" disabled={page === lastPage} onClick={() => setPageIndex(page + 1)}>Next points</button>
    </nav> : null}
  </details>;
}

export function PointTimelineEvidence({ analysis }: { analysis: PointAnalysis | null | undefined }) {
  if (!analysis) return null;
  const { summary } = analysis;
  return <section className="analytics-section" aria-labelledby="point-analysis-title">
    <div className="analytics-section-heading">
      <div><span className="analytics-kicker">Provisional point and serve evidence</span><h2 id="point-analysis-title">Points in context.</h2></div>
      <p>Independent accuracy has not been validated. These records support review; they do not establish a confirmed score or result.</p>
    </div>
    {!analysis.authoritativeEventsEnabled ? <p className="analytics-availability-note" role="status">Point scoring is unavailable because the required event evidence is not established.</p> : null}
    <dl className="analytics-player-metrics">
      <div><dt>Recorded completed points</dt><dd>{analysis.authoritativeEventsEnabled ? summary.totalPointsPlayed ?? '—' : '—'}</dd></div>
      <div><dt>Partial or review records</dt><dd>{summary.partialOrReviewPoints}</dd></div>
      <div><dt>Completed legal rallies</dt><dd>{analysis.authoritativeEventsEnabled ? summary.totalRallies : '—'}</dd></div>
      <div><dt>Known rally playing time</dt><dd>{analysis.authoritativeEventsEnabled ? summary.effectivePlaySeconds?.toFixed(2) ?? '—' : '—'}<small>s</small></dd></div>
    </dl>
    <p className="analytics-availability-note">{analysis.authoritativeEventsEnabled ? <>Timing coverage: {summary.timedCompletedRallies} of {summary.completedRalliesDurationDenominator} completed rallies. </> : 'Rally timing coverage is unavailable. '}Playing time includes only rallies with supported source timing; it is not the full video's playing-time percentage. A dash means unavailable.</p>
    {analysis.authoritativeEventsEnabled ? <p className="analytics-availability-note">Completed rally length including the successful serve: average {summary.averageRallyLength?.toFixed(1) ?? '—'} strokes; longest {summary.longestRallyLength ?? '—'} strokes.</p> : null}
    {analysis.authoritativeEventsEnabled ? <div className="analytics-player-rows">{([1, 2] as const).map(id => {
      const serve = id === 1 ? analysis.serveByPlayer.player_1 : analysis.serveByPlayer.player_2;
      return <article className={`analytics-player-row analytics-player-${id}`} key={id} aria-label={`Participant ${id === 1 ? 'A' : 'B'} serve evidence`}>
        <div className="analytics-player-identity"><i aria-hidden="true">{id === 1 ? 'A' : 'B'}</i><h3>Participant {id === 1 ? 'A' : 'B'}</h3><span>Match participant</span></div>
        <dl className="analytics-player-metrics">
          <div><dt>First serve attempts</dt><dd>{serve.firstServesAttempted}</dd></div>
          <div><dt>Second serve attempts</dt><dd>{serve.secondServesAttempted}</dd></div>
          <div><dt>Double faults</dt><dd>{serve.doubleFaults}</dd></div>
          <div><dt>Resolved first serves in</dt><dd>{serve.firstServePercentage?.toFixed(1) ?? '—'}<small>%</small></dd></div>
        </dl>
        <p className="analytics-player-unavailable">First-serve percentage: {serve.firstServesIn} in / {serve.firstServePercentageDenominator} resolved first attempts. {serve.serviceLets} service lets are excluded; {serve.unresolvedExecutions} executions remain unresolved.</p>
      </article>;
    })}</div> : <p className="analytics-availability-note">Serve statistics are unavailable. Empty event records do not establish zero serve attempts or double faults.</p>}
    <PointRecords key={`${analysis.analysisId}:${analysis.analysisRevision}:${analysis.scoringRevision}`} analysis={analysis} />
  </section>;
}

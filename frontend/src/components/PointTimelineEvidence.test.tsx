import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AnalysisRunSchema, PointAnalysisSchema, type PointAnalysis } from '../data/contracts';
import { parseLiveRun, snapshot } from '../data/repository';
import { PointTimelineEvidence } from './PointTimelineEvidence';

// Synthetic contract fixture: this exercises presentation, not event accuracy.
function evidence(): PointAnalysis {
  const serve = {
    serverId: 1 as const, contactExecutions: 4, serviceLets: 1, firstServesAttempted: 2,
    secondServesAttempted: 1, firstServesIn: 1, firstServesFaulted: 1, secondServesIn: 0,
    doubleFaults: 1, unresolvedExecutions: 0, firstServePercentage: 50,
    firstServePercentageDenominator: 2,
  };
  return {
    schemaVersion: '1.0', status: 'UNQUALIFIED', analysisId: snapshot.defaultRunId,
    analysisRevision: 3, scoringRevision: 2, sourceSha256: 'a'.repeat(64),
    independentAccuracyValidated: false, authoritativeEventsEnabled: true,
    summary: { totalPointsPlayed: 3, partialOrReviewPoints: 1, totalRallies: 2,
      averageRallyLength: 3, longestRallyLength: 3, effectivePlaySeconds: .97,
      timedCompletedRallies: 1, completedRalliesDurationDenominator: 2,
      scoreSummary: null, matchComplete: null, winnerId: null },
    serveByPlayer: { player_1: serve, player_2: { ...serve, serverId: 2,
      contactExecutions: 0, serviceLets: 0, firstServesAttempted: 0, secondServesAttempted: 0,
      firstServesIn: 0, firstServesFaulted: 0, doubleFaults: 0,
      firstServePercentage: null, firstServePercentageDenominator: 0 } },
    points: [
      { id: 'point-rally', status: 'COMPLETED', anchorEventId: 1, terminalEventId: 4,
        serverId: 1, receiverId: 2, setNumber: 1, gameNumber: 1, serviceSide: 'DEUCE', winnerId: 1,
        endingReason: 'The return landed out.', isReviewPending: false,
        totalStrokesIncludingServe: 3, rallyHitsExcludingServe: 2, durationSeconds: .97 },
      { id: 'point-double-fault', status: 'COMPLETED', anchorEventId: 5, terminalEventId: 8,
        serverId: 1, receiverId: 2, setNumber: 1, gameNumber: 1, serviceSide: 'AD', winnerId: 2,
        endingReason: 'Double fault.', isReviewPending: false,
        totalStrokesIncludingServe: 0, rallyHitsExcludingServe: 0, durationSeconds: null },
      { id: 'point-partial', status: 'PARTIAL_UNANCHORED', anchorEventId: null, terminalEventId: null,
        serverId: null, receiverId: null, setNumber: null, gameNumber: null, serviceSide: null,
        winnerId: null, endingReason: null, isReviewPending: true,
        totalStrokesIncludingServe: 0, rallyHitsExcludingServe: 0, durationSeconds: null },
      { id: 'point-untimed', status: 'COMPLETED', anchorEventId: 20, terminalEventId: 24,
        serverId: 2, receiverId: 1, setNumber: 1, gameNumber: 2, serviceSide: 'DEUCE', winnerId: 1,
        endingReason: 'The return landed out.', isReviewPending: false,
        totalStrokesIncludingServe: 3, rallyHitsExcludingServe: 2, durationSeconds: null },
    ],
    rallies: [{ id: 'rally-1', pointId: 'point-rally', status: 'COMPLETED', serverId: 1,
      receiverId: 2, serveAttempt: 1, winnerId: 1, endingReason: 'The return landed out.',
      startFrame: 100, terminalEventId: 4, durationSeconds: .97, durationReason: null,
      totalStrokesIncludingServe: 3, rallyHitsExcludingServe: 2 },
      { id: 'rally-2', pointId: 'point-untimed', status: 'COMPLETED', serverId: 2,
        receiverId: 1, serveAttempt: 1, winnerId: 1, endingReason: 'The return landed out.',
        startFrame: 200, terminalEventId: 24, durationSeconds: null,
        durationReason: 'SOURCE_TIMING_UNAVAILABLE', totalStrokesIncludingServe: 3, rallyHitsExcludingServe: 2 }],
    serveExecutions: [
      { id: 'attempt-1', pointId: 'point-double-fault', status: 'FIRST_SERVE_FAULT', officialAttempt: 1,
        replayIndex: 0, contactEventId: 5, resultEventId: 6, serverId: 1, serverEnd: 'FAR',
        serviceSide: 'AD', rallyId: null },
      { id: 'attempt-2', pointId: 'point-double-fault', status: 'SERVICE_LET', officialAttempt: 2,
        replayIndex: 0, contactEventId: 7, resultEventId: 8, serverId: 1, serverEnd: 'FAR',
        serviceSide: 'AD', rallyId: null },
      { id: 'attempt-3', pointId: 'point-double-fault', status: 'DOUBLE_FAULT', officialAttempt: 2,
        replayIndex: 1, contactEventId: 9, resultEventId: 10, serverId: 1, serverEnd: 'FAR',
        serviceSide: 'AD', rallyId: null },
    ],
  };
}

describe('provisional native point evidence', () => {
  it('preserves optional absence and explicit null for older analyses', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    expect(AnalysisRunSchema.parse(run).pointAnalysis).toBeUndefined();
    expect(parseLiveRun({ ...run, pointAnalysis: null }).pointAnalysis).toBeNull();
    const { container } = render(<PointTimelineEvidence analysis={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('retains nullable timing, separate serve executions, and the unqualified status', () => {
    const value = PointAnalysisSchema.parse(evidence());
    expect(value.points[1]!.durationSeconds).toBeNull();
    expect(value.points[1]!.rallyHitsExcludingServe).toBe(0);
    expect(value.points[1]!.totalStrokesIncludingServe).toBe(0);
    expect(value.serveExecutions).toHaveLength(3);
    expect(value.serveByPlayer.player_2.firstServePercentage).toBeNull();
    expect(value.independentAccuracyValidated).toBe(false);
  });

  it.each([
    { status: 'QUALIFIED' }, { independentAccuracyValidated: true },
    { summary: { ...evidence().summary, effectivePlaySeconds: -1 } },
    { sourceSha256: 'unbound' },
  ])('isolates invalid point evidence without hiding replay: %j', patch => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    const parsed = parseLiveRun({ ...run, pointAnalysis: { ...evidence(), ...patch } });
    expect(parsed.pointAnalysis).toBeNull();
    expect(parsed.videoUrl).toBe(run.videoUrl);
    expect(parsed.warnings).toContain('Point and serve evidence is unavailable for this analysis.');
  });

  it('rejects foreign or stale point evidence while preserving other sections', () => {
    const run = snapshot.details[snapshot.defaultRunId]!;
    for (const patch of [{ analysisId: 'another-run' }, { analysisRevision: 99 }]) {
      const parsed = parseLiveRun({ ...run, analysisRevision: 3, pointAnalysis: { ...evidence(), ...patch } });
      expect(parsed.pointAnalysis).toBeNull();
      expect(parsed.detections).toEqual(run.detections);
    }
  });

  it('shows timing and percentage denominators without claiming validated accuracy', () => {
    render(<PointTimelineEvidence analysis={evidence()} />);
    expect(screen.getByText('Independent accuracy has not been validated.', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Timing coverage: 1 of 2 completed rallies.', { exact: false })).toHaveTextContent("not the full video's playing-time percentage");
    const participantA = screen.getByRole('article', { name: 'Participant A serve evidence' });
    expect(participantA).toHaveTextContent('First-serve percentage: 1 in / 2 resolved first attempts. 1 service lets are excluded');
    const participantB = screen.getByRole('article', { name: 'Participant B serve evidence' });
    expect(participantB).toHaveTextContent('0 in / 0 resolved first attempts');
    expect(within(participantB).getByText('—')).toBeInTheDocument();
  });

  it('keeps faults and lets separate from legal rally timing and does not invent a seek', () => {
    render(<PointTimelineEvidence analysis={evidence()} />);
    expect(screen.getByText('First serve: First fault')).toBeInTheDocument();
    expect(screen.getByText('Second serve: Service let')).toBeInTheDocument();
    expect(screen.getByText('Second serve · replay 1: Double fault')).toBeInTheDocument();
    expect(screen.getByText('Point 3 · Human review needed')).toBeInTheDocument();
    expect(screen.getAllByText('Unavailable')).toHaveLength(3);
    expect(screen.getByText('Point-specific replay is unavailable', { exact: false })).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /seek|replay/i })).not.toBeInTheDocument();
  });

  it('does not turn disabled authority and empty evidence into known zero points', () => {
    const analysis = evidence();
    analysis.authoritativeEventsEnabled = false;
    analysis.summary.totalPointsPlayed = null;
    analysis.summary.effectivePlaySeconds = null;
    analysis.points = [];
    render(<PointTimelineEvidence analysis={analysis} />);
    expect(screen.getByRole('status')).toHaveTextContent('Point scoring is unavailable');
    expect(screen.getByText('No point records are available.', { exact: false })).toHaveTextContent('does not establish that no points were played');
    const label = screen.getByText('Recorded completed points');
    expect(label.parentElement).toHaveTextContent('—');
    expect(screen.getByText('Completed legal rallies').parentElement).toHaveTextContent('—');
    expect(screen.getByText('Serve statistics are unavailable.', { exact: false })).toHaveTextContent('do not establish zero serve attempts or double faults');
    expect(screen.queryByRole('article', { name: 'Participant A serve evidence' })).not.toBeInTheDocument();
  });

  it('bounds the rendered timeline and pages every record in source order', () => {
    const analysis = evidence();
    analysis.points = Array.from({ length: 45 }, (_, index) => ({ ...analysis.points[0]!, id: `point-${index}` }));
    render(<PointTimelineEvidence analysis={analysis} />);
    const list = screen.getByRole('list', { name: 'Point evidence' });
    expect(list.children).toHaveLength(20);
    expect(screen.getByRole('button', { name: 'Previous points' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Next points' }));
    expect(screen.getByText('21–40 of 45 points')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Next points' }));
    expect(list.children).toHaveLength(5);
    expect(screen.getByText('Point 45 · Recorded completed point')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next points' })).toBeDisabled();
  });

  it('bounds a point execution preview even when many service lets are retained', () => {
    const analysis = evidence();
    analysis.serveExecutions = Array.from({ length: 1000 }, (_, index) => ({
      ...analysis.serveExecutions[1]!, id: `let-${index}`, replayIndex: index,
    }));
    render(<PointTimelineEvidence analysis={analysis} />);
    const list = screen.getByRole('list', { name: 'Serve executions for point 2' });
    expect(list.children).toHaveLength(10);
    expect(screen.getByText('Showing the first 10 of 1000 serve executions.', { exact: false })).toBeInTheDocument();
  });
});

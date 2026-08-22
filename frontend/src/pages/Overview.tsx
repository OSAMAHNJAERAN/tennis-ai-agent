import { ArrowRight, CheckCircle, ClockCounterClockwise, Pulse, TennisBall } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { useAnalysis } from "../app/AnalysisContext";
import { CourtDiagram } from "../components/CourtDiagram";
import { DecisionBadge } from "../components/DecisionBadge";
import { Card, EmptyState, Metric, PageHeader, SectionHeading, StatusBadge, UtilityCard } from "../components/ui";
import { deriveCoverage } from "../lib/analytics";
import { formatConfidence, formatDuration, formatNumber, formatPercent, humanize } from "../lib/format";

function scoreValue(player: Record<string, unknown> | undefined) {
  return typeof player?.points_display === "string" ? player.points_display : "0";
}

export default function Overview() {
  const { run, selectedEventId, selectEvent } = useAnalysis();
  if (!run) return null;
  const state = run.matchState;
  const coverage = run.trajectories ? deriveCoverage(run.trajectories.ballTrajectory) : null;
  const playerMetrics = run.playerMetrics;
  const recentEvents = run.events.slice(-4).reverse();

  return (
    <div className="page overview-page">
      <PageHeader
        eyebrow="Match intelligence / live artifact"
        title="A seven-second rally, examined from every angle."
        description="A high-density view of the selected pipeline run, grounded in the committed offline snapshot. Every metric traces back to an existing analysis artifact."
        action={<Link className="button button-primary" to={`/analysis?run=${run.summary.id}`}><TennisBall size={17} weight="fill" /> Open replay</Link>}
      />

      <section className="overview-command-grid" aria-label="Match summary">
        <Card className="score-command-card">
          <div className="score-card-top">
            <div><p className="eyebrow">Current match state</p><h2>{state ? humanize(state.point_state) : "Scoring unavailable"}</h2></div>
            <StatusBadge tone="signal">{state ? `Set ${state.current_set} · Game ${state.current_game}` : run.summary.phase}</StatusBadge>
          </div>
          <div className="scoreboard">
            <div className="score-player"><span>Player 1 {state?.server_id === 1 ? "· Serving" : ""}</span><strong>{scoreValue(state?.player_1)}</strong><small>{playerMetrics ? `${formatNumber(playerMetrics.player_1.total_distance_m, " m")} covered` : "Movement unavailable"}</small></div>
            <span className="score-divider">:</span>
            <div className="score-player"><span>Player 2 {state?.server_id === 2 ? "· Serving" : ""}</span><strong>{scoreValue(state?.player_2)}</strong><small>{playerMetrics ? `${formatNumber(playerMetrics.player_2.total_distance_m, " m")} covered` : "Movement unavailable"}</small></div>
          </div>
          <div className="score-context">
            <span><strong>{state ? humanize(state.service_side) : "Unavailable"}</strong><small>Service side</small></span>
            <span><strong>{state?.serve_attempt ?? "Unavailable"}</strong><small>Serve attempt</small></span>
            <span><strong>{run.events.length}</strong><small>Detected events</small></span>
          </div>
        </Card>

        <Card className="overview-court-card">
          <div className="court-card-head"><span>Canonical evidence map</span><StatusBadge>{run.court?.isValid ? "Court valid" : "Invalid court"}</StatusBadge></div>
          <CourtDiagram run={run} selectedEventId={selectedEventId} compact />
        </Card>

        <div className="overview-kpi-stack">
          <UtilityCard className="overview-kpi"><Metric primary label="Pipeline throughput" value={formatNumber(run.pipelineMetrics?.pipelineFps, " FPS")} meta={`${run.summary.frames} frames processed`} /></UtilityCard>
          <UtilityCard className="overview-kpi"><Metric label="Ball coverage" value={formatPercent(coverage)} meta="Observed or reconstructed frames" /></UtilityCard>
          <UtilityCard className="overview-kpi"><Metric label="Court reprojection" value={formatNumber(run.court?.reprojectionErrorPx, " px")} meta={run.court?.isValid ? "Geometry marked valid" : "Validation unavailable"} /></UtilityCard>
        </div>
      </section>

      <section className="overview-lower-grid">
        <Card className="panel-padding events-panel">
          <SectionHeading eyebrow="Semantic timeline" title="Recent events" detail="Select an event to carry its evidence across views." action={<Link className="text-link" to={`/events?run=${run.summary.id}`}>All events <ArrowRight size={14} /></Link>} />
          {recentEvents.length ? <div className="event-rail">{recentEvents.map((event) => (
            <button className={`event-rail-item ${selectedEventId === event.event_id ? "event-rail-selected" : ""}`} key={event.event_id} type="button" onClick={() => selectEvent(event.event_id)}>
              <span className="event-symbol"><Pulse size={16} weight="duotone" /></span>
              <span><strong>{humanize(event.event_type)}</strong><small>Frame {event.frame} · {event.timestamp_s.toFixed(2)} s</small></span>
              <span>{formatConfidence(event.confidence)}</span>
            </button>
          ))}</div> : <EmptyState title="No event artifact" detail="This pipeline generation does not include semantic event detection." />}
        </Card>

        <Card className="panel-padding comparison-panel">
          <SectionHeading eyebrow="Movement" title="Player comparison" detail="Court-plane estimates from player_metrics.json." />
          {playerMetrics ? <div className="player-comparison">
            {[playerMetrics.player_1, playerMetrics.player_2].map((player, index) => (
              <div className="player-comparison-row" key={index}>
                <div><strong>Player {index + 1}</strong><span>{formatNumber(player.avg_speed_kmh, " km/h avg")}</span></div>
                <div className="comparison-track"><span style={{ width: `${Math.min(100, player.total_distance_m / 25 * 100)}%` }} /></div>
                <strong>{formatNumber(player.total_distance_m, " m")}</strong>
              </div>
            ))}
          </div> : <EmptyState title="Player metrics unavailable" detail="Select a run with a player metrics artifact." />}
        </Card>

        <Card className="panel-padding call-summary-panel">
          <SectionHeading eyebrow="Decisions" title="Line-call summary" detail={`${run.lineCalls.length} bounce decisions in this run.`} />
          {run.lineCalls.length ? <div className="line-summary-list">{run.lineCalls.map((call) => (
            <Link to={`/line-calls?run=${run.summary.id}&event=${call.event_id}`} className="line-summary-item" key={call.event_id}>
              <DecisionBadge decision={call.decision} />
              <span><strong>Frame {call.bounce_frame}</strong><small>{formatNumber(call.ball_edge_margin_cm, " cm margin")}</small></span>
              <ArrowRight size={15} aria-hidden="true" />
            </Link>
          ))}</div> : <div className="no-review-state"><CheckCircle size={22} weight="duotone" /><span><strong>No line-call artifact</strong><small>This generation predates line calling.</small></span></div>}
        </Card>
      </section>

      <footer className="run-footnote">
        <ClockCounterClockwise size={16} aria-hidden="true" />
        <span><strong>{run.summary.label}</strong> · {formatDuration(run.summary.durationSeconds)} source · Static snapshot</span>
      </footer>
    </div>
  );
}


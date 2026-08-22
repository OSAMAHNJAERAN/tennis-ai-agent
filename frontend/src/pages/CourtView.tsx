import { Crosshair, MapTrifold, Path, TennisBall } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { CourtDiagram } from "../components/CourtDiagram";
import { DecisionBadge } from "../components/DecisionBadge";
import { Card, EmptyState, PageHeader, SectionHeading, StatusBadge } from "../components/ui";
import { formatConfidence, formatNumber, humanize } from "../lib/format";

export default function CourtView() {
  const { run, selectedEventId, selectEvent } = useAnalysis();
  if (!run) return null;
  const selectedEvent = run.events.find((event) => event.event_id === selectedEventId) ?? run.events[0] ?? null;
  const selectedCall = run.lineCalls.find((call) => call.event_id === selectedEvent?.event_id) ?? null;

  return (
    <div className="page court-page">
      <PageHeader eyebrow="Court plane / canonical geometry" title="Court view" description="Movement and ball evidence projected onto a regulation 10.97 m by 23.77 m doubles court. Geometry remains proportional at every viewport." action={<StatusBadge tone={run.court?.isValid ? "signal" : "neutral"}>{run.court?.isValid ? "Valid homography" : "Geometry unavailable"}</StatusBadge>} />
      {!run.court ? <EmptyState title="Court geometry unavailable" detail="This run has no court_geometry.json artifact." /> : (
        <div className="court-workspace">
          <Card className="court-primary-panel">
            <div className="court-panel-toolbar">
              <span><MapTrifold size={17} /> Canonical evidence</span>
              <div className="court-legend"><span><i className="legend-player-one" /> Player 1</span><span><i className="legend-player-two" /> Player 2</span><span><i className="legend-ball" /> Ball</span></div>
            </div>
            <CourtDiagram run={run} selectedEventId={selectedEvent?.event_id} />
          </Card>
          <aside className="court-evidence-panel">
            <Card className="panel-padding">
              <SectionHeading eyebrow="Evidence selector" title="Semantic events" detail="Crosses mark contacts. Pluses mark bounces." />
              <div className="court-event-list">{run.events.map((event) => <button className={event.event_id === selectedEvent?.event_id ? "court-event-active" : ""} type="button" key={event.event_id} onClick={() => selectEvent(event.event_id)}><span><Crosshair size={16} /><strong>{humanize(event.event_type)}</strong></span><small>F{event.frame} · {formatConfidence(event.confidence)}</small></button>)}</div>
            </Card>
            <Card className="panel-padding selected-evidence-card">
              <SectionHeading eyebrow="Selected event" title={selectedEvent ? humanize(selectedEvent.event_type) : "None"} detail={selectedEvent ? `Frame ${selectedEvent.frame} at ${selectedEvent.timestamp_s.toFixed(3)} s` : "No event artifact"} />
              {selectedEvent ? <dl className="compact-facts"><div><dt>Court coordinate</dt><dd>{selectedEvent.court_position_m ? `${selectedEvent.court_position_m[0].toFixed(2)}, ${selectedEvent.court_position_m[1].toFixed(2)} m` : "Unavailable"}</dd></div><div><dt>Tracker state</dt><dd>{humanize(selectedEvent.trajectory_state)}</dd></div><div><dt>Participant</dt><dd>{selectedEvent.player_id ? `Player ${selectedEvent.player_id}` : "Ball bounce"}</dd></div></dl> : null}
              {selectedCall ? <div className="selected-call"><DecisionBadge decision={selectedCall.decision} /><p>{selectedCall.reason}</p><strong>{formatNumber(selectedCall.ball_edge_margin_cm, " cm signed margin")}</strong></div> : <p className="unavailable-copy">No line-call decision is attached to this event.</p>}
            </Card>
            <div className="court-stat-strip"><span><Path size={18} /><strong>{run.trajectories?.ballTrajectory.filter((point) => point.court_x_m != null).length ?? 0}</strong><small>ball positions</small></span><span><TennisBall size={18} /><strong>{run.lineCalls.length}</strong><small>line calls</small></span></div>
          </aside>
        </div>
      )}
    </div>
  );
}


import { CaretLeft, CaretRight, Crosshair, Info } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";
import { VideoEvidencePlayer } from "../components/VideoEvidencePlayer";
import { EmptyState, PageHeader, StatusBadge, UtilityCard } from "../components/ui";
import { getAdjacentEvent } from "../lib/analytics";
import { formatConfidence, humanize } from "../lib/format";

export default function MatchAnalysis() {
  const { run, selectedEventId, selectEvent } = useAnalysis();
  if (!run) return null;
  const selectedEvent = run.events.find((event) => event.event_id === selectedEventId) ?? run.events[0] ?? null;
  const selectAdjacent = (direction: -1 | 1) => {
    const event = getAdjacentEvent(run.events, selectedEvent?.event_id ?? null, direction);
    if (event) selectEvent(event.event_id);
  };

  return (
    <div className="page analysis-page">
      <PageHeader eyebrow="Synchronized evidence" title="Match analysis" description="The original H.264 source stays untouched while the browser draws only the active frame’s detections, geometry, event labels, and short ball trail." action={<StatusBadge tone="signal">{run.summary.frames} frames · {run.summary.fps} FPS</StatusBadge>} />
      {!run.videoUrl ? <EmptyState title="Replay unavailable for this generation" detail="Only the Phase 5 source video is bundled. Select Phase 5 Scoring to inspect synchronized evidence." /> : (
        <div className="analysis-workspace">
          <VideoEvidencePlayer run={run} selectedEventId={selectedEvent?.event_id ?? null} onSelectEvent={selectEvent} />
          <aside className="evidence-sidebar" aria-label="Selected event evidence">
            <div className="evidence-side-head"><span><Crosshair size={16} /> Selected evidence</span><StatusBadge>{selectedEvent ? `Event ${selectedEvent.event_id}` : "None"}</StatusBadge></div>
            {selectedEvent ? <>
              <div className="event-identity"><span>{selectedEvent.timestamp_s.toFixed(3)} s</span><h2>{humanize(selectedEvent.event_type)}</h2><p>Frame {selectedEvent.frame} · Player {selectedEvent.player_id ?? "Unavailable"}</p></div>
              <dl className="evidence-facts">
                <div><dt>Confidence</dt><dd>{formatConfidence(selectedEvent.confidence)}</dd></div>
                <div><dt>Tracker state</dt><dd>{humanize(selectedEvent.trajectory_state)}</dd></div>
                <div><dt>Court X</dt><dd>{selectedEvent.court_position_m?.[0].toFixed(2) ?? "Unavailable"} m</dd></div>
                <div><dt>Court Y</dt><dd>{selectedEvent.court_position_m?.[1].toFixed(2) ?? "Unavailable"} m</dd></div>
              </dl>
              <div className="evidence-step"><button type="button" onClick={() => selectAdjacent(-1)} aria-label="Previous event"><CaretLeft size={17} /> Previous</button><button type="button" onClick={() => selectAdjacent(1)}>Next <CaretRight size={17} /></button></div>
              <div className="evidence-note"><Info size={17} /><p>Continuous playback time stays local. The selected run and event remain encoded in the URL for sharing.</p></div>
            </> : <EmptyState title="No semantic events" detail="This run has no event artifact to step through." />}
            <div className="timeline-record-list">{run.events.map((event) => <button type="button" className={event.event_id === selectedEvent?.event_id ? "timeline-record-selected" : ""} key={event.event_id} onClick={() => selectEvent(event.event_id)}><span>{event.timestamp_s.toFixed(2)}</span><strong>{humanize(event.event_type)}</strong><small>F{event.frame}</small></button>)}</div>
          </aside>
        </div>
      )}
      <div className="analysis-method-grid">
        <UtilityCard><span>Playback source</span><strong>H.264 / AVC</strong><small>Committed browser-compatible demo media</small></UtilityCard>
        <UtilityCard><span>Overlay scheduling</span><strong>Video frame callback</strong><small>Callbacks cancel when the view unmounts</small></UtilityCard>
        <UtilityCard><span>Trajectory window</span><strong>18 frames</strong><small>Short temporal tail only</small></UtilityCard>
      </div>
    </div>
  );
}

